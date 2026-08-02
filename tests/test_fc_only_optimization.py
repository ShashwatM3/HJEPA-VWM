"""Behavioral contracts for fixed-representation F_c-only training."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class IdentityEncoder(nn.Module):
    """Frozen-encoder stand-in returning already encoded feature tokens."""

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """Return pre-encoded tokens without changing their values.

        Args:
            clip: Synthetic ``(B, N_e, D_e)`` detailed features.
        Returns:
            The same detailed-feature tensor.
        """
        return clip


def _small_fc_only_run() -> tuple[object, tuple[nn.Module, ...], torch.optim.Optimizer]:
    """Build a CPU-sized Fc-only recipe through the production module interfaces.

    Returns:
        Configuration, canonical module bundle, and Fc-only AdamW optimizer.
    """
    config = importlib.import_module("config")
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = config.Config()
    cfg.model.h = 32
    cfg.model.w = 32
    cfg.model.n_c = 4
    cfg.model.d_e = 16
    cfg.model.d_c = 8
    cfg.model.bottleneck_mixer_dim = 8
    cfg.model.bottleneck_cross_attn_heads = 2
    cfg.model.f_c_blocks = 1
    cfg.model.f_c_heads = 2
    cfg.model.decoder_dim = 8
    cfg.model.decoder_blocks = 1
    cfg.model.decoder_heads = 2
    cfg.train.optimization_scope = "fc_only"
    cfg.train.predict_residual = True
    cfg.train.lambda_var = 0.5
    cfg.train.lambda_cov = 0.01
    cfg.train.lambda_recon = 1.0
    cfg.train.lambda_recon_pred = 0.0
    cfg.train.recon_warmup_steps = 0
    cfg.train.precision = "fp32"
    _, bottleneck, target_bottleneck, coarse_flow, decoder = models.build_phase1_modules(
        cfg, load_encoder=False
    )
    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    modules = (IdentityEncoder(), bottleneck, target_bottleneck, coarse_flow, decoder)
    return cfg, modules, optimizer


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone one module state for independent byte-level change assertions.

    Args:
        module: Module whose parameters and buffers should be snapshotted.
    Returns:
        Detached cloned state tensors keyed by canonical state-dict name.
    """
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def test_fc_only_step_updates_only_fc_and_optimizes_only_flow(monkeypatch):
    """One paid-equivalent step leaves B, B_EMA, and D byte-identical."""
    train = importlib.import_module("train")
    cfg, modules, optimizer = _small_fc_only_run()
    _, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    before = {
        "B": _state(bottleneck),
        "B_EMA": _state(target_bottleneck),
        "F_c": _state(coarse_flow),
        "D": _state(decoder),
    }
    frozen_hashes = train.capture_frozen_state_hashes(modules, cfg)
    monkeypatch.setattr(
        train,
        "_update_ema",
        lambda *_args, **_kwargs: pytest.fail("fc_only attempted an EMA update"),
    )
    batch = (
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
    )

    metrics = train.train_step(
        batch,
        modules,
        optimizer,
        step=100,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    train.verify_frozen_state_hashes(modules, cfg, frozen_hashes)

    for name, module in (("B", bottleneck), ("B_EMA", target_bottleneck), ("D", decoder)):
        assert all(
            torch.equal(value, before[name][key]) for key, value in module.state_dict().items()
        )
    assert any(
        not torch.equal(value, before["F_c"][key])
        for key, value in coarse_flow.state_dict().items()
    )
    assert all(param.grad is None for param in bottleneck.parameters())
    assert all(param.grad is None for param in target_bottleneck.parameters())
    assert all(param.grad is None for param in decoder.parameters())
    assert any(param.grad is not None for param in coarse_flow.parameters())
    assert metrics["loss"] == pytest.approx(metrics["L_flow"])
    assert set(frozen_hashes) == {"B", "B_EMA", "D"}


def test_fc_only_frozen_state_guard_rejects_mutation():
    """Checkpoint publication fails if any supposedly frozen module changes."""
    train = importlib.import_module("train")
    cfg, modules, _ = _small_fc_only_run()
    frozen_hashes = train.capture_frozen_state_hashes(modules, cfg)
    decoder = modules[4]
    with torch.no_grad():
        next(decoder.parameters()).add_(1.0)

    with pytest.raises(RuntimeError, match="frozen module state changed.*D"):
        train.verify_frozen_state_hashes(modules, cfg, frozen_hashes)


def test_fc_only_resume_rejects_tampered_frozen_state_before_loading(tmp_path):
    """A checkpoint cannot redefine the fixed coordinates and become its own baseline."""
    train = importlib.import_module("train")
    cfg, modules, optimizer = _small_fc_only_run()
    frozen_hashes = train.capture_frozen_state_hashes(modules, cfg)
    provenance = {
        "schema": "hjepa-run-provenance-v1",
        "common_identity": "fixed-run",
        "common": {"frozen_state_hashes": frozen_hashes},
        "optimization_contract": {"scope": "fc_only"},
        "frozen_state_hashes": frozen_hashes,
    }
    path = tmp_path / "tampered.pt"
    train.save_checkpoint(
        path,
        1,
        modules,
        optimizer,
        cfg,
        run_provenance=provenance,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    first_key = next(iter(checkpoint["decoder"]))
    checkpoint["decoder"][first_key] = checkpoint["decoder"][first_key] + 1.0
    torch.save(checkpoint, path)

    _, fresh_modules, fresh_optimizer = _small_fc_only_run()
    fresh_before = _state(fresh_modules[1])
    with pytest.raises(RuntimeError, match="frozen-state hash mismatch.*D"):
        train.load_checkpoint(
            path,
            fresh_modules,
            fresh_optimizer,
            expected_run_provenance=provenance,
        )
    assert all(
        torch.equal(value, fresh_before[name])
        for name, value in fresh_modules[1].state_dict().items()
    )
