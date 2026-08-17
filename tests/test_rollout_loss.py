"""Contracts for the two-step differentiable rollout endpoint loss."""

from __future__ import annotations

import importlib
import sys

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class IdentityEncoder(nn.Module):
    """Return synthetic detailed tokens as already-encoded features."""

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """Return the input detailed-token tensor unchanged."""
        return clip


def _fixed_present_bundle():
    """Build a small production-path fixed-present F_c-only configuration."""
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
    cfg.train.flow_source = "present"
    cfg.train.flow_bottleneck_checkpoint = "fixed-contract-active"
    cfg.train.lambda_rollout = 0.2
    cfg.train.rollout_ramp_steps = 1_500
    cfg.train.lambda_var = 0.0
    cfg.train.lambda_cov = 0.0
    cfg.train.lambda_recon = 0.0
    cfg.train.precision = "fp32"
    _, bottleneck, target_bottleneck, coarse_flow, decoder = models.build_phase1_modules(
        cfg, load_encoder=False
    )
    for parameter in bottleneck.parameters():
        parameter.requires_grad_(False)
    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    modules = (IdentityEncoder(), bottleneck, target_bottleneck, coarse_flow, decoder)
    return cfg, modules, optimizer


def test_two_step_helper_preserves_midpoint_gradient_and_uses_fixed_times():
    """The second F_c call backpropagates through the generated first Euler state."""
    train = importlib.import_module("train")

    class RecordingFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.25))
            self.inputs = []
            self.times = []

        def forward(self, state, tau, condition, *, condition_drop):
            self.inputs.append(state)
            self.times.append(tau.detach().clone())
            return self.weight * state + condition

    flow = RecordingFlow()
    present = torch.randn(2, 3, 4)
    endpoint, first_displacement, second_displacement = train.two_step_rollout_endpoint(
        flow, present, present
    )
    midpoint = flow.inputs[1]
    midpoint.retain_grad()
    endpoint.square().mean().backward()

    assert torch.equal(flow.times[0], torch.zeros(2))
    assert torch.equal(flow.times[1], torch.full((2,), 0.5))
    assert midpoint.grad is not None
    assert torch.isfinite(midpoint.grad).all()
    assert flow.weight.grad is not None and torch.isfinite(flow.weight.grad)
    assert endpoint.shape == first_displacement.shape == second_displacement.shape == present.shape


def test_train_step_adds_ramped_rollout_loss_and_updates_only_fc():
    """The enabled experiment adds its exact weighted loss without unfreezing B or D."""
    train = importlib.import_module("train")
    cfg, modules, optimizer = _fixed_present_bundle()
    _, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    batch = (
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
    )

    metrics = train.train_step(
        batch, modules, optimizer, step=750, cfg=cfg, device=torch.device("cpu")
    )

    assert metrics["lambda_rollout_effective"] == pytest.approx(0.1)
    assert metrics["weighted_rollout_loss"] == pytest.approx(0.1 * metrics["L_rollout"])
    assert metrics["loss"] == pytest.approx(metrics["L_flow"] + metrics["weighted_rollout_loss"])
    assert metrics["rollout_2step_endpoint_mse"] == pytest.approx(metrics["L_rollout"])
    if metrics["rollout_2step_copy_ratio_valid"]:
        assert metrics["rollout_2step_copy_ratio"] == pytest.approx(
            metrics["L_rollout"] / metrics["rollout_2step_copy_mse"]
        )
    else:
        assert metrics["rollout_2step_copy_mse"] <= 1e-8
        assert torch.isnan(torch.tensor(metrics["rollout_2step_copy_ratio"]))
    assert all(parameter.grad is None for parameter in bottleneck.parameters())
    assert all(parameter.grad is None for parameter in target_bottleneck.parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())
    assert any(parameter.grad is not None for parameter in coarse_flow.parameters())


def test_zero_weight_does_not_execute_rollout_forwards(monkeypatch):
    """The shipped zero default preserves the baseline F_c call graph."""
    train = importlib.import_module("train")
    cfg, modules, optimizer = _fixed_present_bundle()
    cfg.train.lambda_rollout = 0.0
    monkeypatch.setattr(
        train,
        "two_step_rollout_endpoint",
        lambda *_args, **_kwargs: pytest.fail("zero-weight rollout executed"),
    )
    batch = (
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
    )

    metrics = train.train_step(
        batch, modules, optimizer, step=0, cfg=cfg, device=torch.device("cpu")
    )

    assert metrics["L_rollout"] == 0.0
    assert metrics["lambda_rollout_effective"] == 0.0
    assert metrics["weighted_rollout_loss"] == 0.0


def test_rollout_config_requires_fixed_present_fc_only_and_valid_ramp(tmp_path):
    """An active rollout weight cannot escape the verified experiment contract."""
    config = importlib.import_module("config")
    train = importlib.import_module("train")
    checkpoint = tmp_path / "fixed.pt"
    checkpoint.touch()
    cfg = config.Config()
    cfg.train.lambda_rollout = 0.1
    cfg.train.flow_bottleneck_checkpoint = str(checkpoint)

    with pytest.raises(ValueError, match="fixed present-source"):
        train.finalize_training_config(cfg)

    cfg.train.flow_source = "present"
    with pytest.raises(ValueError, match="optimization_scope=fc_only"):
        train.finalize_training_config(cfg)

    cfg.train.optimization_scope = "fc_only"
    cfg.train.rollout_ramp_steps = -1
    with pytest.raises(ValueError, match="rollout_ramp_steps"):
        train.finalize_training_config(cfg)


def test_rollout_cli_flags_parse(monkeypatch):
    """Both approved experiment controls are available as direct CLI overrides."""
    train = importlib.import_module("train")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--lambda-rollout", "0.1", "--rollout-ramp-steps", "1500"],
    )
    args = train.parse_args()
    assert args.lambda_rollout == pytest.approx(0.1)
    assert args.rollout_ramp_steps == 1_500
