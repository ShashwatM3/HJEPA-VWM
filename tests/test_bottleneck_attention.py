"""Bottleneck attention contracts for the sharpened Perceiver-style latent stack."""

from __future__ import annotations

import importlib
import math

import pytest

torch = pytest.importorskip("torch")


def _small_model_cfg(latent_blocks: int = 3, mixer_dim: int = 8):
    """Return a tiny bottleneck config with the same shape relationships as Phase 1."""
    config = importlib.import_module("config")
    cfg = config.Config()
    cfg.model.h = 32
    cfg.model.w = 32
    cfg.model.n_c = 4
    cfg.model.d_e = 16
    cfg.model.d_c = 8
    cfg.model.bottleneck_mixer_dim = mixer_dim
    cfg.model.bottleneck_cross_attn_heads = 2
    cfg.model.bottleneck_latent_blocks = latent_blocks
    return cfg.model


def test_bottleneck_uses_mixer_dim_for_the_complete_internal_stream():
    """A wide bottleneck keeps memory and slots wide until the external code projection."""
    models = importlib.import_module("models")
    cfg = _small_model_cfg(latent_blocks=2, mixer_dim=16)

    bottleneck = models.Bottleneck(cfg)

    assert bottleneck.to_kv.in_features == 16
    assert bottleneck.to_kv.out_features == 16
    assert bottleneck.queries.shape == (cfg.n_c, 16)
    assert all(block.norm_cross.normalized_shape == (16,) for block in bottleneck.latent_blocks)
    assert bottleneck.abstract_proj.in_features == 16
    assert bottleneck.abstract_proj.out_features == cfg.d_c


def test_wide_bottleneck_opens_an_input_dependent_path_after_one_update():
    """Wide zero-init residuals start stable, then learn input-dependent external codes."""
    torch.manual_seed(0)
    models = importlib.import_module("models")
    cfg = _small_model_cfg(latent_blocks=2, mixer_dim=16)
    bottleneck = models.Bottleneck(cfg)
    optimizer = torch.optim.SGD(bottleneck.parameters(), lr=0.1)
    detailed = torch.randn(2, cfg.n_ctx, cfg.d_e)
    target = torch.randn(2, cfg.n_c, cfg.d_c)

    before = bottleneck(detailed)
    assert before.shape == (2, cfg.n_c, cfg.d_c)
    assert torch.isfinite(before).all()
    assert torch.allclose(before[0], before[1], atol=1e-6)

    torch.nn.functional.mse_loss(before, target).backward()
    assert bottleneck.abstract_proj.weight.grad.abs().sum().item() > 0.0
    assert all(
        block.cross_attn.o_proj.weight.grad.abs().sum().item() > 0.0
        for block in bottleneck.latent_blocks
    )
    optimizer.step()

    after = bottleneck(detailed)
    assert not torch.allclose(after[0], after[1], atol=1e-6)


@pytest.mark.parametrize("latent_blocks", [1, 2, 3])
def test_bottleneck_starts_as_normalized_slot_identities(latent_blocks):
    """At init, every zero-init residual branch leaves only normalized query slots."""
    models = importlib.import_module("models")
    cfg = _small_model_cfg(latent_blocks)
    bottleneck = models.Bottleneck(cfg)
    detailed_a = torch.randn(2, cfg.n_ctx, cfg.d_e)
    detailed_b = torch.randn(2, cfg.n_ctx, cfg.d_e)

    out_a, attn = bottleneck(detailed_a, return_attn=True)
    out_b = bottleneck(detailed_b)
    expected = bottleneck.norm(bottleneck.queries)[None].expand_as(out_a)

    assert len(bottleneck.latent_blocks) == latent_blocks
    assert out_a.shape == (2, cfg.n_c, cfg.d_c)
    assert attn.shape == (2, cfg.bottleneck_cross_attn_heads, cfg.n_c, cfg.n_ctx)
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-6)
    assert torch.allclose(out_a, expected, atol=1e-6)
    assert torch.allclose(out_b, expected, atol=1e-6)
    assert isinstance(bottleneck.abstract_proj, torch.nn.Identity)
    assert not any(key.startswith("abstract_proj.") for key in bottleneck.state_dict())
    assert bottleneck.pos_emb.shape == (1, cfg.n_ctx, cfg.bottleneck_mixer_dim)
    assert bottleneck.pos_emb.float().std(unbiased=False).item() > 0.1
    for block in bottleneck.latent_blocks:
        assert block.cross_attn.logit_scale.exp().item() == pytest.approx(1 / 0.07)
        assert block.cross_attn.o_proj.weight.abs().sum().item() == 0.0
        assert block.self_attn.out_proj.weight.abs().sum().item() == 0.0
        assert block.mlp[-1].weight.abs().sum().item() == 0.0


def test_latent_blocks_read_compete_refine_all_open_with_training():
    """Every residual branch (cross-read, slot self-attention, MLP) receives gradient."""
    models = importlib.import_module("models")
    cfg = _small_model_cfg(2)
    bottleneck = models.Bottleneck(cfg)
    detailed = torch.randn(2, cfg.n_ctx, cfg.d_e)
    target = torch.randn(2, cfg.n_c, cfg.d_c)

    loss = torch.nn.functional.mse_loss(bottleneck(detailed), target)
    loss.backward()

    for block in bottleneck.latent_blocks:
        assert block.cross_attn.o_proj.weight.grad.abs().sum().item() > 0.0
        assert block.self_attn.out_proj.weight.grad.abs().sum().item() > 0.0
        assert block.mlp[-1].weight.grad.abs().sum().item() > 0.0


def test_sharp_attention_temperature_gets_gradient_after_zero_bridge_opens():
    """The learned sharpness knobs participate in training once o_proj has moved."""
    models = importlib.import_module("models")
    cfg = _small_model_cfg(2)
    bottleneck = models.Bottleneck(cfg)
    detailed = torch.randn(2, cfg.n_ctx, cfg.d_e)
    target = torch.randn(2, cfg.n_c, cfg.d_c)
    optimizer = torch.optim.SGD(bottleneck.parameters(), lr=0.1)

    first_loss = torch.nn.functional.mse_loss(bottleneck(detailed), target)
    first_loss.backward()
    for block in bottleneck.latent_blocks:
        assert block.cross_attn.o_proj.weight.grad.abs().sum().item() > 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = torch.nn.functional.mse_loss(bottleneck(detailed), target)
    second_loss.backward()
    for block in bottleneck.latent_blocks:
        grad = block.cross_attn.logit_scale.grad
        assert grad is not None
        assert math.isfinite(float(grad.item()))
        assert abs(float(grad.item())) > 0.0


def test_bottleneck_rejects_nonpositive_latent_block_count():
    """A latent-processor depth below 1 is a configuration error, not a silent no-op."""
    models = importlib.import_module("models")
    cfg = _small_model_cfg(0)
    with pytest.raises(ValueError, match="bottleneck_latent_blocks"):
        models.Bottleneck(cfg)


def test_slot_diversity_logs_raw_and_centered_versions():
    """Slot diversity exposes the mechanical raw rank and the mean-removed rank."""
    diagnostics = importlib.import_module("diagnostics")
    abstract = torch.randn(3, 4, 8)

    metrics = diagnostics.slot_diversity_rank(abstract)

    assert set(metrics) == {"c_slot_diversity_rank", "c_slot_diversity_rank_centered"}
    assert all(math.isfinite(value) for value in metrics.values())
