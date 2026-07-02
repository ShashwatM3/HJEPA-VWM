"""Bottleneck attention contracts for the sharpened slot readout."""

from __future__ import annotations

import importlib
import math

import pytest

torch = pytest.importorskip("torch")


def _small_model_cfg():
    """Return a tiny bottleneck config with the same shape relationships as Phase 1."""
    config = importlib.import_module("config")
    cfg = config.Config()
    cfg.model.h = 32
    cfg.model.w = 32
    cfg.model.n_c = 4
    cfg.model.d_e = 16
    cfg.model.d_c = 8
    cfg.model.bottleneck_mixer_dim = 8
    cfg.model.bottleneck_cross_attn_heads = 2
    return cfg.model


def test_bottleneck_starts_as_normalized_slot_identities():
    """At init, the zero attention bridge leaves only normalized query slots."""
    models = importlib.import_module("models")
    cfg = _small_model_cfg()
    bottleneck = models.Bottleneck(cfg)
    detailed_a = torch.randn(2, cfg.n_ctx, cfg.d_e)
    detailed_b = torch.randn(2, cfg.n_ctx, cfg.d_e)

    out_a, attn = bottleneck(detailed_a, return_attn=True)
    out_b = bottleneck(detailed_b)
    expected = bottleneck.norm(bottleneck.queries)[None].expand_as(out_a)

    assert out_a.shape == (2, cfg.n_c, cfg.d_c)
    assert attn.shape == (2, cfg.bottleneck_cross_attn_heads, cfg.n_c, cfg.n_ctx)
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-6)
    assert torch.allclose(out_a, expected, atol=1e-6)
    assert torch.allclose(out_b, expected, atol=1e-6)
    assert bottleneck.pos_emb.shape == (1, cfg.n_ctx, cfg.bottleneck_mixer_dim)
    assert bottleneck.pos_emb.float().std(unbiased=False).item() > 0.1
    assert bottleneck.cross_attn.logit_scale.exp().item() == pytest.approx(1 / 0.07)


def test_sharp_attention_temperature_gets_gradient_after_zero_bridge_opens():
    """The learned sharpness knob participates in training once o_proj has moved."""
    models = importlib.import_module("models")
    cfg = _small_model_cfg()
    bottleneck = models.Bottleneck(cfg)
    detailed = torch.randn(2, cfg.n_ctx, cfg.d_e)
    target = torch.randn(2, cfg.n_c, cfg.d_c)
    optimizer = torch.optim.SGD(bottleneck.parameters(), lr=0.1)

    first_loss = torch.nn.functional.mse_loss(bottleneck(detailed), target)
    first_loss.backward()
    assert bottleneck.cross_attn.o_proj.weight.grad.abs().sum().item() > 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = torch.nn.functional.mse_loss(bottleneck(detailed), target)
    second_loss.backward()
    grad = bottleneck.cross_attn.logit_scale.grad
    assert grad is not None
    assert math.isfinite(float(grad.item()))
    assert abs(float(grad.item())) > 0.0


def test_slot_diversity_logs_raw_and_centered_versions():
    """Slot diversity exposes the mechanical raw rank and the mean-removed rank."""
    diagnostics = importlib.import_module("diagnostics")
    abstract = torch.randn(3, 4, 8)

    metrics = diagnostics.slot_diversity_rank(abstract)

    assert set(metrics) == {"c_slot_diversity_rank", "c_slot_diversity_rank_centered"}
    assert all(math.isfinite(value) for value in metrics.values())
