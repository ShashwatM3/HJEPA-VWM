"""Tests for the reconstruction decoder architecture."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def _small_cfg():
    config = importlib.import_module("config")
    cfg = config.Config()
    cfg.model.h = 32
    cfg.model.w = 32
    cfg.model.n_c = 4
    cfg.model.d_e = 16
    cfg.model.d_c = 8
    cfg.model.decoder_dim = 8
    cfg.model.decoder_blocks = 1
    cfg.model.decoder_heads = 2
    return cfg


def test_decoder_uses_fixed_position_buffer_not_learned_output_queries():
    """The decoder may know output positions but cannot learn per-token content queries."""
    models = importlib.import_module("models")
    cfg = _small_cfg()
    decoder = models.Decoder(cfg.model)

    named_parameters = dict(decoder.named_parameters())
    named_buffers = dict(decoder.named_buffers())

    assert "queries" not in named_parameters
    assert "fixed_pos" in named_buffers
    assert "fixed_pos" not in named_parameters
    assert decoder.fixed_pos.shape == (cfg.model.n_ctx, cfg.model.decoder_dim)
    assert decoder.fixed_pos.requires_grad is False


def test_decoder_position_does_not_create_output_content_by_itself():
    """Identical c-derived values must decode identically across output positions.

    This catches the specific loophole from the proposal: if fixed position is ever
    added into the hidden stream as content, a zero latent can emit position-specific
    features. With position used only as an attention query, identical values from
    `c` remain identical no matter how position changes the attention weights.
    """
    models = importlib.import_module("models")
    cfg = _small_cfg()
    decoder = models.Decoder(cfg.model)
    latent = torch.zeros(2, cfg.model.n_c, cfg.model.d_c)

    pred_detailed = decoder(latent)
    centered = pred_detailed - pred_detailed[:, :1]

    assert pred_detailed.shape == (2, cfg.model.n_ctx, cfg.model.d_e)
    assert centered.abs().max().item() < 1e-6


def test_decoder_reconstruction_gradients_reach_latent_and_trainable_decoder_weights():
    """The fixed-position buffer is static, while D and its c input remain trainable."""
    models = importlib.import_module("models")
    cfg = _small_cfg()
    decoder = models.Decoder(cfg.model)
    latent = torch.randn(2, cfg.model.n_c, cfg.model.d_c, requires_grad=True)
    target = torch.randn(2, cfg.model.n_ctx, cfg.model.d_e)

    loss = (decoder(latent) - target).pow(2).mean()
    loss.backward()

    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert latent.grad.abs().sum().item() > 0.0
    assert decoder.fixed_pos.grad is None
    assert any(param.grad is not None for param in decoder.parameters())
