"""Tests for the reconstruction decoder architecture."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def _encoder_spec(*, frame_based: bool):
    from encoders import EncoderSpec, FeatureLayout

    layout = (
        FeatureLayout(8, 16, 16, "time_y_x", "frame", 1, 1)
        if frame_based
        else FeatureLayout(4, 16, 16, "time_y_x", "tubelet", 2, 2)
    )
    return EncoderSpec(
        family="fake-frame" if frame_based else "fake-tubelet",
        repo_id="offline/fake",
        requested_revision="a" * 40,
        resolved_revision="a" * 40,
        input_frames=8,
        input_height=256,
        input_width=256,
        feature_dim=768 if frame_based else 1024,
        layout=layout,
        normalization_id="test",
        normalization_mean=(0.0, 0.0, 0.0),
        normalization_std=(1.0, 1.0, 1.0),
        preprocess_version="test-v1",
        inference_precision="fp32",
        frame_microbatch=8,
        attention_implementation="sdpa",
        cache_dir="/tmp/cache",
        parameter_count=0,
    )


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


@pytest.mark.parametrize("frame_based", [False, True])
def test_bottleneck_and_decoder_resolve_all_detailed_geometry_from_encoder_spec(frame_based):
    """Tubelet and frame layouts traverse the same B/D modules without config geometry."""
    models = importlib.import_module("models")
    cfg = _small_cfg()
    spec = _encoder_spec(frame_based=frame_based)
    bottleneck = models.Bottleneck(cfg.model, spec)
    decoder = models.Decoder(cfg.model, spec)
    detailed = torch.randn(1, spec.layout.n_tokens, spec.feature_dim)

    abstract = bottleneck(detailed)
    reconstructed = decoder(abstract)

    assert abstract.shape == (1, cfg.model.n_c, cfg.model.d_c)
    assert reconstructed.shape == (1, spec.layout.n_tokens, spec.feature_dim)
    assert decoder.fixed_pos.shape == (spec.layout.n_tokens, cfg.model.decoder_dim)
