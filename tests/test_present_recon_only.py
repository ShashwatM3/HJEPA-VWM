"""Tests for the present-only reconstruction experiment switch."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class CountingIdentityEncoder(nn.Module):
    """Tiny frozen encoder stand-in that records how many clips were encoded."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, clip):
        self.calls += 1
        return clip


def _small_cfg():
    config = importlib.import_module("config")
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
    cfg.train.present_recon_only = True
    cfg.train.lambda_recon = 1.0
    cfg.train.lambda_recon_pred = 1.0  # ignored by finalize_training_config
    cfg.train.lambda_var = 0.0
    cfg.train.lambda_sigreg = 0.0
    cfg.train.recon_warmup_steps = 0
    return cfg


def test_present_recon_only_skips_prediction_path_and_leaves_fc_unchanged():
    """The switch trains present D(B(e_t))->e_t without touching future/F_c."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    train.finalize_training_config(cfg)

    encoder = CountingIdentityEncoder()
    _, bottleneck, target_bottleneck, coarse_flow, decoder = models.build_phase1_modules(
        cfg, load_encoder=False
    )
    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    modules = (encoder, bottleneck, target_bottleneck, coarse_flow, decoder)
    batch = (
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
    )
    fc_before = {name: param.detach().clone() for name, param in coarse_flow.named_parameters()}
    decoder_before = {name: param.detach().clone() for name, param in decoder.named_parameters()}

    metrics = train.train_step(
        batch, modules, optimizer, step=100, cfg=cfg, device=torch.device("cpu")
    )

    assert encoder.calls == 1
    assert metrics["present_recon_only"] == 1.0
    assert metrics["prediction_active"] == 0.0
    assert metrics["L_flow"] == 0.0
    assert metrics["L_recon"] > 0.0
    assert metrics["L_recon_pred"] == 0.0
    assert all(
        torch.equal(param, fc_before[name]) for name, param in coarse_flow.named_parameters()
    )
    assert any(
        not torch.equal(param, decoder_before[name]) for name, param in decoder.named_parameters()
    )


def test_present_recon_only_requires_present_recon_weight():
    """Fail before launch when the mode would otherwise train no reconstruction path."""
    train = importlib.import_module("train")
    cfg = _small_cfg()
    cfg.train.lambda_recon = 0.0

    with pytest.raises(ValueError, match=r"requires train\.lambda_recon > 0"):
        train.finalize_training_config(cfg)
