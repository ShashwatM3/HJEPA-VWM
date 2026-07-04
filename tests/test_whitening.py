"""Fixed offline feature-whitening contracts (tmp/changes_bottleneck <2>)."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def _small_cfg():
    """Tiny Phase 1 config shared by the whitening wiring tests."""
    config = importlib.import_module("config")
    cfg = config.Config()
    cfg.model.h = 32
    cfg.model.w = 32
    cfg.model.n_c = 4
    cfg.model.d_e = 16
    cfg.model.d_c = 8
    cfg.model.bottleneck_mixer_dim = 8
    cfg.model.bottleneck_cross_attn_heads = 2
    cfg.model.bottleneck_latent_blocks = 2
    cfg.model.f_c_blocks = 1
    cfg.model.f_c_heads = 2
    cfg.model.decoder_dim = 8
    cfg.model.decoder_blocks = 1
    cfg.model.decoder_heads = 2
    return cfg


def _configured_whitener(d: int, rows: int = 4096, seed: int = 0):
    """Build a whitener from synthetic anisotropic data; return (whitener, data)."""
    models = importlib.import_module("models")
    torch.manual_seed(seed)
    scales = torch.linspace(0.05, 3.0, d)
    data = torch.randn(rows, d) * scales + torch.arange(d).float() * 0.1
    eigvals, eigvecs = torch.linalg.eigh(torch.cov(data.t()))
    whitener = models.FeatureWhitener(d)
    whitener.configure(data.mean(dim=0), eigvals, eigvecs, eps=1e-6)
    return whitener, data


def test_whitener_has_no_parameters_and_requires_configure():
    """Buffers only (never optimizable), and use-before-configure fails loudly."""
    models = importlib.import_module("models")
    whitener = models.FeatureWhitener(16)
    assert sum(1 for _ in whitener.parameters()) == 0
    with pytest.raises(RuntimeError, match="configure"):
        whitener.whiten(torch.randn(2, 16))


def test_whitener_round_trip_and_unit_covariance():
    """whiten/unwhiten invert each other and whitened stats are ~N(0, I)."""
    whitener, data = _configured_whitener(16)
    x = torch.randn(3, 5, 16) * 2.0 + 1.0

    assert torch.allclose(whitener.unwhiten(whitener.whiten(x)), x, atol=1e-4)

    whitened = whitener.whiten(data)
    assert whitened.mean(dim=0).abs().max().item() < 1e-3
    cov = torch.cov(whitened.t())
    assert torch.allclose(cov, torch.eye(16), atol=0.05)


def test_whitener_preserves_dtype_and_rejects_bad_stats():
    """fp32 math returns the input dtype; malformed stats/eps fail at configure."""
    models = importlib.import_module("models")
    whitener, _ = _configured_whitener(16)
    half = torch.randn(2, 16, dtype=torch.float16)
    assert whitener.whiten(half).dtype == torch.float16

    fresh = models.FeatureWhitener(16)
    with pytest.raises(ValueError, match="do not match"):
        fresh.configure(torch.zeros(8), torch.ones(8), torch.eye(8), eps=1e-4)
    with pytest.raises(ValueError, match="eps"):
        fresh.configure(torch.zeros(16), torch.ones(16), torch.eye(16), eps=0.0)


def test_finalize_training_config_validates_whitening():
    """whiten_features demands a stats path and a positive eigenvalue floor."""
    train = importlib.import_module("train")
    cfg = _small_cfg()
    cfg.train.whiten_features = True
    cfg.train.whiten_stats_path = ""
    with pytest.raises(ValueError, match="whiten-stats-path"):
        train.finalize_training_config(cfg)
    cfg.train.whiten_stats_path = "/tmp/stats.pt"
    cfg.train.whiten_eps = 0.0
    with pytest.raises(ValueError, match="whiten_eps"):
        train.finalize_training_config(cfg)
    cfg.train.whiten_eps = 1e-4
    train.finalize_training_config(cfg)


def test_build_whitener_loads_stats_file_and_checks_shape(tmp_path):
    """_build_whitener returns None when off, loads valid stats, rejects mismatches."""
    train = importlib.import_module("train")
    cfg = _small_cfg()
    device = torch.device("cpu")

    assert train._build_whitener(cfg, device) is None

    cfg.train.whiten_features = True
    cfg.train.whiten_stats_path = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="whiten_stats.py"):
        train._build_whitener(cfg, device)

    d = cfg.model.d_e
    torch.manual_seed(0)
    data = torch.randn(2048, d) * torch.linspace(0.1, 2.0, d)
    eigvals, eigvecs = torch.linalg.eigh(torch.cov(data.t()))
    stats_path = tmp_path / "stats.pt"
    torch.save({"mean": data.mean(dim=0), "eigvals": eigvals, "eigvecs": eigvecs}, stats_path)
    cfg.train.whiten_stats_path = str(stats_path)
    whitener = train._build_whitener(cfg, device)
    assert bool(whitener.initialized)

    torch.save(
        {"mean": torch.zeros(d + 1), "eigvals": torch.ones(d + 1), "eigvecs": torch.eye(d + 1)},
        stats_path,
    )
    with pytest.raises(ValueError, match="d_e"):
        train._build_whitener(cfg, device)


class _IdentityEncoder(torch.nn.Module):
    """Stand-in frozen encoder mapping a synthetic clip batch to fixed features."""

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, clip):
        return self.features[: clip.shape[0]]


def test_train_step_whitens_both_branches_and_checkpoints_the_whitener(tmp_path):
    """The whitened space reaches B, recon targets, and the checkpoint round-trips."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg()
    cfg.train.whiten_features = True
    cfg.train.whiten_stats_path = "unused-in-this-test"
    cfg.train.lambda_recon = 0.05
    cfg.train.recon_warmup_steps = 0
    device = torch.device("cpu")
    torch.manual_seed(0)

    whitener, _ = _configured_whitener(cfg.model.d_e)
    _, bottleneck, target_bottleneck, coarse_flow, decoder = models.build_phase1_modules(
        cfg, load_encoder=False
    )
    target_bottleneck.copy_weights_from(bottleneck)
    features = torch.randn(2, cfg.model.n_ctx, cfg.model.d_e) * 3.0 + 0.5
    encoder = _IdentityEncoder(features)
    modules = (encoder, bottleneck, target_bottleneck, coarse_flow, decoder)
    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    batch = (
        torch.randn(2, cfg.model.t_ctx, 3, cfg.model.h, cfg.model.w),
        torch.randn(2, cfg.model.t_ctx, 3, cfg.model.h, cfg.model.w),
    )

    # Missing whitener while the flag is on must fail loudly, not silently train raw.
    with pytest.raises(RuntimeError, match="whitener"):
        train.train_step(batch, modules, optimizer, 0, cfg, device)

    metrics = train.train_step(batch, modules, optimizer, 0, cfg, device, whitener=whitener)
    assert metrics["whiten_active"] == 1.0
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values() if isinstance(v, float))

    # run_diagnostics enforces the same pairing and runs in whitened space.
    with pytest.raises(RuntimeError, match="whitener"):
        train.run_diagnostics(batch, modules, cfg, device)
    diag = train.run_diagnostics(batch, modules, cfg, device, whitener=whitener)
    assert "L_recon_present" in diag

    # The checkpoint carries the whitener so offline evaluators reproduce the space.
    path = tmp_path / "ckpt.pt"
    train.save_checkpoint(path, 1, modules, optimizer, cfg, whitener=whitener)
    fresh = models.FeatureWhitener(cfg.model.d_e)
    train.load_checkpoint(path, modules, whitener=fresh)
    assert bool(fresh.initialized)
    assert torch.allclose(fresh.whiten_mat, whitener.whiten_mat)

    # The bottleneck consumed WHITENED features: its latent for whitened input must
    # match a manual whitened forward, not the raw-feature forward.
    with torch.no_grad():
        manual = bottleneck(whitener.whiten(features))
        raw = bottleneck(features)
        from_step = bottleneck(whitener.whiten(features))
    assert torch.allclose(manual, from_step)
    assert not torch.allclose(manual, raw, atol=1e-3)
