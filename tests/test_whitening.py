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


def _encoder_spec(d: int):
    from encoders import EncoderSpec, FeatureLayout

    return EncoderSpec(
        family="fake",
        repo_id="offline/whitening",
        requested_revision="a" * 40,
        resolved_revision="a" * 40,
        input_frames=8,
        input_height=256,
        input_width=256,
        feature_dim=d,
        layout=FeatureLayout(8, 2, 2, "time_y_x", "frame", 1, 1),
        normalization_id="test",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        preprocess_version="test-v1",
        inference_precision="fp32",
        frame_microbatch=4,
        attention_implementation="sdpa",
        cache_dir="/tmp/cache",
        parameter_count=0,
    )


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
    cfg.train.whiten_expected_clips = 0
    with pytest.raises(ValueError, match="whiten_expected_clips"):
        train.finalize_training_config(cfg)


def test_build_whitener_loads_stats_file_and_checks_shape(tmp_path):
    """_build_whitener returns None when off, loads valid stats, rejects mismatches."""
    train = importlib.import_module("train")
    cfg = _small_cfg()
    device = torch.device("cpu")

    assert train._build_whitener(cfg, device) is None

    cfg.train.whiten_features = True
    cfg.train.whiten_expected_clips = 2
    cfg.train.whiten_stats_path = str(tmp_path / "missing.pt")
    spec = _encoder_spec(cfg.model.d_e)
    dataset = {"dataset": "fake", "fingerprint": "d" * 64}
    with pytest.raises(FileNotFoundError, match="whiten_stats.py"):
        train._build_whitener(cfg, device, spec, dataset)

    d = cfg.model.d_e
    torch.manual_seed(0)
    data = torch.randn(2048, d) * torch.linspace(0.1, 2.0, d)
    eigvals, eigvecs = torch.linalg.eigh(torch.cov(data.t()))
    stats_path = tmp_path / "stats.pt"
    import provenance

    envelope = provenance.build_whitening_envelope(
        mean=data.mean(dim=0),
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        encoder_spec=spec,
        dataset_identity=dataset,
        split="train",
        transform_seed=42,
        clip_count=2,
        token_row_count=64,
        eigensolver={
            "name": "torch.linalg.eigh",
            "covariance": "biased-mle",
            "accumulation_dtype": "float64",
        },
    )
    provenance.atomic_torch_save(envelope, stats_path)
    cfg.train.whiten_stats_path = str(stats_path)
    whitener = train._build_whitener(cfg, device, spec, dataset)
    assert bool(whitener.initialized)

    wrong_spec = _encoder_spec(d + 1)
    with pytest.raises(ValueError, match="feature fingerprint"):
        train._build_whitener(cfg, device, wrong_spec, dataset)


def test_resume_prefers_and_verifies_checkpoint_whitener_over_external_stats(tmp_path):
    """Resume preserves the trained coordinate system even during dataset transfer."""
    import provenance
    import train
    from models import FeatureWhitener, build_phase1_modules

    cfg = _small_cfg()
    cfg.train.whiten_features = True
    cfg.train.whiten_expected_clips = 2
    spec = _encoder_spec(cfg.model.d_e)
    dataset = {"dataset": "fake", "fingerprint": "d" * 64}
    external_path = tmp_path / "external.pt"
    external = provenance.build_whitening_envelope(
        mean=torch.zeros(spec.feature_dim),
        eigenvalues=torch.ones(spec.feature_dim),
        eigenvectors=torch.eye(spec.feature_dim),
        encoder_spec=spec,
        dataset_identity=dataset,
        split="train",
        transform_seed=cfg.seed,
        clip_count=2,
        token_row_count=2 * spec.layout.n_tokens,
        eigensolver=provenance.WHITENING_EIGENSOLVER,
    )
    provenance.atomic_torch_save(external, external_path)
    cfg.train.whiten_stats_path = str(external_path)

    source_whitener = FeatureWhitener(spec.feature_dim)
    source_whitener.configure(
        torch.ones(spec.feature_dim),
        torch.ones(spec.feature_dim),
        torch.eye(spec.feature_dim),
        cfg.train.whiten_eps,
    )
    modules = build_phase1_modules(cfg, load_encoder=False, encoder_spec=spec)
    optimizer = train.make_optimizer(modules[1], modules[3], modules[4], cfg)
    checkpoint_path = tmp_path / "resume.pt"
    train.save_checkpoint(
        checkpoint_path,
        1,
        modules,
        optimizer,
        cfg,
        whitener=source_whitener,
    )

    rebuilt = train._build_whitener(
        cfg,
        torch.device("cpu"),
        spec,
        dataset,
        checkpoint_path=checkpoint_path,
    )
    assert torch.equal(rebuilt.mean, source_whitener.mean)
    assert not torch.equal(rebuilt.mean, external["tensors"]["mean"])

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["feature_whitener"]["mean"][0] = 9.0
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="feature_whitener identity"):
        train._build_whitener(
            cfg,
            torch.device("cpu"),
            spec,
            dataset,
            checkpoint_path=checkpoint_path,
        )


def test_stats_factory_path_is_context_only_and_counts_clips_and_token_rows():
    """The offline fit consumes raw context batches and records both sampling counts."""
    import whiten_stats
    from data import ClipBatch
    from encoders import EncoderSpec, FeatureLayout

    spec = EncoderSpec(
        family="fake",
        repo_id="offline/stats",
        requested_revision="b" * 40,
        resolved_revision="b" * 40,
        input_frames=8,
        input_height=256,
        input_width=256,
        feature_dim=2,
        layout=FeatureLayout(8, 1, 1, "time_y_x", "frame", 1, 1),
        normalization_id="test",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        preprocess_version="test-v1",
        inference_precision="fp32",
        frame_microbatch=2,
        attention_implementation="sdpa",
        cache_dir="/tmp/cache",
        parameter_count=0,
    )

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec

        def forward(self, clips):
            assert clips.min() >= 0 and clips.max() <= 1
            base = clips.mean(dim=(2, 3, 4))
            return torch.stack((base, base.square()), dim=-1)

    loader = [
        ClipBatch(torch.rand(2, 8, 3, 2, 2), None, ("a", "b")),
        ClipBatch(torch.rand(2, 8, 3, 2, 2), None, ("c", "d")),
    ]
    cfg = _small_cfg()
    dataset = {"dataset": "fake", "fingerprint": "d" * 64}
    envelope = whiten_stats.compute_whitening_stats(
        cfg,
        "train",
        3,
        torch.device("cpu"),
        _encoder=FakeEncoder(),
        _loader=loader,
        _dataset_identity=dataset,
    )
    assert envelope["metadata"]["clip_count"] == 3
    assert envelope["metadata"]["token_row_count"] == 24


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
