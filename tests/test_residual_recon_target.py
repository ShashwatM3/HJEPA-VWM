"""Tests for the residual reconstruction target (investigation_013, the run-052 fix)."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class IdentityEncoder(nn.Module):
    """Frozen-encoder stand-in that returns feature-shaped clips unchanged."""

    def forward(self, clip):
        return clip


class CountingIdentityEncoder(nn.Module):
    """Frozen-encoder stand-in that records diagnostic forward calls."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, clip):
        self.inputs.append(clip)
        return clip.clone()


def _small_cfg(present_only: bool = True):
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
    cfg.train.present_recon_only = present_only
    cfg.train.lambda_recon = 1.0
    cfg.train.lambda_recon_pred = 0.0
    cfg.train.lambda_var = 0.0
    cfg.train.lambda_sigreg = 0.0
    cfg.train.recon_warmup_steps = 0
    cfg.train.recon_residual_target = True
    return cfg


def _build(cfg, seed: int = 0):
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    torch.manual_seed(seed)
    _, bottleneck, target_bottleneck, coarse_flow, decoder = models.build_phase1_modules(
        cfg, load_encoder=False
    )
    optimizer = train.make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    modules = (IdentityEncoder(), bottleneck, target_bottleneck, coarse_flow, decoder)
    tracker = models.FeatureMeanTracker(
        cfg.model.n_ctx, cfg.model.d_e, cfg.train.recon_mean_momentum
    )
    return modules, optimizer, tracker


def test_tracker_first_update_copies_then_ema_lerps():
    """The first batch initializes the mean directly; later batches EMA toward it."""
    models = importlib.import_module("models")
    tracker = models.FeatureMeanTracker(3, 4, momentum=0.9)
    assert not bool(tracker.initialized)

    first = torch.randn(2, 3, 4)
    tracker.update(first)
    assert bool(tracker.initialized)
    assert torch.allclose(tracker.mean, first.mean(dim=0))

    second = torch.randn(5, 3, 4)
    expected = 0.9 * first.mean(dim=0) + 0.1 * second.mean(dim=0)
    tracker.update(second)
    assert torch.allclose(tracker.mean, expected, atol=1e-6)


def test_tracker_has_no_parameters_and_subtract_keeps_dtype():
    """The tracker can never enter an optimizer, and residuals match input dtype."""
    models = importlib.import_module("models")
    tracker = models.FeatureMeanTracker(3, 4, momentum=0.99)
    assert sum(1 for _ in tracker.parameters()) == 0
    assert not tracker.mean.requires_grad

    features = torch.randn(2, 3, 4)
    tracker.update(features)
    residual = tracker.subtract(features.to(torch.bfloat16))
    assert residual.dtype == torch.bfloat16
    exact = tracker.subtract(features)
    assert torch.allclose(exact, features - tracker.mean, atol=1e-6)


def test_tracker_rejects_bad_momentum_and_bad_shapes():
    """Construction and update fail loudly instead of corrupting the mean."""
    models = importlib.import_module("models")
    with pytest.raises(ValueError, match="recon_mean_momentum"):
        models.FeatureMeanTracker(3, 4, momentum=1.0)

    tracker = models.FeatureMeanTracker(3, 4, momentum=0.99)
    with pytest.raises(ValueError, match="Expected"):
        tracker.update(torch.randn(2, 5, 4))


def test_train_step_residual_mode_uses_tracker_and_changes_the_target():
    """Residual mode warms the mean and produces a different L_recon than absolute."""
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=True)
    train.finalize_training_config(cfg)
    batch = (
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
    )

    modules, optimizer, tracker = _build(cfg, seed=0)
    metrics = train.train_step(
        batch,
        modules,
        optimizer,
        step=100,
        cfg=cfg,
        device=torch.device("cpu"),
        mean_tracker=tracker,
    )
    assert bool(tracker.initialized)
    assert metrics["recon_target_residual"] == 1.0
    assert metrics["recon_mean_norm"] > 0.0
    assert metrics["L_recon"] > 0.0

    cfg_abs = _small_cfg(present_only=True)
    cfg_abs.train.recon_residual_target = False
    train.finalize_training_config(cfg_abs)
    modules_abs, optimizer_abs, _ = _build(cfg_abs, seed=0)
    metrics_abs = train.train_step(
        batch, modules_abs, optimizer_abs, step=100, cfg=cfg_abs, device=torch.device("cpu")
    )
    assert metrics_abs["recon_target_residual"] == 0.0
    assert metrics_abs["recon_mean_norm"] == 0.0
    # Same seed, same batch, different target: the losses must differ.
    assert metrics["L_recon"] != pytest.approx(metrics_abs["L_recon"])


def test_train_step_residual_mode_requires_tracker():
    """Residual mode without a tracker fails fast instead of training absolute."""
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=True)
    train.finalize_training_config(cfg)
    modules, optimizer, _ = _build(cfg)
    batch = (
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(2, cfg.model.n_ctx, cfg.model.d_e),
    )

    with pytest.raises(RuntimeError, match="mean_tracker"):
        train.train_step(batch, modules, optimizer, step=0, cfg=cfg, device=torch.device("cpu"))


def test_full_prediction_residual_mode_reaches_fc_through_pred_anchor():
    """Residual targets keep the option-3 gradient contract: F_c and D both train."""
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=False)
    cfg.train.lambda_recon_pred = 1.0
    train.finalize_training_config(cfg)
    modules, optimizer, tracker = _build(cfg)
    _, _, _, coarse_flow, decoder = modules
    fc_before = {name: param.detach().clone() for name, param in coarse_flow.named_parameters()}
    decoder_before = {name: param.detach().clone() for name, param in decoder.named_parameters()}
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
        mean_tracker=tracker,
    )

    assert metrics["recon_target_residual"] == 1.0
    assert metrics["L_recon"] > 0.0
    assert metrics["L_recon_pred"] > 0.0
    assert any(
        not torch.equal(param, fc_before[name]) for name, param in coarse_flow.named_parameters()
    )
    assert any(
        not torch.equal(param, decoder_before[name]) for name, param in decoder.named_parameters()
    )


def test_finalize_rejects_residual_target_without_recon_anchor():
    """The flag with no active reconstruction loss is a config error, not a no-op."""
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=False)
    cfg.train.lambda_recon = 0.0
    cfg.train.lambda_recon_pred = 0.0

    with pytest.raises(ValueError, match="recon-residual-target"):
        train.finalize_training_config(cfg)


def test_finalize_rejects_out_of_range_momentum():
    """A momentum outside (0, 1) fails before an expensive launch."""
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=True)
    cfg.train.recon_mean_momentum = 1.5

    with pytest.raises(ValueError, match="recon_mean_momentum"):
        train.finalize_training_config(cfg)


def test_checkpoint_round_trips_tracker_and_tolerates_old_checkpoints(tmp_path):
    """The mean survives save/load; checkpoints without it load without crashing."""
    models = importlib.import_module("models")
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=True)
    train.finalize_training_config(cfg)
    modules, optimizer, tracker = _build(cfg)
    tracker.update(torch.randn(4, cfg.model.n_ctx, cfg.model.d_e))

    path = tmp_path / "with_tracker.pt"
    train.save_checkpoint(path, 7, modules, optimizer, cfg, mean_tracker=tracker)
    fresh = models.FeatureMeanTracker(cfg.model.n_ctx, cfg.model.d_e, cfg.train.recon_mean_momentum)
    assert train.load_checkpoint(path, modules, optimizer, mean_tracker=fresh) == 7
    assert bool(fresh.initialized)
    assert torch.allclose(fresh.mean, tracker.mean)

    old_path = tmp_path / "without_tracker.pt"
    train.save_checkpoint(old_path, 3, modules, optimizer, cfg)  # no tracker key
    untouched = models.FeatureMeanTracker(
        cfg.model.n_ctx, cfg.model.d_e, cfg.train.recon_mean_momentum
    )
    assert train.load_checkpoint(old_path, modules, optimizer, mean_tracker=untouched) == 3
    assert not bool(untouched.initialized)


def test_diagnostics_score_residual_target_and_log_video_gap():
    """Present-only diagnostics use the residual target and emit the honesty probe."""
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=True)
    train.finalize_training_config(cfg)
    modules, optimizer, tracker = _build(cfg)
    batch = (
        torch.randn(4, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(4, cfg.model.n_ctx, cfg.model.d_e),
    )
    train.train_step(
        batch,
        modules,
        optimizer,
        step=100,
        cfg=cfg,
        device=torch.device("cpu"),
        mean_tracker=tracker,
    )
    mean_before = tracker.mean.clone()

    metrics = train.run_diagnostics(batch, modules, cfg, torch.device("cpu"), mean_tracker=tracker)

    # Diagnostics read the tracker but never update it (no val-batch leakage).
    assert torch.equal(tracker.mean, mean_before)
    for key in ("L_recon_present", "L_recon_shuffled_c", "L_recon_video_gap"):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key]))
    assert metrics["L_recon_video_gap"] == pytest.approx(
        metrics["L_recon_shuffled_c"] - metrics["L_recon_present"]
    )

    with pytest.raises(RuntimeError, match="mean_tracker"):
        train.run_diagnostics(batch, modules, cfg, torch.device("cpu"))


def test_full_prediction_diagnostics_emit_residual_readouts():
    """The full-prediction readouts score residual targets and include the probe."""
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=False)
    cfg.train.lambda_recon_pred = 1.0
    train.finalize_training_config(cfg)
    modules, optimizer, tracker = _build(cfg)
    batch = (
        torch.randn(4, cfg.model.n_ctx, cfg.model.d_e),
        torch.randn(4, cfg.model.n_ctx, cfg.model.d_e),
    )
    train.train_step(
        batch,
        modules,
        optimizer,
        step=100,
        cfg=cfg,
        device=torch.device("cpu"),
        mean_tracker=tracker,
    )

    metrics = train.run_diagnostics(batch, modules, cfg, torch.device("cpu"), mean_tracker=tracker)

    for key in (
        "L_recon_present",
        "L_recon_cplus",
        "L_recon_chat",
        "L_recon_shuffled_c",
        "L_recon_video_gap",
    ):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key]))


@pytest.mark.parametrize("present_only, expected_encoder_calls", [(True, 1), (False, 2)])
def test_diagnostics_measure_encoder_and_latent_cosine_without_extra_forward(
    monkeypatch, present_only, expected_encoder_calls
):
    """Both cosine keys use their intended tensors from the same diagnostic pass."""
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=present_only)
    if not present_only:
        cfg.train.lambda_recon_pred = 1.0
    train.finalize_training_config(cfg)
    modules, _, tracker = _build(cfg)
    encoder = CountingIdentityEncoder()
    modules = (encoder, *modules[1:])
    batch = (
        torch.randn(4, cfg.model.n_ctx, cfg.model.d_e, requires_grad=True),
        torch.randn(4, cfg.model.n_ctx, cfg.model.d_e, requires_grad=True),
    )
    calls = []

    def cosine_spy(representation):
        calls.append(representation)
        value = 0.25 if representation.shape[1:] == batch[0].shape[1:] else 0.75
        return {"c_cross_video_cosine": value}

    monkeypatch.setattr(train, "cross_video_cosine", cosine_spy)

    metrics = train.run_diagnostics(batch, modules, cfg, torch.device("cpu"), mean_tracker=tracker)

    assert metrics["e_cross_video_cosine"] == pytest.approx(0.25)
    assert metrics["c_cross_video_cosine"] == pytest.approx(0.75)
    assert len(calls) == 2
    detailed, abstract = calls
    assert detailed.shape == (4, cfg.model.n_ctx, cfg.model.d_e)
    assert abstract.shape == (4, cfg.model.n_c, cfg.model.d_c)
    assert detailed is not abstract
    assert all(not representation.requires_grad for representation in calls)
    assert all(representation.grad_fn is None for representation in calls)
    assert len(encoder.inputs) == expected_encoder_calls
    assert encoder.inputs[0].data_ptr() == batch[0].data_ptr()
    if not present_only:
        assert encoder.inputs[1].data_ptr() == batch[1].data_ptr()


def test_default_config_keeps_baseline_untouched():
    """Flag off: no tracker use, no residual metrics, identical mode flags."""
    train = importlib.import_module("train")
    cfg = _small_cfg(present_only=True)
    cfg.train.recon_residual_target = False
    train.finalize_training_config(cfg)
    modules, optimizer, tracker = _build(cfg)
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
        mean_tracker=tracker,
    )

    assert not bool(tracker.initialized)  # never updated on the baseline
    assert metrics["recon_target_residual"] == 0.0
    assert metrics["recon_mean_norm"] == 0.0
