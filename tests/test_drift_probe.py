"""Unit tests for the within-video temporal drift probe (drift_probe.py).

Covers the pure computation layer only: offset parsing, window indexing, probe
selection determinism, drift matrices, aggregation, Spearman, checkpoint-config
rebuild, and the latent path through a real (small-geometry) Bottleneck. No
dataset, frozen encoder, checkpoint file, or matplotlib is touched.
"""

from __future__ import annotations

import importlib
import math

import pytest

torch = pytest.importorskip("torch")


def _probe():
    return importlib.import_module("drift_probe")


def test_parse_offset_list_sorts_and_dedupes():
    probe = _probe()
    assert probe.parse_offset_list("12,4,4,2") == [2, 4, 12]
    with pytest.raises(ValueError):
        probe.parse_offset_list("4,-2")
    with pytest.raises(ValueError):
        probe.parse_offset_list("")


def test_overlap_boundary_and_default_graph2_offsets():
    probe = _probe()
    # v0.2 contract: 8 frames at stride 2 -> windows share frames up to k=14.
    assert probe.overlap_boundary(8, 2) == 14
    offsets = [2, 4, 8, 12, 16, 24, 32]
    assert probe.default_graph2_offsets(offsets, 8, 2) == [16, 24, 32]
    # Fallback: when every offset overlaps, keep them all rather than none.
    assert probe.default_graph2_offsets([2, 4], 8, 2) == [2, 4]


def test_window_frame_indices_match_training_contract():
    probe = _probe()
    # Window ending at frame 20: 8 frames at stride 2 -> 6, 8, ..., 20.
    assert probe.window_frame_indices(20, 8, 2, 100) == [6, 8, 10, 12, 14, 16, 18, 20]
    # Defensive clamp mirrors data.py pad-by-repeat; never negative or past the end.
    clamped = probe.window_frame_indices(4, 8, 2, 6)
    assert clamped[0] == 0 and clamped[-1] == 4 and all(0 <= i <= 5 for i in clamped)


def test_select_probe_order_is_deterministic_and_seed_sensitive():
    probe = _probe()
    paths = [f"validation/{i}.webm" for i in range(50)]
    assert probe.select_probe_order(paths, 42) == probe.select_probe_order(paths, 42)
    assert probe.select_probe_order(paths, 42) != probe.select_probe_order(paths, 7)
    # Permutation, no loss (compare as sets/sorted both sides: "10" < "5" lexicographically,
    # so the numerically-ordered construction list is NOT sorted() order).
    assert sorted(probe.select_probe_order(paths, 42)) == sorted(paths)
    assert probe.select_probe_order(paths, 42) is not paths  # input never mutated in place


def test_rank_and_drift_share_one_default_feature_cache_path(tmp_path):
    """Both offline tools must converge on the same cache without re-encoding."""
    probe = _probe()
    tag = "siglip2_vitb16_ssv2_validation_n64_seed42"
    expected = tmp_path / f"encoder_features_{tag}.pt"
    assert probe.default_feature_cache_path(tmp_path, tag) == expected


def test_drift_matrix_recovers_known_cosine_distances():
    probe = _probe()
    videos = [{"path": "a.webm", "anchor_end": 14}]
    offsets = [4, 8]
    ex = torch.tensor([1.0, 0.0])
    # anchor -> ex; k=4 -> same direction (drift 0); k=8 -> orthogonal (drift 1).
    units = {
        probe.feature_key("a.webm", 14): ex,
        probe.feature_key("a.webm", 18): ex.clone(),
        probe.feature_key("a.webm", 22): torch.tensor([0.0, 1.0]),
    }
    drift = probe.drift_matrix(units, videos, offsets)
    assert drift.shape == (1, 2)
    assert math.isclose(drift[0, 0].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(drift[0, 1].item(), 1.0, abs_tol=1e-6)


def test_pooled_and_flat_unit_vectors_are_unit_norm():
    probe = _probe()
    tokens = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))
    pooled = probe.pooled_unit_vector(tokens)
    assert pooled.shape == (32,)
    assert math.isclose(pooled.norm().item(), 1.0, rel_tol=1e-5)
    latent = torch.randn(4, 8, generator=torch.Generator().manual_seed(1))
    flat = probe.flat_unit_vector(latent)
    assert flat.shape == (32,)
    assert math.isclose(flat.norm().item(), 1.0, rel_tol=1e-5)


def test_summaries_aggregate_the_right_axis():
    probe = _probe()
    drift = torch.tensor([[0.0, 1.0], [1.0, 3.0]])  # (V=2, K=2)
    per_offset = probe.summarize_per_offset(drift)
    assert per_offset["mean"] == [0.5, 2.0]
    per_video = probe.per_video_summary(drift, [4, 8], [8])
    assert per_video.tolist() == [1.0, 3.0]
    both = probe.per_video_summary(drift, [4, 8], [4, 8])
    assert both.tolist() == [0.5, 2.0]


def test_spearman_correlation_endpoints():
    probe = _probe()
    a = torch.tensor([0.1, 0.2, 0.5, 0.9])
    up = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(probe.spearman_correlation(a, up), 1.0, abs_tol=1e-6)
    assert math.isclose(probe.spearman_correlation(a, -up), -1.0, abs_tol=1e-6)
    # Constant input is degenerate: average ranks make it rank-constant -> 0, never
    # a fabricated correlation (the naive argsort ranking bug).
    assert probe.spearman_correlation(a, torch.zeros(4)) == 0.0
    assert probe.spearman_correlation(torch.zeros(4), a) == 0.0


def test_spearman_correlation_handles_ties_with_average_ranks():
    probe = _probe()
    tied = torch.tensor([1.0, 2.0, 2.0, 3.0])
    # Identical tied vectors are perfectly rank-correlated.
    assert math.isclose(probe.spearman_correlation(tied, tied.clone()), 1.0, abs_tol=1e-6)
    ranks = probe._average_ranks(tied)
    assert ranks.tolist() == [0.0, 1.5, 1.5, 3.0]  # tie group shares its mean rank
    assert probe._average_ranks(torch.zeros(3)).tolist() == [1.0, 1.0, 1.0]


def test_model_config_from_checkpoint_dict_filters_and_applies_fields():
    probe = _probe()
    cfg = probe.model_config_from_checkpoint_dict(
        {"n_c": 64, "decoder_dim": 512, "some_removed_legacy_field": 7}
    )
    assert cfg.n_c == 64
    assert cfg.decoder_dim == 512
    assert cfg.d_c == 256  # missing fields fall back to current defaults
    assert probe.model_config_from_checkpoint_dict({}).n_c == 32  # empty dict is safe


def test_latent_drift_through_real_bottleneck_small_geometry():
    """End-to-end latent path on a tiny config: cached features -> B -> drift."""
    probe = _probe()
    from config import ModelConfig
    from models import Bottleneck

    cfg = ModelConfig(h=64, w=64)  # n_ctx = (8/2) * (64/16)^2 = 64 tokens
    torch.manual_seed(0)
    bottleneck = Bottleneck(cfg)
    videos = [
        {"path": "a.webm", "anchor_end": 14},
        {"path": "b.webm", "anchor_end": 20},
    ]
    offsets = [4]
    features = {}
    for video in videos:
        for end in (video["anchor_end"], video["anchor_end"] + 4):
            features[probe.feature_key(video["path"], end)] = torch.randn(
                cfg.n_ctx, cfg.d_e, dtype=torch.float16
            )
    keys = list(features.keys())
    units = probe.compute_latent_unit_vectors(
        bottleneck, features, keys, torch.device("cpu"), latent_batch=2
    )
    assert set(units) == set(keys)
    assert all(u.shape == (cfg.n_c * cfg.d_c,) for u in units.values())
    drift = probe.drift_matrix(units, videos, offsets)
    assert drift.shape == (2, 1)
    assert torch.isfinite(drift).all()
    assert (drift >= -1e-5).all() and (drift <= 2.0 + 1e-5).all()
    # The measurement path is no-grad: no gradient may accumulate on the module.
    assert all(p.grad is None for p in bottleneck.parameters())
    assert not bottleneck.training  # compute_latent_unit_vectors pins eval mode


def test_checkpoint_whitener_identity_is_verified_before_drift_use(tmp_path):
    """Latent drift refuses a self-inconsistent embedded whitening transform."""
    probe = _probe()
    import train
    from config import Config
    from models import FeatureWhitener, _legacy_encoder_spec, build_phase1_modules

    cfg = Config()
    cfg.model.h = 64
    cfg.model.w = 64
    cfg.model.n_c = 4
    cfg.model.d_c = 8
    cfg.model.bottleneck_mixer_dim = 8
    cfg.model.bottleneck_cross_attn_heads = 2
    cfg.model.bottleneck_latent_blocks = 1
    cfg.model.f_c_blocks = 1
    cfg.model.f_c_heads = 2
    cfg.model.decoder_dim = 8
    cfg.model.decoder_blocks = 1
    cfg.model.decoder_heads = 2
    cfg.train.whiten_features = True
    built = build_phase1_modules(cfg, load_encoder=False)

    class SpecOnlyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = _legacy_encoder_spec(cfg.model)

    modules = (SpecOnlyEncoder(), *built[1:])
    optimizer = train.make_optimizer(modules[1], modules[3], modules[4], cfg)
    whitener = FeatureWhitener(cfg.model.d_e)
    whitener.configure(
        torch.zeros(cfg.model.d_e),
        torch.ones(cfg.model.d_e),
        torch.eye(cfg.model.d_e),
        1e-4,
    )
    path = tmp_path / "checkpoint.pt"
    train.save_checkpoint(path, 1, modules, optimizer, cfg, whitener=whitener)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["feature_whitener"]["mean"][0] = 7.0
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match="feature_whitener identity"):
        probe.load_bottleneck_from_checkpoint(path)
