"""Unit tests for the frozen-encoder effective-rank probe (rank_probe.py).

Covers only the pure computation layer and manifest-default contract. No SSv2
dataset, frozen encoder, feature cache, checkpoint file, or matplotlib is touched.
"""

from __future__ import annotations

import importlib
import math

import pytest

torch = pytest.importorskip("torch")


def _probe():
    return importlib.import_module("rank_probe")


def test_entropy_effective_rank_delegates_to_diagnostics_formula():
    probe = _probe()
    from diagnostics import effective_rank

    rows = torch.randn(17, 5, generator=torch.Generator().manual_seed(0))
    expected = effective_rank(rows[None])["c_effective_rank"]
    assert probe.entropy_effective_rank(rows) == expected


def test_covariance_spectrum_matches_effective_rank_formula():
    probe = _probe()
    rows = torch.randn(31, 7, generator=torch.Generator().manual_seed(1))

    spectrum = probe.covariance_spectrum(rows)
    assert spectrum.shape == (7,)
    assert torch.all(spectrum[:-1] >= spectrum[1:])

    probs = spectrum / spectrum.sum().clamp_min(1e-8)
    entropy_rank = float(torch.exp(-(probs * (probs + 1e-8).log()).sum()).item())
    assert math.isclose(entropy_rank, probe.entropy_effective_rank(rows), rel_tol=1e-6)


def test_rank_at_energy_counts_leading_eigenvalues():
    probe = _probe()
    spectrum = torch.tensor([4.0, 1.0, 0.0])
    assert probe.rank_at_energy(spectrum, 0.80) == 1
    assert probe.rank_at_energy(spectrum, 0.81) == 2
    assert probe.rank_at_energy(torch.zeros(3), 0.90) == 0
    assert probe.rank_at_energy(torch.tensor([float("nan"), 1.0]), 0.90) == 0


def test_rank_report_returns_json_ready_views_and_ceilings():
    probe = _probe()
    generator = torch.Generator().manual_seed(2)
    video_tokens = [torch.randn(4, 3, generator=generator) for _ in range(5)]

    report = probe.rank_report(video_tokens)

    assert report["n_videos"] == 5
    assert report["n_ctx"] == 4
    assert report["d_e"] == 3
    assert report["e_effective_rank_ceiling"] == 3
    assert report["e_within_video_rank_ceiling"] == 3
    assert report["e_cross_video_rank_ceiling"] == 4
    assert len(report["e_within_video_rank_per_video"]) == 5
    assert len(report["pooled_spectrum"]) == 3
    assert 1 <= report["pooled_rank_at_90pct_energy"] <= 3
    assert 1 <= report["pooled_rank_at_99pct_energy"] <= 3
    assert math.isfinite(report["e_effective_rank"])
    assert math.isfinite(report["e_within_video_rank_mean"])
    assert math.isfinite(report["e_cross_video_rank"])


def test_rank_report_rejects_empty_or_mismatched_inputs():
    probe = _probe()

    with pytest.raises(ValueError, match="at least one"):
        probe.rank_report([])
    with pytest.raises(ValueError, match=r"\(N_ctx, D_e\)"):
        probe.rank_report([torch.randn(3)])
    with pytest.raises(ValueError, match="same shape"):
        probe.rank_report([torch.randn(4, 3), torch.randn(5, 3)])


def test_rank_probe_manifest_default_matches_drift_probe_default_offsets():
    probe = _probe()
    drift_probe = importlib.import_module("drift_probe")

    assert probe.DEFAULT_MANIFEST_MAX_OFFSET == max(drift_probe.DEFAULT_OFFSETS)
