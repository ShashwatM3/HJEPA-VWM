"""Effective-rank probe for the frozen V-JEPA embeddings `e` (offline diagnostic).

Answers "what is the effective rank of the encoder side?" with the SAME mechanism
the training dashboard uses for `c_effective_rank` (diagnostics.effective_rank:
centered covariance -> eigenvalues -> exp(entropy of the eigenvalue distribution)),
applied to the frozen `e_t` tokens of a fixed probe set instead of `c_t`. The
encoder is frozen, so like the drift probe's e-side curve these numbers are a
constant of the dataset — a rank *budget* to compare `c_effective_rank` against.

Three views, all through the identical formula:

- `e_effective_rank` — the direct analog of `c_effective_rank`: every token of
  every probe video stacked into one (B*N_ctx, D_e) matrix. `c_effective_rank`
  does exactly this with (B*N_c, D_c). Ceiling: D_e = 1024.
- `e_within_video_rank` — rank of one video's own (N_ctx, D_e) tokens, reported
  as mean/min/max over the probe set (plus the per-video list in the JSON). The
  e-side analog of `c_slot_diversity_rank_centered` (centered, like everything
  here). Ceiling: min(N_ctx - 1, D_e) = 1023.
- `e_cross_video_rank` — rank of the B mean-pooled per-video vectors (B, D_e):
  how many independent directions separate whole videos from each other.
  Ceiling: B - 1 = 63 for the default 64-video probe set.

Design contracts (inherited from drift_probe.py):

- Pure ADD-ON analysis helper: training code does not import this file.
- The probe set is pinned by the SAME manifest JSON as the drift probe, and the
  frozen features come from the SAME `vjepa_features_<tag>.pt` cache. With the
  default --out-dir this is a free cache hit after any drift-probe run (only the
  anchor windows are needed, and the drift probe always encodes those); it also
  runs locally on a fetched cache without the dataset or encoder.
- Windows follow the training data contract exactly (via drift_probe helpers).

Outputs (under --out-dir):

- `rank_results_<tag>.json` — all three metrics, per-video ranks, and the pooled
  eigenvalue spectrum with energy-based ranks (rank@90%/99% variance).
- `rank_spectrum_<tag>.png` — pooled eigenvalue spectrum (log y) with the
  effective-rank marker, unless --plot off.

Typical usage (same tag conventions as the drift probe):

    python rank_probe.py --data ssv2 --probe-videos 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError:  # pragma: no cover - local docs-only environments.
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]

from config import Config
from diagnostics import effective_rank
from drift_probe import (
    DEFAULT_OFFSETS,
    _pyplot,
    compute_encoder_features,
    feature_key,
    load_or_build_manifest,
)

DEFAULT_MANIFEST_MAX_OFFSET = max(DEFAULT_OFFSETS)


def _require_torch() -> None:
    """Fail fast when the probe is used without PyTorch installed."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for rank_probe.py. Install requirements.txt on RunPod."
        )


# ---------------------------------------------------------------------------
# Pure computation layer (unit-tested without any dataset or encoder I/O)
# ---------------------------------------------------------------------------


def entropy_effective_rank(rows: Tensor) -> float:
    """Effective rank of a (N, D) sample matrix via the `c_effective_rank` formula.

    Delegates to diagnostics.effective_rank — the exact function behind the
    training metric — so the e-side and c-side numbers can never be computed by
    subtly different math. That function only reshapes to (-1, D), centers, and
    ranks the covariance, so feeding (N, D) directly is equivalent to its
    (B, N_c, D_c) training input.

    Args:
        rows: (N, D) sample matrix (rows are observations).
    Returns:
        exp(entropy(eigenvalue distribution)) of the centered covariance, or NaN
        if the covariance is non-finite.
    """
    _require_torch()
    return effective_rank(rows[None])["c_effective_rank"]


def covariance_spectrum(rows: Tensor) -> Tensor:
    """Eigenvalues of the centered covariance of a (N, D) sample matrix.

    Same centering and covariance as diagnostics.effective_rank (kept in lockstep
    by tests), returned descending for spectrum reporting/plotting.

    Args:
        rows: (N, D) sample matrix.
    Returns:
        (D,) eigenvalues, descending, clamped at 0.
    """
    _require_torch()
    flat = rows.float().reshape(-1, rows.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = flat.t() @ flat / max(1, flat.shape[0] - 1)
    if not torch.isfinite(cov).all():
        return torch.full((flat.shape[-1],), float("nan"), dtype=torch.float32, device=flat.device)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    return eig.flip(0)


def rank_at_energy(spectrum: Tensor, fraction: float) -> int:
    """Smallest number of leading eigenvalues capturing `fraction` of the variance.

    A second, entropy-free readout of the spectrum: rank@0.90 says how many
    principal directions carry 90% of the energy. Complements the entropy
    effective rank, which can be dragged up by a long flat noise tail.

    Args:
        spectrum: (D,) eigenvalues, descending.
        fraction: Target cumulative variance fraction in (0, 1].
    Returns:
        Count of leading eigenvalues needed (0 for an all-zero spectrum).
    """
    _require_torch()
    if not torch.isfinite(spectrum).all():
        return 0
    total = float(spectrum.sum().item())
    if total <= 0:
        return 0
    cumulative = torch.cumsum(spectrum, dim=0) / total
    return int((cumulative < fraction).sum().item()) + 1


def rank_report(video_tokens: list[Tensor]) -> dict:
    """All three effective-rank views over one probe set of frozen features.

    Args:
        video_tokens: One (N_ctx, D_e) frozen-encoder token tensor per probe
            video (the anchor window), any float dtype.
    Returns:
        JSON-serializable dict with pooled/within-video/cross-video effective
        ranks, per-video ranks, ceilings, and the pooled spectrum + energy ranks.
    """
    _require_torch()
    if not video_tokens:
        raise ValueError("rank_report requires at least one video tensor")
    expected_shape = video_tokens[0].shape
    if len(expected_shape) != 2:
        raise ValueError(f"Expected each video tensor to be (N_ctx, D_e); got {expected_shape}")
    for i, tokens in enumerate(video_tokens):
        if tokens.shape != expected_shape:
            raise ValueError(
                f"All video tensors must have the same shape; index 0 has {expected_shape}, "
                f"index {i} has {tokens.shape}"
            )
    stacked = torch.stack([t.float() for t in video_tokens])  # (B, N_ctx, D_e)
    b, n_ctx, d_e = stacked.shape

    pooled_rows = stacked.reshape(b * n_ctx, d_e)
    pooled_rank = entropy_effective_rank(pooled_rows)
    pooled_spectrum = covariance_spectrum(pooled_rows)

    per_video = [entropy_effective_rank(tokens) for tokens in stacked]

    video_means = stacked.mean(dim=1)  # (B, D_e)
    cross_video_rank = entropy_effective_rank(video_means)

    return {
        "n_videos": b,
        "n_ctx": n_ctx,
        "d_e": d_e,
        "e_effective_rank": pooled_rank,
        "e_effective_rank_ceiling": d_e,
        "e_within_video_rank_mean": float(sum(per_video) / len(per_video)),
        "e_within_video_rank_min": float(min(per_video)),
        "e_within_video_rank_max": float(max(per_video)),
        "e_within_video_rank_ceiling": min(n_ctx - 1, d_e),
        "e_within_video_rank_per_video": per_video,
        "e_cross_video_rank": cross_video_rank,
        "e_cross_video_rank_ceiling": b - 1,
        "pooled_rank_at_90pct_energy": rank_at_energy(pooled_spectrum, 0.90),
        "pooled_rank_at_99pct_energy": rank_at_energy(pooled_spectrum, 0.99),
        "pooled_spectrum": [float(v) for v in pooled_spectrum],
    }


# ---------------------------------------------------------------------------
# Plot + CLI
# ---------------------------------------------------------------------------


def plot_spectrum(out_path: Path, report: dict, tag: str) -> None:
    """Pooled eigenvalue spectrum (log y) with the effective-rank marker."""
    plt = _pyplot()
    if plt is None:
        print("matplotlib not available; skipping spectrum plot.")
        return
    spectrum = report["pooled_spectrum"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(spectrum) + 1), spectrum, color="black", lw=1.5)
    ax.set_yscale("log")
    ax.axvline(
        report["e_effective_rank"],
        color="tab:red",
        ls="--",
        label=f"e_effective_rank = {report['e_effective_rank']:.1f}",
    )
    ax.axvline(
        report["pooled_rank_at_99pct_energy"],
        color="tab:blue",
        ls=":",
        label=f"rank@99% energy = {report['pooled_rank_at_99pct_energy']}",
    )
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel("covariance eigenvalue")
    ax.set_title(f"V-JEPA e pooled-token spectrum ({report['n_videos']} videos) — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse the rank-probe CLI (probe-set knobs mirror drift_probe.py)."""
    parser = argparse.ArgumentParser(
        description="Effective rank of frozen V-JEPA embeddings over a fixed probe set."
    )
    parser.add_argument(
        "--data", choices=["ssv2", "ssv2_tiny", "ego4d", "ego4d_tiny"], default="ssv2_tiny"
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Dataset split the probe set is drawn from (default: validation).",
    )
    parser.add_argument(
        "--probe-videos",
        type=int,
        default=64,
        help="Probe set size; only used when building a new manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Probe-selection seed; only used when building a new manifest.",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/drift_probe",
        help="Output directory; defaults to the drift probe's so the "
        "manifest and feature cache are shared.",
    )
    parser.add_argument(
        "--manifest", default=None, help="Probe manifest path (default: derived inside --out-dir)."
    )
    parser.add_argument(
        "--feature-cache",
        default=None,
        help="Encoder feature cache path (default: derived inside --out-dir).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Filename tag (default: dataset/probe/seed derived, matching "
        "drift_probe so cached features are found).",
    )
    parser.add_argument(
        "--manifest-max-offset",
        type=int,
        default=DEFAULT_MANIFEST_MAX_OFFSET,
        help="Largest temporal offset the manifest must support. The rank probe "
        "only encodes anchor windows, but the default matches drift_probe.py's "
        "default offset ladder so a rank-only first run does not create an "
        "incompatible manifest.",
    )
    parser.add_argument(
        "--encoder-batch",
        type=int,
        default=4,
        help="Windows per frozen-encoder forward (cache misses only).",
    )
    parser.add_argument(
        "--plot",
        choices=["on", "off"],
        default="on",
        help="Write the pooled eigenvalue spectrum PNG.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Override device for encoder cache misses and rank math (default: cuda if available).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the rank probe end to end: manifest -> features -> ranks -> JSON + plot."""
    args = parse_args()  # before _require_torch so --help works on torch-less machines
    _require_torch()
    if args.manifest_max_offset < 0:
        raise SystemExit("--manifest-max-offset must be non-negative")
    cfg = Config()
    cfg.data.dataset = args.data
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.data}_{args.split}_n{args.probe_videos}_seed{args.seed}"
    manifest_path = Path(args.manifest) if args.manifest else out_dir / f"manifest_{tag}.json"
    cache_path = (
        Path(args.feature_cache) if args.feature_cache else out_dir / f"vjepa_features_{tag}.pt"
    )

    # Only the anchor windows are needed (offsets=[]), so a manifest/cache left by
    # any drift-probe run over the same tag satisfies this probe without touching
    # the dataset or constructing the encoder.
    manifest = load_or_build_manifest(
        manifest_path, cfg, args.split, args.probe_videos, args.manifest_max_offset, args.seed
    )
    features = compute_encoder_features(cfg, manifest, [], cache_path, device, args.encoder_batch)
    video_tokens = [
        features[feature_key(video["path"], video["anchor_end"])].to(device, dtype=torch.float32)
        for video in manifest["videos"]
    ]

    print(f"Computing effective-rank metrics on {device}...")
    report = rank_report(video_tokens)
    print(
        f"e effective ranks over {report['n_videos']} probe videos "
        f"(anchor windows, N_ctx={report['n_ctx']}, D_e={report['d_e']}):"
    )
    print(
        f"  pooled tokens (c_effective_rank analog): "
        f"{report['e_effective_rank']:.1f} / {report['e_effective_rank_ceiling']}"
    )
    print(
        f"  within-video tokens: mean {report['e_within_video_rank_mean']:.1f} "
        f"(min {report['e_within_video_rank_min']:.1f}, "
        f"max {report['e_within_video_rank_max']:.1f}) "
        f"/ {report['e_within_video_rank_ceiling']}"
    )
    print(
        f"  cross-video pooled vectors: {report['e_cross_video_rank']:.1f} "
        f"/ {report['e_cross_video_rank_ceiling']}"
    )
    print(
        f"  pooled rank@90%/99% energy: {report['pooled_rank_at_90pct_energy']} / "
        f"{report['pooled_rank_at_99pct_energy']}"
    )

    results = {
        "manifest_path": str(manifest_path),
        "feature_cache": str(cache_path),
        "tag": tag,
        **report,
    }
    results_path = out_dir / f"rank_results_{tag}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {results_path}")
    if args.plot == "on":
        plot_spectrum(out_dir / f"rank_spectrum_{tag}.png", report, tag)


if __name__ == "__main__":
    main()
