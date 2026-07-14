"""Effective-rank probe for selected frozen detailed features (offline diagnostic).

Answers "what is the effective rank of the encoder side?" with the SAME mechanism
the training dashboard uses for `c_effective_rank` (diagnostics.effective_rank:
centered covariance -> eigenvalues -> exp(entropy of the eigenvalue distribution)),
applied to the frozen `e_t` tokens of a fixed probe set instead of `c_t`. The
encoder is frozen, so like the drift probe's e-side curve these numbers are a
constant of the dataset — a rank *budget* to compare `c_effective_rank` against.

Three views, all through the identical formula:

- `detailed_effective_rank` — the direct analog of `c_effective_rank`: every token
  of every probe video is stacked into one `(B*N_e,D_e)` matrix. Its ceiling is
  the selected encoder's resolved `D_e`.
- `e_within_video_rank` — rank of one video's own `(N_e,D_e)` tokens, reported
  as mean/min/max over the probe set (plus the per-video list in the JSON). The
  e-side analog of `c_slot_diversity_rank_centered` (centered, like everything
  here). Ceiling: `min(N_e - 1,D_e)`.
- `e_cross_video_rank` — rank of the B mean-pooled per-video vectors (B, D_e):
  how many independent directions separate whole videos from each other.
  Ceiling: B - 1 = 63 for the default 64-video probe set.

Design contracts (inherited from drift_probe.py):

- Pure ADD-ON analysis helper: training code does not import this file.
- The probe set is pinned by the SAME manifest JSON as the drift probe, and the
  frozen features come from the same versioned encoder-feature cache. With the
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
    default_feature_cache_path,
    feature_key,
    load_or_build_manifest,
)
from encoders import FeatureLayout

DEFAULT_MANIFEST_MAX_OFFSET = max(DEFAULT_OFFSETS)


def rank_cache_offsets(manifest_max_offset: int) -> list[int]:
    """Return the shared drift-cache ladder after validating manifest coverage."""
    if manifest_max_offset < DEFAULT_MANIFEST_MAX_OFFSET:
        raise ValueError(
            "--manifest-max-offset must be at least "
            f"{DEFAULT_MANIFEST_MAX_OFFSET} because the shared rank/drift cache records "
            "the default drift offset ladder."
        )
    return list(DEFAULT_OFFSETS)


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


def rank_report(video_tokens: list[Tensor], layout: FeatureLayout | None = None) -> dict:
    """All three effective-rank views over one probe set of frozen features.

    Args:
        video_tokens: One `(N_e,D_e)` frozen-encoder token tensor per probe
            video (the anchor window), any float dtype.
        layout: Optional resolved token lattice. Frame layouts add statistics at
            each temporal index before those frame-token rows are concatenated.
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
    if layout is not None and layout.n_tokens != n_ctx:
        raise ValueError(
            f"Encoder layout token count {layout.n_tokens} does not match feature rows {n_ctx}."
        )

    pooled_rows = stacked.reshape(b * n_ctx, d_e)
    pooled_rank = entropy_effective_rank(pooled_rows)
    pooled_spectrum = covariance_spectrum(pooled_rows)

    per_video = [entropy_effective_rank(tokens) for tokens in stacked]

    video_means = stacked.mean(dim=1)  # (B, D_e)
    cross_video_rank = entropy_effective_rank(video_means)

    report = {
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
        "detailed_raw_rank": int(torch.linalg.matrix_rank(pooled_rows).item()),
        "detailed_effective_rank": pooled_rank,
        "detailed_effective_rank_fraction": pooled_rank / d_e,
        "detailed_feature_dim": d_e,
        "pooled_rank_at_90pct_energy": rank_at_energy(pooled_spectrum, 0.90),
        "pooled_rank_at_99pct_energy": rank_at_energy(pooled_spectrum, 0.99),
        "pooled_spectrum": [float(v) for v in pooled_spectrum],
    }
    if layout is not None:
        report.update(
            {
                "detailed_temporal_unit": layout.temporal_unit,
                "detailed_temporal_count": layout.temporal,
            }
        )
    if layout is not None and layout.temporal_unit == "frame":
        spatial_tokens = layout.height * layout.width
        per_frame = stacked.reshape(b, layout.temporal, spatial_tokens, d_e)
        token_norms = per_frame.norm(dim=-1)
        frame_ranks = [
            entropy_effective_rank(per_frame[:, frame].reshape(b * spatial_tokens, d_e))
            for frame in range(layout.temporal)
        ]
        report.update(
            {
                "detailed_frame_count": layout.temporal,
                "detailed_spatial_tokens_per_frame": spatial_tokens,
                "per_frame_token_norm_mean": [
                    float(value) for value in token_norms.mean(dim=(0, 2))
                ],
                "per_frame_token_norm_std": [
                    float(value) for value in token_norms.std(dim=(0, 2), unbiased=False)
                ],
                "per_frame_effective_rank": frame_ranks,
                "per_frame_effective_rank_fraction": [rank / d_e for rank in frame_ranks],
                "per_frame_effective_rank_mean": float(sum(frame_ranks) / len(frame_ranks)),
                "per_frame_effective_rank_min": float(min(frame_ranks)),
                "per_frame_effective_rank_max": float(max(frame_ranks)),
            }
        )
    return report


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
    ax.set_title(f"Frozen detailed-feature spectrum ({report['n_videos']} videos) — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse the rank-probe CLI (probe-set knobs mirror drift_probe.py)."""
    parser = argparse.ArgumentParser(
        description="Effective rank of frozen detailed features over a fixed probe set."
    )
    parser.add_argument(
        "--data", choices=["ssv2", "ssv2_tiny", "ego4d", "ego4d_tiny"], default="ssv2_tiny"
    )
    parser.add_argument(
        "--encoder",
        choices=("vjepa2_vitl16", "dinov3_vitb16", "siglip2_vitb16"),
        default="vjepa2_vitl16",
    )
    parser.add_argument("--encoder-revision", default=None)
    parser.add_argument("--encoder-precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--encoder-frame-microbatch", type=int, default=None)
    parser.add_argument("--encoder-attention-implementation", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
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
    parser.add_argument("--cache-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--plot",
        choices=["on", "off"],
        default="on",
        help="Write the pooled eigenvalue spectrum PNG.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Override device for encoder cache misses and rank math (default: cuda if available).",
    )
    parser.add_argument("--wandb-artifact", action="store_true")
    parser.add_argument("--wandb-project", default="hjepa-vwm")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the rank probe end to end: manifest -> features -> ranks -> JSON + plot."""
    args = parse_args()  # before _require_torch so --help works on torch-less machines
    _require_torch()
    try:
        cache_offsets = rank_cache_offsets(args.manifest_max_offset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cfg = Config()
    cfg.data.dataset = args.data
    cfg.encoder.alias = args.encoder
    cfg.encoder.revision = args.encoder_revision
    if args.encoder_precision:
        cfg.encoder.precision = args.encoder_precision
    if args.encoder_frame_microbatch is not None:
        cfg.encoder.frame_microbatch = args.encoder_frame_microbatch
    if args.encoder_attention_implementation:
        cfg.encoder.attention_implementation = args.encoder_attention_implementation
    if args.hf_cache_dir:
        cfg.hf_cache_dir = args.hf_cache_dir
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (
        f"{args.encoder}_{args.data}_{args.split}_n{args.probe_videos}_seed{args.seed}"
    )
    manifest_path = Path(args.manifest) if args.manifest else out_dir / f"manifest_{tag}.json"
    cache_path = (
        Path(args.feature_cache) if args.feature_cache else default_feature_cache_path(out_dir, tag)
    )

    # Rank math uses only anchors. The shared cache is nevertheless materialized
    # with drift's default ladder so either tool can validate and reuse the same
    # exact envelope; a strict cache hit still resolves the encoder identity.
    manifest = load_or_build_manifest(
        manifest_path, cfg, args.split, args.probe_videos, args.manifest_max_offset, args.seed
    )
    features = compute_encoder_features(
        cfg,
        manifest,
        cache_offsets,
        cache_path,
        device,
        args.encoder_batch,
        args.cache_dtype,
    )
    cache_envelope = torch.load(cache_path, map_location="cpu")
    video_tokens = [
        features[feature_key(video["path"], video["anchor_end"])].to(device, dtype=torch.float32)
        for video in manifest["videos"]
    ]

    print(f"Computing effective-rank metrics on {device}...")
    from provenance import encoder_spec_from_dict

    encoder_spec = encoder_spec_from_dict(cache_envelope["metadata"]["encoder_spec"])
    report = rank_report(video_tokens, layout=encoder_spec.layout)
    print(
        f"detailed-feature ranks over {report['n_videos']} probe videos "
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
    if report.get("detailed_temporal_unit") == "frame":
        frame_norms = report["per_frame_token_norm_mean"]
        print(
            "  pre-concatenation frame rank: "
            f"mean {report['per_frame_effective_rank_mean']:.1f} "
            f"(min {report['per_frame_effective_rank_min']:.1f}, "
            f"max {report['per_frame_effective_rank_max']:.1f}); "
            f"token-norm means {min(frame_norms):.3f}..{max(frame_norms):.3f}"
        )

    results = {
        "manifest_path": str(manifest_path),
        "feature_cache": str(cache_path),
        "tag": tag,
        "encoder_spec": cache_envelope["metadata"]["encoder_spec"],
        "feature_fingerprint": cache_envelope["metadata"]["feature_fingerprint"],
        "dataset_identity": cache_envelope["metadata"]["dataset_identity"],
        "cache_payload_fingerprint": cache_envelope["payload_fingerprint"],
        **report,
    }
    results_path = out_dir / f"rank_results_{tag}.json"
    from provenance import atomic_json_save

    atomic_json_save(results, results_path)
    print(f"Wrote {results_path}")
    if args.plot == "on":
        plot_spectrum(out_dir / f"rank_spectrum_{tag}.png", report, tag)
    if args.wandb_artifact:
        import wandb

        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, job_type="rank-probe"
        )
        artifact = wandb.Artifact(f"rank-{tag}", type="probe-report")
        artifact.add_file(str(results_path))
        artifact.add_file(str(manifest_path))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
