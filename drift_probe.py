"""Within-video temporal drift probe for HJEPA-VWM (offline diagnostic).

Measures how far the frozen V-JEPA embedding of a video window travels over time
(`1 - cos(e_t, e_{t+k})` for a ladder of temporal offsets `k`) and, optionally, how
far the bottleneck's abstract latent travels over the SAME windows
(`1 - cos(c_t, c_{t+k})`, with `c = B(e)` loaded from a training checkpoint).

This is WITHIN-video drift only: every distance compares two windows of the same
video at different times. It never compares two different videos.

Why it exists (KANBAN context): the Phase 1 copy baseline wins because `c` barely
moves over the training horizon. The encoder-side drift curve is the ground-truth
"signal budget" — how much change V-JEPA actually sees at each horizon — and the
latent-side curve shows what fraction of that change survives the bottleneck. The
encoder is frozen, so the e-side is a constant of the dataset: it is computed once
on a fixed probe set and cached; each checkpoint evaluation afterwards only costs
bottleneck forwards over the cached features.

Design contracts:

- Pure ADD-ON analysis helper (like `parse_logs.py` / `run_history.py`). Training
  code does not import this file and no training behavior changes.
- The probe set is pinned by a manifest JSON (video paths + anchor windows), so
  every invocation — any run, any checkpoint, any time — measures identical inputs.
- Windows follow the training data contract exactly: `t_ctx` frames at
  `frame_stride`, validation-style transforms (resize-256, center crop, encoder
  normalization, no jitter, no flips).
- Checkpoints are self-describing: the bottleneck is rebuilt from the config
  serialized INSIDE the checkpoint, so no architecture flags are needed.

Outputs (per invocation, under --out-dir):

- `results_<tag>.json` — the full raw drift matrices plus all aggregates.
- `graph1_<tag>.png` — X = offset k, Y = mean drift across probe videos
  (V-JEPA reference curve + one latent curve per checkpoint).
- `graph2_<label>_<tag>.png` per checkpoint — X = probe videos sorted by V-JEPA
  drift, Y = per-video mean drift over --graph2-offsets (e staircase vs c dots).

Typical usage (see README "Within-video drift probe"):

    python drift_probe.py --data ssv2 --probe-videos 64 \
        --ckpt /workspace/ckpt/inv011_cosine_residual/phase1_step7500.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
from pathlib import Path

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError:  # pragma: no cover - local docs-only environments.
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]

from config import Config, ModelConfig
from data import (
    _crop,
    _decode_frames,
    _normalize_encoder,
    _open_video_reader,
    _resize_shorter_side,
)

# Windows ending <= this many frames apart share raw frames with the anchor window
# ((t_ctx - 1) * frame_stride with the v0.2 defaults). Drift below this boundary is
# mechanically small, which is exactly why horizon_k=4 made the copy baseline cheap.
DEFAULT_OFFSETS = (2, 4, 8, 12, 16, 24, 32)


def _require_torch() -> None:
    """Fail fast when the probe is used without PyTorch installed."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for drift_probe.py. Install requirements.txt on RunPod."
        )


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without any dataset, encoder, or checkpoint I/O)
# ---------------------------------------------------------------------------


def parse_offset_list(text: str) -> list[int]:
    """Parse a CLI offset list like "2,4,8" into sorted unique positive ints.

    Args:
        text: Comma-separated offsets in ORIGINAL frames.
    Returns:
        Sorted list of unique positive integer offsets.
    """
    try:
        offsets = sorted({int(part) for part in text.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError(f"Bad offset list {text!r}; expected e.g. '2,4,8,12'") from exc
    if not offsets or any(k <= 0 for k in offsets):
        raise ValueError(f"Offsets must be positive integers; got {text!r}")
    return offsets


def overlap_boundary(t_ctx: int, frame_stride: int) -> int:
    """Largest offset at which context/target windows still share raw frames.

    A window ending at `t` spans `(t_ctx - 1) * frame_stride` frames, so a window
    ending at `t + k` overlaps it whenever `k <= (t_ctx - 1) * frame_stride`.

    Args:
        t_ctx: Frames per window (8).
        frame_stride: Frame stride inside a window (2).
    Returns:
        The boundary offset (14 with v0.2 defaults).
    """
    return (t_ctx - 1) * frame_stride


def default_graph2_offsets(offsets: list[int], t_ctx: int, frame_stride: int) -> list[int]:
    """Choose the offsets averaged into Graph 2's one-point-per-video summary.

    Defaults to the non-overlapping offsets (past `overlap_boundary`) so the
    static-vs-dynamic contrast between videos is not diluted by offsets where
    drift is mechanically tiny. Falls back to all offsets if none qualify.

    Args:
        offsets: Full measured offset ladder.
        t_ctx: Frames per window.
        frame_stride: Frame stride inside a window.
    Returns:
        Non-empty subset of `offsets`.
    """
    boundary = overlap_boundary(t_ctx, frame_stride)
    non_overlapping = [k for k in offsets if k > boundary]
    return non_overlapping if non_overlapping else list(offsets)


def window_frame_indices(end_frame: int, t_ctx: int, frame_stride: int, num_frames: int) -> list[int]:
    """Frame indices for one window ending at `end_frame` (training contract).

    Mirrors `data.SSV2Dataset._window_indices`: `t_ctx` frames at `frame_stride`
    ending at `end_frame`, clamped to the video's last frame as a defensive
    pad-by-repeat (manifest anchors are chosen so clamping never fires).

    Args:
        end_frame: Index of the window's last frame.
        t_ctx: Frames per window.
        frame_stride: Frame stride inside a window.
        num_frames: Total decoded frames in the video.
    Returns:
        List of `t_ctx` frame indices, non-decreasing.
    """
    last = num_frames - 1
    return [
        max(0, min(last, end_frame - (t_ctx - 1 - i) * frame_stride)) for i in range(t_ctx)
    ]


def select_probe_order(paths: list[str], seed: int) -> list[str]:
    """Deterministically shuffle candidate video paths for probe selection.

    The manifest builder walks this order and keeps the first N videos long
    enough for the largest offset, so the probe set is reproducible from
    (sorted file listing, seed) alone.

    Args:
        paths: Sorted candidate paths.
        seed: Probe-selection seed.
    Returns:
        Shuffled copy of `paths`.
    """
    order = list(paths)
    random.Random(seed).shuffle(order)
    return order


def feature_key(rel_path: str, end_frame: int) -> str:
    """Cache key for one (video, window-end) pair of frozen encoder features."""
    return f"{rel_path}::end{end_frame}"


def model_config_from_checkpoint_dict(model_dict: dict) -> ModelConfig:
    """Rebuild a ModelConfig from the config dict serialized inside a checkpoint.

    Checkpoints store the full dataclass config as nested dicts
    (`train.save_checkpoint`), so the bottleneck can be reconstructed with the
    right `n_c`/dims without CLI architecture flags. Unknown keys (from older or
    newer code eras) are dropped; missing keys fall back to current defaults.

    Args:
        model_dict: The checkpoint's `config["model"]` dict (possibly empty).
    Returns:
        ModelConfig matching the checkpointed architecture fields.
    """
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    kwargs = {k: v for k, v in (model_dict or {}).items() if k in field_names}
    return ModelConfig(**kwargs)


def pooled_unit_vector(features: Tensor) -> Tensor:
    """Mean-pool detailed tokens into one unit vector per window.

    Args:
        features: (N_ctx, D_e) frozen encoder tokens for one window.
    Returns:
        unit: (D_e,) L2-normalized fp32 mean token.
    """
    _require_torch()
    pooled = features.float().mean(dim=0)
    return pooled / pooled.norm().clamp_min(1e-8)


def flat_unit_vector(latent: Tensor) -> Tensor:
    """Flatten an abstract latent into one unit vector per window.

    Matches how `diagnostics.cross_video_cosine` treats `c` (flatten slots and
    dims), so latent drift is cosine distance in the same geometry the collapse
    metrics already use.

    Args:
        latent: (N_c, D_c) abstract latent for one window.
    Returns:
        unit: (N_c * D_c,) L2-normalized fp32 vector.
    """
    _require_torch()
    flat = latent.float().reshape(-1)
    return flat / flat.norm().clamp_min(1e-8)


def drift_matrix(unit_vectors: dict[str, Tensor], videos: list[dict], offsets: list[int]) -> Tensor:
    """Within-video drift `1 - cos(v_t, v_{t+k})` for every (video, offset) pair.

    Args:
        unit_vectors: `feature_key(path, end)` -> (D,) unit vector (encoder-pooled
            or latent-flattened; both sides use this same function).
        videos: Manifest video entries with `path` and `anchor_end`.
        offsets: Measured offsets in original frames.
    Returns:
        drift: (V, K) fp32 tensor, `drift[i, j] = 1 - cos(anchor_i, anchor_i + k_j)`.
    """
    _require_torch()
    rows = []
    for video in videos:
        anchor = unit_vectors[feature_key(video["path"], video["anchor_end"])]
        row = [
            1.0 - float(anchor @ unit_vectors[feature_key(video["path"], video["anchor_end"] + k)])
            for k in offsets
        ]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def summarize_per_offset(drift: Tensor) -> dict[str, list[float]]:
    """Aggregate a (V, K) drift matrix across videos for the Graph-1 curve.

    Args:
        drift: (V, K) per-video per-offset drift.
    Returns:
        Dict with `mean`, `p25`, `p75` lists, one value per offset.
    """
    _require_torch()
    return {
        "mean": drift.mean(dim=0).tolist(),
        "p25": drift.quantile(0.25, dim=0).tolist(),
        "p75": drift.quantile(0.75, dim=0).tolist(),
    }


def per_video_summary(drift: Tensor, offsets: list[int], graph2_offsets: list[int]) -> Tensor:
    """Average a (V, K) drift matrix over the Graph-2 offsets for one point per video.

    Args:
        drift: (V, K) per-video per-offset drift.
        offsets: The K measured offsets (column labels of `drift`).
        graph2_offsets: Subset of `offsets` to average (CLI `--graph2-offsets`).
    Returns:
        summary: (V,) per-video mean drift over the chosen offsets.
    """
    _require_torch()
    columns = [offsets.index(k) for k in graph2_offsets]
    return drift[:, columns].mean(dim=1)


def _average_ranks(x: Tensor) -> Tensor:
    """Ranks with ties assigned their group's average rank (Spearman convention).

    Naive `argsort` ranking hands TIED values arbitrary distinct ranks, which
    fabricates variance — a constant vector would rank as [0, 1, 2, ...] and
    correlate perfectly with anything monotone. Averaging within tie groups makes
    a constant vector rank constant (degenerate -> correlation 0 downstream).

    Args:
        x: (N,) values.
    Returns:
        ranks: (N,) fp32 average ranks; ties share their group's mean rank.
    """
    _require_torch()
    n = x.numel()
    order = torch.argsort(x)
    sorted_x = x[order]
    ranks = torch.empty(n, dtype=torch.float32)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_correlation(a: Tensor, b: Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors (no scipy dependency).

    Answers "does the latent move when and only when V-JEPA moves": average-ranks
    both sides (ties handled per the standard Spearman convention), then computes
    the Pearson correlation of the ranks.

    Args:
        a: (N,) values.
        b: (N,) values, same length.
    Returns:
        Correlation in [-1, 1] (0.0 for degenerate/constant inputs).
    """
    _require_torch()
    ra = _average_ranks(a.reshape(-1).float())
    rb = _average_ranks(b.reshape(-1).float())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(ra.norm() * rb.norm())
    if denom < 1e-12:
        return 0.0
    return float((ra @ rb) / denom)


# ---------------------------------------------------------------------------
# Manifest: the pinned probe set (video paths + anchor windows)
# ---------------------------------------------------------------------------


def build_manifest(
    cfg: Config, split: str, n_videos: int, max_offset: int, seed: int
) -> dict:
    """Select the fixed probe set and pin it to a manifest dict.

    Walks the split's videos in `select_probe_order`, keeps the first `n_videos`
    long enough to fit the anchor window plus `max_offset` extra frames, and
    records a deterministic center-ish anchor per video. This is the ONLY place
    randomness enters the probe; every later invocation replays the manifest.

    Args:
        cfg: Global config (dataset root, `t_ctx`, `frame_stride`).
        split: Dataset split, normally `validation`.
        n_videos: Probe set size.
        max_offset: Largest offset the manifest must support, in original frames.
        seed: Probe-selection seed.
    Returns:
        Manifest dict (JSON-serializable) with per-video path/frames/anchor.
    """
    _require_torch()
    root = Path(cfg.data.dataset_root())
    paths = sorted(str(p.relative_to(root)) for p in (root / split).glob("*.webm"))
    if not paths:
        raise FileNotFoundError(f"No .webm files found in {root / split}")
    t_ctx, stride = cfg.model.t_ctx, cfg.train.frame_stride
    span = overlap_boundary(t_ctx, stride) + max_offset
    videos: list[dict] = []
    skipped = 0
    for rel_path in select_probe_order(paths, seed):
        reader = _open_video_reader(root / rel_path)
        num_frames = len(reader)
        if num_frames < span + 1:
            skipped += 1
            continue
        start = (num_frames - 1 - span) // 2
        videos.append(
            {
                "index": len(videos),
                "path": rel_path,
                "num_frames": num_frames,
                "anchor_end": start + overlap_boundary(t_ctx, stride),
            }
        )
        if len(videos) >= n_videos:
            break
    if len(videos) < n_videos:
        raise RuntimeError(
            f"Only {len(videos)}/{n_videos} probe videos have >= {span + 1} frames "
            f"(needed for max offset {max_offset}); lower --probe-videos or the offsets."
        )
    return {
        "dataset": cfg.data.dataset,
        "split": split,
        "seed": seed,
        "n_videos": n_videos,
        "t_ctx": t_ctx,
        "frame_stride": stride,
        "max_offset": max_offset,
        "skipped_too_short": skipped,
        "videos": videos,
    }


def load_or_build_manifest(
    manifest_path: Path, cfg: Config, split: str, n_videos: int, max_offset: int, seed: int
) -> dict:
    """Load the pinned manifest, or build and save it on first use.

    A loaded manifest is validated against the current invocation so two
    incompatible probe definitions can never be silently mixed.
    """
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        mismatches = {
            "dataset": (manifest.get("dataset"), cfg.data.dataset),
            "split": (manifest.get("split"), split),
            "t_ctx": (manifest.get("t_ctx"), cfg.model.t_ctx),
            "frame_stride": (manifest.get("frame_stride"), cfg.train.frame_stride),
        }
        bad = {k: v for k, v in mismatches.items() if v[0] != v[1]}
        if bad:
            raise RuntimeError(f"Manifest {manifest_path} does not match this invocation: {bad}")
        if manifest.get("max_offset", 0) < max_offset:
            raise RuntimeError(
                f"Manifest {manifest_path} supports offsets up to {manifest.get('max_offset')}, "
                f"but {max_offset} was requested. Delete it or pass --manifest for a new file."
            )
        return manifest
    manifest = build_manifest(cfg, split, n_videos, max_offset, seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote probe manifest: {manifest_path} ({manifest['n_videos']} videos)")
    return manifest


# ---------------------------------------------------------------------------
# Frozen encoder features (computed once per probe set, cached to disk)
# ---------------------------------------------------------------------------


def _load_window_clip(cfg: Config, video_path: Path, indices: list[int]) -> Tensor:
    """Decode and preprocess one window with validation-style transforms.

    Args:
        cfg: Global config (frame size).
        video_path: Absolute path to the .webm file.
        indices: `t_ctx` frame indices from `window_frame_indices`.
    Returns:
        clip: (T, 3, H, W) encoder-normalized float32 window.
    """
    _require_torch()
    reader = _open_video_reader(video_path)
    frames = torch.from_numpy(_decode_frames(reader, indices))
    frames = frames.float().permute(0, 3, 1, 2) / 255.0
    frames = _resize_shorter_side(frames, cfg.model.h)
    frames = _crop(frames, cfg.model.h, "validation")
    return _normalize_encoder(frames)


def compute_encoder_features(
    cfg: Config,
    manifest: dict,
    offsets: list[int],
    cache_path: Path,
    device: torch.device,
    encoder_batch: int,
) -> dict[str, Tensor]:
    """Return frozen V-JEPA features for every (probe video, window end), cached.

    The encoder is frozen, so features never go stale: the cache is keyed by
    (video path, window end) and only missing windows are computed. The heavy
    encoder is not even constructed on a full cache hit, which is what makes
    re-evaluating checkpoints cheap.

    Args:
        cfg: Global config.
        manifest: Probe manifest from `load_or_build_manifest`.
        offsets: Offset ladder (windows needed: anchor and anchor + each offset).
        cache_path: Feature cache file (.pt, fp16 CPU tensors).
        device: Device for encoder forwards.
        encoder_batch: Windows per encoder forward.
    Returns:
        `feature_key -> (N_ctx, D_e)` fp16 CPU tensors.
    """
    _require_torch()
    features: dict[str, Tensor] = {}
    if cache_path.exists():
        features = torch.load(cache_path, map_location="cpu")
    needed: list[tuple[dict, int]] = []
    for video in manifest["videos"]:
        for end in [video["anchor_end"]] + [video["anchor_end"] + k for k in offsets]:
            if feature_key(video["path"], end) not in features:
                needed.append((video, end))
    if not needed:
        return features
    print(f"Encoding {len(needed)} probe windows with the frozen encoder (one-time cost)...")
    from models import FrozenEncoder

    encoder = FrozenEncoder(cfg.model).to(device)
    root = Path(cfg.data.dataset_root())
    t_ctx, stride = manifest["t_ctx"], manifest["frame_stride"]
    with torch.no_grad():
        for i in range(0, len(needed), encoder_batch):
            batch_items = needed[i : i + encoder_batch]
            clips = torch.stack(
                [
                    _load_window_clip(
                        cfg,
                        root / video["path"],
                        window_frame_indices(end, t_ctx, stride, video["num_frames"]),
                    )
                    for video, end in batch_items
                ]
            ).to(device)
            out = encoder(clips)
            for (video, end), tokens in zip(batch_items, out, strict=True):
                features[feature_key(video["path"], end)] = tokens.detach().to("cpu", torch.float16)
            print(f"  encoded {min(i + encoder_batch, len(needed))}/{len(needed)} windows")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)
    print(f"Wrote feature cache: {cache_path}")
    return features


# ---------------------------------------------------------------------------
# Latent side: bottleneck loaded from a training checkpoint
# ---------------------------------------------------------------------------


def load_bottleneck_from_checkpoint(ckpt_path: Path, use_ema: bool = False):
    """Rebuild the checkpointed bottleneck for latent-drift evaluation.

    Args:
        ckpt_path: A `phase1_step*.pt` checkpoint saved by `train.save_checkpoint`.
        use_ema: Load `target_bottleneck` (B_EMA) weights instead of the online B.
    Returns:
        `(bottleneck, label, step)`: eval-mode, grad-free Bottleneck; a plot label
        derived from the checkpoint path; and the saved global step.
    """
    _require_torch()
    from models import Bottleneck

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "bottleneck" not in ckpt:
        raise RuntimeError(f"{ckpt_path} has no 'bottleneck' state; not a Phase 1 checkpoint.")
    model_cfg = model_config_from_checkpoint_dict((ckpt.get("config") or {}).get("model", {}))
    bottleneck = Bottleneck(model_cfg)
    if use_ema:
        state = {
            key[len("bottleneck.") :]: value
            for key, value in ckpt["target_bottleneck"].items()
            if key.startswith("bottleneck.")
        }
    else:
        state = ckpt["bottleneck"]
    try:
        bottleneck.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Bottleneck state in {ckpt_path} does not match the current architecture. "
            "Checkpoints from before the sharp-slot attention change (commit cc0e318) "
            "need the matching code checkout to evaluate."
        ) from exc
    bottleneck.eval()
    for param in bottleneck.parameters():
        param.requires_grad_(False)
    step = int(ckpt.get("global_step", -1))
    suffix = "-ema" if use_ema else ""
    label = f"{ckpt_path.parent.name}/step{step}{suffix}"
    return bottleneck, label, step


def compute_latent_unit_vectors(
    bottleneck,
    features: dict[str, Tensor],
    keys: list[str],
    device: torch.device,
    latent_batch: int,
) -> dict[str, Tensor]:
    """Run the bottleneck over cached windows and return flattened unit latents.

    Args:
        bottleneck: Eval-mode Bottleneck from `load_bottleneck_from_checkpoint`.
        features: Cached `feature_key -> (N_ctx, D_e)` fp16 tensors.
        keys: Which cached windows to embed (probe order).
        device: Device for bottleneck forwards.
        latent_batch: Windows per bottleneck forward.
    Returns:
        `feature_key -> (N_c * D_c,)` fp32 CPU unit vectors.
    """
    _require_torch()
    bottleneck = bottleneck.to(device).eval()  # measurement is always eval-mode, no-grad
    units: dict[str, Tensor] = {}
    with torch.no_grad():
        for i in range(0, len(keys), latent_batch):
            chunk = keys[i : i + latent_batch]
            detailed = torch.stack([features[k].float() for k in chunk]).to(device)
            abstract = bottleneck(detailed)
            for key, latent in zip(chunk, abstract, strict=True):
                units[key] = flat_unit_vector(latent).cpu()
    return units


# ---------------------------------------------------------------------------
# Plots (matplotlib optional: JSON is always written, plots are best-effort)
# ---------------------------------------------------------------------------


def _pyplot():
    """Import matplotlib lazily so a missing install degrades to JSON-only output."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        print("WARN: matplotlib not installed; skipping PNG plots (JSON still written).")
        return None


def plot_graph1(
    out_path: Path,
    offsets: list[int],
    encoder_summary: dict[str, list[float]] | None,
    checkpoint_curves: list[tuple[str, dict[str, list[float]]]],
    boundary: int,
) -> None:
    """Graph 1: mean drift vs temporal offset (V-JEPA reference + latent curves).

    Args:
        out_path: PNG destination.
        offsets: X values (original frames).
        encoder_summary: `summarize_per_offset` output for the e-side (None = off).
        checkpoint_curves: `(label, summarize_per_offset output)` per checkpoint.
        boundary: `overlap_boundary` — the shaded window-overlap region ends here.
    """
    plt = _pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(0, boundary, alpha=0.08, color="gray")
    ax.text(boundary / 2, 0.98, "windows share frames", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="gray")
    if encoder_summary is not None:
        ax.plot(offsets, encoder_summary["mean"], "o-", color="black", label="V-JEPA e (frozen)")
        ax.fill_between(offsets, encoder_summary["p25"], encoder_summary["p75"],
                        color="black", alpha=0.12)
    for label, summary in checkpoint_curves:
        ax.plot(offsets, summary["mean"], "s--", label=f"latent c — {label}")
    ax.set_xlabel("temporal offset k (original frames)")
    ax.set_ylabel("mean within-video drift (1 − cosine)")
    ax.set_title("Within-video temporal drift vs offset (probe set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_graph2(
    out_path: Path,
    e_summary: Tensor,
    c_summary: Tensor,
    graph2_offsets: list[int],
    label: str,
    spearman: float,
) -> None:
    """Graph 2: per-video drift, videos sorted by V-JEPA drift (staircase vs dots).

    Args:
        out_path: PNG destination.
        e_summary: (V,) per-video V-JEPA drift over `graph2_offsets`.
        c_summary: (V,) per-video latent drift over the same offsets.
        graph2_offsets: Offsets averaged into each point (for the title).
        label: Checkpoint label.
        spearman: Per-video Spearman between the two summaries.
    """
    plt = _pyplot()
    if plt is None:
        return
    order = torch.argsort(e_summary)
    x = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, e_summary[order].tolist(), "o", color="black", ms=4, label="V-JEPA e (frozen)")
    ax.plot(x, c_summary[order].tolist(), "o", ms=4, alpha=0.8, label=f"latent c — {label}")
    ax.set_xlabel("probe videos, sorted by V-JEPA drift →")
    ax.set_ylabel(f"mean drift over offsets {graph2_offsets} (1 − cosine)")
    ax.set_title(f"Per-video drift — {label}  (Spearman ρ = {spearman:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse the drift-probe CLI."""
    parser = argparse.ArgumentParser(
        description="Within-video temporal drift probe (V-JEPA embeddings vs bottleneck latents)."
    )
    parser.add_argument("--data", choices=["ssv2", "ssv2_tiny"], default="ssv2_tiny")
    parser.add_argument("--split", default="validation", choices=["train", "validation"],
                        help="Dataset split the probe set is drawn from (default: validation).")
    parser.add_argument("--probe-videos", type=int, default=64,
                        help="Probe set size; only used when building a new manifest.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Probe-selection seed; only used when building a new manifest.")
    parser.add_argument("--offsets", default=",".join(str(k) for k in DEFAULT_OFFSETS),
                        help="Comma-separated temporal offsets in ORIGINAL frames.")
    parser.add_argument("--graph2-offsets", default=None,
                        help="Offsets averaged into Graph 2's per-video points "
                        "(default: the non-overlapping offsets, k > 14).")
    parser.add_argument("--encoder-curve", choices=["on", "off"], default="on",
                        help="Include the V-JEPA drift curve in Graph 1 (features are "
                        "computed/cached regardless — the latent side needs them).")
    parser.add_argument("--latent-curve", choices=["on", "off"], default="on",
                        help="Evaluate bottleneck checkpoints (--ckpt) for latent drift.")
    parser.add_argument("--ckpt", action="append", default=[],
                        help="Path to a phase1_step*.pt checkpoint; repeatable to compare "
                        "several checkpoints/runs in one invocation.")
    parser.add_argument("--use-ema", action="store_true",
                        help="Load the EMA bottleneck (B_EMA) instead of the online B.")
    parser.add_argument("--out-dir", default="logs/drift_probe",
                        help="Output directory for manifest, cache, JSON, and PNGs.")
    parser.add_argument("--manifest", default=None,
                        help="Probe manifest path (default: derived inside --out-dir).")
    parser.add_argument("--feature-cache", default=None,
                        help="Encoder feature cache path (default: derived inside --out-dir).")
    parser.add_argument("--tag", default=None,
                        help="Filename tag for outputs (default: dataset/probe/seed derived).")
    parser.add_argument("--encoder-batch", type=int, default=4,
                        help="Windows per frozen-encoder forward.")
    parser.add_argument("--latent-batch", type=int, default=16,
                        help="Windows per bottleneck forward.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="Override device (default: cuda if available).")
    return parser.parse_args()


def main() -> None:
    """Run the drift probe end to end: manifest -> features -> drift -> JSON + plots."""
    args = parse_args()  # before _require_torch so --help works on torch-less machines
    _require_torch()
    offsets = parse_offset_list(args.offsets)
    cfg = Config()
    cfg.data.dataset = args.data
    t_ctx, stride = cfg.model.t_ctx, cfg.train.frame_stride
    graph2_offsets = (
        parse_offset_list(args.graph2_offsets)
        if args.graph2_offsets
        else default_graph2_offsets(offsets, t_ctx, stride)
    )
    if not set(graph2_offsets).issubset(offsets):
        raise SystemExit(f"--graph2-offsets {graph2_offsets} must be a subset of --offsets {offsets}")
    if args.latent_curve == "on" and not args.ckpt:
        raise SystemExit("--latent-curve on requires at least one --ckpt "
                         "(or pass --latent-curve off for a V-JEPA-only probe).")
    if args.latent_curve == "off" and args.ckpt:
        print("WARN: --latent-curve off; ignoring --ckpt arguments.")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.data}_{args.split}_n{args.probe_videos}_seed{args.seed}"
    manifest_path = Path(args.manifest) if args.manifest else out_dir / f"manifest_{tag}.json"
    cache_path = (
        Path(args.feature_cache) if args.feature_cache else out_dir / f"vjepa_features_{tag}.pt"
    )

    manifest = load_or_build_manifest(
        manifest_path, cfg, args.split, args.probe_videos, max(offsets), args.seed
    )
    features = compute_encoder_features(
        cfg, manifest, offsets, cache_path, device, args.encoder_batch
    )
    videos = manifest["videos"]
    window_keys = [
        feature_key(v["path"], end)
        for v in videos
        for end in [v["anchor_end"]] + [v["anchor_end"] + k for k in offsets]
    ]

    encoder_units = {key: pooled_unit_vector(features[key]) for key in window_keys}
    e_drift = drift_matrix(encoder_units, videos, offsets)
    e_offset_summary = summarize_per_offset(e_drift)
    e_video_summary = per_video_summary(e_drift, offsets, graph2_offsets)
    print("V-JEPA drift by offset (mean over probe videos):")
    for k, mean in zip(offsets, e_offset_summary["mean"], strict=True):
        print(f"  k={k:>3}: {mean:.4f}")

    results = {
        "manifest_path": str(manifest_path),
        "feature_cache": str(cache_path),
        "offsets": offsets,
        "graph2_offsets": graph2_offsets,
        "overlap_boundary": overlap_boundary(t_ctx, stride),
        "encoder_drift": {
            "per_video_per_offset": e_drift.tolist(),
            "per_offset": e_offset_summary,
            "per_video_graph2": e_video_summary.tolist(),
        },
        "checkpoints": [],
    }

    checkpoint_curves: list[tuple[str, dict[str, list[float]]]] = []
    if args.latent_curve == "on":
        for ckpt_arg in args.ckpt:
            ckpt_path = Path(ckpt_arg)
            bottleneck, label, step = load_bottleneck_from_checkpoint(ckpt_path, args.use_ema)
            latent_units = compute_latent_unit_vectors(
                bottleneck, features, window_keys, device, args.latent_batch
            )
            c_drift = drift_matrix(latent_units, videos, offsets)
            c_offset_summary = summarize_per_offset(c_drift)
            c_video_summary = per_video_summary(c_drift, offsets, graph2_offsets)
            rho_all = spearman_correlation(e_drift.reshape(-1), c_drift.reshape(-1))
            rho_video = spearman_correlation(e_video_summary, c_video_summary)
            rho_per_offset = {
                str(k): spearman_correlation(e_drift[:, j], c_drift[:, j])
                for j, k in enumerate(offsets)
            }
            print(f"{label}: spearman(all pairs)={rho_all:.3f}  spearman(per-video)={rho_video:.3f}")
            results["checkpoints"].append(
                {
                    "path": str(ckpt_path),
                    "label": label,
                    "step": step,
                    "use_ema": args.use_ema,
                    "latent_drift": {
                        "per_video_per_offset": c_drift.tolist(),
                        "per_offset": c_offset_summary,
                        "per_video_graph2": c_video_summary.tolist(),
                    },
                    "spearman_all_pairs": rho_all,
                    "spearman_per_video": rho_video,
                    "spearman_per_offset": rho_per_offset,
                }
            )
            checkpoint_curves.append((label, c_offset_summary))
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
            plot_graph2(
                out_dir / f"graph2_{safe_label}_{tag}.png",
                e_video_summary,
                c_video_summary,
                graph2_offsets,
                label,
                rho_video,
            )

    results_path = out_dir / f"results_{tag}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {results_path}")
    plot_graph1(
        out_dir / f"graph1_{tag}.png",
        offsets,
        e_offset_summary if args.encoder_curve == "on" else None,
        checkpoint_curves,
        overlap_boundary(t_ctx, stride),
    )


if __name__ == "__main__":
    main()
