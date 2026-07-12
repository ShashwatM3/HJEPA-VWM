"""Create the EGO4D-tiny smoke subset (symlinks into the chunked EGO4D corpus).

The EGO4D sibling of `make_subset.py`: SSv2-tiny stratifies by class label, but EGO4D
chunks carry no class taxonomy, so this samples SOURCE VIDEOS deterministically and caps
chunks per video — a diversity guard, since chunks of one long video are near-duplicates
and uniform chunk sampling would let a few multi-hour videos dominate the subset.

Operates on `<data_root>/ego4d/{train,validation}` (real chunk files written by
`chunk_ego4d.py`, named `<video_uid>_<index:05d>.mp4`) and creates the symlink-only
subset at `<data_root>/ego4d_tiny` plus `manifest.json`. Never copies or re-encodes.
Idempotent reruns are safe. Targets mirror SSv2-tiny's scale (~4,000 train / ~350 val)
so smoke-run wall-clock expectations carry over.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def group_chunks_by_source(split_dir: Path) -> dict[str, list[Path]]:
    """Group chunk files by their source-video UID prefix.

    Args:
        split_dir: `<data_root>/ego4d/<split>` directory of `<uid>_<index>.mp4` files.
    Returns:
        Mapping video_uid -> sorted chunk paths for that source video.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for item in sorted(split_dir.glob("*.mp4")):
        uid = item.name[: item.name.rfind("_")]
        groups[uid].append(item)
    if not groups:
        raise FileNotFoundError(f"No .mp4 chunks found in {split_dir}")
    return dict(groups)


def _symlink_resolved(src: Path, dst: Path) -> None:
    """Create an idempotent symlink to the resolved chunk file (mirrors make_subset.py)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def select_for_split(
    groups: dict[str, list[Path]],
    target_chunks: int,
    per_video_cap: int,
    seed: int,
    split_name: str,
    tiny_root: Path,
) -> dict[str, Any]:
    """Select capped per-video chunks until the target count and create tiny symlinks.

    Deterministic: source videos visit in seeded-shuffle order; within a video, chunks
    are a seeded sample of at most `per_video_cap`, re-sorted for stable naming.

    Args:
        groups: video_uid -> chunk paths for the full split.
        target_chunks: Total symlinks to create for this split.
        per_video_cap: Max chunks taken from any one source video.
        seed: Deterministic seed.
        split_name: `train` or `validation`.
        tiny_root: Output root `<data_root>/ego4d_tiny`.
    Returns:
        Manifest fragment for this split.
    """
    rng = random.Random(seed)
    uids = sorted(groups)
    rng.shuffle(uids)
    out_dir = tiny_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[str]] = {}
    total = 0
    for uid in uids:
        if total >= target_chunks:
            break
        chunks = groups[uid]
        take = min(per_video_cap, len(chunks), target_chunks - total)
        picked = sorted(rng.sample(chunks, take), key=lambda p: p.name)
        selected[uid] = []
        for src in picked:
            _symlink_resolved(src, out_dir / src.name)
            selected[uid].append(src.name)
        total += take
    if total < target_chunks:
        print(
            f"WARN: {split_name} exhausted at {total} < {target_chunks} chunks "
            f"(raise --per-video-cap or chunk more source hours)."
        )
    return {
        "count": total,
        "source_videos": len(selected),
        "per_video_cap": per_video_cap,
        "per_video": selected,
    }


def create_subset(
    data_root: str | Path,
    train_chunks: int = 4000,
    val_chunks: int = 350,
    per_video_cap: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Create `<data_root>/ego4d_tiny` from `<data_root>/ego4d` chunk files.

    Args:
        data_root: Parent containing `ego4d/` and receiving `ego4d_tiny/`.
        train_chunks: Train symlinks to create.
        val_chunks: Validation symlinks to create.
        per_video_cap: Max chunks per source video (diversity guard).
        seed: Deterministic seed.
    Returns:
        Manifest dictionary also written to `ego4d_tiny/manifest.json`.
    """
    data_root = Path(data_root)
    full_root = data_root / "ego4d"
    tiny_root = data_root / "ego4d_tiny"
    manifest: dict[str, Any] = {
        "seed": seed,
        "source_root": str(full_root),
        "train_chunks": train_chunks,
        "val_chunks": val_chunks,
        "per_video_cap": per_video_cap,
        "splits": {},
    }
    for split_name, count in (("train", train_chunks), ("validation", val_chunks)):
        groups = group_chunks_by_source(full_root / split_name)
        manifest["splits"][split_name] = select_for_split(
            groups, count, per_video_cap, seed, split_name, tiny_root
        )
    manifest_path = tiny_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for split_name, info in manifest["splits"].items():
        print(f"{split_name}: {info['count']} symlinks from {info['source_videos']} videos")
    for split_name in ("train", "validation"):
        for link in (tiny_root / split_name).glob("*.mp4"):
            assert link.resolve().exists(), f"Broken tiny symlink: {link}"
    print(f"Wrote {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the EGO4D-tiny subset creation CLI (GUIDE.md Stage 5)."""
    parser = argparse.ArgumentParser(description="Create the HJEPA-VWM EGO4D-tiny subset.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("JEPA_DATA_ROOT", "/workspace/data"),
        help="Parent containing ego4d/ and receiving ego4d_tiny/.",
    )
    parser.add_argument("--train-chunks", type=int, default=4000)
    parser.add_argument("--val-chunks", type=int, default=350)
    parser.add_argument("--per-video-cap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Create EGO4D-tiny from CLI arguments (all behavior lives in `create_subset`)."""
    args = parse_args()
    create_subset(
        args.data_root, args.train_chunks, args.val_chunks, args.per_video_cap, args.seed
    )


if __name__ == "__main__":
    main()
