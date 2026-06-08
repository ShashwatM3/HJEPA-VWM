"""Create the SSv2-tiny stratified smoke subset.

The script operates on symlink directories under `/workspace/data/ssv2` (or
`JEPA_DATA_ROOT/ssv2`) and creates a smaller symlink-only dataset at
`/workspace/data/ssv2_tiny`. It never copies or re-encodes video bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_labels(path: Path) -> dict[str, str]:
    """Load SSv2 labels as a `video_id -> class label` mapping.

    Args:
        path: JSON file at `<DATA_ROOT>/ssv2/labels.json`.
    Returns:
        Dictionary keyed by video id without `.webm` suffix.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        labels: dict[str, str] = {}
        for item in raw:
            video_id = str(item.get("id") or item.get("video_id") or item.get("name"))
            label = str(item.get("template") or item.get("label") or item.get("class") or "")
            if video_id and label:
                labels[video_id] = label
        return labels
    raise TypeError(f"Unsupported labels.json structure: {type(raw)!r}")


def group_split_by_class(split_dir: Path, labels: dict[str, str]) -> dict[str, list[Path]]:
    """Group readable split symlinks by class label.

    Args:
        split_dir: Directory containing one `video_id.webm` symlink per video.
        labels: Mapping from video id to class label.
    Returns:
        Mapping class label -> sorted paths in that class.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for item in sorted(split_dir.glob("*.webm")):
        video_id = item.stem
        label = labels.get(video_id)
        if label is None:
            continue
        if item.is_symlink() and not item.resolve().exists():
            print(f"WARN: skipping broken symlink: {item}")
            continue
        groups[label].append(item)
    return dict(groups)


def _symlink_resolved(src: Path, dst: Path) -> None:
    """Create an idempotent symlink to the resolved raw video target."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def select_for_split(
    groups: dict[str, list[Path]],
    count_per_class: int,
    seed: int,
    split_name: str,
    tiny_root: Path,
) -> dict[str, Any]:
    """Select deterministic per-class items and create tiny symlinks.

    Args:
        groups: Mapping class label -> full split symlinks.
        count_per_class: Maximum items to select per class.
        seed: Deterministic seed.
        split_name: `train` or `validation`.
        tiny_root: Output root `<DATA_ROOT>/ssv2_tiny`.
    Returns:
        Manifest fragment for this split.
    """
    rng = random.Random(seed)
    out_dir = tiny_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class: dict[str, list[str]] = {}
    undersized: dict[str, int] = {}
    for label, paths in sorted(groups.items()):
        paths = list(paths)
        rng.shuffle(paths)
        selected = paths[:count_per_class]
        if len(paths) < count_per_class:
            undersized[label] = len(paths)
        per_class[label] = []
        for src in selected:
            dst = out_dir / src.name
            _symlink_resolved(src, dst)
            per_class[label].append(src.stem)
    counts = [len(v) for v in per_class.values()]
    return {
        "count": sum(counts),
        "classes": len(per_class),
        "per_class": per_class,
        "undersized_classes": undersized,
        "min_per_class": min(counts) if counts else 0,
        "max_per_class": max(counts) if counts else 0,
        "mean_per_class": mean(counts) if counts else 0.0,
    }


def create_subset(
    data_root: str | Path,
    train_per_class: int = 23,
    val_per_class: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Create `/workspace/data/ssv2_tiny` from `/workspace/data/ssv2` symlinks.

    Args:
        data_root: Parent containing `ssv2/` and receiving `ssv2_tiny/`.
        train_per_class: Train symlinks selected per class.
        val_per_class: Validation symlinks selected per class.
        seed: Deterministic class-local shuffle seed.
    Returns:
        Manifest dictionary also written to `ssv2_tiny/manifest.json`.
    """
    data_root = Path(data_root)
    full_root = data_root / "ssv2"
    tiny_root = data_root / "ssv2_tiny"
    labels = load_labels(full_root / "labels.json")
    manifest: dict[str, Any] = {
        "seed": seed,
        "source_root": str(full_root),
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "splits": {},
    }
    for split_name, count in (("train", train_per_class), ("validation", val_per_class)):
        groups = group_split_by_class(full_root / split_name, labels)
        manifest["splits"][split_name] = select_for_split(
            groups, count, seed, split_name, tiny_root
        )
    manifest_path = tiny_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for split_name, info in manifest["splits"].items():
        print(
            f"{split_name}: {info['count']} symlinks across {info['classes']} classes "
            f"(min={info['min_per_class']}, max={info['max_per_class']}, "
            f"mean={info['mean_per_class']:.2f})"
        )
    for split_name in ("train", "validation"):
        for link in (tiny_root / split_name).glob("*.webm"):
            assert link.resolve().exists(), f"Broken tiny symlink: {link}"
    print(f"Wrote {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the SSv2-tiny subset creation CLI.

    The CLI mirrors PHASE_1.md §5 so the human can run one deterministic command
    on the RunPod volume after `/workspace/data/ssv2` is migrated.

    Returns:
        Parsed arguments for data root, per-class counts, and seed.
    """
    parser = argparse.ArgumentParser(description="Create the HJEPA-VWM SSv2-tiny subset.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("JEPA_DATA_ROOT", "/workspace/data"),
        help="Parent containing ssv2/ and receiving ssv2_tiny/.",
    )
    parser.add_argument("--train-per-class", type=int, default=23)
    parser.add_argument("--val-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Create SSv2-tiny from CLI arguments.

    This entry point is intentionally tiny: all behavior lives in `create_subset`
    so tests can exercise the symlink and manifest contract without shelling out.
    """
    args = parse_args()
    create_subset(args.data_root, args.train_per_class, args.val_per_class, args.seed)


if __name__ == "__main__":
    main()
