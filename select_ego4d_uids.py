"""Select EGO4D source-video UIDs for the HJEPA-VWM chunk corpus.

Reads the official `ego4d.json` metadata (downloaded by the Ego4D CLI), filters out
unusable videos (v2.1 grouped videos, stereo captures, non-30fps streams, very short
recordings), greedily
selects a scenario-diverse subset totaling `--target-hours`, splits the selection into
train/validation at the SOURCE-VIDEO level (chunks of one long video are near-duplicates,
so a chunk-level split would leak train content into validation), and partitions the
selection into `--batches` hour-balanced download batches for the batched
download -> chunk -> delete loop in AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md Stages 3-4.

Outputs in `--out-dir`:
  train_uids.txt / val_uids.txt      one UID per line (Ego4D CLI --video_uid_file format)
  batch_<b>_uids.txt                 hour-balanced partition of train+val, whole videos only
  selection_manifest.json            seed, filters, per-scenario hours, per-UID facts

Pure stdlib; no torch dependency (this runs before any training environment exists).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

SECONDS_PER_HOUR = 3600.0


def load_video_records(metadata_path: Path) -> list[dict[str, Any]]:
    """Load the `videos` list from the official ego4d.json metadata file.

    Fails loudly on a missing/renamed top-level key instead of guessing: the Ego4D
    metadata schema is external and versioned, so silent tolerance would produce an
    empty selection rather than an actionable error.

    Args:
        metadata_path: Path to `ego4d.json` (CLI download, `<output_dir>/ego4d.json`).
    Returns:
        List of per-video metadata dicts.
    """
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "videos" not in raw:
        raise KeyError(
            f"{metadata_path} has no top-level 'videos' list; got keys "
            f"{sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}. "
            "The ego4d.json schema may have changed — inspect the file before rerunning."
        )
    videos = raw["videos"]
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"{metadata_path} 'videos' is empty or not a list.")
    return videos


def load_downloadable_uids(manifest_path: Path) -> set[str]:
    """Load the authoritative video UIDs from an EGO4D dataset-tier manifest CSV."""
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "video_uid" not in reader.fieldnames:
            raise KeyError(
                f"{manifest_path} has no 'video_uid' column; got {reader.fieldnames}."
            )
        uids = {row["video_uid"].strip() for row in reader if row.get("video_uid", "").strip()}
    if not uids:
        raise ValueError(f"{manifest_path} contains no downloadable video UIDs.")
    return uids


def _video_fps(video: dict[str, Any]) -> float | None:
    """Return the video's fps if present (top-level or nested video_metadata), else None."""
    fps = video.get("fps")
    if fps is None and isinstance(video.get("video_metadata"), dict):
        fps = video["video_metadata"].get("fps")
    try:
        return float(fps) if fps is not None else None
    except (TypeError, ValueError):
        return None


def primary_scenario(video: dict[str, Any]) -> str:
    """Return the video's first scenario label, or "(none)" when unlabeled."""
    scenarios = video.get("scenarios")
    if isinstance(scenarios, list) and scenarios:
        return str(scenarios[0]).strip().lower()
    return "(none)"


def filter_videos(
    videos: list[dict[str, Any]],
    min_duration_sec: float = 60.0,
    fps_target: float = 30.0,
    fps_tolerance: float = 0.1,
    downloadable_uids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop videos unusable for the chunk corpus, with per-reason counters.

    Strict on identity fields (`video_uid`, `duration_sec` — missing raises), excludes
    v2.1 `grp-*` Goal-Step aggregates that have no `video_540ss` object, and is lenient
    with reporting on optional fields (`is_stereo` absent counts as mono, fps absent is
    not checked), matching the GUIDE Delta-4 contract.

    Args:
        videos: Raw per-video metadata dicts from `load_video_records`.
        min_duration_sec: Drop recordings shorter than this.
        fps_target: Canonical EGO4D frame rate (30).
        fps_tolerance: Allowed |fps - fps_target| when fps metadata is present.
        downloadable_uids: Optional authoritative UID set from the selected download tier.
    Returns:
        (kept, drop_counts) — usable video dicts and a reason -> count report.
    """
    kept: list[dict[str, Any]] = []
    drops: dict[str, int] = defaultdict(int)
    for video in videos:
        if "video_uid" not in video or "duration_sec" not in video:
            raise KeyError(
                "ego4d.json video record missing required 'video_uid'/'duration_sec': "
                f"keys={sorted(video.keys())[:12]}. Schema drift — inspect before rerunning."
            )
        uid = str(video["video_uid"])
        duration = float(video["duration_sec"])
        if uid.startswith("grp-"):
            drops["grouped_video_not_in_video_540ss"] += 1
            continue
        if downloadable_uids is not None and uid not in downloadable_uids:
            drops["not_in_video_540ss_manifest"] += 1
            continue
        if video.get("is_stereo") is True:
            drops["stereo"] += 1
            continue
        if "is_stereo" not in video:
            drops["is_stereo_missing_assumed_mono"] += 1
        fps = _video_fps(video)
        if fps is not None and abs(fps - fps_target) > fps_tolerance:
            drops["fps_not_30"] += 1
            continue
        if duration < min_duration_sec:
            drops["too_short"] += 1
            continue
        kept.append(video)
    if not kept:
        raise ValueError("Every video was filtered out — check the filters and metadata.")
    return kept, dict(drops)


def select_diverse(
    videos: list[dict[str, Any]],
    target_hours: float,
    seed: int,
    scenario_cap_frac: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Greedily pick a scenario-diverse subset totaling ~target_hours.

    Deterministic seeded shuffle, then a capped pass: accept a video unless its primary
    scenario already holds more than `scenario_cap_frac` of the target hours. If the cap
    leaves the selection underfilled (few scenarios in the corpus), a second relaxed
    pass tops up with a warning, so the target is always reached when enough footage
    survives filtering.

    Args:
        videos: Filtered video dicts.
        target_hours: Total source hours to accumulate.
        seed: Shuffle seed (recorded in the manifest).
        scenario_cap_frac: Max fraction of target hours one primary scenario may hold.
    Returns:
        (selected, scenario_hours) — chosen videos and per-scenario hour totals.
    """
    rng = random.Random(seed)
    order = sorted(videos, key=lambda v: str(v["video_uid"]))
    rng.shuffle(order)
    cap_hours = scenario_cap_frac * target_hours
    selected: list[dict[str, Any]] = []
    scenario_hours: dict[str, float] = defaultdict(float)
    total = 0.0
    remaining: list[dict[str, Any]] = []
    for video in order:
        if total >= target_hours:
            break
        hours = float(video["duration_sec"]) / SECONDS_PER_HOUR
        scenario = primary_scenario(video)
        if scenario_hours[scenario] + hours > cap_hours:
            remaining.append(video)
            continue
        selected.append(video)
        scenario_hours[scenario] += hours
        total += hours
    if total < target_hours:
        print(
            f"WARN: scenario cap left selection at {total:.1f}h < {target_hours:.1f}h; "
            "topping up without the cap."
        )
        for video in remaining:
            if total >= target_hours:
                break
            hours = float(video["duration_sec"]) / SECONDS_PER_HOUR
            selected.append(video)
            scenario_hours[primary_scenario(video)] += hours
            total += hours
    if total < target_hours:
        print(f"WARN: corpus exhausted at {total:.1f}h < requested {target_hours:.1f}h.")
    return selected, dict(scenario_hours)


def split_train_val(
    selected: list[dict[str, Any]], val_fraction: float, seed: int
) -> dict[str, str]:
    """Assign each selected SOURCE VIDEO to train or validation by hours.

    Seeded shuffle, then videos accumulate into validation until it holds
    ~val_fraction of the total hours; everything else is train. Whole videos only —
    the anti-leakage contract.

    Args:
        selected: Videos chosen by `select_diverse`.
        val_fraction: Target fraction of total hours in validation.
        seed: Shuffle seed (offset from the selection seed so the orders differ).
    Returns:
        Mapping video_uid -> "train" | "validation".
    """
    rng = random.Random(seed + 1)
    order = sorted(selected, key=lambda v: str(v["video_uid"]))
    rng.shuffle(order)
    total_hours = sum(float(v["duration_sec"]) for v in order) / SECONDS_PER_HOUR
    val_target = val_fraction * total_hours
    split: dict[str, str] = {}
    val_hours = 0.0
    for video in order:
        uid = str(video["video_uid"])
        hours = float(video["duration_sec"]) / SECONDS_PER_HOUR
        if val_hours < val_target:
            split[uid] = "validation"
            val_hours += hours
        else:
            split[uid] = "train"
    if "train" not in split.values() or "validation" not in split.values():
        raise ValueError("Degenerate split: need at least one train and one validation video.")
    return split


def assign_batches(
    selected: list[dict[str, Any]], split: dict[str, str], n_batches: int
) -> dict[str, int]:
    """Partition selected videos into hour-balanced download batches (1-indexed).

    Longest-processing-time greedy per split: within train and validation separately,
    videos (sorted by duration desc, uid tiebreak) go to the currently lightest batch.
    Running LPT per split keeps every batch carrying both splits at roughly the global
    ratio, so per-batch chunking always populates both output directories.

    Args:
        selected: Videos chosen by `select_diverse`.
        split: video_uid -> split name from `split_train_val`.
        n_batches: Number of download batches.
    Returns:
        Mapping video_uid -> batch number in [1, n_batches].
    """
    batch_hours = [0.0] * n_batches
    assignment: dict[str, int] = {}
    for split_name in ("train", "validation"):
        members = [v for v in selected if split[str(v["video_uid"])] == split_name]
        members.sort(key=lambda v: (-float(v["duration_sec"]), str(v["video_uid"])))
        for video in members:
            lightest = min(range(n_batches), key=lambda b: batch_hours[b])
            uid = str(video["video_uid"])
            assignment[uid] = lightest + 1
            batch_hours[lightest] += float(video["duration_sec"]) / SECONDS_PER_HOUR
    return assignment


def write_outputs(
    out_dir: Path,
    selected: list[dict[str, Any]],
    split: dict[str, str],
    batches: dict[str, int],
    scenario_hours: dict[str, float],
    drop_counts: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Write the UID files and the selection manifest; return the manifest dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {"train": [], "validation": []}
    by_batch: dict[int, list[str]] = defaultdict(list)
    records = []
    for video in sorted(selected, key=lambda v: str(v["video_uid"])):
        uid = str(video["video_uid"])
        by_split[split[uid]].append(uid)
        by_batch[batches[uid]].append(uid)
        records.append(
            {
                "video_uid": uid,
                "duration_sec": float(video["duration_sec"]),
                "primary_scenario": primary_scenario(video),
                "scenarios": video.get("scenarios", []),
                "split": split[uid],
                "batch": batches[uid],
            }
        )
    (out_dir / "train_uids.txt").write_text("\n".join(by_split["train"]) + "\n")
    (out_dir / "val_uids.txt").write_text("\n".join(by_split["validation"]) + "\n")
    for batch_num in sorted(by_batch):
        (out_dir / f"batch_{batch_num}_uids.txt").write_text(
            "\n".join(by_batch[batch_num]) + "\n"
        )
    manifest = {
        "seed": args.seed,
        "target_hours": args.target_hours,
        "val_fraction": args.val_fraction,
        "batches": args.batches,
        "filters": {"min_duration_sec": 60.0, "fps_target": 30.0, "drops": drop_counts},
        "scenario_hours": {k: round(v, 3) for k, v in sorted(scenario_hours.items())},
        "totals": {
            "videos": len(records),
            "hours": round(sum(r["duration_sec"] for r in records) / SECONDS_PER_HOUR, 2),
            "train_videos": len(by_split["train"]),
            "val_videos": len(by_split["validation"]),
        },
        "videos": records,
    }
    (out_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the EGO4D UID-selection CLI (GUIDE.md Stage 3 step 5)."""
    parser = argparse.ArgumentParser(description="Select EGO4D source-video UIDs and batches.")
    parser.add_argument("--metadata", required=True, help="Path to the downloaded ego4d.json.")
    parser.add_argument(
        "--download-manifest",
        help="Optional video_540ss manifest.csv used to exclude metadata-only UIDs.",
    )
    parser.add_argument("--target-hours", type=float, default=210.0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="ego4d_manifests")
    return parser.parse_args()


def main() -> None:
    """Run selection end to end and print the per-scenario hour table."""
    args = parse_args()
    videos = load_video_records(Path(args.metadata))
    downloadable_uids = (
        load_downloadable_uids(Path(args.download_manifest)) if args.download_manifest else None
    )
    kept, drop_counts = filter_videos(videos, downloadable_uids=downloadable_uids)
    selected, scenario_hours = select_diverse(kept, args.target_hours, args.seed)
    split = split_train_val(selected, args.val_fraction, args.seed)
    batches = assign_batches(selected, split, args.batches)
    manifest = write_outputs(
        Path(args.out_dir), selected, split, batches, scenario_hours, drop_counts, args
    )
    print(f"Filtered {len(videos)} -> {len(kept)} usable videos; drops: {drop_counts}")
    print("Per-scenario hours (selection):")
    for scenario, hours in sorted(scenario_hours.items(), key=lambda kv: -kv[1]):
        print(f"  {hours:8.1f} h  {scenario}")
    totals = manifest["totals"]
    print(
        f"Selected {totals['videos']} videos / {totals['hours']} h "
        f"({totals['train_videos']} train / {totals['val_videos']} val) "
        f"into {args.batches} batches -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
