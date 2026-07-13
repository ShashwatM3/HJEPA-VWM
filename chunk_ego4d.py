"""Chunk downloaded EGO4D 540ss videos into SSv2-shaped 4-second training clips.

Turns each long-form EGO4D source video (30 fps, 540 px shorter side, from the CLI's
`video_540ss` dataset) into non-overlapping 4-second clips re-encoded to 12 fps with a
256 px shorter side (GUIDE.md Stage 0 "Option A"), so the training config stays
byte-identical to SSv2: `frame_stride=2` and `horizon_k` keep the same real-time meaning.
Windows overlapping a video's privacy `redacted_intervals` are skipped. Output filenames
are `<video_uid>_<window_index:05d>.mp4` under `<out-root>/<train|validation>/` per the
selection manifest's SOURCE-VIDEO split (the anti-leakage contract).

Batch-friendly by construction (GUIDE.md Stage 4): only manifest UIDs whose raw file is
PRESENT in `--raw-dir` are processed; missing ones are reported as pending, never errors.
Idempotent: existing non-empty outputs are skipped, so interrupted runs simply rerun.
`chunk_manifest.json` is rebuilt by rescanning `--out-root`, so it stays cumulative and
correct across the four batch invocations.

Note on 30 -> 12 fps: a non-integer (2.5x) decimation, so the constant-frame-rate resample
selects source frames at alternating 2/3-frame steps (~83 ms jitter) — accepted; far below
the 167 ms sampling interval the model sees at stride 2.

Pure stdlib + ffmpeg; no torch dependency (runs on a CPU pod).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation")
MAX_DEFAULT_WORKERS = 4


def require_ffmpeg() -> str:
    """Return the ffmpeg path or fail before starting worker processes.

    ffmpeg is an OS-level dependency rather than a Python package, so importing this
    module cannot guarantee that encoding is available.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg is required to chunk EGO4D videos but is not installed. "
            "Run: apt-get update && apt-get install -y ffmpeg"
        )
    return path


def default_workers() -> int:
    """Choose conservative process parallelism because each worker launches ffmpeg."""
    return min(MAX_DEFAULT_WORKERS, os.cpu_count() or MAX_DEFAULT_WORKERS)


def load_split_map(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load `selection_manifest.json` as a `video_uid -> {split, duration_sec}` map."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "videos" not in manifest:
        raise KeyError(f"{manifest_path} has no 'videos' list — not a selection manifest.")
    return {
        str(v["video_uid"]): {"split": v["split"], "duration_sec": float(v["duration_sec"])}
        for v in manifest["videos"]
    }


def load_redactions(metadata_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Load per-video redacted intervals from ego4d.json as (start_sec, end_sec) pairs.

    The interval schema is external, so parsing is defensive: dicts with
    start/end second keys and 2-element sequences are both accepted; anything else is
    reported once and skipped rather than crashing the chunker.
    """
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    redactions: dict[str, list[tuple[float, float]]] = {}
    unparsed = 0
    for video in raw.get("videos", []):
        uid = str(video.get("video_uid", ""))
        intervals = video.get("redacted_intervals") or []
        parsed: list[tuple[float, float]] = []
        for item in intervals:
            if isinstance(item, dict):
                start = item.get("start_sec", item.get("start_time"))
                end = item.get("end_sec", item.get("end_time"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                start, end = item
            else:
                start = end = None
            try:
                parsed.append((float(start), float(end)))
            except (TypeError, ValueError):
                unparsed += 1
        if parsed:
            redactions[uid] = parsed
    if unparsed:
        print(f"WARN: {unparsed} redacted-interval entries had an unrecognized shape; skipped.")
    return redactions


def chunk_windows(duration_sec: float, chunk_seconds: float) -> list[tuple[float, float]]:
    """Non-overlapping [t, t+chunk_seconds) windows covering the video, partial tail dropped.

    Args:
        duration_sec: Source video duration.
        chunk_seconds: Window length (4.0 in the committed recipe).
    Returns:
        List of (start_sec, end_sec) windows.
    """
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start + chunk_seconds <= duration_sec:
        windows.append((start, start + chunk_seconds))
        start += chunk_seconds
    return windows


def overlaps_redaction(
    window: tuple[float, float], intervals: list[tuple[float, float]]
) -> bool:
    """True when the window intersects any privacy-redacted interval."""
    start, end = window
    return any(start < r_end and r_start < end for r_start, r_end in intervals)


def _scale_filter(fps: int, shorter_side: int) -> str:
    """ffmpeg -vf string: CFR resample then shorter-side scale, portrait-safe, even dims."""
    return (
        f"fps={fps},"
        f"scale=w='if(gte(iw,ih),-2,{shorter_side})':h='if(gte(iw,ih),{shorter_side},-2)'"
    )


def encode_window(
    src: Path,
    out: Path,
    start_sec: float,
    chunk_seconds: float,
    fps: int,
    shorter_side: int,
    crf: int,
) -> bool:
    """Encode one window with ffmpeg; returns success, removing partial output on failure.

    `-ss` before `-i` fast-seeks on keyframes then decodes to the exact start; keyframe
    interval == fps (one per second) keeps decord random access cheap; `+faststart` puts
    the container index up front so per-item open cost stays low in the dataloader. ffmpeg
    writes a `.part.mp4` file that is atomically renamed only after a successful encode.
    """
    partial = out.with_name(f"{out.stem}.part{out.suffix}")
    partial.unlink(missing_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-threads",
        "1",
        "-i",
        str(src),
        "-t",
        f"{chunk_seconds:.3f}",
        "-vf",
        _scale_filter(fps, shorter_side),
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps),
        "-an",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not partial.exists() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        print(f"WARN: ffmpeg failed on {src.name} @ {start_sec:.0f}s: {result.stderr.strip()}")
        return False
    partial.replace(out)
    return True


def raise_for_failed_encodes(totals: dict[str, int]) -> None:
    """Fail the batch when any window failed, preventing unsafe raw-file deletion."""
    failed = totals.get("failed", 0)
    if failed:
        raise RuntimeError(
            f"{failed} ffmpeg window encodes failed. Keep the raw videos and rerun the "
            "idempotent chunk command with lower --workers until failed is 0."
        )


def chunk_one_video(
    uid: str,
    split: str,
    duration_sec: float,
    raw_dir: str,
    out_root: str,
    redactions: list[tuple[float, float]],
    chunk_seconds: float,
    fps: int,
    shorter_side: int,
    crf: int,
) -> dict[str, int]:
    """Chunk one source video; returns {encoded, skipped_existing, skipped_redacted, failed}.

    Top-level (picklable) so a ProcessPoolExecutor can parallelize across videos while
    each worker walks its own video's windows sequentially.
    """
    src = Path(raw_dir) / f"{uid}.mp4"
    out_dir = Path(out_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"encoded": 0, "skipped_existing": 0, "skipped_redacted": 0, "failed": 0}
    for index, window in enumerate(chunk_windows(duration_sec, chunk_seconds)):
        out = out_dir / f"{uid}_{index:05d}.mp4"
        if out.exists() and out.stat().st_size > 0:
            counts["skipped_existing"] += 1
            continue
        if overlaps_redaction(window, redactions):
            counts["skipped_redacted"] += 1
            continue
        if encode_window(src, out, window[0], chunk_seconds, fps, shorter_side, crf):
            counts["encoded"] += 1
        else:
            counts["failed"] += 1
    return counts


def rescan_chunk_manifest(out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild chunk_manifest.json from the filesystem (cumulative across batch runs)."""
    manifest: dict[str, Any] = {
        "chunk_seconds": args.chunk_seconds,
        "fps": args.fps,
        "shorter_side": args.shorter_side,
        "crf": args.crf,
        "ffmpeg": "libx264 preset=veryfast pix_fmt=yuv420p -g <fps> -an +faststart",
        "splits": {},
    }
    for split in SPLITS:
        files = sorted((out_root / split).glob("*.mp4")) if (out_root / split).is_dir() else []
        uids = {f.name[: f.name.rfind("_")] for f in files}
        manifest["splits"][split] = {
            "chunks": len(files),
            "source_uids": len(uids),
            "encoded_seconds": len(files) * args.chunk_seconds,
        }
    (out_root / "chunk_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the EGO4D chunker CLI (GUIDE.md Stage 4 step 2)."""
    parser = argparse.ArgumentParser(description="Chunk EGO4D 540ss videos into 4s clips.")
    parser.add_argument("--raw-dir", default="/workspace/ego4d_raw/v2/video_540ss")
    parser.add_argument("--manifest", required=True, help="selection_manifest.json path.")
    parser.add_argument("--metadata", required=True, help="ego4d.json path (redactions).")
    parser.add_argument("--out-root", default="/workspace/data/ego4d")
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--shorter-side", type=int, default=256)
    parser.add_argument("--crf", type=int, default=27)
    parser.add_argument("--workers", type=int, default=default_workers())
    return parser.parse_args()


def main() -> None:
    """Chunk every manifest UID whose raw file is present; report processed vs pending."""
    args = parse_args()
    require_ffmpeg()
    split_map = load_split_map(Path(args.manifest))
    redactions = load_redactions(Path(args.metadata))
    raw_dir = Path(args.raw_dir)
    present = [uid for uid in split_map if (raw_dir / f"{uid}.mp4").exists()]
    pending = sorted(set(split_map) - set(present))
    if pending:
        print(
            f"WARN: {len(pending)} manifest UIDs have no raw file in {raw_dir} "
            "(later batches, or not yet downloaded) — skipped, not failed."
        )
    totals = {"encoded": 0, "skipped_existing": 0, "skipped_redacted": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                chunk_one_video,
                uid,
                split_map[uid]["split"],
                split_map[uid]["duration_sec"],
                str(raw_dir),
                args.out_root,
                redactions.get(uid, []),
                args.chunk_seconds,
                args.fps,
                args.shorter_side,
                args.crf,
            ): uid
            for uid in present
        }
        for done, future in enumerate(as_completed(futures), start=1):
            counts = future.result()
            for key in totals:
                totals[key] += counts[key]
            if done % 25 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] videos done | cumulative {totals}")
    raise_for_failed_encodes(totals)
    manifest = rescan_chunk_manifest(Path(args.out_root), args)
    print(
        f"Processed {len(present)} of {len(split_map)} manifest UIDs "
        f"({len(pending)} still pending). This run: {totals}."
    )
    for split in SPLITS:
        info = manifest["splits"][split]
        print(f"  {split}: {info['chunks']} chunks from {info['source_uids']} source videos")
    print(f"Wrote {Path(args.out_root) / 'chunk_manifest.json'}")


if __name__ == "__main__":
    main()
