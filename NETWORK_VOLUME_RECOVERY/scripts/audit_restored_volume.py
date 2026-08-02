#!/usr/bin/env python3
"""Read-only structural audit for a restored HJEPA-VWM `/workspace`.

The audit intentionally does not decode or hash every video. Use the project provenance and rclone
full-download checks for byte/frame-level proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EXPECTED_TOP_LEVEL = (
    "archive",
    "checkpoints",
    "ckpt",
    "data",
    "ego4d_raw",
    "hf_cache",
    "hierarchal-jepa-flow-world-model",
    "logs",
    "preflight",
    "ssv2_raw",
    "stats",
)

RECOVERY_CONTROL_TOP_LEVEL = (
    "recovery-input",
    "recovery-manifests",
)

DATASET_SPLITS = {
    "ssv2/train": ("data/ssv2/train", ".webm"),
    "ssv2/validation": ("data/ssv2/validation", ".webm"),
    "ssv2_tiny/train": ("data/ssv2_tiny/train", ".webm"),
    "ssv2_tiny/validation": ("data/ssv2_tiny/validation", ".webm"),
    "ego4d/train": ("data/ego4d/train", ".mp4"),
    "ego4d/validation": ("data/ego4d/validation", ".mp4"),
    "ego4d_tiny/train": ("data/ego4d_tiny/train", ".mp4"),
    "ego4d_tiny/validation": ("data/ego4d_tiny/validation", ".mp4"),
}

KEY_FILES = (
    "data/ssv2/labels.json",
    "data/ssv2_tiny/manifest.json",
    "data/ego4d/chunk_manifest.json",
    "data/ego4d_tiny/manifest.json",
    "ego4d_raw/ego4d.json",
    "ego4d_raw/video_540ss_manifest.csv",
    "ego4d_raw/manifests/selection_manifest.json",
)


@dataclass
class SplitAudit:
    exists: bool = False
    matching_entries: int = 0
    symlinks: int = 0
    regular_files: int = 0
    broken_symlinks: int = 0
    other_types: int = 0
    resolved_bytes: int = 0


@dataclass
class VolumeAudit:
    workspace: str
    expected_top_level_present: list[str] = field(default_factory=list)
    expected_top_level_missing: list[str] = field(default_factory=list)
    unexpected_top_level: list[str] = field(default_factory=list)
    top_level_types: dict[str, str] = field(default_factory=dict)
    dataset_splits: dict[str, SplitAudit] = field(default_factory=dict)
    raw_ssv2_webm: int = 0
    raw_ego4d_transient_mp4: int = 0
    key_files: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    whitening_files: list[dict[str, Any]] = field(default_factory=list)
    hf_snapshots: list[str] = field(default_factory=list)
    repo: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def type_name(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(root: Path, suffix: str) -> SplitAudit:
    result = SplitAudit(exists=root.is_dir())
    if not root.is_dir():
        return result
    with os.scandir(root) as iterator:
        for entry in iterator:
            if not entry.name.endswith(suffix):
                continue
            result.matching_entries += 1
            entry_path = Path(entry.path)
            if entry.is_symlink():
                result.symlinks += 1
                try:
                    target_stat = entry_path.stat()
                except FileNotFoundError:
                    result.broken_symlinks += 1
                else:
                    result.resolved_bytes += target_stat.st_size
            elif entry.is_file(follow_symlinks=False):
                result.regular_files += 1
                result.resolved_bytes += entry.stat(follow_symlinks=False).st_size
            else:
                result.other_types += 1
    return result


def list_files(root: Path, suffix: str) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    return [
        {"path": str(path), "size": path.stat().st_size}
        for path in sorted(root.rglob(f"*{suffix}"))
        if path.is_file()
    ]


def git_output(repo: Path, *args: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code": None, "error": str(exc)}
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout.rstrip(),
        "stderr": completed.stderr.rstrip(),
    }


def audit(workspace: Path) -> VolumeAudit:
    workspace = workspace.resolve(strict=True)
    result = VolumeAudit(workspace=str(workspace))
    actual_names: set[str] = set()
    for child in sorted(workspace.iterdir(), key=lambda path: path.name):
        actual_names.add(child.name)
        result.top_level_types[child.name] = type_name(child)
    result.expected_top_level_present = sorted(set(EXPECTED_TOP_LEVEL) & actual_names)
    result.expected_top_level_missing = sorted(set(EXPECTED_TOP_LEVEL) - actual_names)
    result.unexpected_top_level = sorted(
        actual_names - set(EXPECTED_TOP_LEVEL) - set(RECOVERY_CONTROL_TOP_LEVEL)
    )
    if result.expected_top_level_missing:
        result.warnings.append(
            "Expected historical top-level roots are missing; reconcile them with the source manifest: "
            + ", ".join(result.expected_top_level_missing)
        )

    for name, (relative, suffix) in DATASET_SPLITS.items():
        split = audit_split(workspace / relative, suffix)
        result.dataset_splits[name] = split
        if not split.exists or split.matching_entries == 0:
            result.errors.append(f"dataset split missing or empty: {name}")
        if split.broken_symlinks:
            result.errors.append(
                f"dataset split has {split.broken_symlinks} broken symlinks: {name}"
            )
        if split.other_types:
            result.errors.append(
                f"dataset split has {split.other_types} unexpected entry types: {name}"
            )

    raw_ssv2 = workspace / "ssv2_raw" / "20bn-something-something-v2"
    if raw_ssv2.is_dir():
        result.raw_ssv2_webm = sum(
            1 for path in raw_ssv2.iterdir() if path.is_file() and path.suffix == ".webm"
        )
    if result.raw_ssv2_webm == 0:
        result.errors.append("SSv2 raw video root is missing or contains no .webm files")

    transient = workspace / "ego4d_raw" / "v2" / "video_540ss"
    if transient.is_dir():
        result.raw_ego4d_transient_mp4 = sum(
            1 for path in transient.iterdir() if path.is_file() and path.suffix == ".mp4"
        )
    if result.raw_ego4d_transient_mp4:
        result.warnings.append(
            f"{result.raw_ego4d_transient_mp4} transient EGO4D raw MP4 files remain; preserve and inspect"
        )

    for relative in KEY_FILES:
        path = workspace / relative
        if path.is_file():
            result.key_files[relative] = {
                "exists": True,
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        else:
            result.key_files[relative] = {"exists": False}
            result.errors.append(f"key manifest/file missing: {relative}")

    result.checkpoints = list_files(workspace / "ckpt", ".pt") + list_files(
        workspace / "checkpoints", ".pt"
    )
    if not result.checkpoints:
        result.warnings.append(
            "no .pt checkpoints found under /workspace/ckpt or /workspace/checkpoints"
        )
    result.whitening_files = list_files(workspace / "stats", ".pt")
    if not result.whitening_files:
        result.warnings.append("no standalone whitening .pt files found under /workspace/stats")

    hf_cache = workspace / "hf_cache"
    if hf_cache.is_dir():
        snapshots = set()
        for snapshots_root in hf_cache.glob("**/snapshots"):
            if not snapshots_root.is_dir():
                continue
            for child in snapshots_root.iterdir():
                if child.is_dir() and len(child.name) == 40:
                    snapshots.add(str(child.relative_to(workspace)))
        result.hf_snapshots = sorted(snapshots)
    if not result.hf_snapshots:
        result.warnings.append("no 40-character Hugging Face snapshot directories detected")

    repo = workspace / "hierarchal-jepa-flow-world-model"
    result.repo = {
        "exists": repo.is_dir(),
        "git_directory_exists": (repo / ".git").exists(),
    }
    if repo.is_dir() and (repo / ".git").exists():
        result.repo["head"] = git_output(repo, "rev-parse", "HEAD")
        result.repo["status"] = git_output(repo, "status", "--short")
    else:
        result.errors.append("expected Git repository/.git is missing")
    return result


def serialize(result: VolumeAudit) -> dict[str, Any]:
    payload = asdict(result)
    return payload


def print_summary(result: VolumeAudit) -> None:
    print(f"workspace: {result.workspace}")
    print(f"expected roots present: {len(result.expected_top_level_present)}")
    print(f"expected roots missing: {result.expected_top_level_missing}")
    print(f"unexpected roots: {result.unexpected_top_level}")
    print("dataset splits:")
    for name, split in result.dataset_splits.items():
        print(
            f"  {name}: entries={split.matching_entries} links={split.symlinks} "
            f"regular={split.regular_files} broken={split.broken_symlinks} "
            f"bytes={split.resolved_bytes}"
        )
    print(f"raw SSv2 .webm: {result.raw_ssv2_webm}")
    print(f"transient raw EGO4D .mp4: {result.raw_ego4d_transient_mp4}")
    print(f"checkpoints: {len(result.checkpoints)}")
    print(f"whitening files: {len(result.whitening_files)}")
    print(f"HF snapshots detected: {len(result.hf_snapshots)}")
    for warning in result.warnings:
        print(f"WARN: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"warnings: {len(result.warnings)}")
    print(f"errors: {len(result.errors)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.workspace)
        payload = serialize(result)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, args.json_out)
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print_summary(result)
    print(f"JSON report: {args.json_out}")
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
