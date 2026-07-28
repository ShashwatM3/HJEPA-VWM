#!/usr/bin/env python3
"""Rebuild HJEPA-VWM tiny-dataset symlinks from a preserved manifest.

Supported manifests are produced by this repository's `make_subset.py` (SSv2) and
`make_ego4d_subset.py` (EGO4D). The script validates manifest counts and target existence. It never
replaces an existing regular file, directory, or incorrect symlink.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LinkSpec:
    split: str
    filename: str
    target: Path


@dataclass
class Summary:
    manifest_entries: int = 0
    already_correct: int = 0
    would_create: int = 0
    created: int = 0
    missing_links_in_check: int = 0
    missing_targets: int = 0
    collisions: int = 0

    @property
    def errors(self) -> int:
        return self.missing_links_in_check + self.missing_targets + self.collisions


def safe_filename(value: Any, suffix: str, *, allow_missing_suffix: bool) -> str:
    filename = str(value).strip()
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError(f"unsafe/empty manifest filename: {filename!r}")
    if allow_missing_suffix and not filename.endswith(suffix):
        filename += suffix
    if not filename.endswith(suffix):
        raise ValueError(f"manifest filename has wrong suffix: {filename!r}; expected {suffix}")
    return filename


def require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object")
    return value


def require_list(value: Any, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{description} must be a JSON list")
    return value


def ssv2_specs(payload: dict[str, Any], workspace: Path) -> Iterable[LinkSpec]:
    splits = require_mapping(payload.get("splits"), "manifest.splits")
    raw_root = workspace / "ssv2_raw" / "20bn-something-something-v2"
    for split in ("train", "validation"):
        info = require_mapping(splits.get(split), f"manifest.splits.{split}")
        per_class = require_mapping(info.get("per_class"), f"manifest.splits.{split}.per_class")
        entries: list[LinkSpec] = []
        for label, ids in per_class.items():
            for video_id in require_list(ids, f"manifest {split} class {label!r}"):
                filename = safe_filename(video_id, ".webm", allow_missing_suffix=True)
                entries.append(LinkSpec(split, filename, raw_root / filename))
        validate_declared_count(info, entries, split)
        yield from entries


def ego4d_specs(payload: dict[str, Any], workspace: Path) -> Iterable[LinkSpec]:
    splits = require_mapping(payload.get("splits"), "manifest.splits")
    full_root = workspace / "data" / "ego4d"
    uid_split: dict[str, str] = {}
    for split in ("train", "validation"):
        info = require_mapping(splits.get(split), f"manifest.splits.{split}")
        per_video = require_mapping(info.get("per_video"), f"manifest.splits.{split}.per_video")
        entries: list[LinkSpec] = []
        for raw_uid, filenames in per_video.items():
            uid = str(raw_uid).strip()
            if not uid or uid in {".", ".."} or "/" in uid or "\\" in uid:
                raise ValueError(f"unsafe/empty EGO4D source UID: {uid!r}")
            previous_split = uid_split.get(uid)
            if previous_split is not None and previous_split != split:
                raise ValueError(
                    f"EGO4D source UID crosses splits: {uid!r} in {previous_split} and {split}"
                )
            uid_split[uid] = split
            for value in require_list(filenames, f"manifest {split} video {uid!r}"):
                filename = safe_filename(value, ".mp4", allow_missing_suffix=False)
                if not filename.startswith(f"{uid}_"):
                    raise ValueError(
                        f"EGO4D manifest filename {filename!r} does not belong to UID {uid!r}"
                    )
                entries.append(LinkSpec(split, filename, full_root / split / filename))
        validate_declared_count(info, entries, split)
        yield from entries


def validate_declared_count(info: dict[str, Any], entries: list[LinkSpec], split: str) -> None:
    declared = info.get("count")
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise TypeError(f"manifest split {split!r} has no integer count")
    if declared != len(entries):
        raise ValueError(
            f"manifest split {split!r} count mismatch: declared={declared}, listed={len(entries)}"
        )


def load_specs(manifest: Path, dataset: str, workspace: Path) -> list[LinkSpec]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    root = require_mapping(payload, "manifest")
    specs = list(ssv2_specs(root, workspace) if dataset == "ssv2" else ego4d_specs(root, workspace))
    if not specs:
        raise ValueError("manifest contains no tiny-dataset entries")
    seen: set[str] = set()
    for spec in specs:
        if spec.filename in seen:
            raise ValueError(
                f"duplicate/cross-split manifest filename: {spec.split}/{spec.filename}"
            )
        seen.add(spec.filename)
    return specs


def require_safe_directory_path(workspace: Path, directory: Path) -> None:
    """Refuse parent components that could redirect writes outside workspace."""
    relative = directory.relative_to(workspace)
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink parent collision: {current}")
        if current.exists() and not current.is_dir():
            raise ValueError(f"non-directory parent collision: {current}")


def rebuild(
    manifest: Path,
    dataset: str,
    workspace: Path,
    *,
    dry_run: bool,
    check: bool,
) -> Summary:
    workspace = workspace.resolve(strict=True)
    specs = load_specs(manifest, dataset, workspace)
    tiny_root = workspace / "data" / f"{dataset}_tiny"
    for split in ("train", "validation"):
        require_safe_directory_path(workspace, tiny_root / split)
    if dataset == "ssv2":
        require_safe_directory_path(
            workspace, workspace / "ssv2_raw" / "20bn-something-something-v2"
        )
    else:
        for split in ("train", "validation"):
            require_safe_directory_path(workspace, workspace / "data" / "ego4d" / split)
    summary = Summary(manifest_entries=len(specs))
    for spec in specs:
        link = tiny_root / spec.split / spec.filename
        target = spec.target
        if target.is_symlink() or not target.is_file():
            summary.missing_targets += 1
            print(
                f"ERROR: target missing/not a non-symlink regular file for {link}: {target}",
                file=sys.stderr,
            )
            continue
        if link.is_symlink():
            current_text = os.readlink(link)
            resolved = (link.parent / current_text).resolve(strict=False)
            if resolved == target and target.exists():
                summary.already_correct += 1
            else:
                summary.collisions += 1
                print(
                    f"ERROR: incorrect existing symlink: {link} -> {current_text}; expected {target}",
                    file=sys.stderr,
                )
            continue
        if link.exists():
            summary.collisions += 1
            print(f"ERROR: existing non-symlink collision: {link}", file=sys.stderr)
            continue
        if check:
            summary.missing_links_in_check += 1
            print(f"ERROR: expected link is missing: {link} -> {target}", file=sys.stderr)
            continue
        if dry_run:
            summary.would_create += 1
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(target), str(link))
        summary.created += 1
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ssv2", "ego4d"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and report creations only.")
    mode.add_argument("--check", action="store_true", help="Require every manifest link to exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = rebuild(
            args.manifest,
            args.dataset,
            args.workspace,
            dry_run=args.dry_run,
            check=args.check,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print("tiny-link reconstruction summary:")
    for key, value in asdict(summary).items():
        print(f"  {key}: {value}")
    print(f"  errors: {summary.errors}")
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
