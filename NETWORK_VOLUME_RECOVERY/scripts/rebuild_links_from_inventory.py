#!/usr/bin/env python3
"""Rebuild HJEPA-VWM's documented symlink views from an rclone CSV inventory.

Expected inventory input is produced by:
    rclone lsf SOURCE -R --files-only --format sp --csv

The script recognizes only these link families:
    data/ssv2/{train,validation}/*.webm
    data/ssv2_tiny/{train,validation}/*.webm
    data/ego4d_tiny/{train,validation}/*.mp4

It never replaces an existing regular file, directory, or incorrect symlink.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


@dataclass
class Summary:
    inventory_rows: int = 0
    recognized: int = 0
    already_correct: int = 0
    would_create: int = 0
    created: int = 0
    missing_links_in_check: int = 0
    missing_targets: int = 0
    target_size_mismatches: int = 0
    collisions: int = 0
    unsafe_paths: int = 0
    duplicate_paths: int = 0

    @property
    def errors(self) -> int:
        return (
            self.missing_links_in_check
            + self.missing_targets
            + self.target_size_mismatches
            + self.collisions
            + self.unsafe_paths
            + self.duplicate_paths
        )


@dataclass(frozen=True)
class InventoryEntry:
    size: int
    path: str


def parse_inventory(path: Path) -> list[InventoryEntry]:
    """Parse rclone `--format sp --csv` (also tolerates `shp`) without a header."""
    entries: list[InventoryEntry] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) == 2:
                size_text, object_path = row
            elif len(row) == 3:
                size_text, _hash_text, object_path = row
            else:
                raise ValueError(
                    f"{path}:{line_number}: expected 2 (sp) or 3 (shp) CSV fields, got {len(row)}"
                )
            try:
                size = int(size_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: first CSV field is not an integer size: {size_text!r}"
                ) from exc
            entries.append(InventoryEntry(size=size, path=object_path))
    return entries


def classify_link(object_path: str, family: str) -> tuple[PurePosixPath, PurePosixPath] | None:
    """Return (link-relative-path, target-relative-path) for a recognized safe view."""
    candidate = PurePosixPath(object_path)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ValueError(f"unsafe inventory path: {object_path!r}")
    parts = candidate.parts
    if len(parts) != 4 or parts[0] != "data" or parts[2] not in {"train", "validation"}:
        return None
    dataset, split, filename = parts[1], parts[2], parts[3]
    if "/" in filename or filename in {"", ".", ".."}:
        raise ValueError(f"unsafe filename: {object_path!r}")

    if dataset in {"ssv2", "ssv2_tiny"} and filename.endswith(".webm"):
        if family not in {"all", "ssv2"}:
            return None
        target = PurePosixPath("ssv2_raw/20bn-something-something-v2") / filename
        return candidate, target
    if dataset == "ego4d_tiny" and filename.endswith(".mp4"):
        if family not in {"all", "ego4d"}:
            return None
        target = PurePosixPath("data/ego4d") / split / filename
        return candidate, target
    return None


def iter_recognized(
    entries: Iterable[InventoryEntry], family: str, summary: Summary
) -> Iterable[tuple[InventoryEntry, PurePosixPath, PurePosixPath]]:
    seen: set[str] = set()
    for entry in entries:
        summary.inventory_rows += 1
        try:
            classified = classify_link(entry.path, family)
        except ValueError as exc:
            summary.unsafe_paths += 1
            print(f"ERROR: {exc}", file=sys.stderr)
            continue
        if classified is None:
            continue
        if entry.path in seen:
            summary.duplicate_paths += 1
            print(f"ERROR: duplicate recognized inventory path: {entry.path}", file=sys.stderr)
            continue
        seen.add(entry.path)
        summary.recognized += 1
        link_rel, target_rel = classified
        yield entry, link_rel, target_rel


def first_bad_parent(
    workspace: Path, relative_parent: PurePosixPath
) -> tuple[Path, str] | None:
    """Return a symlink/non-directory ancestor without following it outside workspace."""
    current = workspace
    for part in relative_parent.parts:
        current = current / part
        if current.is_symlink():
            return current, "symlink"
        if current.exists() and not current.is_dir():
            return current, "non-directory"
    return None


def rebuild(
    inventory: Path,
    workspace: Path,
    family: str,
    *,
    dry_run: bool,
    check: bool,
) -> Summary:
    workspace = workspace.resolve(strict=True)
    summary = Summary()
    entries = parse_inventory(inventory)

    for entry, link_rel, target_rel in iter_recognized(entries, family, summary):
        link = workspace.joinpath(*link_rel.parts)
        target = workspace.joinpath(*target_rel.parts)

        bad = first_bad_parent(workspace, link_rel.parent)
        if bad is None:
            bad = first_bad_parent(workspace, target_rel.parent)
        if bad is not None:
            summary.collisions += 1
            print(
                f"ERROR: {bad[1]} parent collision while rebuilding {link}: {bad[0]}",
                file=sys.stderr,
            )
            continue

        try:
            target_stat = target.stat()
        except FileNotFoundError:
            summary.missing_targets += 1
            print(f"ERROR: target missing for {link}: {target}", file=sys.stderr)
            continue
        if target.is_symlink() or not target.is_file():
            summary.missing_targets += 1
            print(
                f"ERROR: target is not a non-symlink regular file for {link}: {target}",
                file=sys.stderr,
            )
            continue
        if target_stat.st_size != entry.size:
            summary.target_size_mismatches += 1
            print(
                f"ERROR: target size mismatch for {link}: inventory={entry.size}, "
                f"target={target_stat.st_size}, target_path={target}",
                file=sys.stderr,
            )
            continue

        if link.is_symlink():
            current_text = os.readlink(link)
            current_resolved = (link.parent / current_text).resolve(strict=False)
            if current_resolved == target and target.exists():
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
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--family", choices=("all", "ssv2", "ego4d"), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and report creations only.")
    mode.add_argument(
        "--check", action="store_true", help="Require every recognized link to exist."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = rebuild(
            args.inventory,
            args.workspace,
            args.family,
            dry_run=args.dry_run,
            check=args.check,
        )
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print("link reconstruction summary:")
    for key, value in asdict(summary).items():
        print(f"  {key}: {value}")
    print(f"  errors: {summary.errors}")
    if summary.recognized == 0:
        print("FATAL: inventory contained no recognized link-family paths", file=sys.stderr)
        return 2
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
