#!/usr/bin/env python3
"""Rebuild safe directory paths from an rclone one-column CSV inventory.

Expected input is produced by:
    rclone lsf SOURCE -R --dirs-only --format p --csv

The helper never removes or replaces anything, refuses absolute/parent-traversal paths, refuses to
walk through symlinks, and deliberately skips RunPod's service-owned `.s3compat_uploads` tree.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


@dataclass
class Summary:
    inventory_rows: int = 0
    already_present: int = 0
    would_create: int = 0
    created: int = 0
    missing_in_check: int = 0
    service_internal_skipped: int = 0
    unsafe_paths: int = 0
    collisions: int = 0
    duplicates: int = 0

    @property
    def errors(self) -> int:
        return self.missing_in_check + self.unsafe_paths + self.collisions + self.duplicates


def parse_relative_directory(raw_value: str) -> PurePosixPath:
    value = raw_value.rstrip("/")
    raw_parts = value.split("/")
    if (
        not value
        or "\x00" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError(f"unsafe/empty directory path: {raw_value!r}")
    return PurePosixPath(*raw_parts)


def read_inventory(path: Path) -> list[PurePosixPath]:
    result: list[PurePosixPath] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 1:
                raise ValueError(
                    f"{path}:{line_number}: expected one CSV path field, got {len(row)}"
                )
            try:
                result.append(parse_relative_directory(row[0]))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return result


def first_bad_component(workspace: Path, relative: PurePosixPath) -> tuple[Path, str] | None:
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return current, "symlink"
        if current.exists() and not current.is_dir():
            return current, "non-directory"
    return None


def create_one(workspace: Path, relative: PurePosixPath) -> None:
    current = workspace
    for part in relative.parts:
        current = current / part
        if not current.exists():
            current.mkdir()


def rebuild(inventory: Path, workspace: Path, *, dry_run: bool, check: bool) -> Summary:
    workspace = workspace.resolve(strict=True)
    paths = read_inventory(inventory)
    summary = Summary(inventory_rows=len(paths))
    seen: set[str] = set()
    for relative in paths:
        key = relative.as_posix()
        if key in seen:
            summary.duplicates += 1
            print(f"ERROR: duplicate directory path: {key}", file=sys.stderr)
            continue
        seen.add(key)
        if relative.parts[0] == ".s3compat_uploads":
            summary.service_internal_skipped += 1
            continue
        bad = first_bad_component(workspace, relative)
        if bad is not None:
            summary.collisions += 1
            print(
                f"ERROR: {bad[1]} collision while rebuilding {relative}: {bad[0]}",
                file=sys.stderr,
            )
            continue
        target = workspace.joinpath(*relative.parts)
        if target.is_dir():
            summary.already_present += 1
        elif check:
            summary.missing_in_check += 1
            print(f"ERROR: expected directory missing: {target}", file=sys.stderr)
        elif dry_run:
            summary.would_create += 1
        else:
            create_one(workspace, relative)
            summary.created += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--workspace", default=Path("/workspace"), type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        summary = rebuild(
            args.inventory,
            args.workspace,
            dry_run=args.dry_run,
            check=args.check,
        )
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print("directory reconstruction summary:")
    for key, value in asdict(summary).items():
        print(f"  {key}: {value}")
    print(f"  errors: {summary.errors}")
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
