#!/usr/bin/env python3
"""Restore tracked regular-file execute bits from a recovered Git index.

Content is never replaced. Untracked files and Git symlinks are left unchanged.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path


def indexed_entries(repo: Path) -> list[tuple[str, Path]]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--stage", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    entries: list[tuple[str, Path]] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        metadata, separator, raw_path = raw.partition(b"\t")
        if not separator:
            raise ValueError(f"unexpected git ls-files record: {raw!r}")
        mode, _object_id, stage_number = metadata.decode("ascii").split()
        if stage_number != "0":
            raise ValueError(f"unmerged index entry at {os.fsdecode(raw_path)!r}")
        entries.append((mode, repo / Path(os.fsdecode(raw_path))))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    changed = 0
    already = 0
    skipped = 0
    errors = 0
    try:
        entries = indexed_entries(repo)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    for git_mode, path in entries:
        if git_mode not in {"100644", "100755"}:
            skipped += 1
            continue
        if not path.exists() or path.is_symlink() or not path.is_file():
            errors += 1
            print(f"ERROR: tracked regular file missing/wrong type: {path}", file=sys.stderr)
            continue
        current = stat.S_IMODE(path.stat().st_mode)
        if git_mode == "100755":
            wanted = current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        else:
            wanted = current & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if wanted == current:
            already += 1
            continue
        print(f"{'WOULD ' if args.dry_run else ''}CHMOD {current:04o} -> {wanted:04o} {path}")
        if not args.dry_run:
            os.chmod(path, wanted, follow_symlinks=False)
        changed += 1
    print(f"indexed entries: {len(entries)}")
    print(f"changed/would-change: {changed}")
    print(f"already correct: {already}")
    print(f"non-regular modes skipped: {skipped}")
    print(f"errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
