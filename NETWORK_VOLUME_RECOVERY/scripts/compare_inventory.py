#!/usr/bin/env python3
"""Compare two rclone `lsf --csv` inventories by path, size, and usable MD5."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

MD5 = re.compile(r"^[0-9a-fA-F]{32}$")


@dataclass(frozen=True)
class Entry:
    size: int
    hash: str | None


@dataclass
class Report:
    source_objects: int = 0
    destination_objects: int = 0
    source_bytes: int = 0
    destination_bytes: int = 0
    missing_on_destination: int = 0
    extra_on_destination: int = 0
    size_differences: int = 0
    comparable_hashes: int = 0
    hash_differences: int = 0


def read_inventory(path: Path) -> dict[str, Entry]:
    result: dict[str, Entry] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) == 2:
                size_text, object_path = row
                hash_text = ""
            elif len(row) == 3:
                size_text, hash_text, object_path = row
            else:
                raise ValueError(f"{path}:{line_number}: expected 2 or 3 columns, got {len(row)}")
            size = int(size_text)
            if object_path in result:
                raise ValueError(f"{path}:{line_number}: duplicate path {object_path!r}")
            usable_hash = hash_text.lower() if MD5.fullmatch(hash_text) else None
            result[object_path] = Entry(size=size, hash=usable_hash)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--details-out", type=Path)
    args = parser.parse_args()
    try:
        source = read_inventory(args.source)
        destination = read_inventory(args.destination)
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    report = Report(
        source_objects=len(source),
        destination_objects=len(destination),
        source_bytes=sum(entry.size for entry in source.values()),
        destination_bytes=sum(entry.size for entry in destination.values()),
    )
    details: list[str] = []
    for path in sorted(source.keys() - destination.keys()):
        report.missing_on_destination += 1
        details.append(f"MISSING_DEST\t{path}")
    for path in sorted(destination.keys() - source.keys()):
        report.extra_on_destination += 1
        details.append(f"EXTRA_DEST\t{path}")
    for path in sorted(source.keys() & destination.keys()):
        left, right = source[path], destination[path]
        if left.size != right.size:
            report.size_differences += 1
            details.append(f"SIZE\t{left.size}\t{right.size}\t{path}")
        if left.hash is not None and right.hash is not None:
            report.comparable_hashes += 1
            if left.hash != right.hash:
                report.hash_differences += 1
                details.append(f"MD5\t{left.hash}\t{right.hash}\t{path}")

    payload = asdict(report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.details_out:
        args.details_out.write_text("\n".join(details) + ("\n" if details else ""))
    mismatches = (
        report.missing_on_destination
        + report.extra_on_destination
        + report.size_differences
        + report.hash_differences
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
