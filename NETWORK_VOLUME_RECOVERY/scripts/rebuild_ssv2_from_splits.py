#!/usr/bin/env python3
"""Rebuild the full SSv2 split symlinks and labels map from official split JSON.

The script refuses to replace any existing non-symlink path or incorrect symlink. It accepts the
usual list-of-records format with `id` and `template`/`label`, or an ID-to-label mapping whose IDs
resolve to raw videos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Record:
    video_id: str
    label: str


def _validate_id(value: Any, source: Path) -> str:
    video_id = str(value).strip()
    if not video_id or video_id in {".", ".."} or "/" in video_id or "\\" in video_id:
        raise ValueError(f"{source}: unsafe/empty video id {video_id!r}")
    if video_id.endswith(".webm"):
        video_id = video_id[:-5]
    return video_id


def read_records(path: Path, raw_root: Path) -> list[Record]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[Record] = []
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise TypeError(f"{path}: record {index} is not an object")
            raw_id = item.get("id") or item.get("video_id") or item.get("name")
            label = item.get("template") or item.get("label") or item.get("class")
            if raw_id is None or label is None or not str(label).strip():
                raise ValueError(f"{path}: record {index} lacks id and template/label")
            records.append(Record(_validate_id(raw_id, path), str(label)))
    elif isinstance(payload, dict):
        for raw_id, label in payload.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"{path}: mapping does not look like video-id -> string label; "
                    f"bad value for {raw_id!r}"
                )
            records.append(Record(_validate_id(raw_id, path), label))
    else:
        raise TypeError(f"{path}: expected JSON list or object, got {type(payload).__name__}")

    seen: set[str] = set()
    for record in records:
        if record.video_id in seen:
            raise ValueError(f"{path}: duplicate video id {record.video_id}")
        seen.add(record.video_id)
        target = raw_root / f"{record.video_id}.webm"
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError(f"{path}: raw target missing for {record.video_id}: {target}")
    return sorted(records, key=lambda record: record.video_id)


def expected_label_bytes(labels: dict[str, str]) -> bytes:
    return (json.dumps(labels, indent=2, sort_keys=True) + "\n").encode("utf-8")


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


def run(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve(strict=True)
    raw_root = workspace / "ssv2_raw" / "20bn-something-something-v2"
    require_safe_directory_path(workspace, raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw SSv2 directory missing: {raw_root}")
    for split in ("train", "validation"):
        require_safe_directory_path(workspace, workspace / "data" / "ssv2" / split)
    train = read_records(args.train_json, raw_root)
    validation = read_records(args.validation_json, raw_root)
    train_ids = {record.video_id for record in train}
    val_ids = {record.video_id for record in validation}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"train/validation IDs overlap: {overlap[:10]}")

    labels = {record.video_id: record.label for record in [*train, *validation]}
    collisions = 0
    missing = 0
    created = 0
    already = 0
    for split, records in (("train", train), ("validation", validation)):
        for record in records:
            link = workspace / "data" / "ssv2" / split / f"{record.video_id}.webm"
            target = raw_root / f"{record.video_id}.webm"
            if link.is_symlink():
                resolved = (link.parent / os.readlink(link)).resolve(strict=False)
                if resolved == target and target.exists():
                    already += 1
                else:
                    collisions += 1
                    print(
                        f"ERROR: incorrect symlink {link} -> {os.readlink(link)}", file=sys.stderr
                    )
                continue
            if link.exists():
                collisions += 1
                print(f"ERROR: non-symlink collision: {link}", file=sys.stderr)
                continue
            if args.check:
                missing += 1
                print(f"ERROR: missing expected link: {link}", file=sys.stderr)
            elif not args.dry_run:
                link.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(target), str(link))
                created += 1

    labels_path = workspace / "data" / "ssv2" / "labels.json"
    wanted = expected_label_bytes(labels)
    if labels_path.is_symlink():
        collisions += 1
        print(f"ERROR: labels map is a symlink; refusing it: {labels_path}", file=sys.stderr)
    elif labels_path.exists():
        try:
            existing_payload = json.loads(labels_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            collisions += 1
            print(
                f"ERROR: existing labels map is unreadable: {labels_path}: {exc}", file=sys.stderr
            )
        else:
            existing_labels = (
                {str(key): str(value) for key, value in existing_payload.items()}
                if isinstance(existing_payload, dict)
                else None
            )
            if existing_labels != labels:
                collisions += 1
                print(
                    f"ERROR: existing labels map differs; refusing to overwrite {labels_path}",
                    file=sys.stderr,
                )
    elif args.check:
        missing += 1
        print(f"ERROR: missing labels map: {labels_path}", file=sys.stderr)
    elif not args.dry_run:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = labels_path.with_suffix(".json.recovery-tmp")
        temporary.write_bytes(wanted)
        os.replace(temporary, labels_path)

    print(f"train records: {len(train)}")
    print(f"validation records: {len(validation)}")
    print(f"labels: {len(labels)}")
    print(f"already-correct links: {already}")
    print(f"created links: {created}")
    print(
        f"would-create links: {0 if not args.dry_run else len(train) + len(validation) - already}"
    )
    print(f"missing in check mode: {missing}")
    print(f"collisions: {collisions}")
    return 1 if collisions or missing else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        return run(parse_args())
    except (OSError, TypeError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
