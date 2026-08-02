"""Parse a training log (`step=N {dict}` per line) into a structured JSON.

Each line of the input is expected to look like::

    step=42 {'loss': 1.23, 'L_flow': 1.20, ...}

where the dict is a Python literal. Lines that don't match this shape (blank
lines, wandb banners, shell prompt residue, `real/user/sys` timing, etc.) are
skipped, but a count of skipped lines is preserved in the output for sanity.

Different steps may log different keys (e.g. step 0 often carries extra
diagnostics). The script unions all keys across lines so every entry in the
output has the same schema, with missing values written as ``null``.

Usage::

    python parse_logs.py logs/1.txt                 # writes logs/1.json
    python parse_logs.py logs/1.txt -o out/foo.json # custom output path
    python parse_logs.py logs/1.txt --stdout        # print JSON to stdout
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

STEP_LINE_RE = re.compile(r"^\s*step\s*=\s*(\d+)\s*(\{.*\})\s*$")


def parse_line(line: str) -> tuple[int, dict[str, Any]] | None:
    """Return ``(step, metrics)`` if the line matches, else ``None``."""
    match = STEP_LINE_RE.match(line)
    if not match:
        return None
    step = int(match.group(1))
    try:
        metrics = ast.literal_eval(match.group(2))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(metrics, dict):
        return None
    return step, {str(k): v for k, v in metrics.items()}


def parse_log(path: Path) -> dict[str, Any]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    entries: list[dict[str, Any]] = []
    field_order: list[str] = ["step"]
    seen: set[str] = {"step"}
    skipped = 0

    for line in raw_lines:
        if not line.strip():
            continue
        parsed = parse_line(line)
        if parsed is None:
            skipped += 1
            continue
        step, metrics = parsed
        for key in metrics:
            if key not in seen:
                seen.add(key)
                field_order.append(key)
        entries.append({"step": step, **metrics})

    entries.sort(key=lambda e: e["step"])

    normalized = [{key: entry.get(key, None) for key in field_order} for entry in entries]

    steps = [e["step"] for e in normalized]
    return {
        "source_file": str(path),
        "num_steps": len(normalized),
        "step_range": [min(steps), max(steps)] if steps else None,
        "fields": field_order,
        "skipped_lines": skipped,
        "steps": normalized,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="Path to the .txt log file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (defaults to <input>.json next to the input)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2; pass 0 for compact)",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 1

    payload = parse_log(args.input)
    indent = args.indent if args.indent > 0 else None
    serialized = json.dumps(payload, indent=indent, ensure_ascii=False)

    if args.stdout:
        print(serialized)
    else:
        out_path = args.output or args.input.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(serialized + "\n", encoding="utf-8")
        print(
            f"parsed {payload['num_steps']} steps "
            f"({len(payload['fields']) - 1} metric fields, "
            f"{payload['skipped_lines']} non-step lines skipped) -> {out_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
