"""Download full metric history for a W&B run via the Public API.

Uses ``wandb.Api`` and ``run.scan_history()`` (unsampled) per the W&B docs:
https://docs.wandb.ai/models/track/public-api-guide

Default target is investigation 005 resume run ``drawn-elevator-16``
(``0n5mx3qf`` on project ``hjepa-vwm``).

Auth: ``wandb login`` or ``WANDB_API_KEY`` in the environment.

Examples::

    python run_history.py
    python run_history.py --run 0n5mx3qf
    python run_history.py --run smahalanobis-uc-davis/hjepa-vwm/0n5mx3qf --report
    python run_history.py --run 0n5mx3qf -o logs/drawn-elevator-16.json
    python run_history.py --run 0n5mx3qf --format parse_logs -o logs/drawn-elevator-16/output.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

DEFAULT_ENTITY = "smahalanobis-uc-davis"
DEFAULT_PROJECT = "hjepa-vwm"
DEFAULT_RUN_ID = "0n5mx3qf"

# Metrics logged every ``log_every`` steps (train.py).
CORE_METRICS = (
    "loss",
    "L_flow",
    "L_var",
    "L_cov",
    "L_slot",
    "L_sigreg",
    "L_recon",
    "L_recon_pred",
    "grad_norm",
    "grad_skipped",
    "instability_warn",
    "lr_mult",
    "loss/rollout",
    "rollout/lambda_effective",
    "rollout/copy_ratio",
    "rollout/displacement_cosine",
    "rollout/displacement_norm_ratio",
    "rollout/target_displacement_valid",
)

# Extra diagnostics logged every ``diag_every`` steps (train.py).
DIAG_METRICS = (
    "c_std_mean",
    "c_cross_video_cosine",
    "c_effective_rank",
    "c_plus_effective_rank",
    "coarse_copy_loss",
    "coarse_vs_copy_ratio",
    "coarse_vs_batch_mean_ratio",
    "coarse_condition_shuffle_degradation",
    "teacher_forced_random_tau_coarse_copy_loss",
    "teacher_forced_random_tau_coarse_vs_copy_ratio",
    "teacher_forced_random_tau_coarse_vs_batch_mean_ratio",
    "teacher_forced_random_tau_coarse_condition_shuffle_degradation",
    "grad_has_nan",
    "L_recon_present",
    "L_recon_shuffled_c",
    "L_recon_cplus",
    "L_recon_chat",
)

PHASE1_REPORT_METRICS = (
    "coarse_vs_copy_ratio",
    "coarse_vs_batch_mean_ratio",
    "c_effective_rank",
    "c_cross_video_cosine",
    "c_std_mean",
    "c_dead_dim_frac",
    "grad_norm",
    "grad_skipped",
    "loss",
)


def _json_default(value: Any) -> Any:
    """Serialize W&B config/summary values for JSON output."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    return str(value)


def resolve_run_path(
    run: str | None,
    *,
    entity: str,
    project: str,
    default_run_id: str,
) -> str:
    """Return ``entity/project/run_id`` from a run id or full path."""
    raw = (run or default_run_id).strip().strip("/")
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3:
            return "/".join(parts)
        if len(parts) == 2:
            return f"{entity}/{parts[0]}/{parts[1]}"
        raise ValueError(f"invalid run path {raw!r}; expected run_id or entity/project/run_id")
    return f"{entity}/{project}/{raw}"


def fetch_run(api: Any, run_path: str) -> Any:
    """Load a W&B run object."""
    return api.run(run_path)


def run_metadata(run: Any, run_path: str) -> dict[str, Any]:
    """Collect run identity, config, and summary (no history download)."""
    entity, project, run_id = run_path.split("/", 2)
    config = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
    summary: dict[str, Any]
    if hasattr(run.summary, "_json_dict"):
        summary = dict(run.summary._json_dict)
    else:
        summary = {k: v for k, v in dict(run.summary).items() if not str(k).startswith("_")}

    url = getattr(run, "url", None) or f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
    return {
        "path": run_path,
        "entity": entity,
        "project": project,
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": url,
        "created_at": str(getattr(run, "created_at", "")),
        "heartbeat_at": str(getattr(run, "heartbeat_at", "")),
        "runtime_seconds": (
            _safe_float(summary.get("_wandb", {}).get("runtime"))
            if isinstance(summary.get("_wandb"), dict)
            else _safe_float(summary.get("_runtime"))
        ),
        "config": _json_default(config),
        "summary": _json_default(summary),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _normalize_history_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map W&B history row to parse_logs-style ``{step, metric...}``."""
    out: dict[str, Any] = {}
    step = row.get("_step", row.get("step"))
    if step is not None:
        out["step"] = int(step)
    for key, value in row.items():
        if key in {"_step", "step"}:
            continue
        if str(key).startswith("_"):
            continue
        if value is None:
            out[str(key)] = None
            continue
        if isinstance(value, (int, float, bool, str)):
            out[str(key)] = value
            continue
        out[str(key)] = _json_default(value)
    return out


def fetch_history(
    run: Any,
    *,
    page_size: int,
    min_step: int,
    max_step: int | None,
    keys: list[str] | None,
) -> list[dict[str, Any]]:
    """Download unsampled history via ``run.scan_history()``."""
    kwargs: dict[str, Any] = {
        "page_size": page_size,
        "min_step": min_step,
    }
    if max_step is not None:
        kwargs["max_step"] = max_step
    if keys:
        kwargs["keys"] = keys

    rows = [_normalize_history_row(dict(row)) for row in run.scan_history(**kwargs)]
    rows.sort(key=lambda r: r.get("step", -1))
    return rows


def union_field_order(rows: list[dict[str, Any]]) -> list[str]:
    """Stable union of metric keys across history rows (like parse_logs.py)."""
    order = ["step"]
    seen = {"step"}
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                order.append(key)
    return order


def normalize_rows(rows: list[dict[str, Any]], field_order: list[str]) -> list[dict[str, Any]]:
    return [{key: row.get(key) for key in field_order} for row in rows]


def build_payload(
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    field_order = union_field_order(rows)
    normalized = normalize_rows(rows, field_order)
    steps = [r["step"] for r in normalized if r.get("step") is not None]
    return {
        "run": meta,
        "history": {
            "num_steps": len(normalized),
            "step_range": [min(steps), max(steps)] if steps else None,
            "fields": field_order,
            "steps": normalized,
        },
    }


def format_parse_logs(rows: list[dict[str, Any]]) -> str:
    """Emit ``step=N {dict}`` lines compatible with parse_logs.py."""
    lines: list[str] = []
    for row in rows:
        step = row.get("step")
        if step is None:
            continue
        metrics = {k: v for k, v in row.items() if k != "step"}
        lines.append(f"step={step} {metrics!r}")
    return "\n".join(lines) + ("\n" if lines else "")


def _last_row_with(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for row in reversed(rows):
        if row.get(key) is not None:
            return row
    return None


def _rows_with(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get(key) is not None]


def phase1_report(meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Human-readable Phase 1 gate snapshot from downloaded history."""
    lines: list[str] = []
    lines.append(f"Run: {meta['name']} ({meta['id']}) — {meta['state']}")
    lines.append(f"URL: {meta['url']}")
    if rows:
        steps = [r["step"] for r in rows if r.get("step") is not None]
        lines.append(f"History: {len(rows)} logged rows, steps {min(steps)}–{max(steps)}")
    else:
        lines.append("History: (empty)")
        return "\n".join(lines)

    diag_rows = _rows_with(rows, "c_effective_rank")
    if diag_rows:
        last_diag = diag_rows[-1]
        lines.append("")
        lines.append("Latest diagnostic row (diag_every metrics):")
        for key in PHASE1_REPORT_METRICS:
            if key in last_diag and last_diag[key] is not None:
                lines.append(f"  {key}: {last_diag[key]}")

    skipped = [r for r in rows if r.get("grad_skipped")]
    if skipped:
        first = skipped[0]["step"]
        lines.append("")
        lines.append(
            f"grad_skipped: {len(skipped)}/{len(rows)} rows "
            f"({100.0 * len(skipped) / len(rows):.1f}%), first at step {first}"
        )
    else:
        lines.append("")
        lines.append("grad_skipped: 0 rows (healthy)")

    post_10k = [r for r in diag_rows if (r.get("step") or 0) >= 10_000]
    if post_10k:
        copy_ratios = [
            r["coarse_vs_copy_ratio"] for r in post_10k if r.get("coarse_vs_copy_ratio") is not None
        ]
        batch_ratios = [
            r["coarse_vs_batch_mean_ratio"]
            for r in post_10k
            if r.get("coarse_vs_batch_mean_ratio") is not None
        ]
        ranks = [r["c_effective_rank"] for r in post_10k if r.get("c_effective_rank") is not None]
        lines.append("")
        lines.append("Post-10k diagnostic aggregates:")
        if copy_ratios:
            lines.append(
                f"  coarse_vs_copy_ratio: min={min(copy_ratios):.4f}, "
                f"median={sorted(copy_ratios)[len(copy_ratios) // 2]:.4f}, "
                f"max={max(copy_ratios):.4f} (gate ≤ 0.70)"
            )
        if batch_ratios:
            lines.append(
                f"  coarse_vs_batch_mean_ratio: min={min(batch_ratios):.4f}, "
                f"median={sorted(batch_ratios)[len(batch_ratios) // 2]:.4f}, "
                f"max={max(batch_ratios):.4f} (gate ≤ 0.50)"
            )
        if ranks:
            lines.append(
                f"  c_effective_rank: min={min(ranks):.2f}, "
                f"median={sorted(ranks)[len(ranks) // 2]:.2f}, "
                f"max={max(ranks):.2f} (gate > 60)"
            )

    summary = meta.get("summary") or {}
    final_step = summary.get("_step")
    if final_step is not None:
        lines.append("")
        lines.append(f"Summary _step: {final_step}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download unsampled W&B run history (Public API).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run",
        default=None,
        help=f"Run id or entity/project/run_id (default: {DEFAULT_RUN_ID})",
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="W&B entity when --run is an id")
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT, help="W&B project when --run is an id"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write output to this path (extension hints format if --format auto)",
    )
    parser.add_argument("--stdout", action="store_true", help="Print output to stdout")
    parser.add_argument(
        "--format",
        choices=("auto", "json", "parse_logs"),
        default="auto",
        help="Output format (default: json, or parse_logs if -o ends with .log/.txt)",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (0 = compact)")
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Fetch run metadata (config, summary) without downloading history",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a Phase 1 acceptance snapshot to stderr after download",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Optional metric keys for scan_history (see W&B docs: only steps with ALL keys)",
    )
    parser.add_argument("--page-size", type=int, default=500, help="scan_history page_size")
    parser.add_argument("--min-step", type=int, default=0, help="scan_history min_step (inclusive)")
    parser.add_argument(
        "--max-step", type=int, default=None, help="scan_history max_step (exclusive)"
    )
    args = parser.parse_args(argv)

    try:
        import wandb
    except ModuleNotFoundError:
        print("error: wandb is not installed. pip install -r requirements.txt", file=sys.stderr)
        return 1

    try:
        run_path = resolve_run_path(
            args.run,
            entity=args.entity,
            project=args.project,
            default_run_id=DEFAULT_RUN_ID,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        api = wandb.Api()
        run = fetch_run(api, run_path)
    except (wandb.errors.CommError, wandb.errors.UsageError) as exc:  # type: ignore[attr-defined]
        print(
            "error: W&B API request failed. Run `wandb login` or set WANDB_API_KEY.\n" f"  {exc}",
            file=sys.stderr,
        )
        return 1

    meta = run_metadata(run, run_path)
    if args.info_only:
        payload = {"run": meta}
        out_format = "json"
    else:
        rows = fetch_history(
            run,
            page_size=args.page_size,
            min_step=args.min_step,
            max_step=args.max_step,
            keys=args.keys,
        )
        payload = build_payload(meta, rows)
        if args.report:
            print(phase1_report(meta, rows), file=sys.stderr)

    if args.format == "auto":
        if args.output is not None and args.output.suffix in {".log", ".txt"}:
            out_format = "parse_logs"
        else:
            out_format = "json"
    else:
        out_format = args.format

    if out_format == "json":
        indent = args.indent if args.indent > 0 else None
        serialized = json.dumps(payload, indent=indent, ensure_ascii=False, default=_json_default)
    else:
        if args.info_only:
            print("error: --format parse_logs requires history (omit --info-only)", file=sys.stderr)
            return 1
        serialized = format_parse_logs(payload["history"]["steps"])

    if args.stdout or args.output is None:
        print(serialized, end="" if serialized.endswith("\n") else "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            serialized if serialized.endswith("\n") else serialized + "\n", encoding="utf-8"
        )
        if out_format == "json":
            hist = payload.get("history", {})
            print(
                f"fetched run {meta['name']} ({meta['id']}): "
                f"{hist.get('num_steps', 0)} history rows -> {args.output}",
                file=sys.stderr,
            )
        else:
            print(f"wrote parse_logs format -> {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
