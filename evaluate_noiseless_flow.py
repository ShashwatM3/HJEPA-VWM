"""Inference-only Gate-B validation for a fixed bottleneck and fresh CoarseFlow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config
from data import SSV2Dataset
from losses import reconstruction_loss
from models import build_phase1_modules
from provenance import encoder_spec_from_dict, sha256_file


def _maximum_assignment(scores: torch.Tensor) -> list[int]:
    """Return an exact maximum-weight row-to-column assignment for a square matrix."""
    cost = (-scores.double()).cpu().tolist()
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j] = current
                        way[j] = j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def _configure(cfg: Config, checkpoint: dict, cache: Path, horizon: int) -> None:
    """Apply the checkpoint's resolved architecture and immutable encoder identity."""
    saved = checkpoint["config"]
    for key, value in saved["model"].items():
        if hasattr(cfg.model, key):
            setattr(cfg.model, key, value)
    cfg.encoder.alias = "dinov3_vitb16"
    cfg.encoder.revision = "5931719e67bbdb9737e363e781fb0c67687896bc"
    cfg.encoder.precision = "fp32"
    cfg.encoder.frame_microbatch = 8
    cfg.encoder.hf_cache_dir = str(cache)
    cfg.hf_cache_dir = str(cache)
    cfg.train.horizon_k = horizon


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Evaluate online/EMA geometry and a freshly initialized flow without training."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config()
    _configure(cfg, checkpoint, args.hf_cache, args.horizon)
    spec = encoder_spec_from_dict(checkpoint["encoder_spec"])
    encoder, online, ema, _saved_flow, decoder = build_phase1_modules(cfg, encoder_spec=spec)
    assert encoder is not None
    online.load_state_dict(checkpoint["bottleneck"], strict=True)
    ema.load_state_dict(checkpoint["target_bottleneck"], strict=True)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    # Deliberately do not load checkpoint CoarseFlow: Run 60 never trained it.
    torch.manual_seed(args.seed)
    _, _, _, fresh_flow, _ = build_phase1_modules(cfg, load_encoder=False, encoder_spec=spec)
    modules = (encoder, online, ema, decoder, fresh_flow)
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    dataset = SSV2Dataset(args.data_root, "validation", cfg)
    records = []
    assignments = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            context = sample.context.unsqueeze(0)
            target = sample.target.unsqueeze(0)
            detailed_present = encoder(context)
            detailed_future = encoder(target)
            present = online(detailed_present)
            future = online(detailed_future)
            present_ema = ema(detailed_present)
            future_ema = ema(detailed_future)
            cosine = F.normalize(present[0], dim=-1) @ F.normalize(future[0], dim=-1).T
            assignment = _maximum_assignment(cosine)
            assignments.append(assignment)
            rows = torch.arange(cosine.shape[0])
            tau = torch.full((1,), 0.5)
            velocity = fresh_flow(0.5 * (present + future), tau, present)
            endpoint = 0.5 * (present + future) + 0.5 * velocity
            records.append(
                {
                    "sample_id": dataset.sample_id_at(index),
                    "same_index_cosine": float(cosine.diag().mean()),
                    "row_best_cosine": float(cosine.max(dim=1).values.mean()),
                    "row_best_identity": float((cosine.argmax(dim=1) == rows).float().mean()),
                    "hungarian_cosine": float(cosine[rows, assignment].mean()),
                    "motion_rms": float(torch.mean((future - present) ** 2).sqrt()),
                    "online_ema_present_rms": float(
                        torch.mean((present_ema - present) ** 2).sqrt()
                    ),
                    "online_ema_future_rms": float(torch.mean((future_ema - future) ** 2).sqrt()),
                    "online_reconstruction": float(
                        reconstruction_loss(decoder(present), detailed_present, mode="cosine")
                    ),
                    "ema_reconstruction": float(
                        reconstruction_loss(decoder(present_ema), detailed_present, mode="cosine")
                    ),
                    "fresh_flow_endpoint_mse": float(torch.mean((endpoint - future) ** 2)),
                    "copy_endpoint_mse": float(torch.mean((present - future) ** 2)),
                }
            )
    numeric = [key for key in records[0] if key != "sample_id"]
    aggregate = {key: sum(row[key] for row in records) / len(records) for key in numeric}
    aggregate["identity_hungarian_gap"] = (
        aggregate["hungarian_cosine"] - aggregate["same_index_cosine"]
    )
    pair_agreement = []
    for left in range(len(assignments)):
        for right in range(left + 1, len(assignments)):
            pair_agreement.append(
                sum(a == b for a, b in zip(assignments[left], assignments[right], strict=True))
                / len(assignments[left])
            )
    aggregate["pairwise_assignment_agreement"] = sum(pair_agreement) / len(pair_agreement)
    return {
        "schema": "hjepa-noiseless-flow-gate-b-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "encoder_revision": cfg.encoder.revision,
        "flow_source": "present",
        "endpoint_bottleneck": "saved_online_bottleneck_for_both_endpoints",
        "coarse_flow": "fresh_initialization_untrained",
        "horizon_k": args.horizon,
        "sample_count": len(records),
        "aggregate": aggregate,
        "samples": records,
    }


def main() -> None:
    """Parse the bounded Gate-B inputs and atomically emit one JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=[12, 16], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
