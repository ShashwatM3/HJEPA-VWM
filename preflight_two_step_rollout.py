"""Bounded CUDA-only memory preflight for the locked treatment graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import Config, load_experiment_config
from losses import flow_matching_loss, interpolate, velocity_target
from models import CoarseFlow
from train import (
    autocast_context,
    finalize_training_config,
    two_step_rollout_endpoint,
    validate_experiment_protocol,
)


def treatment_graph_loss(
    coarse_flow: torch.nn.Module,
    present: torch.Tensor,
    future: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """Build the exact one-call teacher-forced plus two-call rollout loss graph."""
    batch_size = present.shape[0]
    tau = torch.linspace(0.0, 1.0, batch_size + 2, device=present.device, dtype=present.dtype)[1:-1]
    no_drop = torch.zeros(batch_size, device=present.device, dtype=torch.bool)
    z_tau = interpolate(future, present, tau)
    target_velocity = velocity_target(future, present)
    predicted_velocity = coarse_flow(z_tau, tau, present, condition_drop=no_drop)
    teacher_forced = flow_matching_loss(predicted_velocity, target_velocity)
    endpoint, _, _ = two_step_rollout_endpoint(coarse_flow, present, present)
    rollout = torch.mean((endpoint - future) ** 2)
    return teacher_forced + cfg.train.lambda_rollout * rollout


def run_cuda_preflight(config_path: Path) -> dict[str, object]:
    """Measure the exact synthetic treatment graph without optimizer or filesystem writes."""
    if not torch.cuda.is_available():
        raise RuntimeError("two-step rollout GPU preflight requires CUDA; CPU execution is refused")
    experiment = load_experiment_config(config_path)
    cfg = experiment.config
    finalize_training_config(cfg)
    validate_experiment_protocol(cfg, experiment.protocol)
    if cfg.train.lambda_rollout != 0.1:
        raise ValueError(
            "two-step rollout GPU preflight requires treatment train.lambda_rollout=0.1; "
            f"got {cfg.train.lambda_rollout!r}"
        )
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    flow = CoarseFlow(cfg.model).to(device).train()
    present = torch.randn(
        cfg.train.global_batch,
        cfg.model.n_c,
        cfg.model.d_c,
        device=device,
        dtype=torch.bfloat16,
    )
    future = torch.randn_like(present)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    with autocast_context(device, cfg):
        loss = treatment_graph_loss(flow, present, future, cfg)
    loss.backward()
    torch.cuda.synchronize(device)
    report = {
        "schema": "hjepa-two-step-rollout-gpu-preflight-v1",
        "device": torch.cuda.get_device_name(device),
        "batch_size": cfg.train.global_batch,
        "coarse_shape": [cfg.train.global_batch, cfg.model.n_c, cfg.model.d_c],
        "precision": cfg.train.precision,
        "flow_calls": 3,
        "optimizer_created": False,
        "optimizer_step": False,
        "checkpoint_written": False,
        "loss": float(loss.detach().float().item()),
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    return report


def main() -> None:
    """Run the CUDA-only graph and print its bounded memory report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_cuda_preflight(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
