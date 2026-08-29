"""Deterministic checkpoint evaluator for the locked two-step rollout experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from config import ExperimentConfig, load_experiment_config
from data import ClipBatch, build_fixed_diagnostic_batch
from diagnostics import coarse_baselines, flow_euler_rollouts
from losses import interpolate
from provenance import atomic_json_save, sha256_file, state_dict_hash
from train import (
    _build_and_init,
    _fixed_flow_forward,
    _fixed_flow_training_inputs,
    device_for_training,
    finalize_training_config,
    validate_experiment_protocol,
)

SCHEMA = "hjepa-two-step-rollout-evaluation-v1"
SOLVER_STEPS = (1, 2, 4, 8)
CHECKPOINT_STEPS = (2_500, 5_000)


def _canonical_config_hash(config: dict[str, Any]) -> str:
    """Hash resolved scientific configuration while excluding operational paths."""
    comparable = json.loads(json.dumps(config))
    comparable.pop("checkpoint_dir", None)
    comparable.pop("experiment_config_path", None)
    comparable.pop("experiment_config_sha256", None)
    payload = json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_config_matches(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    confirmation: bool,
    step: int,
) -> bool:
    """Allow only the confirmation duration/schedule delta on its referenced step-5000 source."""
    left = json.loads(json.dumps(saved))
    right = json.loads(json.dumps(current))
    for payload in (left, right):
        payload.pop("checkpoint_dir", None)
        payload.pop("experiment_config_path", None)
        payload.pop("experiment_config_sha256", None)
    if confirmation and step == 5_000:
        for payload in (left, right):
            train = payload.get("train") or {}
            train.pop("max_steps", None)
            train.pop("checkpoint_steps", None)
    return left == right


def fixed_batch_identity(batch: ClipBatch, dataset_fingerprint: str) -> dict[str, Any]:
    """Return a stable identity for the exact ordered fixed evaluation examples."""
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "sample_ids": list(batch.sample_ids),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def validate_checkpoint_steps(
    observed: tuple[int, ...], expected: tuple[int, ...] = CHECKPOINT_STEPS
) -> None:
    """Require the protocol-selected persistence checkpoints in exact order."""
    if observed != expected:
        raise ValueError(f"Locked evaluation requires checkpoint steps {expected}; got {observed}")


def build_evaluation_artifact(
    *,
    config_path: Path,
    config_sha256: str,
    arm: str,
    fixed_bottleneck_sha256: str,
    batch_identity: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the complete stable artifact envelope around checkpoint metrics."""
    return {
        "schema": SCHEMA,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "arm": arm,
        "fixed_bottleneck_sha256": fixed_bottleneck_sha256,
        "fixed_batch": batch_identity,
        "solver_grid": list(SOLVER_STEPS),
        "checkpoints": checkpoints,
    }


def structured_rollout_metrics(flat: dict[str, float]) -> dict[str, Any]:
    """Convert native flat rollout diagnostics into the versioned artifact schema."""
    evaluations: dict[str, Any] = {}
    for count in SOLVER_STEPS:
        normal = f"rollout_{count}step_normal"
        evaluations[str(count)] = {
            "endpoint_mse": flat[f"{normal}_endpoint_mse"],
            "endpoint_copy_ratio": flat[f"{normal}_coarse_to_copy_loss_ratio"],
            "endpoint_batch_mean_ratio": flat[f"{normal}_endpoint_to_batch_mean_ratio"],
            "displacement_cosine": flat[f"{normal}_displacement_alignment"],
            "displacement_norm_ratio": flat[f"{normal}_displacement_norm_ratio"],
            "correct_vs_shuffled_degradation": flat[
                f"rollout_{count}step_condition_shuffle_degradation"
            ],
            "shuffled_endpoint_mse": flat[f"rollout_{count}step_shuffled_endpoint_mse"],
            "zero_endpoint_mse": flat[f"rollout_{count}step_zero_endpoint_mse"],
        }
    return {
        "target_displacement_valid": flat["rollout_target_displacement_valid"],
        "copy_endpoint_mse": flat["rollout_copy_present_endpoint_mse"],
        "batch_mean_endpoint_mse": flat["rollout_batch_mean_endpoint_mse"],
        "true_displacement_norm": flat["rollout_true_displacement_norm"],
        "solver_evaluations": evaluations,
    }


def evaluate_latents(
    coarse_flow: torch.nn.Module,
    present: torch.Tensor,
    future: torch.Tensor,
    tau: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate native Euler rollouts and the matched teacher-forced endpoint."""
    flat = flow_euler_rollouts(
        coarse_flow, present, future, present, present, steps=SOLVER_STEPS
    )
    z_tau = interpolate(future, present, tau)
    teacher = coarse_baselines(
        coarse_flow,
        z_tau,
        tau,
        present,
        future,
        present,
        predict_residual=False,
        flow_source="present",
    )
    return {
        "rollout": structured_rollout_metrics(flat),
        "teacher_forced": {
            "endpoint_mse": teacher["coarse_endpoint_mse"],
            "endpoint_copy_ratio": teacher["coarse_endpoint_vs_copy_ratio"],
            "condition_shuffle_degradation": teacher[
                "coarse_condition_shuffle_degradation"
            ],
            "velocity_mse": teacher["coarse_model_loss"],
        },
    }


def _load_locked_experiment(path: Path) -> ExperimentConfig:
    experiment = load_experiment_config(path)
    finalize_training_config(experiment.config)
    validate_experiment_protocol(experiment.config, experiment.protocol)
    return experiment


def evaluate_checkpoints(
    config_path: Path, checkpoint_paths: tuple[Path, ...], output: Path
) -> dict[str, Any]:
    """Evaluate exactly the locked 2500/5000 checkpoints on one fixed batch."""
    experiment = _load_locked_experiment(config_path)
    cfg = experiment.config
    device = device_for_training()
    modules = _build_and_init(cfg, device, load_encoder=True)
    encoder, bottleneck, _, coarse_flow, _ = modules
    batch = build_fixed_diagnostic_batch(
        cfg, batch_size=experiment.protocol.evaluation_batch_size, needs_target=True
    )
    first = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    dataset_identity = first.get("dataset_identity")
    if not isinstance(dataset_identity, dict) or not dataset_identity.get("fingerprint"):
        raise ValueError("Evaluation checkpoint is missing dataset_identity.fingerprint")
    batch_identity = fixed_batch_identity(batch, dataset_identity["fingerprint"])
    context = batch.context.to(device, non_blocking=True)
    if batch.target is None:
        raise RuntimeError("Locked rollout evaluation requires future clips")
    target = batch.target.to(device, non_blocking=True)
    with torch.no_grad():
        present, future, _, _ = _fixed_flow_forward(
            encoder, bottleneck, context, target, whitener=None
        )
    _, _, tau, _ = _fixed_flow_training_inputs(present, future, 900_001, cfg)

    records = []
    observed_steps = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        step = int(checkpoint.get("completed_updates", checkpoint.get("global_step", -1)))
        observed_steps.append(step)
        if (
            checkpoint.get("dataset_identity", {}).get("fingerprint")
            != dataset_identity["fingerprint"]
        ):
            raise ValueError(f"Checkpoint {path} dataset fingerprint differs from fixed batch")
        saved_config = checkpoint.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError(f"Checkpoint {path} is missing resolved config")
        current_config = json.loads(json.dumps(cfg, default=lambda value: value.__dict__))
        if not _checkpoint_config_matches(
            saved_config,
            current_config,
            confirmation=experiment.protocol.name == "two_step_rollout_confirmation_v1",
            step=step,
        ):
            raise ValueError(f"Checkpoint {path} config hash does not match {config_path}")
        flow_state = checkpoint.get("coarse_flow")
        if not isinstance(flow_state, dict):
            raise ValueError(f"Checkpoint {path} is missing coarse_flow state")
        coarse_flow.load_state_dict(flow_state, strict=True)
        coarse_flow.eval()
        metrics = evaluate_latents(coarse_flow, present, future, tau)
        records.append(
            {
                "step": step,
                "checkpoint_path": str(path),
                "checkpoint_sha256": sha256_file(path),
                "config_hash": _canonical_config_hash(saved_config),
                "model_hash": state_dict_hash(flow_state),
                "metrics": metrics,
            }
        )
    validate_checkpoint_steps(tuple(observed_steps), experiment.protocol.evaluation_checkpoints)
    artifact = build_evaluation_artifact(
        config_path=config_path,
        config_sha256=cfg.experiment_config_sha256,
        arm="treatment" if cfg.train.lambda_rollout > 0 else "control",
        fixed_bottleneck_sha256=experiment.protocol.fixed_bottleneck_sha256,
        batch_identity=batch_identity,
        checkpoints=records,
    )
    atomic_json_save(artifact, output)
    return artifact


def main() -> None:
    """Parse one locked arm and atomically write its structured evaluation artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = evaluate_checkpoints(args.config, tuple(args.checkpoints), args.output)
    print(json.dumps({"schema": artifact["schema"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
