"""Offline paired encoder/latent cross-video diagnostics from a Phase 1 checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import Config, DataConfig, EncoderConfig, ModelConfig, TrainConfig
from data import ClipBatch, build_fixed_diagnostic_batch, source_video_id
from diagnostics import cross_video_cosine
from train import _build_and_init, load_checkpoint, set_seed


def _dataclass_from_dict(cls: type[Any], values: dict[str, Any]) -> Any:
    """Construct one config dataclass from the fields serialized in a checkpoint."""
    names = {field.name for field in dataclasses.fields(cls) if field.init}
    return cls(**{name: value for name, value in values.items() if name in names})


def config_from_checkpoint(checkpoint: dict[str, Any]) -> Config:
    """Reconstruct the complete runtime config embedded by ``save_checkpoint``."""
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint has no embedded configuration.")
    cfg = Config(
        model=_dataclass_from_dict(ModelConfig, saved.get("model", {})),
        encoder=_dataclass_from_dict(EncoderConfig, saved.get("encoder", {})),
        train=_dataclass_from_dict(TrainConfig, saved.get("train", {})),
        data=_dataclass_from_dict(DataConfig, saved.get("data", {})),
        debug_shapes=bool(saved.get("debug_shapes", True)),
        checkpoint_dir=str(saved.get("checkpoint_dir", "/workspace/checkpoints")),
        hf_cache_dir=saved.get("hf_cache_dir"),
        seed=int(saved.get("seed", 42)),
    )
    return cfg


def resolved_config_summary(cfg: Config) -> dict[str, Any]:
    """Return the architecture and evaluation fields that must be reported."""
    return {
        "encoder_name": cfg.encoder.alias,
        "encoder_revision": cfg.encoder.revision,
        "bottleneck_type": "memory",
        "bottleneck_dimension": cfg.model.bottleneck_mixer_dim,
        "latent_blocks": cfg.model.bottleneck_latent_blocks,
        "decoder_dimension": cfg.model.decoder_dim,
        "decoder_blocks": cfg.model.decoder_blocks,
        "context_slots": cfg.model.n_c,
        "horizon": cfg.train.horizon_k,
        "present_reconstruction_only": cfg.train.present_recon_only,
        "whitening": cfg.train.whiten_features,
        "lambda_variance": cfg.train.lambda_var,
        "lambda_covariance": cfg.train.lambda_cov,
        "training_precision": cfg.train.precision,
        "encoder_precision": cfg.encoder.precision,
    }


def evaluate_once(
    batch: ClipBatch,
    encoder: nn.Module,
    bottleneck: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Run one encoder forward and feed its output to both paired cosine metrics."""
    encoder.eval()
    bottleneck.eval()
    with torch.inference_mode():
        context = batch.context.to(device, non_blocking=True)
        detailed = encoder(context)
        abstract = bottleneck(detailed)
        encoder_value = cross_video_cosine(detailed)["c_cross_video_cosine"]
        latent_value = cross_video_cosine(abstract)["c_cross_video_cosine"]
    difference = latent_value - encoder_value
    ratio = latent_value / encoder_value if abs(encoder_value) > 1e-12 else None
    return {
        "input_shape": list(context.shape),
        "detailed_shape": list(detailed.shape),
        "abstract_shape": list(abstract.shape),
        "e_cross_video_cosine": encoder_value,
        "c_cross_video_cosine": latent_value,
        "latent_minus_encoder": difference,
        "latent_over_encoder": ratio,
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    requested_batch_size: int | None = None,
    device: torch.device | None = None,
    repeat_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Restore one checkpoint and repeat its corrected offline diagnostic readout."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "hjepa-phase1-checkpoint-v2":
        raise ValueError("Offline diagnostics require a current hjepa-phase1-checkpoint-v2 file.")
    cfg = config_from_checkpoint(checkpoint)
    print(
        "Resolved configuration before weight load: "
        + json.dumps(resolved_config_summary(cfg), sort_keys=True),
        file=sys.stderr,
    )
    if cfg.train.whiten_features:
        raise ValueError(
            "Whitened checkpoint evaluation is not supported by this narrow entry point."
        )
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    modules = _build_and_init(cfg, selected_device)
    step = load_checkpoint(
        path,
        modules,
        optimizer=None,
        expected_encoder_spec=modules[0].spec,
        allow_legacy=False,
    )
    for module in modules:
        module.eval()
    batch_size = requested_batch_size or min(16, cfg.train.global_batch)
    batch = build_fixed_diagnostic_batch(
        cfg,
        batch_size=batch_size,
        needs_target=not cfg.train.present_recon_only,
    )
    source_ids = tuple(
        source_video_id(sample_id, cfg.data.dataset) for sample_id in batch.sample_ids
    )
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Corrected diagnostic batch contains duplicate source video IDs.")
    started = time.perf_counter()
    first = evaluate_once(batch, modules[0], modules[1], selected_device)
    second = evaluate_once(batch, modules[0], modules[1], selected_device)
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    runtime_seconds = time.perf_counter() - started
    repeat_deltas = {
        key: abs(float(first[key]) - float(second[key]))
        for key in ("e_cross_video_cosine", "c_cross_video_cosine")
    }
    repeat_consistent = all(delta <= repeat_tolerance for delta in repeat_deltas.values())
    if not repeat_consistent:
        raise RuntimeError(
            f"Repeated diagnostic values differ beyond tolerance {repeat_tolerance}: "
            f"{repeat_deltas}"
        )
    if (
        first["input_shape"][0] != first["detailed_shape"][0]
        or first["input_shape"][0] != first["abstract_shape"][0]
    ):
        raise RuntimeError("Input, detailed, and abstract batch dimensions do not match.")
    return {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_step": step,
        "weights": "live_online_bottleneck",
        "config": resolved_config_summary(cfg),
        "requested_diagnostic_batch_size": batch_size,
        "actual_sample_count": len(batch.sample_ids),
        "unique_source_count": len(set(source_ids)),
        "sample_ids": list(batch.sample_ids),
        "source_uids": list(source_ids),
        "duplicate_source_uids": False,
        **first,
        "repeat_values": {
            "e_cross_video_cosine": second["e_cross_video_cosine"],
            "c_cross_video_cosine": second["c_cross_video_cosine"],
        },
        "repeat_absolute_deltas": repeat_deltas,
        "repeat_tolerance": repeat_tolerance,
        "repeat_consistent": repeat_consistent,
        "runtime_seconds_two_passes": runtime_seconds,
        "device": str(selected_device),
        "cuda_device_name": (
            torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None
        ),
    }


def main() -> None:
    """Parse CLI arguments, run offline evaluation, and print one JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a Phase 1 checkpoint.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--repeat-tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    if args.repeat_tolerance < 0 or not math.isfinite(args.repeat_tolerance):
        parser.error("--repeat-tolerance must be finite and non-negative.")
    device = None if args.device == "auto" else torch.device(args.device)
    if device is not None and device.type == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable.")
    report = evaluate_checkpoint(
        args.checkpoint,
        requested_batch_size=args.batch_size,
        device=device,
        repeat_tolerance=args.repeat_tolerance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
