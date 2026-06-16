"""Phase 1 training entry point for HJEPA-VWM (v0.2 — frozen encoder).

Implements Stage 0 synthetic sanity and Stage 1 coarse-dynamics training only:
frozen V-JEPA 2 encoder + trainable bottleneck + EMA bottleneck + coarse flow F_c,
with a variance floor on c_t. Fine flow, Stages 2-4, VAE, and the frame generator
are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]

from config import Config
from data import build_dataloader
from diagnostics import (
    attention_entropy,
    coarse_baselines,
    cross_video_cosine,
    effective_rank,
    gradient_health,
    slot_diversity_rank,
    variance_stats,
)
from losses import (
    covariance_floor,
    flow_matching_loss,
    interpolate,
    slot_diversity_loss,
    variance_floor,
    velocity_target,
)
from models import build_phase1_modules

# Module bundle order throughout: (encoder E, bottleneck B, target_bottleneck B_EMA,
# coarse_flow F_c).


def _require_torch() -> None:
    """Fail fast when training is invoked without PyTorch installed."""
    if torch is None:
        raise RuntimeError("PyTorch is required for train.py. Install requirements.txt on RunPod.")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Integer seed used for repeatable smoke tests and subset creation.
    """
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ema_cosine(step: int, start: float, end: float, total: int) -> float:
    """Cosine-ramp EMA momentum from `start` toward `end` over `total` steps.

    Args:
        step: Global optimizer step.
        start: Initial EMA momentum (0.996).
        end: Final EMA momentum (0.9999).
        total: Latent-stage schedule denominator (105000).
    Returns:
        Momentum value for the current step.
    """
    progress = min(max(step / total, 0.0), 1.0)
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * progress))


def lr_scale(step: int, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup then cosine-decay LR multiplier for Stage 1."""
    if step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(
    bottleneck: nn.Module, coarse_flow: nn.Module, cfg: Config
) -> torch.optim.Optimizer:
    """Create AdamW param groups for the trainable modules only (E is frozen).

    Args:
        bottleneck: Trainable bottleneck B.
        coarse_flow: Coarse flow F_c.
        cfg: Config with locked Phase 1 learning rates.
    Returns:
        AdamW optimizer over B and F_c (the encoder is not in any group).
    """
    _require_torch()
    return torch.optim.AdamW(
        [
            {"params": bottleneck.parameters(), "lr": cfg.train.lr_bottleneck, "name": "B"},
            {"params": coarse_flow.parameters(), "lr": cfg.train.lr_coarse_flow, "name": "F_c"},
        ],
        betas=cfg.train.adam_betas,
        weight_decay=cfg.train.weight_decay,
    )


def apply_lr_schedule(
    optimizer: torch.optim.Optimizer, base_lrs: list[float], step: int, cfg: Config
) -> float:
    """Apply the Stage 1 LR schedule to optimizer param groups."""
    scale = lr_scale(step, cfg.train.warmup_steps, cfg.train.stage1_steps)
    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
        group["lr"] = base_lr * scale
    return scale


def device_for_training() -> torch.device:
    """Choose CUDA when available, otherwise CPU for local smoke commands."""
    _require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """Update the EMA bottleneck from the online bottleneck.

    Models carry parameters; the training loop owns the optimizer/EMA transitions.

    Args:
        online: Trainable bottleneck B.
        target: EMA bottleneck B_EMA (no backprop).
        momentum: EMA coefficient m.
    """
    with torch.no_grad():
        for p_online, p_target in zip(online.parameters(), target.parameters(), strict=True):
            p_target.data.lerp_(p_online.data, 1.0 - momentum)


def autocast_context(device: torch.device, cfg: Config):
    """Return a bf16 autocast context for CUDA, null context otherwise."""
    _require_torch()
    if device.type == "cuda" and cfg.train.precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def _coarse_forward(
    encoder: nn.Module,
    bottleneck: nn.Module,
    target_bottleneck: nn.Module,
    context_clip: Tensor,
    target_clip: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the online `c_t` and the detached target `c_plus`.

    Args:
        encoder: Frozen V-JEPA 2 encoder E.
        bottleneck: Trainable bottleneck B.
        target_bottleneck: EMA bottleneck B_EMA.
        context_clip: (B, 8, 3, 256, 256) encoder-normalized context window.
        target_clip: (B, 8, 3, 256, 256) encoder-normalized future window.
    Returns:
        (abstract, target_abstract): c_t (grad) and c_plus (stop-grad).
    """
    with torch.no_grad():
        detailed = encoder(context_clip)
    abstract = bottleneck(detailed)
    with torch.no_grad():
        target_detailed = encoder(target_clip)
    target_abstract = target_bottleneck(target_detailed)  # B_EMA + as_target inside
    return abstract, target_abstract


def train_step(
    batch: tuple[Tensor, Tensor],
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    """Run one Stage 1 coarse-dynamics training step.

    Args:
        batch: `(context_clip, target_clip)`, both (B, 8, 3, 256, 256).
        modules: `(encoder, bottleneck, target_bottleneck, coarse_flow)`.
        optimizer: AdamW over B and F_c.
        step: Global Stage 1 step.
        cfg: Training config.
        device: CUDA or CPU device.
    Returns:
        Scalar metrics for logging.
    """
    encoder, bottleneck, target_bottleneck, coarse_flow = modules
    context_clip, target_clip = (x.to(device, non_blocking=True) for x in batch)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, cfg):
        abstract, target_abstract = _coarse_forward(
            encoder, bottleneck, target_bottleneck, context_clip, target_clip
        )
        eps_c = torch.randn_like(target_abstract)
        tau_c = torch.rand(target_abstract.shape[0], device=device)
        z_c = interpolate(target_abstract, eps_c, tau_c)
        u_c = velocity_target(target_abstract, eps_c)
        u_c_hat = coarse_flow(z_c, tau_c, abstract)
        flow_loss = flow_matching_loss(u_c_hat, u_c)
        var_loss = variance_floor(abstract, cfg.train.var_floor_std_target)
        # VICReg-C (Plan Phase 04): always computed so L_cov is logged even on the
        # baseline (Run A) to calibrate lambda_cov; only added to the loss — and
        # thus the gradient — when active. lambda_cov=0.0 => byte-identical baseline.
        cov_loss = covariance_floor(abstract)
        # Slot-diversity (Plan Phase 04): Run A showed the dominant failure is
        # slot collapse (32 slots reading the same near-uniform attention average).
        # Compute for logging/calibration; only add when lambda_slot is active.
        slot_loss = slot_diversity_loss(abstract)
        loss = flow_loss + cfg.train.lambda_var * var_loss
        if cfg.train.lambda_cov > 0.0:
            loss = loss + cfg.train.lambda_cov * cov_loss
        if cfg.train.lambda_slot > 0.0:
            loss = loss + cfg.train.lambda_slot * slot_loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(bottleneck.parameters()) + list(coarse_flow.parameters()), cfg.train.grad_clip
    )
    # Survivability guard added 2026-06-10 after the run-1 explosion (POSTMORTEM_RUN1.md).
    # If the pre-clip gradient is non-finite or larger than `grad_skip_threshold`,
    # skip the optimizer step entirely: zero the gradient, do not update parameters,
    # do not update EMA. One bad batch loses one update, not the whole run. At the
    # current (lr=1e-4 / 2e-4, clip=0.5) settings this should fire ~never; any
    # nonzero `grad_skipped` rate in a healthy run is a stop-and-investigate signal.
    grad_norm_f = float(grad_norm)
    grad_skipped = (
        not math.isfinite(grad_norm_f)
        or grad_norm_f > cfg.train.grad_skip_threshold
    )
    if grad_skipped:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()
    momentum = ema_cosine(
        step, cfg.train.ema_m_start, cfg.train.ema_m_end, cfg.train.ema_schedule_steps
    )
    if not grad_skipped:
        _update_ema(bottleneck, target_bottleneck, momentum)
    return {
        "loss": float(loss.detach().float().item()),
        "L_flow": float(flow_loss.detach().float().item()),
        "L_var": float(var_loss.detach().float().item()),
        "L_cov": float(cov_loss.detach().float().item()),
        "L_slot": float(slot_loss.detach().float().item()),
        "grad_norm": grad_norm_f,
        "grad_skipped": float(grad_skipped),
        "ema_m": momentum,
    }


def save_checkpoint(
    path: Path,
    step: int,
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> None:
    """Save a Phase 1 checkpoint (encoder excluded — reloaded from HF).

    Args:
        path: Destination checkpoint path.
        step: Global step saved.
        modules: Phase 1 modules.
        optimizer: AdamW state.
        cfg: Config serialized as nested dicts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _, bottleneck, target_bottleneck, coarse_flow = modules
    torch.save(
        {
            "global_step": step,
            "bottleneck": bottleneck.state_dict(),
            "target_bottleneck": target_bottleneck.state_dict(),
            "coarse_flow": coarse_flow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": json.loads(json.dumps(cfg, default=lambda o: getattr(o, "__dict__", str(o)))),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load a Phase 1 checkpoint and return its global step (encoder untouched)."""
    ckpt = torch.load(path, map_location="cpu")
    _, bottleneck, target_bottleneck, coarse_flow = modules
    bottleneck.load_state_dict(ckpt["bottleneck"])
    target_bottleneck.load_state_dict(ckpt["target_bottleneck"])
    coarse_flow.load_state_dict(ckpt["coarse_flow"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("global_step", 0))


def _build_and_init(cfg: Config, device: torch.device, load_encoder: bool = True):
    """Build modules on device and initialize B_EMA from B.

    Args:
        cfg: Global config.
        device: Target device.
        load_encoder: When False, encoder is None (for non-encoder smoke paths).
    Returns:
        Module bundle `(encoder, bottleneck, target_bottleneck, coarse_flow)`.
    """
    encoder, bottleneck, target_bottleneck, coarse_flow = build_phase1_modules(
        cfg, load_encoder=load_encoder
    )
    bottleneck = bottleneck.to(device)
    target_bottleneck = target_bottleneck.to(device)
    coarse_flow = coarse_flow.to(device)
    if encoder is not None:
        encoder = encoder.to(device)
    target_bottleneck.copy_weights_from(bottleneck)
    return encoder, bottleneck, target_bottleneck, coarse_flow


def run_stage0(cfg: Config) -> None:
    """Run synthetic Stage 0 sanity: load+freeze E, forward, backward, EMA update."""
    _require_torch()
    set_seed(cfg.seed)
    device = device_for_training()
    modules = _build_and_init(cfg, device, load_encoder=True)
    encoder, bottleneck, target_bottleneck, coarse_flow = modules
    assert sum(p.numel() for p in encoder.parameters() if p.requires_grad) == 0, "E not frozen"
    optimizer = make_optimizer(bottleneck, coarse_flow, cfg)
    batch = (
        torch.randn(2, cfg.model.t_ctx, 3, cfg.model.h, cfg.model.w, device=device),
        torch.randn(2, cfg.model.t_ctx, 3, cfg.model.h, cfg.model.w, device=device),
    )
    enc_before = next(encoder.parameters()).detach().clone()
    ema_before = next(target_bottleneck.parameters()).detach().clone()
    metrics = train_step(batch, modules, optimizer, 0, cfg, device)
    enc_after = next(encoder.parameters()).detach().clone()
    ema_after = next(target_bottleneck.parameters()).detach().clone()
    assert math.isfinite(metrics["loss"]), metrics
    assert torch.equal(enc_before, enc_after), "Frozen encoder params changed"
    assert not torch.equal(ema_before, ema_after), "B_EMA did not update"
    print(f"Stage 0 sanity passed: {metrics}")


def run_diagnostics(
    batch: tuple[Tensor, Tensor],
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    """Run Phase 1 diagnostics on a fixed validation batch."""
    encoder, bottleneck, target_bottleneck, coarse_flow = modules
    context_clip, target_clip = (x.to(device, non_blocking=True) for x in batch)
    with torch.no_grad():
        abstract, target_abstract = _coarse_forward(
            encoder, bottleneck, target_bottleneck, context_clip, target_clip
        )
        eps_c = torch.randn_like(target_abstract)
        tau_c = torch.rand(target_abstract.shape[0], device=device)
        z_c = interpolate(target_abstract, eps_c, tau_c)
    metrics: dict[str, float] = {}
    metrics.update(variance_stats(abstract))
    metrics.update(cross_video_cosine(abstract))
    metrics.update(effective_rank(abstract))
    metrics.update(slot_diversity_rank(abstract))
    metrics.update(coarse_baselines(coarse_flow, z_c, tau_c, abstract, target_abstract, eps_c))
    metrics.update(gradient_health(nn.ModuleList([bottleneck, coarse_flow])))
    # Attention entropy needs the bottleneck's pre-attention input (e_t); the
    # encoder forward is cheap at diag cadence and avoids threading `detailed`
    # out of `_coarse_forward` (which the training step does not need).
    with torch.no_grad():
        detailed = encoder(context_clip)
    metrics.update(attention_entropy(bottleneck, detailed))
    return metrics


def run_training(cfg: Config, steps: int, resume: str | None = None) -> None:
    """Run Stage 1 training on SSv2 or SSv2-tiny."""
    _require_torch()
    set_seed(cfg.seed)
    device = device_for_training()
    modules = _build_and_init(cfg, device, load_encoder=True)
    _, bottleneck, target_bottleneck, coarse_flow = modules
    optimizer = make_optimizer(bottleneck, coarse_flow, cfg)
    start_step = load_checkpoint(resume, modules, optimizer) if resume else 0
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "validation", batch_size=min(16, cfg.train.global_batch))
    val_batch = next(iter(val_loader))
    checkpoint_dir = Path(cfg.checkpoint_dir)
    try:
        import wandb

        wandb.init(
            project="hjepa-vwm",
            config=json.loads(json.dumps(cfg, default=lambda o: getattr(o, "__dict__", str(o)))),
        )
    except Exception as exc:  # pragma: no cover - W&B optional for smoke.
        wandb = None
        print(f"WARN: W&B disabled: {exc}")
    step = start_step
    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break
            lr_mult = apply_lr_schedule(optimizer, base_lrs, step, cfg)
            metrics = train_step(batch, modules, optimizer, step, cfg, device)
            metrics["lr_mult"] = lr_mult
            if step % cfg.train.diag_every == 0:
                metrics.update(run_diagnostics(val_batch, modules, cfg, device))
            if step % cfg.train.log_every == 0:
                print(f"step={step} {metrics}")
                if wandb is not None:
                    wandb.log(metrics, step=step)
            if step > 0 and step % cfg.train.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"phase1_step{step}.pt", step, modules, optimizer, cfg
                )
            step += 1
    save_checkpoint(checkpoint_dir / f"phase1_step{steps}.pt", steps, modules, optimizer, cfg)


def parse_args() -> argparse.Namespace:
    """Parse the Phase 1 training CLI."""
    parser = argparse.ArgumentParser(description="Train HJEPA-VWM Phase 1 (v0.2).")
    parser.add_argument("--data", choices=["ssv2", "ssv2_tiny"], default="ssv2_tiny")
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage0-only", action="store_true")
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Print + W&B-log frequency (overrides cfg.train.log_every).",
    )
    parser.add_argument(
        "--diag-every",
        type=int,
        default=None,
        help="Diagnostic-batch frequency (overrides cfg.train.diag_every).",
    )
    # Plan Phase 04 empirical knobs: defaults leave both penalties off; Run A
    # logs L_cov/L_slot to calibrate them. Init fixes are baked into the model,
    # not flagged. See EXECUTION_PHASES.md.
    parser.add_argument(
        "--lambda-cov",
        type=float,
        default=None,
        help="VICReg-C weight on c_t (Run B). 0 = baseline; ~0.01-0.1 once "
        "calibrated against Run A's logged L_cov.",
    )
    parser.add_argument(
        "--lambda-slot",
        type=float,
        default=None,
        help="Within-video slot-diversity weight on c_t. 0 = off; use after "
        "Run A showed slot collapse / near-uniform bottleneck attention.",
    )
    parser.add_argument(
        "--horizon-k",
        type=int,
        default=None,
        help="Prediction horizon in ORIGINAL frames (cfg.train.horizon_k). "
        "Default 4 (target overlaps context heavily); 12 = harder task with "
        "slight overlap, 16+ = non-overlapping. Single fixed horizon (Phases 1-3).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested Phase 1 command (Stage 0 sanity or Stage 1 training)."""
    args = parse_args()
    cfg = Config()
    cfg.data.dataset = args.data
    cfg.seed = args.seed
    cfg.train.max_steps = args.steps
    if args.log_every is not None:
        cfg.train.log_every = args.log_every
    if args.diag_every is not None:
        cfg.train.diag_every = args.diag_every
    if args.lambda_cov is not None:
        cfg.train.lambda_cov = args.lambda_cov
    if args.lambda_slot is not None:
        cfg.train.lambda_slot = args.lambda_slot
    if args.horizon_k is not None:
        cfg.train.horizon_k = args.horizon_k
    if args.stage0_only:
        run_stage0(cfg)
    else:
        run_training(cfg, args.steps, args.resume)


if __name__ == "__main__":
    main()
