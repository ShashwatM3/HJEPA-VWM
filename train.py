"""Phase 1 training entry point for HJEPA-VWM.

Implements Stage 0 synthetic sanity and Stage 1 coarse-dynamics training only.
Fine flow, Stage 2/3, VAE, and frame generator are intentionally not included.
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
from diagnostics import coarse_baselines, gradient_health, latent_std_stats
from losses import flow_matching_loss, interpolate, phase1_total_loss, velocity_target
from models import _build_phase1_modules


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
    """Cosine-ramp EMA momentum from 0.996 toward 0.9999.

    Args:
        step: Global optimizer step.
        start: Initial EMA momentum.
        end: Final EMA momentum.
        total: Latent-stage schedule denominator, 105000.
    Returns:
        Momentum value for the current step.
    """
    progress = min(max(step / total, 0.0), 1.0)
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * progress))


def lr_scale(step: int, warmup_steps: int, max_steps: int) -> float:
    """Phase 1 linear warmup then cosine decay LR multiplier."""
    if step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(
    online_encoder: nn.Module,
    bottleneck: nn.Module,
    coarse_flow: nn.Module,
    cfg: Config,
) -> torch.optim.Optimizer:
    """Create AdamW param groups with locked Phase 1 learning rates."""
    _require_torch()
    return torch.optim.AdamW(
        [
            {"params": online_encoder.parameters(), "lr": cfg.train.lr_encoder, "name": "E"},
            {"params": bottleneck.parameters(), "lr": cfg.train.lr_bottleneck, "name": "B"},
            {"params": coarse_flow.parameters(), "lr": cfg.train.lr_coarse_flow, "name": "F_c"},
        ],
        betas=cfg.train.adam_betas,
        weight_decay=cfg.train.weight_decay,
    )


def apply_lr_schedule(
    optimizer: torch.optim.Optimizer, base_lrs: list[float], step: int, cfg: Config
) -> float:
    """Apply Phase 1 LR schedule to optimizer param groups."""
    scale = lr_scale(step, cfg.train.warmup_steps, cfg.train.stage1_steps)
    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
        group["lr"] = base_lr * scale
    return scale


def device_for_training() -> torch.device:
    """Choose CUDA when available, otherwise CPU for local smoke commands.

    The production target is RunPod GPU, but this helper keeps Stage 0 runnable
    on a laptop when only synthetic checks are needed.

    Returns:
        PyTorch device used for module construction and batch tensors.
    """
    _require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """Update one EMA module from its online counterpart.

    Keeping EMA in `train.py` preserves the CODE_DESIGN boundary: models carry
    parameters, while the training loop owns optimizer/EMA state transitions.

    Args:
        online: Trainable online module.
        target: EMA module with no backprop.
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


def train_step(
    batch: tuple[Tensor, Tensor],
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    """Run one Stage 1 coarse-dynamics training step.

    Args:
        batch: `(context_clip, future_frame)` with shapes (B, 4, 3, 128, 128) and (B, 3, 128, 128).
        modules: `(patch_embed, online_encoder, bottleneck, target_branch, coarse_flow)`.
        optimizer: AdamW over E/B/F_c.
        step: Global Stage 1 step.
        cfg: Training config.
        device: CUDA or CPU device.
    Returns:
        Scalar metrics for logging.
    """
    patch_embed, online_encoder, bottleneck, target_branch, coarse_flow = modules
    context_clip, future_frame = (x.to(device, non_blocking=True) for x in batch)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, cfg):
        tokens, kept_mask = patch_embed.forward_context(context_clip)
        detailed = online_encoder(tokens)
        abstract = bottleneck(detailed, kept_mask)
        target_detailed, target_abstract = target_branch(future_frame)
        eps_c = torch.randn_like(target_abstract)
        tau_c = torch.rand(target_abstract.shape[0], device=device)
        z_c = interpolate(target_abstract, eps_c, tau_c)
        u_c = velocity_target(target_abstract, eps_c)
        u_c_hat = coarse_flow(z_c, tau_c, abstract)
        coarse_loss = flow_matching_loss(u_c_hat, u_c)
        loss, parts = phase1_total_loss(
            coarse_loss,
            detailed,
            abstract,
            cfg.train.lambda_e_reg,
            cfg.train.lambda_c_reg,
            cfg.train.sigreg_m,
            cfg.train.sigreg_knots,
        )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(online_encoder.parameters())
        + list(bottleneck.parameters())
        + list(coarse_flow.parameters()),
        cfg.train.grad_clip,
    )
    optimizer.step()
    momentum = ema_cosine(
        step, cfg.train.ema_m_start, cfg.train.ema_m_end, cfg.train.ema_schedule_steps
    )
    _update_ema(online_encoder, target_branch.target_encoder, momentum)
    _update_ema(bottleneck, target_branch.target_bottleneck, momentum)
    metrics = {name: float(value.detach().float().item()) for name, value in parts.items()}
    metrics.update(
        {
            "loss": float(loss.detach().float().item()),
            "grad_norm": float(grad_norm),
            "ema_m": momentum,
        }
    )
    return metrics


def save_checkpoint(
    path: Path,
    step: int,
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> None:
    """Save Phase 1 checkpoint to `/workspace/checkpoints`.

    Args:
        path: Destination checkpoint path.
        step: Global step saved.
        modules: Phase 1 modules.
        optimizer: AdamW state.
        cfg: Config serialized as nested dicts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    patch_embed, online_encoder, bottleneck, target_branch, coarse_flow = modules
    torch.save(
        {
            "global_step": step,
            "patch_embed": patch_embed.state_dict(),
            "online_encoder": online_encoder.state_dict(),
            "bottleneck": bottleneck.state_dict(),
            "target_branch": target_branch.state_dict(),
            "coarse_flow": coarse_flow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": json.loads(json.dumps(cfg, default=lambda o: getattr(o, "__dict__", str(o)))),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load a Phase 1 checkpoint and return its global step."""
    ckpt = torch.load(path, map_location="cpu")
    patch_embed, online_encoder, bottleneck, target_branch, coarse_flow = modules
    patch_embed.load_state_dict(ckpt["patch_embed"])
    online_encoder.load_state_dict(ckpt["online_encoder"])
    bottleneck.load_state_dict(ckpt["bottleneck"])
    target_branch.load_state_dict(ckpt["target_branch"])
    coarse_flow.load_state_dict(ckpt["coarse_flow"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("global_step", 0))


def run_stage0(cfg: Config) -> None:
    """Run synthetic Stage 0 sanity: forward, backward, optimizer, EMA update."""
    _require_torch()
    set_seed(cfg.seed)
    device = device_for_training()
    modules = tuple(module.to(device) for module in _build_phase1_modules(cfg))
    _, online_encoder, bottleneck, target_branch, coarse_flow = modules
    target_branch.copy_weights_from_online(online_encoder, bottleneck)
    optimizer = make_optimizer(online_encoder, bottleneck, coarse_flow, cfg)
    batch = (
        torch.randn(2, 4, 3, 128, 128, device=device),
        torch.randn(2, 3, 128, 128, device=device),
    )
    before = next(target_branch.target_encoder.parameters()).detach().clone()
    metrics = train_step(batch, modules, optimizer, 0, cfg, device)
    after = next(target_branch.target_encoder.parameters()).detach().clone()
    assert math.isfinite(metrics["loss"]), metrics
    assert not torch.equal(before, after), "EMA target parameters did not update"
    print(f"Stage 0 sanity passed: {metrics}")


def run_training(cfg: Config, steps: int, resume: str | None = None) -> None:
    """Run Stage 1 training on SSv2 or SSv2-tiny."""
    _require_torch()
    set_seed(cfg.seed)
    device = device_for_training()
    modules = tuple(module.to(device) for module in _build_phase1_modules(cfg))
    _, online_encoder, bottleneck, target_branch, coarse_flow = modules
    target_branch.copy_weights_from_online(online_encoder, bottleneck)
    optimizer = make_optimizer(online_encoder, bottleneck, coarse_flow, cfg)
    start_step = load_checkpoint(resume, modules, optimizer) if resume else 0
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "validation", batch_size=min(16, cfg.train.global_batch))
    val_batch = next(iter(val_loader))
    checkpoint_dir = Path(cfg.checkpoint_dir)
    try:
        import wandb

        wandb.init(
            project="hjepa-vwm", config=json.loads(json.dumps(cfg, default=lambda o: o.__dict__))
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


def run_diagnostics(
    batch: tuple[Tensor, Tensor],
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    """Run Phase 1 diagnostics on a fixed validation batch."""
    patch_embed, online_encoder, bottleneck, target_branch, coarse_flow = modules
    context_clip, future_frame = (x.to(device, non_blocking=True) for x in batch)
    with torch.no_grad():
        tokens, kept_mask = patch_embed.forward_context(context_clip)
        detailed = online_encoder(tokens)
        abstract = bottleneck(detailed, kept_mask)
        _, target_abstract = target_branch(future_frame)
        eps_c = torch.randn_like(target_abstract)
        tau_c = torch.rand(target_abstract.shape[0], device=device)
        z_c = interpolate(target_abstract, eps_c, tau_c)
    metrics = latent_std_stats(detailed, abstract)
    metrics.update(coarse_baselines(coarse_flow, z_c, tau_c, abstract, target_abstract, eps_c))
    metrics.update(gradient_health(nn.ModuleList([online_encoder, bottleneck, coarse_flow])))
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse the Phase 1 training CLI.

    The phase docs require `--data`, `--steps`, `--resume`, `--seed`, and
    `--stage0-only` so the same entry point handles sanity, smoke, and full runs.

    Returns:
        Parsed CLI namespace used to populate `Config`.
    """
    parser = argparse.ArgumentParser(description="Train HJEPA-VWM Phase 1.")
    parser.add_argument("--data", choices=["ssv2", "ssv2_tiny"], default="ssv2_tiny")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage0-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the requested Phase 1 command.

    The entry point selects Stage 0 synthetic sanity or Stage 1 training while
    keeping later phases out of scope until their phase docs are opened.
    """
    args = parse_args()
    cfg = Config()
    cfg.data.dataset = args.data
    cfg.seed = args.seed
    cfg.train.max_steps = args.steps
    if args.stage0_only:
        run_stage0(cfg)
    else:
        run_training(cfg, args.steps, args.resume)


if __name__ == "__main__":
    main()
