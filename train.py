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
    _no_drop,
    apply_trainable_agc,
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
    reconstruction_loss,
    residual_target,
    sigreg_loss,
    slot_diversity_loss,
    variance_floor,
    velocity_target,
)
from models import build_phase1_modules

# Module bundle order throughout: (encoder E, bottleneck B, target_bottleneck B_EMA,
# coarse_flow F_c, decoder D). D is the reconstruction-anchor decoder (option 1);
# it is built/saved/loaded always but only trained when lambda_recon > 0.


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
    bottleneck: nn.Module, coarse_flow: nn.Module, decoder: nn.Module, cfg: Config
) -> torch.optim.Optimizer:
    """Create AdamW param groups for the trainable modules only (E is frozen).

    The decoder D is always in the optimizer so it trains/saves/loads consistently;
    when lambda_recon=0 it simply receives no gradient (AdamW skips ``None`` grads),
    so its parameters never move on the baseline.

    Args:
        bottleneck: Trainable bottleneck B.
        coarse_flow: Coarse flow F_c.
        decoder: Reconstruction decoder D.
        cfg: Config with locked Phase 1 learning rates.
    Returns:
        AdamW optimizer over B, F_c, and D (the encoder is not in any group).
    """
    _require_torch()
    return torch.optim.AdamW(
        [
            {"params": bottleneck.parameters(), "lr": cfg.train.lr_bottleneck, "name": "B"},
            {"params": coarse_flow.parameters(), "lr": cfg.train.lr_coarse_flow, "name": "F_c"},
            {"params": decoder.parameters(), "lr": cfg.train.lr_decoder, "name": "D"},
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


def peak_base_lrs(cfg: Config) -> list[float]:
    """Peak (pre-schedule) LRs for B, F_c, and D — order matches make_optimizer groups."""
    return [cfg.train.lr_bottleneck, cfg.train.lr_coarse_flow, cfg.train.lr_decoder]


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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the online `c_t`, the detached target `c_plus`, and both detailed tensors.

    Args:
        encoder: Frozen V-JEPA 2 encoder E.
        bottleneck: Trainable bottleneck B.
        target_bottleneck: EMA bottleneck B_EMA.
        context_clip: (B, 8, 3, 256, 256) encoder-normalized context window.
        target_clip: (B, 8, 3, 256, 256) encoder-normalized future window.
    Returns:
        (abstract, target_abstract, detailed, target_detailed): c_t (grad), c_plus
        (stop-grad), and the two frozen detailed tensors e_t / e_plus (no-grad). The
        detailed tensors are returned (not just c_t) so the reconstruction anchor and
        diagnostics can reuse them without a second frozen-encoder forward.
    """
    with torch.no_grad():
        detailed = encoder(context_clip)
    abstract = bottleneck(detailed)
    with torch.no_grad():
        target_detailed = encoder(target_clip)
    target_abstract = target_bottleneck(target_detailed)  # B_EMA + as_target inside
    return abstract, target_abstract, detailed, target_detailed


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
        batch: `(context_clip, target_clip)`, both (B, 8, 3, 256, 256).
        modules: `(encoder, bottleneck, target_bottleneck, coarse_flow)`.
        optimizer: AdamW over B and F_c.
        step: Global Stage 1 step.
        cfg: Training config.
        device: CUDA or CPU device.
    Returns:
        Scalar metrics for logging.
    """
    encoder, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    context_clip, target_clip = (x.to(device, non_blocking=True) for x in batch)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, cfg):
        abstract, target_abstract, detailed, target_detailed = _coarse_forward(
            encoder, bottleneck, target_bottleneck, context_clip, target_clip
        )
        if cfg.train.predict_residual:
            # investigation_009: predict the temporal residual Δ = c_{t+k} - c_t (both from
            # B_EMA -> purely temporal, fully detached) instead of the full future latent.
            # Scale the flow noise to Δ so the velocity target isn't noise-dominated; the
            # option-3 recon add-back below becomes ĉ = c_t + Δ̂.
            target_present = target_bottleneck(detailed)  # B_EMA(e_t), detached
            flow_target, residual_sigma = residual_target(target_abstract, target_present)
            eps_c = residual_sigma.to(flow_target.dtype) * torch.randn_like(flow_target)
        else:
            flow_target = target_abstract
            eps_c = torch.randn_like(target_abstract)
        tau_c = torch.rand(target_abstract.shape[0], device=device)
        z_c = interpolate(flow_target, eps_c, tau_c)
        u_c = velocity_target(flow_target, eps_c)
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
        # SIGReg (investigation_008): isotropic-Gaussian regularizer on c_t, the
        # principled replacement for the variance floor's active anti-collapse role.
        # Always computed so L_sigreg is logged on the baseline (for λ calibration);
        # only added to the loss when active. A DEDICATED per-step generator keeps
        # SIGReg's randperm/randn OFF the global RNG (WALK_FIXES F3), so lambda_sigreg=0
        # stays byte-identical to the pre-SIGReg baseline and eps_c/τ/condition-dropout
        # don't shift across λ values (clean ablation, reproducible sketch).
        sigreg_gen = torch.Generator(device=device)
        sigreg_gen.manual_seed(cfg.seed * 1_000_003 + step)
        sigreg_l = sigreg_loss(abstract, generator=sigreg_gen)
        loss = flow_loss + cfg.train.lambda_var * var_loss
        if cfg.train.lambda_cov > 0.0:
            loss = loss + cfg.train.lambda_cov * cov_loss
        if cfg.train.lambda_slot > 0.0:
            loss = loss + cfg.train.lambda_slot * slot_loss
        if cfg.train.lambda_sigreg > 0.0:
            loss = loss + cfg.train.lambda_sigreg * sigreg_l
        # Reconstruction anchor (option 1): decode the ONLINE c_t back to e_hat and
        # penalize MSE against the frozen e_t. The gradient flows into D and B only
        # (abstract is online; detailed is frozen/no-grad; F_c is untouched because
        # we decode c_t, not c_hat). Ramp linearly over recon_warmup_steps to protect
        # the fragile early phase. Default off => decoder is NOT run here (skips the
        # heavy N_ctx x D_e forward), so the baseline stays byte-identical; L_recon_*
        # is logged from run_diagnostics instead.
        recon_loss_val = 0.0
        recon_pred_loss_val = 0.0
        recon_scale = 0.0
        if cfg.train.lambda_recon > 0.0 or cfg.train.lambda_recon_pred > 0.0:
            recon_scale = min(1.0, step / max(1, cfg.train.recon_warmup_steps))
        if cfg.train.lambda_recon > 0.0:
            recon_loss = reconstruction_loss(decoder(abstract), detailed)
            loss = loss + cfg.train.lambda_recon * recon_scale * recon_loss
            recon_loss_val = float(recon_loss.detach().float().item())
        # Reconstruction anchor (option 3): decode the PREDICTED future latent c_hat and
        # penalize MSE against the frozen future features e_{t+k} (target_detailed). c_hat
        # is the rectified-flow one-step endpoint estimate built from the SAME u_c_hat
        # already computed for flow_loss (no extra F_c forward), so the gradient flows
        # through F_c AND — via the F_c conditioning on c_t — into B. The conditioning
        # path is intentionally NOT detached (VITA-style joint training: c is shaped to be
        # predictable, not merely reconstructable). target_detailed is frozen (detached
        # inside reconstruction_loss). Reuses the present anchor's warmup ramp. Default
        # off => not run, so the option-1 baseline stays byte-identical. With this on, the
        # diag readout L_recon_chat should drop (the gradient now acts on it).
        if cfg.train.lambda_recon_pred > 0.0:
            endpoint = z_c + (1.0 - tau_c.reshape(-1, 1, 1)) * u_c_hat
            # Residual mode reconstructs the future from ĉ = c_t + Δ̂ (online c_t, so the
            # recon gradient still reaches B via the add-back and the F_c conditioning).
            c_hat = (abstract + endpoint) if cfg.train.predict_residual else endpoint
            recon_pred_loss = reconstruction_loss(decoder(c_hat), target_detailed)
            loss = loss + cfg.train.lambda_recon_pred * recon_scale * recon_pred_loss
            recon_pred_loss_val = float(recon_pred_loss.detach().float().item())
    loss.backward()
    agc_metrics = apply_trainable_agc(
        bottleneck,
        coarse_flow,
        decoder,
        enabled=cfg.train.agc_enabled,
        lambda_bottleneck=cfg.train.agc_lambda_bottleneck,
        lambda_coarse_flow=cfg.train.agc_lambda_coarse_flow,
        lambda_decoder=cfg.train.agc_lambda_decoder,
        eps=cfg.train.agc_eps,
    )
    trainable = (
        list(bottleneck.parameters())
        + list(coarse_flow.parameters())
        + list(decoder.parameters())
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
    # Survivability guard (POSTMORTEM_RUN1.md; retuned after elated-snowflake-15).
    # AGC runs first and clips per-tensor spikes; this checks the post-AGC global
    # norm returned by clip_grad_norm_ (total norm before the 0.5 global rescale).
    # Skip only tail catastrophes — not the 30–100 band that elated showed at
    # L_flow≈1.9 (step 8450 grad≈65).
    grad_norm_f = float(grad_norm)
    grad_skipped = (
        not math.isfinite(grad_norm_f)
        or grad_norm_f > cfg.train.grad_skip_threshold
    )
    instability_warn = (
        grad_norm_f > cfg.train.instability_warn_grad_norm
        and float(flow_loss.detach().float().item()) > cfg.train.instability_warn_l_flow
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
        "L_sigreg": float(sigreg_l.detach().float().item()),
        "L_recon": recon_loss_val,
        "L_recon_pred": recon_pred_loss_val,
        "recon_scale": recon_scale,
        "grad_norm": grad_norm_f,
        "grad_skipped": float(grad_skipped),
        "instability_warn": float(instability_warn),
        "ema_m": momentum,
        **agc_metrics,
    }


def save_checkpoint(
    path: Path,
    step: int,
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
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
    _, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    torch.save(
        {
            "global_step": step,
            "bottleneck": bottleneck.state_dict(),
            "target_bottleneck": target_bottleneck.state_dict(),
            "coarse_flow": coarse_flow.state_dict(),
            "decoder": decoder.state_dict(),
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
    """Load a Phase 1 checkpoint and return its global step (encoder untouched)."""
    ckpt = torch.load(path, map_location="cpu")
    _, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    bottleneck.load_state_dict(ckpt["bottleneck"])
    target_bottleneck.load_state_dict(ckpt["target_bottleneck"])
    coarse_flow.load_state_dict(ckpt["coarse_flow"])
    # Backward-compat: pre-reconstruction checkpoints have no "decoder"; leave D at
    # its fresh init (it is only meaningful once lambda_recon > 0 has trained it).
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
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
    encoder, bottleneck, target_bottleneck, coarse_flow, decoder = build_phase1_modules(
        cfg, load_encoder=load_encoder
    )
    bottleneck = bottleneck.to(device)
    target_bottleneck = target_bottleneck.to(device)
    coarse_flow = coarse_flow.to(device)
    decoder = decoder.to(device)
    if encoder is not None:
        encoder = encoder.to(device)
    target_bottleneck.copy_weights_from(bottleneck)
    return encoder, bottleneck, target_bottleneck, coarse_flow, decoder


def run_stage0(cfg: Config) -> None:
    """Run synthetic Stage 0 sanity: load+freeze E, forward, backward, EMA update."""
    _require_torch()
    set_seed(cfg.seed)
    device = device_for_training()
    modules = _build_and_init(cfg, device, load_encoder=True)
    encoder, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    assert sum(p.numel() for p in encoder.parameters() if p.requires_grad) == 0, "E not frozen"
    optimizer = make_optimizer(bottleneck, coarse_flow, decoder, cfg)
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
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    """Run Phase 1 diagnostics on a fixed validation batch."""
    encoder, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    context_clip, target_clip = (x.to(device, non_blocking=True) for x in batch)
    with torch.no_grad():
        abstract, target_abstract, detailed, target_detailed = _coarse_forward(
            encoder, bottleneck, target_bottleneck, context_clip, target_clip
        )
        if cfg.train.predict_residual:
            target_present = target_bottleneck(detailed)
            flow_target, residual_sigma = residual_target(target_abstract, target_present)
            eps_c = residual_sigma.to(flow_target.dtype) * torch.randn_like(flow_target)
        else:
            flow_target = target_abstract
            eps_c = torch.randn_like(target_abstract)
        tau_c = torch.rand(target_abstract.shape[0], device=device)
        z_c = interpolate(flow_target, eps_c, tau_c)
    metrics: dict[str, float] = {}
    metrics.update(variance_stats(abstract))
    metrics.update(cross_video_cosine(abstract))
    metrics.update(effective_rank(abstract))
    metrics.update(slot_diversity_rank(abstract))
    metrics.update(
        coarse_baselines(
            coarse_flow, z_c, tau_c, abstract, flow_target, eps_c,
            predict_residual=cfg.train.predict_residual,
        )
    )
    metrics.update(gradient_health(nn.ModuleList([bottleneck, coarse_flow, decoder])))
    metrics.update(reconstruction_readouts(decoder, coarse_flow, abstract, target_abstract,
                                           detailed, target_detailed, z_c, tau_c,
                                           predict_residual=cfg.train.predict_residual))
    # Reuse the detailed tensor from `_coarse_forward` (no second encoder forward).
    metrics.update(attention_entropy(bottleneck, detailed))
    return metrics


def reconstruction_readouts(
    decoder: nn.Module,
    coarse_flow: nn.Module,
    abstract: Tensor,
    target_abstract: Tensor,
    detailed: Tensor,
    target_detailed: Tensor,
    z_c: Tensor,
    tau_c: Tensor,
    predict_residual: bool = False,
) -> dict[str, float]:
    """Log-only reconstruction readouts that also SCOPE the option-3 decision.

    Computed under no-grad at diag cadence so the baseline (lambda_recon=0) still
    gets `L_recon_present` for calibrating lambda_recon. In option 1 the decoder is
    trained ONLY on the present (`c_t` -> `e_t`); evaluating it on the true future
    (`c_plus` -> `e_plus`) and the PREDICTED future (`c_hat` -> `e_plus`) tells us,
    from data, whether routing reconstruction through `F_c` (option 3) is warranted:
    a large `L_recon_chat` - `L_recon_cplus` gap means `F_c`'s one-step guess lands
    where the representation reconstructs poorly. `c_hat` is the rectified-flow
    one-step endpoint estimate; condition dropout is disabled for a clean readout.

    Returns:
        Metrics dict: `L_recon_present`, `L_recon_cplus`, `L_recon_chat`.
    """
    _require_torch()
    with torch.no_grad():
        u_c_hat = coarse_flow(z_c, tau_c, abstract, condition_drop=_no_drop(abstract))
        endpoint = z_c + (1.0 - tau_c.reshape(-1, 1, 1)) * u_c_hat
        # Residual mode: the predicted future latent is ĉ = c_t + Δ̂ (target_abstract stays
        # the true c_plus for the L_recon_cplus readout below).
        c_hat = (abstract + endpoint) if predict_residual else endpoint
        return {
            "L_recon_present": float(reconstruction_loss(decoder(abstract), detailed).item()),
            "L_recon_cplus": float(
                reconstruction_loss(decoder(target_abstract), target_detailed).item()
            ),
            "L_recon_chat": float(
                reconstruction_loss(decoder(c_hat), target_detailed).item()
            ),
        }


def run_training(cfg: Config, steps: int, resume: str | None = None) -> None:
    """Run Stage 1 training on SSv2 or SSv2-tiny."""
    _require_torch()
    set_seed(cfg.seed)
    device = device_for_training()
    modules = _build_and_init(cfg, device, load_encoder=True)
    _, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    optimizer = make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    start_step = load_checkpoint(resume, modules, optimizer) if resume else 0
    # Peak LRs always come from config/CLI — not checkpoint param_group["lr"], which
    # stores the *scheduled* LR at save time and would double-apply cosine decay on resume.
    base_lrs = peak_base_lrs(cfg)
    if resume:
        print(
            f"Resumed step {start_step}; peak base LRs B={base_lrs[0]:.2e}, F_c={base_lrs[1]:.2e}"
        )
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
    parser.add_argument(
        "--lambda-var",
        type=float,
        default=None,
        help="Variance-floor (VICReg V) weight on c_t (cfg.train.lambda_var, "
        "default 0.10). The primary anti-collapse lever: raise it when c_std_mean "
        "sits well below var_floor_std_target (1.0) and cross_video_cosine climbs.",
    )
    parser.add_argument(
        "--lambda-sigreg",
        type=float,
        default=None,
        help="SIGReg (isotropic-Gaussian) weight on c_t (cfg.train.lambda_sigreg, "
        "investigation_008). 0 = baseline (L_sigreg logged at diag cadence only). "
        "Nonzero drives the pooled c_t toward N(0,I) to break the c_effective_rank "
        "~13/256 ceiling; sweep geometrically (~0.3-10), calibrate vs L_flow scale.",
    )
    parser.add_argument(
        "--lambda-recon",
        type=float,
        default=None,
        help="Reconstruction-anchor weight (option 1: decode c_t -> e_t, grad into "
        "B only, NOT F_c). 0 = baseline (decoder runs at diag cadence for "
        "L_recon_present calibration only). Nonzero ramps in over --recon-warmup-steps; "
        "calibrate against the logged baseline L_recon_present magnitude.",
    )
    parser.add_argument(
        "--lambda-recon-pred",
        type=float,
        default=None,
        help="Prediction-side reconstruction-anchor weight (option 3: decode the "
        "PREDICTED c_hat -> e_{t+k}, grad through F_c and into B via the conditioning). "
        "0 = option-1 baseline (prediction branch not run). Runs alongside --lambda-recon, "
        "reusing the same decoder and --recon-warmup-steps ramp. Watch L_recon_chat drop "
        "and coarse_vs_copy_ratio fall below 1.",
    )
    parser.add_argument(
        "--predict-residual",
        action="store_true",
        help="investigation_009: predict the temporal residual Δ = c_{t+k} - c_t (EMA both "
        "ends) instead of the full future latent. The option-3 recon decodes ĉ = c_t + Δ̂, "
        "the copy baseline becomes the zero residual (ratio stays comparable), and the flow "
        "noise is scaled to Δ. Default off = full-latent prediction (byte-identical).",
    )
    parser.add_argument(
        "--recon-warmup-steps",
        type=int,
        default=None,
        help="Linear ramp length for lambda_recon / lambda_recon_pred "
        "(cfg.train.recon_warmup_steps, default 2000). Protects the fragile early phase "
        "(royal-cherry-17 8600 cliff).",
    )
    parser.add_argument(
        "--lr-bottleneck",
        type=float,
        default=None,
        help="Peak LR for bottleneck B (cfg.train.lr_bottleneck, default 1e-4). "
        "Applied after --resume; overrides checkpoint scheduled LR.",
    )
    parser.add_argument(
        "--lr-coarse-flow",
        type=float,
        default=None,
        help="Peak LR for coarse flow F_c (cfg.train.lr_coarse_flow, default 2e-4). "
        "Applied after --resume; overrides checkpoint scheduled LR.",
    )
    parser.add_argument(
        "--no-agc",
        action="store_true",
        help="Disable adaptive gradient clipping (cfg.train.agc_enabled).",
    )
    parser.add_argument(
        "--agc-lambda-bottleneck",
        type=float,
        default=None,
        help="AGC λ for bottleneck B (cfg.train.agc_lambda_bottleneck, default 0.20).",
    )
    parser.add_argument(
        "--agc-lambda-coarse-flow",
        type=float,
        default=None,
        help="AGC λ for coarse flow F_c (cfg.train.agc_lambda_coarse_flow, default 0.10).",
    )
    parser.add_argument(
        "--grad-skip-threshold",
        type=float,
        default=None,
        help="Post-AGC global grad-norm tail guard (cfg.train.grad_skip_threshold, "
        "default 150). Steps above this skip optimizer.step().",
    )
    # investigation_007 capacity-floor sweep: decoder size, latent size, checkpoint dir.
    # All architecture knobs (set on cfg.model BEFORE the modules are built); changing
    # any of them makes a checkpoint shape-incompatible — do NOT --resume across them.
    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=None,
        help="Reconstruction decoder D width (cfg.model.decoder_dim, default 256). "
        "Capacity-floor sweep axis: tests whether D is too thin to expand c->e.",
    )
    parser.add_argument(
        "--decoder-blocks",
        type=int,
        default=None,
        help="Reconstruction decoder D depth (cfg.model.decoder_blocks, default 2). "
        "Capacity-floor sweep axis (depth probe).",
    )
    parser.add_argument(
        "--n-c",
        type=int,
        default=None,
        help="Abstract latent slot count n_c (cfg.model.n_c, default 32). Latent "
        "BANDWIDTH = n_c * d_c; raising it widens the information channel (128:1 -> "
        "64:1 at 64). Tests the capacity-bound hypothesis. Audited: no hard-coded 32.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint output dir (cfg.checkpoint_dir, default /workspace/checkpoints). "
        "MANDATORY per-run when launching parallel sweeps — a shared dir clobbers "
        "phase1_step*.pt across runs in real time.",
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
    if args.lambda_var is not None:
        cfg.train.lambda_var = args.lambda_var
    if args.lambda_sigreg is not None:
        cfg.train.lambda_sigreg = args.lambda_sigreg
    if args.lambda_recon is not None:
        cfg.train.lambda_recon = args.lambda_recon
    if args.lambda_recon_pred is not None:
        cfg.train.lambda_recon_pred = args.lambda_recon_pred
    if args.predict_residual:
        cfg.train.predict_residual = True
    if args.recon_warmup_steps is not None:
        cfg.train.recon_warmup_steps = args.recon_warmup_steps
    if args.lr_bottleneck is not None:
        cfg.train.lr_bottleneck = args.lr_bottleneck
    if args.lr_coarse_flow is not None:
        cfg.train.lr_coarse_flow = args.lr_coarse_flow
    if args.no_agc:
        cfg.train.agc_enabled = False
    if args.agc_lambda_bottleneck is not None:
        cfg.train.agc_lambda_bottleneck = args.agc_lambda_bottleneck
    if args.agc_lambda_coarse_flow is not None:
        cfg.train.agc_lambda_coarse_flow = args.agc_lambda_coarse_flow
    if args.grad_skip_threshold is not None:
        cfg.train.grad_skip_threshold = args.grad_skip_threshold
    # Architecture knobs (investigation_007) — set on cfg.model before modules are built.
    if args.decoder_dim is not None:
        cfg.model.decoder_dim = args.decoder_dim
    if args.decoder_blocks is not None:
        cfg.model.decoder_blocks = args.decoder_blocks
    if args.n_c is not None:
        cfg.model.n_c = args.n_c
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.stage0_only:
        run_stage0(cfg)
    else:
        run_training(cfg, args.steps, args.resume)


if __name__ == "__main__":
    main()
