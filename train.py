"""Phase 1 training entry point for HJEPA-VWM (v0.2 — pluggable frozen encoder).

Implements Stage 0 synthetic sanity and Stage 1 coarse-dynamics training only:
frozen dense encoder + trainable bottleneck + EMA bottleneck + coarse flow F_c,
with a variance floor on c_t. Fine flow, Stages 2-4, VAE, and the frame generator
are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import numpy as np
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]

from config import Config, load_experiment_config
from data import ClipBatch, build_dataloader
from diagnostics import (
    _no_drop,
    apply_trainable_agc,
    attention_entropy,
    coarse_baselines,
    cross_video_cosine,
    effective_rank,
    gradient_health,
    partition_decay_params,
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

if TYPE_CHECKING:
    from encoders import EncoderSpec

EXPERIMENT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "train.yaml"

# Module bundle order throughout: (encoder E, bottleneck B, target_bottleneck B_EMA,
# coarse_flow F_c, decoder D). D is the reconstruction-anchor decoder (option 1);
# it is built/saved/loaded always but only trained when lambda_recon > 0.
# A parameter-free models.FeatureMeanTracker travels OUTSIDE the bundle (optional
# kwarg + "recon_feature_mean" checkpoint key): it holds the EMA per-position
# feature mean for the residual reconstruction target (recon_residual_target) and
# is only updated/used when that flag is on.


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


def linear_ramp_scale(step: int, warmup_steps: int) -> float:
    """Return a 0->1 linear ramp, with non-positive warmup meaning full strength."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, step / warmup_steps))


def _trainable_param_groups(
    bottleneck: nn.Module, coarse_flow: nn.Module, decoder: nn.Module, cfg: Config
) -> list[dict]:
    """Build the AdamW param-group spec: a decay and a no-decay group per module.

    Issue 9: weight decay is applied ONLY to the genuine Linear/Conv weight matrices.
    Biases, LayerNorm scale/shift, the learned query/null/type/slot embeddings, and
    the zero-init adaLN gates go in a parallel ``weight_decay=0`` group (the split
    comes from :func:`diagnostics.partition_decay_params`, which shares its predicate
    with AGC so the two exclusion sets cannot drift). Decaying learned query slots and
    normalization scales fights c's representation geometry — exactly the surface
    SIGReg is trying to shape.

    Groups are emitted in a fixed order — for each of (B, F_c, D), its decay group
    then its no-decay group — so :func:`peak_base_lrs` (which rebuilds from the same
    spec) stays index-aligned with ``optimizer.param_groups`` for the LR schedule.
    """
    _require_torch()
    spec = [
        ("B", bottleneck, cfg.train.lr_bottleneck),
        ("F_c", coarse_flow, cfg.train.lr_coarse_flow),
        ("D", decoder, cfg.train.lr_decoder),
    ]
    groups: list[dict] = []
    for tag, module, lr in spec:
        decay, no_decay = partition_decay_params(module)
        if decay:
            groups.append(
                {
                    "params": decay,
                    "lr": lr,
                    "weight_decay": cfg.train.weight_decay,
                    "name": f"{tag}/decay",
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": f"{tag}/no_decay",
                }
            )
    return groups


def make_optimizer(
    bottleneck: nn.Module, coarse_flow: nn.Module, decoder: nn.Module, cfg: Config
) -> torch.optim.Optimizer:
    """Create AdamW param groups for the trainable modules only (E is frozen).

    The decoder D is always in the optimizer so it trains/saves/loads consistently;
    when lambda_recon=0 it simply receives no gradient (AdamW skips ``None`` grads),
    so its parameters never move on the baseline. Each module contributes a decay and
    a no-decay group (Issue 9 — see :func:`_trainable_param_groups`).

    NOTE: the decay/no-decay split changes the optimizer's param-group COUNT, so an
    optimizer state_dict saved before this change (3 groups) cannot be loaded into the
    new optimizer (up to 6 groups). Resume across this change with a fresh optimizer
    (do not rely on ``--resume`` carrying optimizer state) — consistent with the
    existing "do not --resume across architecture knobs" guidance.

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
        _trainable_param_groups(bottleneck, coarse_flow, decoder, cfg),
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


def peak_base_lrs(
    bottleneck: nn.Module, coarse_flow: nn.Module, decoder: nn.Module, cfg: Config
) -> list[float]:
    """Peak (pre-schedule) LR per optimizer group, index-aligned with make_optimizer.

    Rebuilds the group spec from the same modules/config so the order and COUNT match
    ``optimizer.param_groups`` exactly (decay then no-decay, per module). Derived from
    config — NOT from a resumed checkpoint's ``param_group["lr"]`` (which stores the
    SCHEDULED lr at save time and would double-apply cosine decay on resume).
    """
    return [group["lr"] for group in _trainable_param_groups(bottleneck, coarse_flow, decoder, cfg)]


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


def _assert_ema_transition(
    online_after: list[Tensor],
    target_before: list[Tensor],
    target_after: list[Tensor],
    *,
    momentum: float,
    updated: bool,
) -> None:
    """Assert the exact dtype-rounded EMA transition used by the training step.

    Early warmup updates can imply EMA deltas below fp32 resolution, so requiring
    visible target movement is a false negative. Replaying the same lerp checks the
    transition itself while preserving the real parameter dtype and rounding.
    """
    for index, (online, before, after) in enumerate(
        zip(online_after, target_before, target_after, strict=True)
    ):
        expected = before.clone()
        if updated:
            expected.lerp_(online, 1.0 - momentum)
        assert torch.equal(expected, after), f"B_EMA transition incorrect at parameter {index}"


def autocast_context(device: torch.device, cfg: Config):
    """Return a bf16 autocast context for CUDA, null context otherwise."""
    _require_torch()
    if device.type == "cuda" and cfg.train.precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def _batch_clips(batch: ClipBatch | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor | None]:
    """Normalize legacy tuple and typed raw dataloader batches at the train seam."""
    if isinstance(batch, ClipBatch):
        return batch.context, batch.target
    if isinstance(batch, tuple) and len(batch) == 2:
        return batch[0], batch[1]
    raise TypeError("Training batches must be data.ClipBatch or a legacy two-tensor tuple.")


def _coarse_forward(
    encoder: nn.Module,
    bottleneck: nn.Module,
    target_bottleneck: nn.Module,
    context_clip: Tensor,
    target_clip: Tensor,
    whitener: nn.Module | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the online `c_t`, the detached target `c_plus`, and both detailed tensors.

    Args:
        encoder: Selected frozen dense encoder E.
        bottleneck: Trainable bottleneck B.
        target_bottleneck: EMA bottleneck B_EMA.
        context_clip: (B, 8, 3, 256, 256) raw `[0,1]` context window.
        target_clip: (B, 8, 3, 256, 256) raw `[0,1]` future window.
        whitener: Optional `models.FeatureWhitener`. When given, BOTH encoder
            outputs are mapped to whitened space here, at the single seam where
            frozen features enter the trainable stack — so `B`, `B_EMA`, the flow
            targets, the reconstruction targets, and every diagnostic downstream
            all see the same (whitened) feature space consistently.
    Returns:
        (abstract, target_abstract, detailed, target_detailed): c_t (grad), c_plus
        (stop-grad), and the two frozen detailed tensors e_t / e_plus (no-grad,
        whitened when a whitener is given). The detailed tensors are returned (not
        just c_t) so the reconstruction anchor and diagnostics can reuse them
        without a second frozen-encoder forward.
    """
    with torch.no_grad():
        detailed = encoder(context_clip)
        if whitener is not None:
            detailed = whitener.whiten(detailed)
    abstract = bottleneck(detailed)
    with torch.no_grad():
        target_detailed = encoder(target_clip)
        if whitener is not None:
            target_detailed = whitener.whiten(target_detailed)
    target_abstract = target_bottleneck(target_detailed)  # B_EMA + as_target inside
    return abstract, target_abstract, detailed, target_detailed


def _present_forward(
    encoder: nn.Module,
    bottleneck: nn.Module,
    context_clip: Tensor,
    whitener: nn.Module | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute only the present detailed and abstract latents for recon-only runs.

    Present-only reconstruction intentionally removes the future branch and F_c
    prediction path, so it should not encode the target clip or construct c_plus.

    Args:
        encoder: Selected frozen dense encoder E.
        bottleneck: Trainable bottleneck B.
        context_clip: (B, 8, 3, 256, 256) raw `[0,1]` context window.
        whitener: Optional `models.FeatureWhitener`; same seam as `_coarse_forward`.
    Returns:
        (abstract, detailed): c_t (grad) and frozen e_t (no-grad, whitened when a
        whitener is given).
    """
    with torch.no_grad():
        detailed = encoder(context_clip)
        if whitener is not None:
            detailed = whitener.whiten(detailed)
    abstract = bottleneck(detailed)
    return abstract, detailed


def train_step(
    batch: tuple[Tensor, Tensor],
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Config,
    device: torch.device,
    mean_tracker: nn.Module | None = None,
    whitener: nn.Module | None = None,
) -> dict[str, float]:
    """Run one Stage 1 coarse-dynamics training step.

    Args:
        batch: `(context_clip, target_clip)`, both (B, 8, 3, 256, 256).
        modules: `(encoder, bottleneck, target_bottleneck, coarse_flow)`.
        optimizer: AdamW over B and F_c.
        step: Global Stage 1 step.
        cfg: Training config.
        device: CUDA or CPU device.
        mean_tracker: `models.FeatureMeanTracker` for the residual reconstruction
            target. Required when `cfg.train.recon_residual_target` is True; unused
            (may be None) otherwise.
        whitener: `models.FeatureWhitener` with fixed offline training-set stats.
            Required when `cfg.train.whiten_features` is True; unused (may be
            None) otherwise. Whitening happens inside the forward helpers, so
            every downstream tensor (c_t, c_plus, recon targets, tracker mean)
            lives in the same whitened space.
    Returns:
        Scalar metrics for logging.
    """
    # Every stochastic training draw (flow noise/tau, dropout, model dropout) is
    # a pure function of seed+step. Diagnostics and backend construction therefore
    # cannot perturb continuation, and a checkpoint resumes exact future draws.
    torch.manual_seed(cfg.seed * 1_000_003 + step)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed * 1_000_003 + step)
    encoder, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    # Residual reconstruction target (investigation_013): active only when the flag is
    # set AND a reconstruction anchor actually trains (finalize_training_config enforces
    # this pairing, so the check here is a belt-and-braces guard for direct callers).
    residual_recon_active = cfg.train.recon_residual_target and (
        cfg.train.lambda_recon > 0.0
        or (not cfg.train.present_recon_only and cfg.train.lambda_recon_pred > 0.0)
    )
    if residual_recon_active and mean_tracker is None:
        raise RuntimeError(
            "cfg.train.recon_residual_target is True but no mean_tracker was passed to "
            "train_step; construct a models.FeatureMeanTracker and pass it explicitly."
        )
    # Whitening (tmp/changes_bottleneck <2>): gated by the config flag, never by the
    # mere presence of the module, so the default path stays byte-identical.
    whiten_active = cfg.train.whiten_features
    if whiten_active and whitener is None:
        raise RuntimeError(
            "cfg.train.whiten_features is True but no whitener was passed to train_step; "
            "build one via train._build_whitener (offline stats from whiten_stats.py)."
        )
    step_whitener = whitener if whiten_active else None
    context_source, target_source = _batch_clips(batch)
    context_clip = context_source.to(device, non_blocking=True)
    target_clip = None
    if not cfg.train.present_recon_only:
        if target_source is None:
            raise RuntimeError("Full prediction requires a target clip in the batch.")
        target_clip = target_source.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, cfg):
        if cfg.train.present_recon_only:
            abstract, detailed = _present_forward(
                encoder, bottleneck, context_clip, whitener=step_whitener
            )
            flow_loss = abstract.new_zeros(())
        else:
            assert target_clip is not None
            abstract, target_abstract, detailed, target_detailed = _coarse_forward(
                encoder,
                bottleneck,
                target_bottleneck,
                context_clip,
                target_clip,
                whitener=step_whitener,
            )
            if cfg.train.predict_residual:
                # investigation_009: predict the temporal residual Δ = c_{t+k} - c_t
                # (both from B_EMA -> purely temporal, fully detached) instead of the full
                # future latent. Scale the flow noise to Δ so the velocity target isn't
                # noise-dominated; the option-3 recon add-back below becomes ĉ = c_t + Δ̂.
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
        if residual_recon_active:
            # Residual reconstruction target: fold this batch's frozen context features
            # into the per-position mean BEFORE the loss uses it, so step 0 subtracts a
            # real batch mean instead of zeros. Train-step only — the fixed diagnostic
            # validation batch must never leak into the mean. No RNG is involved, so
            # eps_c / tau / condition dropout are unaffected across the flag values.
            mean_tracker.update(detailed)
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
        if cfg.train.present_recon_only:
            loss = abstract.sum() * 0.0
        else:
            loss = flow_loss
        loss = loss + cfg.train.lambda_var * var_loss
        if cfg.train.lambda_cov > 0.0:
            loss = loss + cfg.train.lambda_cov * cov_loss
        if cfg.train.lambda_slot > 0.0:
            loss = loss + cfg.train.lambda_slot * slot_loss
        # SIGReg warmup (Issue 7): ramp the weight linearly over sigreg_warmup_steps so a
        # strong λ_sigreg doesn't move the ONLINE bottleneck's coordinate system faster than
        # the EMA target (the flow target c_plus) can follow — which would give F_c a moving
        # input/output geometry and the rank-up / flow-plateau pattern. Same ramp shape as
        # the recon anchors. sigreg_scale stays 0 when the term is off (logged for clarity).
        sigreg_scale = 0.0
        if cfg.train.lambda_sigreg > 0.0:
            sigreg_scale = linear_ramp_scale(step, cfg.train.sigreg_warmup_steps)
            loss = loss + cfg.train.lambda_sigreg * sigreg_scale * sigreg_l
        # Reconstruction anchor (option 1): decode the ONLINE c_t back to e_hat and
        # penalize per-token cosine distance against the frozen e_t. The gradient
        # flows into D and B only (abstract is online; detailed is frozen/no-grad;
        # F_c is untouched because we decode c_t, not c_hat). Ramp linearly over
        # recon_warmup_steps to protect the fragile early phase. Default off =>
        # decoder is NOT run here (skips the heavy N_ctx x D_e forward), so the
        # baseline stays byte-identical; L_recon_* is logged from run_diagnostics.
        recon_loss_val = 0.0
        recon_pred_loss_val = 0.0
        recon_scale = 0.0
        if cfg.train.lambda_recon > 0.0 or (
            not cfg.train.present_recon_only and cfg.train.lambda_recon_pred > 0.0
        ):
            recon_scale = linear_ramp_scale(step, cfg.train.recon_warmup_steps)
        if cfg.train.lambda_recon > 0.0:
            # Residual mode (investigation_013): the target becomes e_t - mean, so the
            # video-independent template component earns zero and every unit of loss
            # reduction must route video-specific content through c_t. The decoder's
            # output is then interpreted as the residual (e_hat_full = D(c) + mean).
            present_target = mean_tracker.subtract(detailed) if residual_recon_active else detailed
            recon_loss = reconstruction_loss(
                decoder(abstract),
                present_target,
                mode=cfg.train.recon_loss_mode,
            )
            loss = loss + cfg.train.lambda_recon * recon_scale * recon_loss
            recon_loss_val = float(recon_loss.detach().float().item())
        # Reconstruction anchor (option 3): decode the PREDICTED future latent c_hat and
        # penalize per-token cosine distance against the frozen future features
        # e_{t+k} (target_detailed). c_hat is the rectified-flow one-step endpoint
        # estimate built from the SAME u_c_hat already computed for flow_loss (no extra
        # F_c forward), so the gradient flows through F_c AND — via the F_c conditioning
        # on c_t — into B. The conditioning path is intentionally NOT detached
        # (VITA-style joint training: c is shaped to be predictable, not merely
        # reconstructable). target_detailed is frozen (detached inside
        # reconstruction_loss). Reuses the present anchor's warmup ramp. Default off =>
        # not run, so the option-1 baseline stays byte-identical. With this on, the diag
        # readout L_recon_chat should drop (the gradient now acts on it).
        if not cfg.train.present_recon_only and cfg.train.lambda_recon_pred > 0.0:
            endpoint = z_c + (1.0 - tau_c.reshape(-1, 1, 1)) * u_c_hat
            # Residual mode reconstructs the future from ĉ = c_t + Δ̂ (online c_t, so the
            # recon gradient still reaches B via the add-back and the F_c conditioning).
            c_hat = (abstract + endpoint) if cfg.train.predict_residual else endpoint
            # Residual mode uses the SAME tracker for the future target e_{t+k} - mean:
            # context and future clips share the frozen encoder feature distribution.
            future_target = (
                mean_tracker.subtract(target_detailed) if residual_recon_active else target_detailed
            )
            recon_pred_loss = reconstruction_loss(
                decoder(c_hat),
                future_target,
                mode=cfg.train.recon_loss_mode,
            )
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
        list(bottleneck.parameters()) + list(coarse_flow.parameters()) + list(decoder.parameters())
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
    # Survivability guard (POSTMORTEM_RUN1.md; retuned after elated-snowflake-15).
    # AGC runs first and clips per-tensor spikes; this checks the post-AGC global
    # norm returned by clip_grad_norm_ (total norm before the 0.5 global rescale).
    # Skip only tail catastrophes — not the 30–100 band that elated showed at
    # L_flow≈1.9 (step 8450 grad≈65).
    grad_norm_f = float(grad_norm)
    grad_skipped = not math.isfinite(grad_norm_f) or grad_norm_f > cfg.train.grad_skip_threshold
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
        "sigreg_scale": sigreg_scale,
        "L_recon": recon_loss_val,
        "L_recon_pred": recon_pred_loss_val,
        "recon_scale": recon_scale,
        "recon_target_residual": float(residual_recon_active),
        "recon_mean_norm": (
            float(mean_tracker.mean.norm().item()) if residual_recon_active else 0.0
        ),
        "present_recon_only": float(cfg.train.present_recon_only),
        "prediction_active": float(not cfg.train.present_recon_only),
        "whiten_active": float(whiten_active),
        "grad_norm": grad_norm_f,
        "grad_skipped": float(grad_skipped),
        "instability_warn": float(instability_warn),
        "ema_m": momentum,
        **agc_metrics,
    }


def _sampler_position(step: int, train_count: int, batch_size: int) -> dict[str, int]:
    """Map the next global update to its exact drop-last shuffled-epoch position.

    Args:
        step: Next global optimizer update.
        train_count: Number of training clips in the bound dataset inventory.
        batch_size: Physical training batch size.
    Returns:
        Epoch number and number of already-consumed samples in that epoch.
    """
    if step < 0:
        raise ValueError("Sampler step cannot be negative.")
    if train_count <= 0 or batch_size <= 0:
        raise ValueError("Sampler train count and batch size must be positive.")
    batches_per_epoch = train_count // batch_size
    if batches_per_epoch <= 0:
        raise ValueError("Training split is smaller than one drop-last physical batch.")
    return {
        "epoch": step // batches_per_epoch,
        "batch_offset": (step % batches_per_epoch) * batch_size,
    }


def save_checkpoint(
    path: Path,
    step: int,
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    mean_tracker: nn.Module | None = None,
    whitener: nn.Module | None = None,
    dataset_identity: dict[str, Any] | None = None,
    trainable_init_hash: str | None = None,
    wandb_run_id: str | None = None,
    run_provenance: dict[str, Any] | None = None,
) -> None:
    """Save a Phase 1 checkpoint (encoder excluded — reloaded from HF).

    Args:
        path: Destination checkpoint path.
        step: Global step saved.
        modules: Phase 1 modules.
        optimizer: AdamW state.
        cfg: Config serialized as nested dicts.
        mean_tracker: Optional `models.FeatureMeanTracker`; its buffers are saved
            under "recon_feature_mean" so a residual-target run resumes with the
            same per-position mean instead of re-warming it.
        whitener: Optional `models.FeatureWhitener`; its buffers are saved under
            "feature_whitener" so the checkpoint is self-describing — offline
            evaluators (drift_probe.py) and resumes reproduce the exact whitened
            space the bottleneck was trained in without needing the stats file.
        dataset_identity: Exact dataset inventory/manifests bound to the run.
        trainable_init_hash: Initialization fingerprint for the trainable stack.
        wandb_run_id: Optional W&B continuation identity, never a credential.
        run_provenance: Fully resolved scientific and runtime provenance.
    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    from provenance import atomic_torch_save, encoder_spec_to_dict, state_dict_hash

    encoder_spec = getattr(modules[0], "spec", None)
    payload = {
        "schema": "hjepa-phase1-checkpoint-v2",
        "next_step": step,
        "completed_updates": step,
        # Retained only for the narrow reader used by historical checkpoints.
        "global_step": step,
        "bottleneck": bottleneck.state_dict(),
        "target_bottleneck": target_bottleneck.state_dict(),
        "coarse_flow": coarse_flow.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": json.loads(json.dumps(cfg, default=lambda o: getattr(o, "__dict__", str(o)))),
        "encoder_spec": encoder_spec_to_dict(encoder_spec) if encoder_spec is not None else None,
        "feature_fingerprint": encoder_spec.fingerprint if encoder_spec is not None else None,
        "dataset_identity": dataset_identity,
        "trainable_init_hash": trainable_init_hash,
        "wandb_run_id": wandb_run_id,
        "run_provenance": run_provenance,
        "sampler_state": (
            _sampler_position(
                step,
                dataset_identity["splits"]["train"]["count"],
                cfg.train.global_batch,
            )
            if dataset_identity is not None and "splits" in dataset_identity
            else None
        ),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state() if np is not None else None,
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if mean_tracker is not None:
        payload["recon_feature_mean"] = mean_tracker.state_dict()
        payload["recon_feature_mean_identity"] = state_dict_hash(mean_tracker.state_dict())
    if whitener is not None:
        payload["feature_whitener"] = whitener.state_dict()
        payload["feature_whitener_identity"] = state_dict_hash(whitener.state_dict())
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path,
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer | None = None,
    mean_tracker: nn.Module | None = None,
    whitener: nn.Module | None = None,
    *,
    expected_encoder_spec: EncoderSpec | None = None,
    expected_dataset_identity: dict[str, Any] | None = None,
    reset_optimizer: bool = False,
    allow_legacy: bool = True,
    allow_dataset_transfer: bool = False,
    expected_run_provenance: dict[str, Any] | None = None,
    sampler_state_out: dict[str, int] | None = None,
) -> int:
    """Load a Phase 1 checkpoint and return its global step (encoder untouched).

    Checkpoints saved before the residual reconstruction target have no
    "recon_feature_mean" entry; the tracker is then left fresh and re-warms from
    live batches (~1/(1-momentum) steps), which `run_training` reports on resume.
    Checkpoints saved with whitening carry "feature_whitener"; when a whitener is
    passed, that saved state overrides the stats-file-derived buffers so a resume
    reproduces the exact training-time whitened space.

    Args:
        path: Checkpoint path written by :func:`save_checkpoint`.
        modules: Live `(E,B,B_EMA,F_c,D)` module bundle.
        optimizer: Optional optimizer whose compatible state should be restored.
        mean_tracker: Optional residual-target feature-mean module.
        whitener: Optional fixed whitening module to restore exactly.
        expected_encoder_spec: Exact selected encoder identity.
        expected_dataset_identity: Exact selected dataset identity.
        reset_optimizer: Intentionally keep a fresh optimizer state.
        allow_legacy: Permit the narrow historical V-JEPA checkpoint reader.
        allow_dataset_transfer: Permit only explicitly scoped dataset differences.
        expected_run_provenance: Resolved continuation provenance contract.
        sampler_state_out: Optional mutable mapping populated with the validated
            saved epoch/offset after a successful exact-dataset load. Dataset
            transfers and legacy checkpoints intentionally leave it empty.
    Returns:
        Next global training step stored by the checkpoint.
    """
    from provenance import (
        compare_checkpoint_provenance,
        encoder_spec_from_dict,
        state_dict_hash,
    )

    # Checkpoints contain Python/NumPy RNG tuples in addition to tensors. They are
    # trusted local run artifacts, so opt into the full loader explicitly rather
    # than relying on PyTorch's version-dependent default.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    is_current = ckpt.get("schema") == "hjepa-phase1-checkpoint-v2"
    if not is_current:
        if not allow_legacy:
            raise ValueError("Legacy checkpoint requires explicit allow_legacy=True.")
        warnings.warn(
            "Loading a legacy V-JEPA checkpoint without strict encoder/dataset identity.",
            stacklevel=2,
        )
    if expected_encoder_spec is None:
        expected_encoder_spec = getattr(modules[0], "spec", None)
    validated_sampler_state: dict[str, int] | None = None
    dataset_transfer_active = False
    if is_current:
        serialized = ckpt.get("encoder_spec")
        if serialized is not None:
            saved_spec = encoder_spec_from_dict(serialized)
            if (
                expected_encoder_spec is not None
                and saved_spec.fingerprint != expected_encoder_spec.fingerprint
            ):
                raise ValueError(
                    "Checkpoint encoder fingerprint does not match the selected encoder."
                )
        elif expected_encoder_spec is not None:
            raise ValueError("Checkpoint is missing its encoder fingerprint.")
        saved_dataset = ckpt.get("dataset_identity")
        dataset_transfer_active = bool(
            expected_dataset_identity is not None
            and isinstance(saved_dataset, dict)
            and saved_dataset.get("fingerprint") != expected_dataset_identity.get("fingerprint")
        )
        if expected_dataset_identity is not None and not allow_dataset_transfer:
            if not isinstance(saved_dataset, dict) or saved_dataset.get(
                "fingerprint"
            ) != expected_dataset_identity.get("fingerprint"):
                raise ValueError("Checkpoint dataset fingerprint does not match this run.")
        saved_provenance = ckpt.get("run_provenance")
        if expected_run_provenance is not None:
            compare_checkpoint_provenance(
                saved_provenance,
                expected_run_provenance,
                allow_dataset_transfer=allow_dataset_transfer,
            )
        saved_sampler = ckpt.get("sampler_state")
        if saved_sampler is not None:
            if not isinstance(saved_sampler, dict) or set(saved_sampler) != {
                "epoch",
                "batch_offset",
            }:
                raise RuntimeError("Checkpoint sampler state is malformed.")
            saved_config = ckpt.get("config")
            try:
                saved_train_count = int(saved_dataset["splits"]["train"]["count"])
                saved_batch_size = int(saved_config["train"]["global_batch"])
                expected_sampler = _sampler_position(
                    int(ckpt.get("next_step", ckpt.get("global_step", 0))),
                    saved_train_count,
                    saved_batch_size,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Checkpoint sampler state cannot be validated against its saved "
                    "dataset/config contract."
                ) from exc
            if saved_sampler != expected_sampler:
                raise RuntimeError(
                    "Checkpoint sampler state does not match its saved next step, "
                    "dataset size, and physical batch."
                )
            if not dataset_transfer_active:
                validated_sampler_state = dict(expected_sampler)
        elif sampler_state_out is not None:
            raise RuntimeError(
                "Current-format checkpoint is missing the sampler state required "
                "for an exact dataloader resume."
            )

    def require_state_compatible(
        label: str,
        module: nn.Module,
        saved: dict[str, Tensor],
    ) -> None:
        """Reject incompatible module state before mutating any live parameter.

        Args:
            label: Human-readable checkpoint component name.
            module: Live destination module.
            saved: Named tensor state read from the checkpoint.
        Returns:
            None. Key or shape differences raise before any load operation.
        """
        current = module.state_dict()
        if set(saved) != set(current):
            raise RuntimeError(f"Checkpoint {label} state keys do not match this architecture.")
        bad_shapes = {
            name: (tuple(saved[name].shape), tuple(current[name].shape))
            for name in current
            if saved[name].shape != current[name].shape
        }
        if bad_shapes:
            raise RuntimeError(
                f"Checkpoint {label} tensor shapes do not match this architecture: {bad_shapes}."
            )

    # Every compatibility check above happens before the first parameter mutation.
    _, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    require_state_compatible("bottleneck", bottleneck, ckpt["bottleneck"])
    require_state_compatible("target_bottleneck", target_bottleneck, ckpt["target_bottleneck"])
    require_state_compatible("coarse_flow", coarse_flow, ckpt["coarse_flow"])
    if "decoder" in ckpt:
        decoder_state = ckpt["decoder"]
        if "queries" in decoder_state or "fixed_pos" not in decoder_state:
            raise RuntimeError(
                "Checkpoint uses the old learned-query reconstruction decoder. "
                "Start a fresh run, or load a checkpoint without decoder state, when using "
                "the fixed-position decoder architecture."
            )
        require_state_compatible("decoder", decoder, decoder_state)
    if optimizer is not None and "optimizer" in ckpt and not reset_optimizer:
        saved_optimizer = ckpt["optimizer"]
        current_optimizer = optimizer.state_dict()
        saved_groups = saved_optimizer.get("param_groups", [])
        current_groups = current_optimizer.get("param_groups", [])
        if len(current_groups) != len(saved_groups):
            raise RuntimeError(
                "Checkpoint optimizer state is incompatible. Pass reset_optimizer=True "
                "only for an intentional optimizer reset."
            )
        parameter_by_saved_id: dict[int, Tensor] = {}
        for live_group, current_group, saved_group in zip(
            optimizer.param_groups, current_groups, saved_groups, strict=True
        ):
            current_ids = current_group.get("params", [])
            saved_ids = saved_group.get("params", [])
            live_parameters = live_group.get("params", [])
            if len(current_ids) != len(saved_ids) or len(live_parameters) != len(saved_ids):
                raise RuntimeError(
                    "Checkpoint optimizer parameter groups are incompatible. Pass "
                    "reset_optimizer=True only for an intentional optimizer reset."
                )
            parameter_by_saved_id.update(zip(saved_ids, live_parameters, strict=True))
        for saved_id, state in saved_optimizer.get("state", {}).items():
            parameter = parameter_by_saved_id.get(saved_id)
            if parameter is None or not isinstance(state, dict):
                raise RuntimeError("Checkpoint optimizer state references an unknown parameter.")
            for state_name, value in state.items():
                if isinstance(value, Tensor) and value.ndim > 0 and value.shape != parameter.shape:
                    raise RuntimeError(
                        "Checkpoint optimizer state tensor shape is incompatible: "
                        f"parameter={saved_id}, state={state_name}, "
                        f"saved={tuple(value.shape)}, expected={tuple(parameter.shape)}."
                    )
    if mean_tracker is not None and "recon_feature_mean" in ckpt:
        saved_mean = ckpt["recon_feature_mean"]
        require_state_compatible("recon_feature_mean", mean_tracker, saved_mean)
        expected_identity = ckpt.get("recon_feature_mean_identity")
        if expected_identity is not None and state_dict_hash(saved_mean) != expected_identity:
            raise RuntimeError("Checkpoint recon_feature_mean identity is invalid.")
    if whitener is not None and "feature_whitener" in ckpt:
        saved_whitener = ckpt["feature_whitener"]
        require_state_compatible("feature_whitener", whitener, saved_whitener)
        expected_identity = ckpt.get("feature_whitener_identity")
        if expected_identity is not None and state_dict_hash(saved_whitener) != expected_identity:
            raise RuntimeError("Checkpoint feature_whitener identity is invalid.")
    bottleneck.load_state_dict(ckpt["bottleneck"])
    target_bottleneck.load_state_dict(ckpt["target_bottleneck"])
    coarse_flow.load_state_dict(ckpt["coarse_flow"])
    # Backward-compat: pre-reconstruction checkpoints have no "decoder"; leave D at
    # its fresh init (it is only meaningful once lambda_recon > 0 has trained it).
    # Checkpoints from the learned-query decoder era are intentionally incompatible:
    # those stored a trainable per-output-token content table (`queries`), while the
    # fixed-position decoder must start from a non-trainable `fixed_pos` buffer.
    if "decoder" in ckpt:
        decoder.load_state_dict(decoder_state)
    if mean_tracker is not None and "recon_feature_mean" in ckpt:
        mean_tracker.load_state_dict(ckpt["recon_feature_mean"])
    if whitener is not None and "feature_whitener" in ckpt:
        whitener.load_state_dict(ckpt["feature_whitener"])
    if optimizer is not None and "optimizer" in ckpt and not reset_optimizer:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            raise RuntimeError(
                "Checkpoint optimizer state is incompatible. Pass reset_optimizer=True "
                "only for an intentional optimizer reset."
            ) from exc
    rng_state = ckpt.get("rng_state")
    if isinstance(rng_state, dict):
        random.setstate(rng_state["python"])
        if np is not None and rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    if sampler_state_out is not None:
        sampler_state_out.clear()
        if validated_sampler_state is not None:
            sampler_state_out.update(validated_sampler_state)
    return int(ckpt.get("next_step", ckpt.get("global_step", 0)))


def _build_and_init(cfg: Config, device: torch.device, load_encoder: bool = True):
    """Build modules on device and initialize B_EMA from B.

    Args:
        cfg: Global config.
        device: Target device.
        load_encoder: When False, encoder is None (for non-encoder smoke paths).
    Returns:
        Module bundle `(encoder, bottleneck, target_bottleneck, coarse_flow)`.
    """
    encoder = None
    encoder_spec = None
    if load_encoder:
        from encoders import build_frozen_encoder

        encoder = build_frozen_encoder(cfg.encoder).to(device)
        encoder_spec = encoder.spec
    # Loading a backend may consume global RNG internally. Downstream initialization
    # is isolated so shape-matched encoder arms start from byte-identical trainable state.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.seed + 1_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed + 1_000)
        _, bottleneck, target_bottleneck, coarse_flow, decoder = build_phase1_modules(
            cfg, load_encoder=False, encoder_spec=encoder_spec
        )
    bottleneck = bottleneck.to(device)
    target_bottleneck = target_bottleneck.to(device)
    coarse_flow = coarse_flow.to(device)
    decoder = decoder.to(device)
    target_bottleneck.copy_weights_from(bottleneck)
    return encoder, bottleneck, target_bottleneck, coarse_flow, decoder


def _build_whitener(
    cfg: Config,
    device: torch.device,
    encoder_spec: EncoderSpec | None = None,
    dataset_identity: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> nn.Module | None:
    """Build the fixed-stats feature whitener, or None when whitening is off.

    Loads the offline training-set statistics written by `whiten_stats.py` and
    applies the config eigenvalue floor. Returns None on the default path so the
    baseline stays byte-identical (no module built, no checkpoint key written).

    Args:
        cfg: Finalized runtime configuration with the expected stats contract.
        device: Destination device for the fixed whitening buffers.
        encoder_spec: Exact encoder feature identity and geometry.
        dataset_identity: Exact dataset inventory/manifests identity.
        checkpoint_path: Optional resume checkpoint containing saved whitener state.
    Returns:
        Configured fixed FeatureWhitener, or ``None`` when whitening is disabled.

    Raises:
        FileNotFoundError: `cfg.train.whiten_stats_path` does not exist.
        ValueError: The stats file is malformed or does not match `d_e`.
    """
    if not cfg.train.whiten_features:
        return None
    from models import FeatureWhitener
    from provenance import (
        WHITENING_EIGENSOLVER,
        load_whitening_envelope,
        state_dict_hash,
    )

    if encoder_spec is None:
        raise ValueError("Whitening requires the resolved EncoderSpec, not ModelConfig dimensions.")

    stats_path = Path(cfg.train.whiten_stats_path)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        embedded = checkpoint.get("feature_whitener")
        if embedded is None:
            raise RuntimeError("Whitened resume checkpoint has no embedded feature_whitener state.")
        expected_identity = checkpoint.get("feature_whitener_identity")
        if expected_identity is None or state_dict_hash(embedded) != expected_identity:
            raise RuntimeError("Resume checkpoint feature_whitener identity is missing or invalid.")
        whitener = FeatureWhitener(encoder_spec.feature_dim)
        whitener.load_state_dict(embedded)
        return whitener.to(device)
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"whiten_features is on but the stats file {stats_path} does not exist; "
            "run `python whiten_stats.py --data <dataset>` first and set its output "
            "as train.whiten_stats_path in the experiment YAML."
        )
    envelope = load_whitening_envelope(
        stats_path,
        expected_encoder_spec=encoder_spec,
        expected_dataset_identity=dataset_identity,
        expected_transform_seed=cfg.seed,
        expected_clip_count=cfg.train.whiten_expected_clips,
        expected_eigensolver=WHITENING_EIGENSOLVER,
    )
    tensors = envelope["tensors"]
    whitener = FeatureWhitener(encoder_spec.feature_dim)
    whitener.configure(
        tensors["mean"],
        tensors["eigenvalues"],
        tensors["eigenvectors"],
        cfg.train.whiten_eps,
    )
    return whitener.to(device)


def _build_mean_tracker(
    cfg: Config,
    device: torch.device,
    encoder_spec: EncoderSpec,
) -> nn.Module:
    """Construct the per-position feature-mean tracker on the training device.

    Always built (4 MB of fp32 buffers) so checkpoints and call signatures have one
    shape; it is only UPDATED and USED when `cfg.train.recon_residual_target` is on,
    which keeps the default path byte-identical to the pre-tracker baseline.

    Args:
        cfg: Runtime configuration containing the mean-tracker momentum.
        device: Destination device for the tracker buffers.
        encoder_spec: Resolved detailed-token geometry.
    Returns:
        Uninitialized parameter-free FeatureMeanTracker.
    """
    from models import FeatureMeanTracker

    if encoder_spec is None:
        raise ValueError("FeatureMeanTracker requires the resolved EncoderSpec.")
    return FeatureMeanTracker(
        encoder_spec.layout.n_tokens,
        encoder_spec.feature_dim,
        cfg.train.recon_mean_momentum,
    ).to(device)


def run_stage0(cfg: Config) -> None:
    """Run synthetic Stage 0 sanity: load+freeze E, forward, backward, EMA update."""
    _require_torch()
    set_seed(cfg.seed)
    device = device_for_training()
    modules = _build_and_init(cfg, device, load_encoder=True)
    encoder, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    assert sum(p.numel() for p in encoder.parameters() if p.requires_grad) == 0, "E not frozen"
    optimizer = make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    mean_tracker = _build_mean_tracker(cfg, device, encoder.spec)
    dataset_identity = None
    if cfg.train.whiten_features:
        from provenance import build_dataset_identity

        dataset_identity = build_dataset_identity(cfg, require_complete=cfg.data.dataset == "ego4d")
    whitener = _build_whitener(cfg, device, encoder.spec, dataset_identity)
    batch = (
        torch.rand(
            2,
            cfg.encoder.input_frames,
            3,
            cfg.encoder.input_height,
            cfg.encoder.input_width,
            device=device,
        ),
        torch.rand(
            2,
            cfg.encoder.input_frames,
            3,
            cfg.encoder.input_height,
            cfg.encoder.input_width,
            device=device,
        ),
    )
    enc_before = next(encoder.parameters()).detach().clone()
    # Snapshot every target parameter so Stage 0 can replay the exact EMA lerp.
    # A visible-change assertion is invalid here: the tiny first warmup update,
    # multiplied by (1 - momentum), can round back to the old fp32 target value.
    ema_before = [p.detach().clone() for p in target_bottleneck.parameters()]
    metrics = train_step(
        batch, modules, optimizer, 0, cfg, device, mean_tracker=mean_tracker, whitener=whitener
    )
    enc_after = next(encoder.parameters()).detach().clone()
    ema_after = [p.detach().clone() for p in target_bottleneck.parameters()]
    assert math.isfinite(metrics["loss"]), metrics
    assert torch.equal(enc_before, enc_after), "Frozen encoder params changed"
    _assert_ema_transition(
        [p.detach() for p in bottleneck.parameters()],
        ema_before,
        ema_after,
        momentum=metrics["ema_m"],
        updated=metrics["grad_skipped"] == 0.0,
    )
    print(f"Stage 0 sanity passed: {metrics}")


def _run_diagnostics_impl(
    batch: tuple[Tensor, Tensor],
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    cfg: Config,
    device: torch.device,
    mean_tracker: nn.Module | None = None,
    whitener: nn.Module | None = None,
) -> dict[str, float]:
    """Run Phase 1 diagnostics on a fixed validation batch.

    The tracker is READ here (residual reconstruction targets) but never updated:
    the fixed validation batch must not leak into the per-position feature mean.
    The whitener (when active) is applied inside the forward helpers, so every
    readout — variance/rank/cosine on c_t, flow baselines, reconstruction — is
    measured in the same whitened space the training step optimizes.
    """
    encoder, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    residual_recon_active = cfg.train.recon_residual_target and (
        cfg.train.lambda_recon > 0.0
        or (not cfg.train.present_recon_only and cfg.train.lambda_recon_pred > 0.0)
    )
    if residual_recon_active and mean_tracker is None:
        raise RuntimeError(
            "cfg.train.recon_residual_target is True but no mean_tracker was passed to "
            "run_diagnostics; the readouts would silently score the wrong target."
        )
    whiten_active = cfg.train.whiten_features
    if whiten_active and whitener is None:
        raise RuntimeError(
            "cfg.train.whiten_features is True but no whitener was passed to "
            "run_diagnostics; the readouts would silently score raw features against "
            "a bottleneck trained in whitened space."
        )
    diag_whitener = whitener if whiten_active else None
    context_source, target_source = _batch_clips(batch)
    context_clip = context_source.to(device, non_blocking=True)
    if context_clip.shape[0] <= 1:
        raise RuntimeError(
            "Diagnostics require validation batch size > 1 so shuffled-c pairs "
            "different videos instead of becoming an identity operation."
        )
    if cfg.train.present_recon_only:
        with torch.no_grad():
            abstract, detailed = _present_forward(
                encoder, bottleneck, context_clip, whitener=diag_whitener
            )
        metrics: dict[str, float] = {}
        metrics.update(variance_stats(abstract))
        metrics.update(cross_video_cosine(abstract))
        metrics.update(effective_rank(abstract))
        metrics.update(slot_diversity_rank(abstract))
        metrics.update(gradient_health(nn.ModuleList([bottleneck, coarse_flow, decoder])))
        with torch.no_grad():
            present_target = mean_tracker.subtract(detailed) if residual_recon_active else detailed
            metrics["L_recon_present"] = float(
                reconstruction_loss(
                    decoder(abstract),
                    present_target,
                    mode=cfg.train.recon_loss_mode,
                ).item()
            )
            # Honesty probe (run-052 lesson): decode ANOTHER video's c_t against this
            # video's target. A gap near zero means the decoder output barely depends
            # on which video's latent it received — the template-collapse signature.
            # torch.roll is deterministic and RNG-free, so diagnostics stay reproducible.
            shuffled_abstract = torch.roll(abstract, shifts=1, dims=0)
            metrics["L_recon_shuffled_c"] = float(
                reconstruction_loss(
                    decoder(shuffled_abstract),
                    present_target,
                    mode=cfg.train.recon_loss_mode,
                ).item()
            )
            metrics["L_recon_video_gap"] = (
                metrics["L_recon_shuffled_c"] - metrics["L_recon_present"]
            )
        metrics["present_recon_only"] = 1.0
        metrics["prediction_active"] = 0.0
        metrics.update(attention_entropy(bottleneck, detailed))
        return metrics
    if target_source is None:
        raise RuntimeError("Full diagnostics require a target clip in the batch.")
    target_clip = target_source.to(device, non_blocking=True)
    with torch.no_grad():
        abstract, target_abstract, detailed, target_detailed = _coarse_forward(
            encoder,
            bottleneck,
            target_bottleneck,
            context_clip,
            target_clip,
            whitener=diag_whitener,
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
    metrics["present_recon_only"] = 0.0
    metrics["prediction_active"] = 1.0
    metrics.update(variance_stats(abstract))
    metrics.update(cross_video_cosine(abstract))
    metrics.update(effective_rank(abstract))
    metrics.update(slot_diversity_rank(abstract))
    # Issue 7: log effective rank / std for the EMA TARGET c_plus alongside the online
    # c_t (above). SIGReg shapes the online bottleneck; if the EMA target lags, the
    # online rank can look healthy while F_c's actual regression target still has a
    # collapsed/low-rank geometry. Surfacing both makes that lag visible.
    _tgt_var = variance_stats(target_abstract)
    metrics["c_plus_std_mean"] = _tgt_var["c_std_mean"]
    metrics["c_plus_std_median"] = _tgt_var["c_std_median"]
    metrics["c_plus_effective_rank"] = effective_rank(target_abstract)["c_effective_rank"]
    metrics.update(
        coarse_baselines(
            coarse_flow,
            z_c,
            tau_c,
            abstract,
            flow_target,
            eps_c,
            predict_residual=cfg.train.predict_residual,
        )
    )
    metrics.update(gradient_health(nn.ModuleList([bottleneck, coarse_flow, decoder])))
    metrics.update(
        reconstruction_readouts(
            decoder,
            coarse_flow,
            abstract,
            target_abstract,
            detailed,
            target_detailed,
            z_c,
            tau_c,
            predict_residual=cfg.train.predict_residual,
            recon_loss_mode=cfg.train.recon_loss_mode,
            mean_tracker=mean_tracker,
            residual_target=residual_recon_active,
        )
    )
    # Reuse the detailed tensor from `_coarse_forward` (no second encoder forward).
    metrics.update(attention_entropy(bottleneck, detailed))
    return metrics


def run_diagnostics(
    batch: tuple[Tensor, Tensor] | ClipBatch,
    modules: tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module],
    cfg: Config,
    device: torch.device,
    mean_tracker: nn.Module | None = None,
    whitener: nn.Module | None = None,
) -> dict[str, float]:
    """Run diagnostics in a forked RNG stream and restore training RNG exactly."""
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.seed * 1_000_003 + 900_001)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed * 1_000_003 + 900_001)
        return _run_diagnostics_impl(
            batch,
            modules,
            cfg,
            device,
            mean_tracker=mean_tracker,
            whitener=whitener,
        )


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
    recon_loss_mode: str = "cosine",
    mean_tracker: nn.Module | None = None,
    residual_target: bool = False,
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

    When `residual_target` (investigation_013), every readout scores against the
    per-position residual `e - mean` — the same target the train step optimizes —
    so the logged values stay comparable with `L_recon` from training. The shuffled
    readout decodes a ROTATED batch of latents against the unrotated targets; the
    gap to `L_recon_present` measures how video-specific the decode actually is
    (near zero == template collapse, the run-052 failure).

    Returns:
        Metrics dict: `L_recon_present`, `L_recon_cplus`, `L_recon_chat`,
        `L_recon_shuffled_c`, `L_recon_video_gap`.
    """
    _require_torch()
    if residual_target and mean_tracker is None:
        raise RuntimeError("residual_target readouts require a mean_tracker.")
    with torch.no_grad():
        u_c_hat = coarse_flow(z_c, tau_c, abstract, condition_drop=_no_drop(abstract))
        endpoint = z_c + (1.0 - tau_c.reshape(-1, 1, 1)) * u_c_hat
        # Residual mode: the predicted future latent is ĉ = c_t + Δ̂ (target_abstract stays
        # the true c_plus for the L_recon_cplus readout below).
        c_hat = (abstract + endpoint) if predict_residual else endpoint
        present_target = mean_tracker.subtract(detailed) if residual_target else detailed
        future_target = (
            mean_tracker.subtract(target_detailed) if residual_target else target_detailed
        )
        recon_present = float(
            reconstruction_loss(
                decoder(abstract),
                present_target,
                mode=recon_loss_mode,
            ).item()
        )
        # Honesty probe (run-052 lesson): torch.roll pairs each video's target with
        # ANOTHER video's latent, deterministically and without touching the RNG.
        recon_shuffled = float(
            reconstruction_loss(
                decoder(torch.roll(abstract, shifts=1, dims=0)),
                present_target,
                mode=recon_loss_mode,
            ).item()
        )
        return {
            "L_recon_present": recon_present,
            "L_recon_cplus": float(
                reconstruction_loss(
                    decoder(target_abstract),
                    future_target,
                    mode=recon_loss_mode,
                ).item()
            ),
            "L_recon_chat": float(
                reconstruction_loss(
                    decoder(c_hat),
                    future_target,
                    mode=recon_loss_mode,
                ).item()
            ),
            "L_recon_shuffled_c": recon_shuffled,
            "L_recon_video_gap": recon_shuffled - recon_present,
        }


def _prepare_run(
    cfg: Config,
    device: torch.device,
    resume: str | None = None,
    tracking_identity: dict[str, Any] | None = None,
):
    """Resolve all identities before training state is loaded or mutated.

    This ordering guarantees encoder/data/stats/provenance validation occurs before a
    checkpoint can alter the live trainable stack.

    Args:
        cfg: Fully finalized run configuration.
        device: Training device.
        resume: Optional checkpoint path supplying compatible auxiliary state.
        tracking_identity: Credential-free W&B identity fields.
    Returns:
        Modules, optimizer, auxiliary state, dataset identity, initialization hash,
        and resolved run provenance.
    """
    from provenance import (
        WHITENING_EIGENSOLVER,
        build_dataset_identity,
        build_run_provenance,
        load_whitening_envelope,
        trainable_state_hash,
    )

    dataset_identity = build_dataset_identity(cfg, require_complete=cfg.data.dataset == "ego4d")
    modules = _build_and_init(cfg, device, load_encoder=True)
    encoder, bottleneck, _, coarse_flow, decoder = modules
    init_hash = trainable_state_hash((bottleneck, coarse_flow, decoder))
    optimizer = make_optimizer(bottleneck, coarse_flow, decoder, cfg)
    mean_tracker = _build_mean_tracker(cfg, device, encoder.spec)
    whitener = _build_whitener(
        cfg,
        device,
        encoder.spec,
        dataset_identity,
        checkpoint_path=resume,
    )
    whitening_fingerprint = None
    stats_path = Path(cfg.train.whiten_stats_path)
    if cfg.train.whiten_features and resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        whitening_fingerprint = (saved.get("run_provenance") or {}).get(
            "whitening_payload_fingerprint"
        )
        if not whitening_fingerprint:
            raise RuntimeError(
                "Whitened resume checkpoint is missing its whitening payload fingerprint."
            )
    elif cfg.train.whiten_features and stats_path.is_file():
        whitening_fingerprint = load_whitening_envelope(
            stats_path,
            expected_encoder_spec=encoder.spec,
            expected_dataset_identity=dataset_identity,
            expected_transform_seed=cfg.seed,
            expected_clip_count=cfg.train.whiten_expected_clips,
            expected_eigensolver=WHITENING_EIGENSOLVER,
        )["payload_fingerprint"]
    provenance = build_run_provenance(
        cfg,
        encoder_spec=encoder.spec,
        dataset_identity=dataset_identity,
        trainable_init=init_hash,
        whitening_payload_fingerprint=whitening_fingerprint,
        tracking_identity=tracking_identity,
    )
    return modules, optimizer, mean_tracker, whitener, dataset_identity, init_hash, provenance


def materialize_preflight(
    cfg: Config,
    output: str | Path,
    tracking_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact no-step run and atomically write its resolved provenance.

    No optimizer step is taken; this validates immutable identities before paid work.

    Args:
        cfg: Fully finalized run configuration.
        output: Destination provenance JSON path.
        tracking_identity: Credential-free W&B identity fields.
    Returns:
        Resolved run-provenance envelope.
    """
    from provenance import atomic_json_save

    set_seed(cfg.seed)
    device = device_for_training()
    *_, provenance = _prepare_run(cfg, device, tracking_identity=tracking_identity)
    atomic_json_save(provenance, output)
    print(f"Wrote resolved run provenance: {output}")
    return provenance


def _wandb_config_from_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Expose resolved encoder identity beside legacy compatibility configuration.

    ``ModelConfig`` retains historical V-JEPA geometry so old checkpoints and scripts
    remain readable. Alternate encoders are constructed from ``EncoderSpec`` instead,
    so W&B needs an explicitly named resolved block that cannot be mistaken for those
    compatibility fields.

    Args:
        provenance: Validated run-provenance envelope built after encoder loading.
    Returns:
        Independent JSON-compatible W&B config with authoritative resolved identities.
    """
    config = json.loads(json.dumps(provenance["resolved_config"]))
    config["resolved_encoder_spec"] = json.loads(json.dumps(provenance["encoder_spec"]))
    config["resolved_feature_fingerprint"] = provenance["feature_fingerprint"]
    config["resolved_dataset_fingerprint"] = provenance["dataset_identity"]["fingerprint"]
    return config


def _throughput_rates(
    seconds: float,
    batch_size: int,
    input_frames: int,
    detailed_tokens_per_clip: int,
    encoder_passes: int,
) -> dict[str, float]:
    """Convert one measured operation into comparable work-normalized rates.

    Args:
        seconds: Synchronized elapsed device time.
        batch_size: Number of distinct video examples in the operation.
        input_frames: Frames in each raw clip.
        detailed_tokens_per_clip: Dense encoder tokens produced per clip.
        encoder_passes: Encoder clip forwards per example (one in present-only,
            two in full present/future mode).
    Returns:
        Examples, input frames, and dense output tokens processed per second.
    """
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("Throughput seconds must be finite and positive.")
    counts = (batch_size, input_frames, detailed_tokens_per_clip, encoder_passes)
    if any(not isinstance(value, int) or value <= 0 for value in counts):
        raise ValueError("Throughput work counts must be positive integers.")
    return {
        "examples_per_second": batch_size / seconds,
        "frames_per_second": batch_size * input_frames * encoder_passes / seconds,
        "detailed_tokens_per_second": (
            batch_size * detailed_tokens_per_clip * encoder_passes / seconds
        ),
    }


def _timed_operation(device: torch.device, operation: Callable[[], Any]) -> tuple[Any, float]:
    """Run one operation with synchronized CUDA-event timing when available.

    CUDA events measure device execution without adding a synchronization to the
    real training loop; CPU/MPS preflights use a synchronized wall-clock fallback.
    This helper is intentionally confined to explicit resource preflight.

    Args:
        device: Device on which the measured operation executes.
        operation: Zero-argument callable containing only the interval to measure.
    Returns:
        Operation result and synchronized elapsed seconds.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        torch.cuda.synchronize(device)
        seconds = start.elapsed_time(end) / 1_000.0
    else:
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        started = time.perf_counter()
        result = operation()
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        seconds = time.perf_counter() - started
    if seconds <= 0.0:
        raise RuntimeError("Resource preflight produced a non-positive elapsed time.")
    return result, seconds


def run_resource_preflight(cfg: Config, output: str | Path) -> dict[str, Any]:
    """Exercise one exact recipe step and diagnostic without W&B/checkpoint research state.

    The report captures throughput and peak memory for selecting one common paired-run
    batch and frame-microbatch before whitening or paid training.

    Args:
        cfg: Fully finalized candidate recipe.
        output: Destination resource-report JSON path.
    Returns:
        Provenance envelope extended with resource measurements and metrics.
    """
    from provenance import atomic_json_save

    set_seed(cfg.seed)
    device = device_for_training()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prepared = _prepare_run(cfg, device)
    modules, optimizer, mean_tracker, whitener, _, _, provenance = prepared
    encoder = modules[0]
    train_batch = next(
        iter(
            build_dataloader(
                cfg,
                "train",
                batch_size=cfg.train.global_batch,
                needs_target=not cfg.train.present_recon_only,
            )
        )
    )
    validation_batch = next(
        iter(
            build_dataloader(
                cfg,
                "validation",
                batch_size=min(16, cfg.train.global_batch),
                needs_target=not cfg.train.present_recon_only,
            )
        )
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        encoder_baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    _, encoder_seconds = _timed_operation(
        device,
        lambda: encoder(train_batch.context.to(device, non_blocking=True)),
    )
    encoder_peak = (
        torch.cuda.max_memory_allocated(device) - encoder_baseline
        if device.type == "cuda"
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics, elapsed = _timed_operation(
        device,
        lambda: train_step(
            train_batch,
            modules,
            optimizer,
            0,
            cfg,
            device,
            mean_tracker=mean_tracker,
            whitener=whitener,
        ),
    )
    total_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    # Validation diagnostics are deliberately outside the timed/peak training-step
    # interval so their extra encoder/decoder forwards cannot contaminate throughput.
    metrics.update(
        run_diagnostics(
            validation_batch,
            modules,
            cfg,
            device,
            mean_tracker=mean_tracker,
            whitener=whitener,
        )
    )
    encoder_passes = 1 if cfg.train.present_recon_only else 2
    encoder_rates = _throughput_rates(
        encoder_seconds,
        cfg.train.global_batch,
        cfg.encoder.input_frames,
        encoder.spec.layout.n_tokens,
        1,
    )
    training_rates = _throughput_rates(
        elapsed,
        cfg.train.global_batch,
        cfg.encoder.input_frames,
        encoder.spec.layout.n_tokens,
        encoder_passes,
    )
    report = {
        **provenance,
        "resource_preflight": {
            "device": str(device),
            "timing_method": (
                "cuda_events" if device.type == "cuda" else "synchronized_wall_clock"
            ),
            "batch_size": cfg.train.global_batch,
            "encoder_peak_memory_bytes": encoder_peak,
            "total_peak_memory_bytes": total_peak,
            "encoder_seconds": encoder_seconds,
            "encoder_throughput": encoder_rates,
            "step_seconds": elapsed,
            "encoder_passes_per_example": encoder_passes,
            "training_step_throughput": training_rates,
            # Retained for existing report consumers; examples/s is the precise name.
            "clips_per_second": training_rates["examples_per_second"],
            "metrics": metrics,
        },
    }
    atomic_json_save(report, output)
    print(f"Wrote resource preflight: {output}")
    return report


def run_training(
    cfg: Config,
    steps: int,
    resume: str | None = None,
    *,
    require_wandb: bool = False,
    wandb_options: dict[str, Any] | None = None,
    provenance_out: str | None = None,
    reset_optimizer: bool = False,
    allow_dataset_transfer: bool = False,
    allow_legacy_checkpoint: bool = False,
) -> None:
    """Run strict, resumable Stage 1 training on the selected dataset/encoder."""
    _require_torch()
    from provenance import atomic_json_save, sha256_file

    set_seed(cfg.seed)
    device = device_for_training()
    prepared = _prepare_run(cfg, device, resume, tracking_identity=wandb_options)
    modules, optimizer, mean_tracker, whitener, dataset_identity, init_hash, provenance = prepared
    encoder, bottleneck, target_bottleneck, coarse_flow, decoder = modules
    encoder_spec = encoder.spec
    restored_sampler_state: dict[str, int] = {}
    start_step = (
        load_checkpoint(
            resume,
            modules,
            optimizer,
            mean_tracker=mean_tracker,
            whitener=whitener,
            expected_encoder_spec=encoder_spec,
            expected_dataset_identity=dataset_identity,
            reset_optimizer=reset_optimizer,
            allow_legacy=allow_legacy_checkpoint,
            allow_dataset_transfer=allow_dataset_transfer,
            expected_run_provenance=provenance,
            sampler_state_out=restored_sampler_state,
        )
        if resume
        else 0
    )
    transfer_active = False
    if resume:
        resume_checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        source_dataset = resume_checkpoint.get("dataset_identity") or {}
        source_fingerprint = source_dataset.get("fingerprint")
        destination_fingerprint = dataset_identity.get("fingerprint")
        transfer_active = source_fingerprint != destination_fingerprint
        provenance["resume_policy"] = {
            "checkpoint_sha256": sha256_file(resume),
            "allow_dataset_transfer": allow_dataset_transfer,
            "dataset_transfer_active": transfer_active,
            "source_dataset_fingerprint": source_fingerprint,
            "destination_dataset_fingerprint": destination_fingerprint,
            "reset_optimizer": reset_optimizer,
            "allow_legacy_checkpoint": allow_legacy_checkpoint,
        }
        if transfer_active:
            print(
                "WARN: explicit dataset transfer active: "
                f"{source_fingerprint} -> {destination_fingerprint}."
            )
    # Peak LRs always come from config/CLI — not checkpoint param_group["lr"], which
    # stores the *scheduled* LR at save time and would double-apply cosine decay on resume.
    base_lrs = peak_base_lrs(bottleneck, coarse_flow, decoder, cfg)
    if resume:
        print(
            f"Resumed step {start_step}; peak base LRs "
            f"B={cfg.train.lr_bottleneck:.2e}, F_c={cfg.train.lr_coarse_flow:.2e}"
        )
        if cfg.train.recon_residual_target and not bool(mean_tracker.initialized):
            print(
                "WARN: residual reconstruction target is on but the checkpoint has no "
                "recon_feature_mean state; the per-position mean re-warms from live batches."
            )
    val_loader = build_dataloader(
        cfg,
        "validation",
        batch_size=min(16, cfg.train.global_batch),
        needs_target=not cfg.train.present_recon_only,
    )
    val_batch = next(iter(val_loader))
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved_provenance_path = Path(provenance_out or checkpoint_dir / "run_provenance.json")
    atomic_json_save(provenance, resolved_provenance_path)
    options = dict(wandb_options or {})
    if resume and not options.get("id"):
        options["id"] = torch.load(resume, map_location="cpu", weights_only=False).get(
            "wandb_run_id"
        )
    if options.get("id"):
        options["resume"] = "must"
    try:
        import wandb

        run = wandb.init(
            project=options.pop("project", "hjepa-vwm"),
            entity=options.pop("entity", None),
            group=options.pop("group", None),
            name=options.pop("name", None),
            config=_wandb_config_from_provenance(provenance),
            **options,
        )
        run.config.update({"resolved_provenance": provenance}, allow_val_change=True)
        artifact = wandb.Artifact(f"{run.id}-provenance", type="run-provenance")
        artifact.add_file(str(resolved_provenance_path))
        run.log_artifact(artifact)
        if require_wandb and cfg.train.whiten_features:
            if resume:
                run.summary["whitening_payload_fingerprint"] = provenance[
                    "whitening_payload_fingerprint"
                ]
            else:
                stats_artifact = wandb.Artifact(f"{run.id}-whitening", type="whitening-stats")
                stats_artifact.add_file(cfg.train.whiten_stats_path)
                run.log_artifact(stats_artifact)
        wandb_run_id = run.id
    except Exception as exc:  # pragma: no cover - network/auth dependent
        if require_wandb:
            raise RuntimeError(f"Required W&B initialization failed: {exc}") from exc
        wandb = None
        run = None
        wandb_run_id = None
        print(f"WARN: W&B disabled: {exc}")

    def log_to_wandb(metrics: dict[str, float], current_step: int) -> None:
        """Apply identical strict/optional failure policy at every logging cadence."""
        nonlocal wandb, run
        if wandb is None:
            return
        try:
            wandb.log(metrics, step=current_step)
        except Exception as exc:  # pragma: no cover - network/auth dependent
            if require_wandb:
                raise RuntimeError(f"Required W&B logging failed: {exc}") from exc
            print(f"WARN: W&B logging disabled: {exc}")
            wandb = None
            run = None

    step = start_step
    train_count = dataset_identity["splits"]["train"]["count"]
    batches_per_epoch = train_count // cfg.train.global_batch
    if batches_per_epoch <= 0:
        raise RuntimeError("Training split is smaller than one drop-last physical batch.")
    first_resume_position = restored_sampler_state if not transfer_active else {}
    while step < steps:
        if first_resume_position:
            # The checkpointed position—not a fresh derivation—is the authority for
            # the first resumed loader. Its exact step/count/batch consistency was
            # validated before any checkpoint tensor mutated live state.
            epoch = first_resume_position["epoch"]
            batch_offset = first_resume_position["batch_offset"]
            first_resume_position = {}
        else:
            position = _sampler_position(step, train_count, cfg.train.global_batch)
            epoch = position["epoch"]
            batch_offset = position["batch_offset"]
        train_loader = build_dataloader(
            cfg,
            "train",
            needs_target=not cfg.train.present_recon_only,
            epoch=epoch,
            start_offset=batch_offset,
        )
        for batch in train_loader:
            if step >= steps:
                break
            lr_mult = apply_lr_schedule(optimizer, base_lrs, step, cfg)
            metrics = train_step(
                batch,
                modules,
                optimizer,
                step,
                cfg,
                device,
                mean_tracker=mean_tracker,
                whitener=whitener,
            )
            metrics["lr_mult"] = lr_mult
            diagnostic_due = step % cfg.train.diag_every == 0
            if diagnostic_due:
                metrics.update(
                    run_diagnostics(
                        val_batch,
                        modules,
                        cfg,
                        device,
                        mean_tracker=mean_tracker,
                        whitener=whitener,
                    )
                )
            if step % cfg.train.log_every == 0:
                print(f"step={step} {metrics}")
                log_to_wandb(metrics, step)
            elif diagnostic_due:
                log_to_wandb(metrics, step)
            next_step = step + 1
            if next_step % cfg.train.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"phase1_step{next_step}.pt",
                    next_step,
                    modules,
                    optimizer,
                    cfg,
                    mean_tracker=mean_tracker,
                    whitener=whitener,
                    dataset_identity=dataset_identity,
                    trainable_init_hash=init_hash,
                    wandb_run_id=wandb_run_id,
                    run_provenance=provenance,
                )
            step = next_step
    final_checkpoint = checkpoint_dir / f"phase1_step{steps}.pt"
    save_checkpoint(
        final_checkpoint,
        steps,
        modules,
        optimizer,
        cfg,
        mean_tracker=mean_tracker,
        whitener=whitener,
        dataset_identity=dataset_identity,
        trainable_init_hash=init_hash,
        wandb_run_id=wandb_run_id,
        run_provenance=provenance,
    )
    checksum = sha256_file(final_checkpoint)
    if run is not None:
        try:
            run.summary["final_checkpoint_path"] = str(final_checkpoint)
            run.summary["final_checkpoint_sha256"] = checksum
            run.finish()
        except Exception as exc:  # pragma: no cover - network/auth dependent
            if require_wandb:
                raise RuntimeError(f"Required W&B finalization failed: {exc}") from exc
            print(f"WARN: W&B finalization failed: {exc}")
    print(f"Final checkpoint: {final_checkpoint} sha256={checksum}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the compact Phase 1 training CLI.

    Scientific settings live in YAML. Only the five axes repeatedly swept in the latest
    investigations remain as direct overrides: dataset, encoder, slot count, external slot width,
    and internal bottleneck width. The remaining flags are operator controls, not hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Train HJEPA-VWM Phase 1 from the repository's single configs/train.yaml."
    )
    operator = parser.add_argument_group("operator controls")
    scientific = parser.add_argument_group("scientific hot overrides")
    scientific.add_argument(
        "--data",
        choices=["ssv2", "ssv2_tiny", "ego4d", "ego4d_tiny"],
        default=None,
        help="Hot override for the YAML dataset.",
    )
    scientific.add_argument(
        "--encoder",
        choices=("vjepa2_vitl16", "dinov3_vitb16", "siglip2_vitb16"),
        default=None,
        help=(
            "Hot override for the YAML frozen-encoder alias. Every supported alias has an "
            "immutable default revision; --encoder-revision is an explicit override."
        ),
    )
    scientific.add_argument(
        "--n-c",
        type=int,
        default=None,
        help="Hot override for the YAML abstract slot count.",
    )
    scientific.add_argument(
        "--d-c",
        type=int,
        default=None,
        help="Hot override for the YAML external abstract slot width.",
    )
    scientific.add_argument(
        "--bottleneck-mixer-dim",
        type=int,
        default=None,
        help="Hot override for the YAML internal bottleneck width.",
    )
    operator.add_argument("--resume", default=None, help="Operator override for runtime.resume.")
    mode = operator.add_mutually_exclusive_group()
    mode.add_argument("--stage0-only", action="store_true")
    mode.add_argument(
        "--resource-preflight",
        action="store_true",
        help="Run one exact forward/backward/optimizer/diagnostic recipe and write provenance.",
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Materialize and validate resolved provenance without taking a training step.",
    )
    operator.add_argument("--provenance-out", default=None)
    operator.add_argument("--compare-provenance", nargs=2, metavar=("LEFT", "RIGHT"))
    operator.add_argument("--require-wandb", action="store_true")
    operator.add_argument("--wandb-entity", default=None)
    operator.add_argument("--wandb-project", default=None)
    operator.add_argument("--wandb-group", default=None)
    operator.add_argument("--wandb-name", default=None)
    operator.add_argument("--wandb-run-id", default=None)
    operator.add_argument("--reset-optimizer", action="store_true")
    operator.add_argument("--allow-dataset-transfer", action="store_true")
    operator.add_argument("--allow-legacy-checkpoint", action="store_true")
    operator.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Operator override for the YAML checkpoint directory.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse arguments from the compact Phase 1 CLI."""
    return build_arg_parser().parse_args()


_ENCODER_FRAME_MICROBATCH_DEFAULTS = {
    "vjepa2_vitl16": 8,
    "siglip2_vitb16": 8,
    "dinov3_vitb16": 32,
}


def finalize_training_config(cfg: Config) -> None:
    """Resolve automatic values and reject invalid settings before paid work starts."""
    if cfg.encoder.alias not in _ENCODER_FRAME_MICROBATCH_DEFAULTS:
        choices = ", ".join(sorted(_ENCODER_FRAME_MICROBATCH_DEFAULTS))
        raise ValueError(f"cfg.encoder.alias must be one of {choices}; got {cfg.encoder.alias!r}")
    if cfg.encoder.frame_microbatch is None:
        cfg.encoder.frame_microbatch = _ENCODER_FRAME_MICROBATCH_DEFAULTS[cfg.encoder.alias]
    if cfg.encoder.frame_microbatch <= 0:
        raise ValueError("encoder.frame_microbatch must be positive")
    if (cfg.encoder.input_frames, cfg.encoder.input_height, cfg.encoder.input_width) != (
        8,
        256,
        256,
    ):
        raise ValueError("encoder input contract must be exactly 8 frames at 256x256")
    if cfg.encoder.precision not in {"bf16", "fp32"}:
        raise ValueError("cfg.encoder.precision must be 'bf16' or 'fp32'")
    if cfg.encoder.attention_implementation not in {"sdpa", "eager"}:
        raise ValueError("cfg.encoder.attention_implementation must be 'sdpa' or 'eager'")

    if not 0 <= cfg.seed <= 4_294_967_295:
        raise ValueError(f"cfg.seed must be in [0, 4294967295]; got {cfg.seed}")
    internal_width = cfg.model.bottleneck_mixer_dim
    attention_heads = cfg.model.bottleneck_cross_attn_heads
    if internal_width <= 0:
        raise ValueError(f"cfg.model.bottleneck_mixer_dim must be positive; got {internal_width}")
    if attention_heads <= 0:
        raise ValueError("cfg.model.bottleneck_cross_attn_heads must be positive")
    if internal_width % attention_heads != 0:
        raise ValueError(
            "cfg.model.bottleneck_mixer_dim must be divisible by "
            f"bottleneck_cross_attn_heads={attention_heads}; got {internal_width}"
        )
    if cfg.model.bottleneck_convnext_blocks < 0:
        raise ValueError("cfg.model.bottleneck_convnext_blocks must be non-negative")
    if cfg.model.bottleneck_latent_blocks < 1:
        raise ValueError("cfg.model.bottleneck_latent_blocks must be at least 1")
    if cfg.model.n_c <= 0:
        raise ValueError(f"cfg.model.n_c must be positive; got {cfg.model.n_c}")
    if cfg.model.d_c <= 0:
        raise ValueError(f"cfg.model.d_c must be positive; got {cfg.model.d_c}")
    if cfg.model.f_c_blocks <= 0:
        raise ValueError("cfg.model.f_c_blocks must be positive")
    if cfg.model.f_c_heads <= 0:
        raise ValueError("cfg.model.f_c_heads must be positive")
    if cfg.model.d_c % cfg.model.f_c_heads != 0:
        raise ValueError(
            "cfg.model.d_c must be divisible by "
            f"f_c_heads={cfg.model.f_c_heads}; got {cfg.model.d_c}"
        )
    if not 0.0 <= cfg.model.condition_dropout <= 1.0:
        raise ValueError("cfg.model.condition_dropout must be in [0, 1]")
    if cfg.model.decoder_dim <= 0:
        raise ValueError("cfg.model.decoder_dim must be positive")
    if cfg.model.decoder_blocks <= 0:
        raise ValueError("cfg.model.decoder_blocks must be positive")
    if cfg.model.decoder_heads <= 0:
        raise ValueError("cfg.model.decoder_heads must be positive")
    if cfg.model.decoder_dim % cfg.model.decoder_heads != 0:
        raise ValueError(
            "cfg.model.decoder_dim must be divisible by "
            f"decoder_heads={cfg.model.decoder_heads}; got {cfg.model.decoder_dim}"
        )

    if cfg.train.global_batch <= 0:
        raise ValueError("train.global_batch must be positive")
    if cfg.train.global_batch <= 1:
        raise ValueError(
            "train.global_batch must be greater than 1 because the shuffled-c honesty "
            "diagnostic requires another video."
        )
    if cfg.train.stage1_steps <= 0:
        raise ValueError("train.stage1_steps must be positive")
    if not 0 <= cfg.train.warmup_steps < cfg.train.stage1_steps:
        raise ValueError("train.warmup_steps must be in [0, train.stage1_steps)")
    if not 0 < cfg.train.max_steps <= cfg.train.stage1_steps:
        raise ValueError("train.max_steps must be in (0, train.stage1_steps]")
    if cfg.train.total_latent_steps < cfg.train.stage1_steps:
        raise ValueError("train.total_latent_steps must be >= train.stage1_steps")
    for field_name in ("lr_bottleneck", "lr_coarse_flow", "lr_decoder"):
        value = getattr(cfg.train, field_name)
        if value <= 0.0:
            raise ValueError(f"train.{field_name} must be positive")
    if any(not 0.0 <= beta < 1.0 for beta in cfg.train.adam_betas):
        raise ValueError("train.adam_betas entries must be in [0, 1)")
    if not 0.0 <= cfg.train.weight_decay <= 1.0:
        raise ValueError("train.weight_decay must be in [0, 1]")
    if cfg.train.grad_clip <= 0.0:
        raise ValueError("train.grad_clip must be positive")
    if cfg.train.grad_skip_threshold <= cfg.train.grad_clip:
        raise ValueError("train.grad_skip_threshold must be greater than train.grad_clip")
    for field_name in (
        "agc_lambda_bottleneck",
        "agc_lambda_coarse_flow",
        "agc_lambda_decoder",
        "agc_eps",
        "instability_warn_grad_norm",
        "instability_warn_l_flow",
    ):
        value = getattr(cfg.train, field_name)
        if value <= 0.0:
            raise ValueError(f"train.{field_name} must be positive")
    if not 0.0 <= cfg.train.ema_m_start < 1.0:
        raise ValueError("train.ema_m_start must be in [0, 1)")
    if not cfg.train.ema_m_start <= cfg.train.ema_m_end < 1.0:
        raise ValueError("train.ema_m_end must be in [train.ema_m_start, 1)")
    if cfg.train.ema_schedule_steps <= 0:
        raise ValueError("train.ema_schedule_steps must be positive")
    for field_name in (
        "lambda_var",
        "lambda_cov",
        "lambda_slot",
        "lambda_sigreg",
        "lambda_recon",
        "lambda_recon_pred",
    ):
        value = getattr(cfg.train, field_name)
        if value < 0.0:
            raise ValueError(f"train.{field_name} must be non-negative")
    if cfg.train.var_floor_std_target <= 0.0:
        raise ValueError("train.var_floor_std_target must be positive")
    if cfg.train.sigreg_warmup_steps < 0:
        raise ValueError("train.sigreg_warmup_steps must be non-negative")
    if cfg.train.recon_loss_mode not in {"cosine", "relative_mse"}:
        raise ValueError(
            "cfg.train.recon_loss_mode must be 'cosine' or 'relative_mse'; "
            f"got {cfg.train.recon_loss_mode!r}"
        )
    if cfg.train.recon_warmup_steps < 0:
        raise ValueError("train.recon_warmup_steps must be non-negative")
    if not 0.0 < cfg.train.recon_mean_momentum < 1.0:
        raise ValueError("cfg.train.recon_mean_momentum must be in (0, 1)")
    if cfg.train.present_recon_only:
        if cfg.train.lambda_recon <= 0.0:
            raise ValueError("train.present_recon_only requires train.lambda_recon > 0")
        if cfg.train.lambda_recon_pred > 0.0:
            print("present-only mode ignores train.lambda_recon_pred; setting it to 0.0")
            cfg.train.lambda_recon_pred = 0.0
        if cfg.train.predict_residual:
            print("present-only mode ignores train.predict_residual; disabling residual mode")
            cfg.train.predict_residual = False
    if cfg.train.recon_residual_target:
        if cfg.train.lambda_recon <= 0.0 and cfg.train.lambda_recon_pred <= 0.0:
            raise ValueError(
                "train.recon_residual_target requires an active reconstruction anchor "
                "(train.lambda_recon > 0 or train.lambda_recon_pred > 0); without one the "
                "residual target would silently train nothing."
            )
    if cfg.train.whiten_expected_clips <= 0:
        raise ValueError("cfg.train.whiten_expected_clips must be positive")
    if cfg.train.whiten_eps <= 0.0:
        raise ValueError(f"cfg.train.whiten_eps must be > 0; got {cfg.train.whiten_eps}")
    if cfg.train.whiten_features:
        if not cfg.train.whiten_stats_path:
            raise ValueError(
                "train.whiten_features requires train.whiten_stats_path (the offline stats "
                "file written by whiten_stats.py); per-batch whitening is deliberately "
                "not supported."
            )
    if cfg.train.horizon_k < 0:
        raise ValueError("train.horizon_k must be non-negative")
    if cfg.train.frame_stride <= 0:
        raise ValueError("train.frame_stride must be positive")
    if cfg.train.precision not in {"bf16", "fp32"}:
        raise ValueError("cfg.train.precision must be 'bf16' or 'fp32'")
    for field_name in ("log_every", "diag_every", "checkpoint_every"):
        value = getattr(cfg.train, field_name)
        if value <= 0:
            raise ValueError(f"train.{field_name} must be positive")
    if cfg.train.checkpoint_every > cfg.train.max_steps:
        raise ValueError("train.checkpoint_every must be <= train.max_steps")
    if cfg.data.dataset not in {"ssv2", "ssv2_tiny", "ego4d", "ego4d_tiny"}:
        raise ValueError(f"cfg.data.dataset is unsupported: {cfg.data.dataset!r}")
    if cfg.data.num_workers < 0:
        raise ValueError("cfg.data.num_workers must be non-negative")


def main() -> None:
    """Run the requested Phase 1 command (Stage 0 sanity or Stage 1 training)."""
    args = parse_args()
    experiment = load_experiment_config(EXPERIMENT_CONFIG_PATH)
    cfg = experiment.config
    runtime = experiment.runtime
    wandb_config = experiment.wandb

    # The only scientific CLI overrides are the five axes repeatedly varied in the latest
    # KANBAN investigations. Every other recipe value remains an auditable YAML edit.
    if args.data is not None:
        cfg.data.dataset = args.data
    if args.encoder is not None:
        if args.encoder != cfg.encoder.alias and cfg.encoder.revision is not None:
            raise ValueError(
                "--encoder cannot replace encoder.alias while encoder.revision is pinned in YAML; "
                "set encoder.revision to null or edit the pinned revision in configs/train.yaml"
            )
        cfg.encoder.alias = args.encoder
    if args.n_c is not None:
        cfg.model.n_c = args.n_c
    if args.d_c is not None:
        cfg.model.d_c = args.d_c
    if args.bottleneck_mixer_dim is not None:
        cfg.model.bottleneck_mixer_dim = args.bottleneck_mixer_dim
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir

    compare_provenance = args.compare_provenance or runtime.compare_provenance
    if compare_provenance:
        from provenance import compare_run_provenance

        left_path, right_path = map(Path, compare_provenance)
        compare_run_provenance(
            json.loads(left_path.read_text(encoding="utf-8")),
            json.loads(right_path.read_text(encoding="utf-8")),
        )
        print(f"Provenance parity passed: {left_path} == {right_path} (common fields)")
        return

    mode = runtime.mode
    if args.stage0_only:
        mode = "stage0"
    elif args.preflight_only:
        mode = "preflight"
    elif args.resource_preflight:
        mode = "resource_preflight"
    provenance_out = args.provenance_out or runtime.provenance_out
    resume = args.resume or runtime.resume
    require_wandb = args.require_wandb or runtime.require_wandb
    reset_optimizer = args.reset_optimizer or runtime.reset_optimizer
    allow_dataset_transfer = args.allow_dataset_transfer or runtime.allow_dataset_transfer
    allow_legacy_checkpoint = args.allow_legacy_checkpoint or runtime.allow_legacy_checkpoint
    wandb_options = {
        "entity": args.wandb_entity if args.wandb_entity is not None else wandb_config.entity,
        "project": (args.wandb_project if args.wandb_project is not None else wandb_config.project),
        "group": args.wandb_group if args.wandb_group is not None else wandb_config.group,
        "name": args.wandb_name if args.wandb_name is not None else wandb_config.name,
        "id": args.wandb_run_id if args.wandb_run_id is not None else wandb_config.run_id,
    }
    finalize_training_config(cfg)
    if mode == "stage0":
        run_stage0(cfg)
    elif mode == "preflight":
        output = provenance_out or "logs/preflight/run_provenance.json"
        materialize_preflight(
            cfg,
            output,
            tracking_identity=wandb_options,
        )
    elif mode == "resource_preflight":
        output = provenance_out or "logs/preflight/resource_preflight.json"
        run_resource_preflight(cfg, output)
    elif mode == "train":
        run_training(
            cfg,
            cfg.train.max_steps,
            resume,
            require_wandb=require_wandb,
            wandb_options=wandb_options,
            provenance_out=provenance_out,
            reset_optimizer=reset_optimizer,
            allow_dataset_transfer=allow_dataset_transfer,
            allow_legacy_checkpoint=allow_legacy_checkpoint,
        )
    else:
        raise ValueError(
            "runtime.mode must be one of train, stage0, preflight, resource_preflight; "
            f"got {mode!r}"
        )


if __name__ == "__main__":
    main()
