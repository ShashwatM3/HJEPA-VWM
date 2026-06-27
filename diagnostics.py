"""Phase 1 diagnostics for collapse, baselines, and gradient health (v0.2).

The three required minimum monitors (supervisor) are on `c_t`: per-dimension
variance, cross-video cosine similarity, and effective rank. The encoder is frozen,
so `e_t` cannot collapse and is not monitored. All probes are pure and return
`dict[str, float]`.
"""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]

from losses import flow_matching_loss, velocity_target


def _require_torch() -> None:
    """Fail fast when diagnostics are used without PyTorch installed."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for diagnostics.py. Install requirements.txt on RunPod."
        )


def variance_stats(abstract: Tensor) -> dict[str, float]:
    """Per-dimension variance health of `c_t` (§9.1 collapse flags).

    Flattens `c_t` across slots/features per batch element (the same quantity the
    variance floor acts on), computes per-dimension std across the batch, and
    reports the mean/median std and the fraction of near-dead dimensions.

    Args:
        abstract: (B, N_c, D_c) online abstract latent `c_t`.
    Returns:
        Metrics dict: mean/median per-dim std and dead-dimension fraction.
    """
    _require_torch()
    flat = abstract.float().reshape(abstract.shape[0], -1)
    std = flat.std(dim=0, unbiased=False)
    median = std.median().clamp_min(1e-12)
    dead = (std < 0.1 * median).float().mean()
    return {
        "c_std_mean": float(std.mean().item()),
        "c_std_median": float(median.item()),
        "c_dead_dim_frac": float(dead.item()),
    }


def cross_video_cosine(abstract: Tensor) -> dict[str, float]:
    """Mean pairwise cosine similarity of `c_t` across different videos (§9.1b).

    Healthy is well below ~0.5 (distinct videos -> distinct `c_t`). Drift toward 1.0
    signals directional collapse that per-dimension variance alone can miss.

    Args:
        abstract: (B, N_c, D_c) online abstract latent `c_t`, B > 1.
    Returns:
        Metrics dict with the mean off-diagonal pairwise cosine.
    """
    _require_torch()
    flat = abstract.float().reshape(abstract.shape[0], -1)
    if flat.shape[0] < 2:
        return {"c_cross_video_cosine": 0.0}
    normed = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = normed @ normed.t()
    b = sim.shape[0]
    off_diagonal = sim[~torch.eye(b, dtype=torch.bool, device=sim.device)]
    return {"c_cross_video_cosine": float(off_diagonal.mean().item())}


def effective_rank(abstract: Tensor, eps: float = 1e-8) -> dict[str, float]:
    """Covariance effective rank of `c_t` (§9.2 rank floors; healthy > 60).

    Returns `NaN` instead of raising if the covariance has any non-finite
    entries: a model that has gone NaN is a real failure mode, but it should
    surface as a logged NaN value, not a `torch.linalg.eigvalsh` crash that
    takes down the training loop mid-run (see POSTMORTEM_RUN1.md).

    Args:
        abstract: (B, N_c, D_c) abstract latent flattened over batch/slots.
        eps: Numerical floor for eigenvalue normalization.
    Returns:
        Metrics dict with the effective rank `exp(entropy(eigenvalue_distribution))`,
        or `{"c_effective_rank": NaN}` if the covariance is non-finite.
    """
    _require_torch()
    flat = abstract.float().reshape(-1, abstract.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = flat.t() @ flat / max(1, flat.shape[0] - 1)
    if not torch.isfinite(cov).all():
        return {"c_effective_rank": float("nan")}
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    probs = eig / eig.sum().clamp_min(eps)
    entropy = -(probs * (probs + eps).log()).sum()
    return {"c_effective_rank": float(torch.exp(entropy).item())}


def slot_diversity_rank(abstract: Tensor) -> dict[str, float]:
    """Within-video effective rank of the 32 query-slot vectors.

    `effective_rank` pools all slots and videos together, so a low value
    conflates two failures: redundant query slots (all reading similar content)
    vs. correlated feature dimensions. This probe isolates the first: per video,
    it measures how many independent directions the `N_c` slot outputs span
    (max `N_c`), then averages over the batch. A low value points at attention
    saturation (the Fix-1 target); a healthy slot-rank with a low cross-video
    `effective_rank` instead points at feature correlation (the Fix-2 target).
    See KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/DETAILED_UNDERSTAND.md §4.3.

    Args:
        abstract: (B, N_c, D_c) online abstract latent `c_t`.
    Returns:
        Metrics dict with the mean within-video slot effective rank, or NaN if
        a covariance is non-finite.
    """
    _require_torch()
    x = abstract.float()
    if x.shape[0] < 1 or x.shape[1] < 2:
        return {"c_slot_diversity_rank": float("nan")}
    ranks: list[float] = []
    for b in range(x.shape[0]):
        slots = x[b]  # (N_c, D_c)
        slots = slots - slots.mean(dim=0, keepdim=True)
        gram = slots @ slots.t() / max(1, slots.shape[1])  # (N_c, N_c)
        if not torch.isfinite(gram).all():
            return {"c_slot_diversity_rank": float("nan")}
        eig = torch.linalg.eigvalsh(gram).clamp_min(0)
        probs = eig / eig.sum().clamp_min(1e-8)
        entropy = -(probs * (probs + 1e-8).log()).sum()
        ranks.append(float(torch.exp(entropy).item()))
    return {"c_slot_diversity_rank": float(sum(ranks) / len(ranks))}


def attention_entropy(bottleneck: nn.Module, detailed: Tensor) -> dict[str, float]:
    """Normalized PER-HEAD entropy of the bottleneck cross-attention.

    For each (batch, head, query-slot), computes the Shannon entropy of the
    attention distribution over the `N_ctx` memory tokens, normalized by
    `log(N_ctx)` so 1.0 == perfectly uniform (the saturation symptom) and lower
    == sharper / more selective.

    Per-head is deliberate: PyTorch's default head-AVERAGED weights make 8
    sharp-but-different heads look uniform, which masks real selectivity (an
    early version read ~1.0 while `slot_diversity_rank` showed the slots were
    clearly differentiated — see KANBAN/04-FIX-DIMENSIONAL-COLLAPSE). We report
    the mean (overall selectivity) and the min over (head, slot) (does *any*
    head specialize at all).

    Args:
        bottleneck: the online Bottleneck `B`.
        detailed: (B, N_ctx, D_e) frozen-encoder tokens to attend over.
    Returns:
        Metrics dict with mean and min normalized per-head attention entropy.
    """
    _require_torch()
    import math

    with torch.no_grad():
        _, attn = bottleneck(detailed, return_attn=True)  # (B, num_heads, N_c, N_ctx)
    attn = attn.float().clamp_min(1e-12)
    entropy = -(attn * attn.log()).sum(dim=-1)  # (B, num_heads, N_c)
    entropy = entropy / math.log(attn.shape[-1])
    return {
        "c_attn_entropy": float(entropy.mean().item()),
        "c_attn_entropy_min": float(entropy.min().item()),
    }


def adaptive_gradient_clip(
    module: nn.Module,
    clip_factor: float,
    eps: float = 1e-3,
) -> dict[str, float]:
    """Per-tensor AGC: clip gradients when ||g|| exceeds λ(||w|| + ε).

    Implements the Brock et al. rule used in NFNet training: each parameter
    tensor is scaled in-place so no single weight block can propose an update
    far larger than its own weight scale. This absorbs moderate flow-matching
    spikes (elated step-8450 class) instead of freezing training on the first
    global norm above 50.

    Args:
        module: Trainable module with optional `.grad` on its parameters.
        clip_factor: λ — maximum allowed gradient-to-weight norm ratio.
        eps: Floor added to ||w|| in the bound (numerical stability).
    Returns:
        Metrics dict with clipped tensor count and max pre-clip ratio seen.
    """
    _require_torch()
    clipped_tensors = 0
    max_ratio = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        weight_norm = param.detach().float().norm()
        grad_norm = grad.float().norm()
        bound = clip_factor * (weight_norm + eps)
        if bound <= 0:
            continue
        ratio = float((grad_norm / bound).item())
        max_ratio = max(max_ratio, ratio)
        if grad_norm > bound:
            clipped_tensors += 1
            param.grad.mul_(bound / (grad_norm + eps))
    return {
        "agc_clipped_tensors": float(clipped_tensors),
        "agc_max_ratio": max_ratio,
        "agc_any_clipped": float(clipped_tensors > 0),
    }


def apply_trainable_agc(
    bottleneck: nn.Module,
    coarse_flow: nn.Module,
    decoder: nn.Module,
    *,
    enabled: bool,
    lambda_bottleneck: float,
    lambda_coarse_flow: float,
    lambda_decoder: float,
    eps: float,
) -> dict[str, float]:
    """Apply module-specific AGC to bottleneck B, coarse flow F_c, and decoder D.

    Args:
        bottleneck: Trainable bottleneck B.
        coarse_flow: Trainable coarse flow F_c.
        decoder: Reconstruction decoder D (no grads when lambda_recon=0 -> no-op).
        enabled: When False, returns ``agc_active=0`` without touching gradients.
        lambda_bottleneck: λ for B (milder — healthy grads stay ~2–3).
        lambda_coarse_flow: λ for F_c (primary instability source).
        lambda_decoder: λ for D (mirrors the bottleneck).
        eps: AGC denominator floor passed to :func:`adaptive_gradient_clip`.
    Returns:
        Flat metrics dict for W&B logging (``agc_B_*``, ``agc_Fc_*``, ``agc_D_*``).
    """
    _require_torch()
    if not enabled:
        return {"agc_active": 0.0}
    metrics: dict[str, float] = {"agc_active": 1.0}
    for prefix, module, clip_factor in (
        ("agc_B", bottleneck, lambda_bottleneck),
        ("agc_Fc", coarse_flow, lambda_coarse_flow),
        ("agc_D", decoder, lambda_decoder),
    ):
        sub = adaptive_gradient_clip(module, clip_factor, eps)
        metrics[f"{prefix}_clipped"] = sub["agc_clipped_tensors"]
        metrics[f"{prefix}_max_ratio"] = sub["agc_max_ratio"]
        metrics[f"{prefix}_any_clipped"] = sub["agc_any_clipped"]
    return metrics


def gradient_health(model: nn.Module) -> dict[str, float]:
    """Summarize trainable parameter gradient health.

    Args:
        model: Module or container with trainable parameters.
    Returns:
        Metrics dict with the POST-CLIP global grad norm (see note), NaN flag, and
        grad'd param count. For the TRUE gradient magnitude read train_step's
        pre-clip `grad_norm`, not this.
    """
    _require_torch()
    total_sq = 0.0
    has_nan = False
    param_count = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        has_nan = has_nan or bool(torch.isnan(grad).any().item())
        total_sq += float(grad.pow(2).sum().item())
        param_count += 1
    return {
        # NOTE (WALK_FIXES F1): in the training loop this runs AFTER in-place AGC +
        # clip_grad_norm_(0.5), so it is the POST-CLIP norm — pinned near the 0.5 clip
        # whenever clipping fires (≈ every step), NOT the true gradient magnitude.
        # Read train_step's pre-clip `grad_norm` for that. Renamed so it can't be
        # misread as the raw norm.
        "grad_global_norm_postclip": total_sq**0.5,
        "grad_has_nan": float(has_nan),
        "grad_param_count": float(param_count),
    }


def coarse_baselines(
    coarse_flow: nn.Module,
    z_c: Tensor,
    tau_c: Tensor,
    current_abstract: Tensor,
    target_abstract: Tensor,
    eps_c: Tensor,
) -> dict[str, float]:
    """Compare F_c velocity loss against copy and batch-mean baselines (§9.3).

    Args:
        coarse_flow: F_c module producing (B, 32, 256) velocity predictions.
        z_c: (B, 32, 256) noised target abstract latent.
        tau_c: (B,) flow times.
        current_abstract: (B, 32, 256) current c_t.
        target_abstract: (B, 32, 256) target c_plus, detached.
        eps_c: (B, 32, 256) Gaussian noise used to build z_c.
    Returns:
        Metrics dict with model/copy/batch-mean losses and ratios.
    """
    _require_torch()
    with torch.no_grad():
        u_target = velocity_target(target_abstract, eps_c)
        u_hat = coarse_flow(z_c, tau_c, current_abstract, condition_drop=_no_drop(current_abstract))
        model_loss = flow_matching_loss(u_hat, u_target)
        copy_velocity = current_abstract - eps_c
        copy_loss = flow_matching_loss(copy_velocity, u_target)
        batch_mean = target_abstract.mean(dim=0, keepdim=True).expand_as(target_abstract)
        mean_velocity = batch_mean - eps_c
        mean_loss = flow_matching_loss(mean_velocity, u_target)
    return {
        "coarse_model_loss": float(model_loss.item()),
        "coarse_copy_loss": float(copy_loss.item()),
        "coarse_batch_mean_loss": float(mean_loss.item()),
        "coarse_vs_copy_ratio": float((model_loss / copy_loss.clamp_min(1e-8)).item()),
        "coarse_vs_batch_mean_ratio": float((model_loss / mean_loss.clamp_min(1e-8)).item()),
    }


def _no_drop(abstract: Tensor) -> Tensor:
    """Return an all-False condition-drop mask so diagnostics use the real `c_t`."""
    return torch.zeros(abstract.shape[0], dtype=torch.bool, device=abstract.device)


def smoke_test_diagnostics() -> None:
    """Run diagnostics on synthetic Phase 1 tensors."""
    _require_torch()
    c = torch.randn(8, 32, 256)
    metrics = {}
    metrics.update(variance_stats(c))
    metrics.update(cross_video_cosine(c))
    metrics.update(effective_rank(c))
    metrics.update(slot_diversity_rank(c))
    assert all(isinstance(v, float) for v in metrics.values())
    print(metrics)
