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


def _mean_slot_effective_rank(x: Tensor, *, center_slots: bool) -> float:
    """Return mean per-video effective rank across the slot axis.

    Args:
        x: (B, N_c, D_c) abstract latent `c_t`.
        center_slots: Whether to subtract each video's mean slot vector first.
    Returns:
        Mean effective rank over the batch, or NaN if a Gram matrix is non-finite.
    """
    ranks: list[float] = []
    for b in range(x.shape[0]):
        slots = x[b]  # (N_c, D_c)
        if center_slots:
            slots = slots - slots.mean(dim=0, keepdim=True)
        gram = slots @ slots.t() / max(1, slots.shape[1])  # (N_c, N_c)
        if not torch.isfinite(gram).all():
            return float("nan")
        eig = torch.linalg.eigvalsh(gram).clamp_min(0)
        probs = eig / eig.sum().clamp_min(1e-8)
        entropy = -(probs * (probs + 1e-8).log()).sum()
        ranks.append(float(torch.exp(entropy).item()))
    return float(sum(ranks) / len(ranks))


def slot_diversity_rank(abstract: Tensor) -> dict[str, float]:
    """Within-video effective rank of the 32 query-slot vectors.

    The raw value keeps the historical dashboard key but is now expected to jump
    mechanically because fixed slot identities survive into the bottleneck output.
    The centered value subtracts each video's mean slot vector before ranking and
    is the cleaner readout for whether slots carry different information.

    Args:
        abstract: (B, N_c, D_c) online abstract latent `c_t`.
    Returns:
        Metrics dict with raw and centered mean within-video slot effective rank,
        or NaN values if a covariance is non-finite.
    """
    _require_torch()
    x = abstract.float()
    if x.shape[0] < 1 or x.shape[1] < 2:
        nan = float("nan")
        return {"c_slot_diversity_rank": nan, "c_slot_diversity_rank_centered": nan}
    return {
        "c_slot_diversity_rank": _mean_slot_effective_rank(x, center_slots=False),
        "c_slot_diversity_rank_centered": _mean_slot_effective_rank(x, center_slots=True),
    }


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


# Parameters held out of BOTH weight decay (Issue 9) and adaptive gradient
# clipping (Issue 1). Decaying or per-tensor-clipping these fights representation
# geometry instead of stabilizing optimization:
#   * every 1-D tensor — biases and LayerNorm scale/shift (ndim < 2);
#   * learned coordinate systems — bottleneck/decoder query slots, bottleneck memory
#     position embeddings, the F_c null condition, and the F_c token-type /
#     slot-position embeddings (by leaf name);
#   * the zero-init adaLN-Zero gate (AdaLNBlock.mod[-1]) and residual-output
#     projection (Bottleneck.out_mlp[-1]), flagged ``is_zero_init`` at construction.
#     Both are 2-D weights, so neither the ndim nor the name rule catches them — the
#     marker does. While ||w||≈0 their AGC bound collapses to ~clip_factor·eps, which
#     would clamp the very gradients that must "wake up" the residual branches; weight
#     decay on them just re-pins the branch at identity. (Note: AGC scales .grad
#     uniformly per tensor and AdamW then divides by the per-coordinate grad RMS, so
#     a uniform scale largely cancels in the update — but excluding these is the
#     correct NFNet rule regardless of how much it changes any single step.)
_GEOMETRY_LEAF_NAMES = frozenset(
    {"queries", "pos_emb", "null_condition", "slot_pos", "z_type", "cond_type"}
)


def _zero_init_param_ids(module: nn.Module) -> set[int]:
    """Ids of params owned by submodules flagged ``is_zero_init`` at construction.

    Module attributes survive ``.to(device)`` (the Module object is not recreated),
    so the marker is a robust, refactor-proof signal for the zero-init gate/output
    projections — no brittle ``out_mlp.3.weight`` index strings.
    """
    _require_torch()
    ids: set[int] = set()
    for sub in module.modules():
        if getattr(sub, "is_zero_init", False):
            for param in sub.parameters(recurse=False):
                ids.add(id(param))
    return ids


def is_geometry_or_gate_param(name: str, param: Tensor, zero_init_ids: set[int]) -> bool:
    """True when a parameter must be held out of weight decay and AGC.

    See ``_GEOMETRY_LEAF_NAMES`` for the rationale. One shared predicate keeps the
    Issue-1 (AGC) and Issue-9 (weight-decay) exclusion sets identical by construction.
    """
    if param.ndim < 2:
        return True
    if name.rsplit(".", 1)[-1] in _GEOMETRY_LEAF_NAMES:
        return True
    return id(param) in zero_init_ids


def partition_decay_params(module: nn.Module) -> tuple[list, list]:
    """Split a module's trainable params into ``(weight_decay, no_decay)`` lists.

    Consumed by ``train.make_optimizer`` to build per-module decay/no-decay AdamW
    groups (Issue 9). Uses the same predicate as AGC so the two exclusion sets can
    never drift apart.

    Args:
        module: A trainable module (B, F_c, or D).
    Returns:
        ``(decay, no_decay)`` parameter lists; together they cover every
        ``requires_grad`` parameter exactly once.
    """
    _require_torch()
    zero_init_ids = _zero_init_param_ids(module)
    decay: list = []
    no_decay: list = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        target = no_decay if is_geometry_or_gate_param(name, param, zero_init_ids) else decay
        target.append(param)
    return decay, no_decay


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

    Biases, LayerNorm params, learned query/null/type/slot embeddings, and the
    zero-init gate/output projections are EXCLUDED (Issue 1 — see
    ``is_geometry_or_gate_param``): NFNet never clips the final layer, and clipping
    a ~zero-norm tensor to ~clip_factor·eps would keep the zero-init residual gates
    from opening. The genuine Linear/Conv weight matrices are still clipped.

    Args:
        module: Trainable module with optional `.grad` on its parameters.
        clip_factor: λ — maximum allowed gradient-to-weight norm ratio.
        eps: Floor added to ||w|| in the bound (numerical stability).
    Returns:
        Metrics dict with clipped tensor count and max pre-clip ratio seen.
    """
    _require_torch()
    zero_init_ids = _zero_init_param_ids(module)
    clipped_tensors = 0
    max_ratio = 0.0
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if is_geometry_or_gate_param(name, param, zero_init_ids):
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
    flow_target: Tensor,
    eps_c: Tensor,
    predict_residual: bool = False,
) -> dict[str, float]:
    """Compare F_c velocity loss against copy and batch-mean baselines (§9.3).

    Args:
        coarse_flow: F_c module producing (B, 32, 256) velocity predictions.
        z_c: (B, 32, 256) noised flow target (c_plus, or the residual Δ in residual mode).
        tau_c: (B,) flow times.
        current_abstract: (B, 32, 256) current c_t (the conditioning).
        flow_target: (B, 32, 256) the regression target — c_plus normally, or the temporal
            residual Δ = c_{t+k} - c_t when ``predict_residual`` (both detached).
        eps_c: (B, 32, 256) Gaussian noise used to build z_c (scaled to Δ in residual mode).
        predict_residual: investigation_009. When True the copy baseline is "predict ZERO
            residual" (Δ̂=0); otherwise it is "copy c_t forward". Both reduce copy_loss to
            ‖c_t - c_plus‖² = ‖Δ‖², so coarse_vs_copy_ratio stays comparable across runs.
    Returns:
        Metrics dict with model/copy/batch-mean losses and ratios.
    """
    _require_torch()
    with torch.no_grad():
        u_target = velocity_target(flow_target, eps_c)
        u_hat = coarse_flow(z_c, tau_c, current_abstract, condition_drop=_no_drop(current_abstract))
        model_loss = flow_matching_loss(u_hat, u_target)
        # "Predict no change": zero residual (velocity -eps) in residual mode, else copy c_t
        # forward. Both give copy_loss = ‖c_t - c_plus‖² = ‖Δ‖² because the eps term cancels.
        copy_velocity = -eps_c if predict_residual else current_abstract - eps_c
        copy_loss = flow_matching_loss(copy_velocity, u_target)
        batch_mean = flow_target.mean(dim=0, keepdim=True).expand_as(flow_target)
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
