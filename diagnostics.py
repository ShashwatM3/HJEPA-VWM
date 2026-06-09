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

    Args:
        abstract: (B, N_c, D_c) abstract latent flattened over batch/slots.
        eps: Numerical floor for eigenvalue normalization.
    Returns:
        Metrics dict with the effective rank `exp(entropy(eigenvalue_distribution))`.
    """
    _require_torch()
    flat = abstract.float().reshape(-1, abstract.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = flat.t() @ flat / max(1, flat.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    probs = eig / eig.sum().clamp_min(eps)
    entropy = -(probs * (probs + eps).log()).sum()
    return {"c_effective_rank": float(torch.exp(entropy).item())}


def gradient_health(model: nn.Module) -> dict[str, float]:
    """Summarize trainable parameter gradient health.

    Args:
        model: Module or container with trainable parameters.
    Returns:
        Metrics dict with global grad norm, NaN flag, and grad'd param count.
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
        "grad_global_norm": total_sq**0.5,
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
    assert all(isinstance(v, float) for v in metrics.values())
    print(metrics)
