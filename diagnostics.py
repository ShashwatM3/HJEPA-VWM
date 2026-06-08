"""Phase 1 diagnostics for collapse, baselines, and gradient health."""

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


def latent_std_stats(detailed: Tensor, abstract: Tensor) -> dict[str, float]:
    """Measure per-dimension latent standard deviation health.

    Args:
        detailed: (B, N_ctx_post, D_e) online detailed latent e_t.
        abstract: (B, N_c, D_c) online abstract latent c_t.
    Returns:
        Metrics dict with median std and dead-dimension fractions for e_t/c_t.
    """
    _require_torch()

    def stats(x: Tensor, prefix: str) -> dict[str, float]:
        """Compute median std and dead-dimension fraction for one latent."""
        flat = x.float().reshape(-1, x.shape[-1])
        std = flat.std(dim=0, unbiased=False)
        median = std.median().clamp_min(1e-12)
        dead = (std < 0.1 * median).float().mean()
        return {
            f"{prefix}_std_median": float(median.item()),
            f"{prefix}_dead_dim_frac": float(dead.item()),
        }

    return {**stats(detailed, "e"), **stats(abstract, "c")}


def effective_rank(x: Tensor, eps: float = 1e-8) -> float:
    """Compute covariance effective rank for a latent tensor.

    Args:
        x: (B, N, D) latent tensor to flatten over batch/tokens.
        eps: Numerical floor for eigenvalue normalization.
    Returns:
        Effective rank `exp(entropy(eigenvalue_distribution))`.
    """
    _require_torch()
    flat = x.float().reshape(-1, x.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = flat.T @ flat / max(1, flat.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    probs = eig / eig.sum().clamp_min(eps)
    entropy = -(probs * (probs + eps).log()).sum()
    return float(torch.exp(entropy).item())


def gradient_health(model: nn.Module) -> dict[str, float]:
    """Summarize trainable parameter gradient health.

    Args:
        model: Module or container with trainable parameters.
    Returns:
        Metrics dict containing global grad norm and NaN flags.
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
    """Compare F_c velocity loss against copy and batch-mean baselines.

    Args:
        coarse_flow: F_c module producing (B, 32, 256) velocity predictions.
        z_c: (B, 32, 256) noised target abstract latent.
        tau_c: (B,) flow times.
        current_abstract: (B, 32, 256) current c_t.
        target_abstract: (B, 32, 256) target c_plus, detached.
        eps_c: (B, 32, 256) Gaussian noise used to build z_c.
    Returns:
        Metrics dict containing model, copy, batch-mean losses and ratios.
    """
    _require_torch()
    with torch.no_grad():
        u_target = velocity_target(target_abstract, eps_c)
        u_hat = coarse_flow(z_c, tau_c, current_abstract)
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


def smoke_test_diagnostics() -> None:
    """Run diagnostics on synthetic Phase 1 tensors."""
    _require_torch()
    e = torch.randn(4, 154, 384)
    c = torch.randn(4, 32, 256)
    metrics = latent_std_stats(e, c)
    metrics["rank_e"] = effective_rank(e)
    metrics["rank_c"] = effective_rank(c)
    assert all(isinstance(v, float) for v in metrics.values())
    print(metrics)
