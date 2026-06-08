"""Pure tensor losses for HJEPA-VWM."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]


def _require_torch() -> None:
    """Fail fast when tensor losses are used without PyTorch installed."""
    if torch is None:
        raise RuntimeError("PyTorch is required for losses.py. Install requirements.txt on RunPod.")


def _broadcast_tau(tau: Tensor, target: Tensor) -> Tensor:
    """Expand (B,) flow times until they broadcast over a target tensor."""
    while tau.ndim < target.ndim:
        tau = tau.unsqueeze(-1)
    return tau.to(device=target.device, dtype=target.dtype)


def as_target(x: Tensor) -> Tensor:
    """Mark a tensor as a stop-gradient target.

    Used for EMA branch outputs and detached conditioning so every gradient
    boundary is visible through one named helper instead of scattered `.detach()` calls.

    Args:
        x: Any tensor that should be predicted but never optimized.
    Returns:
        Detached tensor with the same shape as `x`.
    """
    return x.detach()


def interpolate(z_target: Tensor, eps: Tensor, tau: Tensor) -> Tensor:
    """Build the rectified-flow interpolation point.

    Args:
        z_target: (B, N, D) stop-gradient target latent.
        eps: (B, N, D) Gaussian noise sample.
        tau: (B,) flow time, broadcast to (B, 1, 1).
    Returns:
        z: (B, N, D) interpolated latent `(1 - tau) * eps + tau * z_target`.
    """
    tau_b = _broadcast_tau(tau, z_target)
    return (1.0 - tau_b) * eps + tau_b * z_target


def velocity_target(z_target: Tensor, eps: Tensor) -> Tensor:
    """Compute the rectified-flow target velocity.

    Args:
        z_target: (B, N, D) future latent target.
        eps: (B, N, D) Gaussian noise source.
    Returns:
        u: (B, N, D) velocity target `z_target - eps`.
    """
    return z_target - eps


def flow_matching_loss(u_hat: Tensor, u_target: Tensor) -> Tensor:
    """Mean-squared error for coarse, fine, and frame flow matching.

    Args:
        u_hat: (B, N, D) predicted velocity.
        u_target: (B, N, D) target velocity.
    Returns:
        Scalar loss averaged over all batch and token dimensions.
    """
    return (u_hat - u_target).pow(2).mean()


def sigreg(latents: Tensor, m: int = 1024, knots: int = 17) -> Tensor:
    """Sketched Isotropic Gaussian Regularization for online latents.

    This implementation follows the Epps-Pulley / characteristic-function spirit
    used by SIGReg: random 1D projections of flattened tokens are compared to a
    standard normal characteristic function across integration knots.

    Args:
        latents: (B, N, D) online detailed or abstract latent tokens.
        m: Number of random projection directions.
        knots: Number of integration knots.
    Returns:
        Scalar regularizer encouraging isotropic Gaussian latent statistics.
    """
    _require_torch()
    x = latents.float().reshape(-1, latents.shape[-1])
    if x.shape[0] < 2:
        return x.new_tensor(0.0)
    x = x - x.mean(dim=0, keepdim=True)
    directions = torch.randn(x.shape[-1], m, device=x.device, dtype=x.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-6)
    proj = x @ directions
    proj = proj / proj.std(dim=0, keepdim=True).clamp_min(1e-6)
    t = torch.linspace(-3.0, 3.0, knots, device=x.device, dtype=x.dtype)
    empirical_real = torch.cos(proj[:, :, None] * t).mean(dim=0)
    empirical_imag = torch.sin(proj[:, :, None] * t).mean(dim=0)
    normal_real = torch.exp(-0.5 * t.pow(2))[None, :]
    stat = (empirical_real - normal_real).pow(2) + empirical_imag.pow(2)
    return stat.mean()


def phase1_total_loss(
    coarse_loss: Tensor,
    detailed: Tensor,
    abstract: Tensor,
    lambda_e_reg: float,
    lambda_c_reg: float,
    sigreg_m: int,
    sigreg_knots: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine Stage 1 coarse flow loss with SIGReg on online latents.

    Args:
        coarse_loss: Scalar L_c from F_c velocity matching.
        detailed: (B, N_ctx_post, D_e) online detailed latent e_t.
        abstract: (B, N_c, D_c) online abstract latent c_t.
        lambda_e_reg: Weight for SIGReg(e_t).
        lambda_c_reg: Weight for SIGReg(c_t).
        sigreg_m: Random projection count.
        sigreg_knots: Integration knot count.
    Returns:
        total: Scalar Phase 1 objective.
        parts: Dict of scalar component tensors for logging.
    """
    reg_e = sigreg(detailed, sigreg_m, sigreg_knots)
    reg_c = sigreg(abstract, sigreg_m, sigreg_knots)
    total = coarse_loss + lambda_e_reg * reg_e + lambda_c_reg * reg_c
    return total, {"L_c": coarse_loss, "SIGReg_e": reg_e, "SIGReg_c": reg_c}
