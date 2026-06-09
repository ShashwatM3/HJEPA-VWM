"""Pure tensor losses for HJEPA-VWM (v0.2).

Flow-matching primitives (rectified flow) plus the variance floor on c_t. SIGReg /
VICReg / covariance losses are intentionally absent (v0.2 supervisor directive —
collapse prevention is a per-dimension variance floor on c_t only). No nn.Parameter
lives here; these are pure functions reused across coarse/fine/frame stages.
"""

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

    Used for the EMA bottleneck (`B_EMA`) output, the frozen-encoder target latent,
    and detached conditioning, so every gradient boundary is visible through one
    named helper instead of scattered `.detach()` calls.

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
        u: (B, N, D) velocity target `z_target - eps` (constant along the trajectory).
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


def variance_floor(abstract: Tensor, std_target: float = 1.0) -> Tensor:
    """Per-dimension variance floor on the abstract latent `c_t` (replaces SIGReg).

    The supervisor's collapse guardrail: flatten `c_t` across slots/features per
    batch element, compute the per-dimension std across the batch, and hinge each
    dimension at `std_target`. This only prevents a *constant* `c_t`; the
    flow-matching objective is what makes `c_t` carry real semantics, so the weight
    `λ_var` is kept small (0.10) and no covariance/SIGReg term is added.

    `L_var = (1/d) Σ_j max(0, std_target - Std(c_j))`,  d = N_c * D_c.

    Args:
        abstract: (B, N_c, D_c) online abstract latent `c_t`.
        std_target: Target per-dimension std (1.0).
    Returns:
        Scalar variance-floor loss.
    """
    _require_torch()
    flat = abstract.reshape(abstract.shape[0], -1).float()
    if flat.shape[0] < 2:
        return flat.new_tensor(0.0)
    std = flat.std(dim=0, unbiased=False)
    return torch.clamp(std_target - std, min=0.0).mean()
