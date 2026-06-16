"""Pure tensor losses for HJEPA-VWM (v0.2).

Flow-matching primitives (rectified flow) plus the anti-collapse regularizers on
c_t: the per-dimension variance floor (VICReg V, always on, λ=0.10), the
off-diagonal feature covariance penalty (VICReg C, flag-gated via `lambda_cov`),
and the within-video slot-diversity penalty (flag-gated via `lambda_slot`).
The covariance term attacks feature-dim correlation; the slot term attacks the
measured Plan Phase 04 failure where all 32 bottleneck slots read out the same
near-uniform attention average. No nn.Parameter lives here; these are pure
functions reused across coarse/fine/frame stages.
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
    dimension at `std_target`. This only prevents a *constant* `c_t`; it does
    not prevent feature-dim correlation or redundant slots. Plan Phase 04 adds
    separate, flag-gated covariance and slot-diversity terms for those axes.

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


def covariance_floor(abstract: Tensor) -> Tensor:
    """VICReg covariance term on `c_t` — the off-diagonal decorrelation penalty.

    `variance_floor` is VICReg's V (each feature dim must carry spread); this is
    VICReg's C: it pushes the off-diagonal covariances of `c_t`'s feature
    dimensions toward zero so the 256 dims encode *distinct* factors instead of
    collapsing onto a correlated low-rank subspace (the `c_effective_rank ≈ 5`
    symptom — see KANBAN/04-FIX-DIMENSIONAL-COLLAPSE). The variance floor alone
    can't do this: it only forbids constant dims, never correlated ones.

    Pools batch × slots into `N = B·N_c` samples over `D = D_c` feature dims,
    centers per dim, forms the `D×D` covariance `(zᵀz)/(N-1)`, and returns the
    sum of squared OFF-diagonal entries divided by `D` (the standard VICReg
    convention: `Σ_{i≠j} cov_ij² / D`). The diagonal (variance) is left to
    `variance_floor`, so the two terms don't fight over scale.

    Args:
        abstract: (B, N_c, D_c) online abstract latent `c_t`.
    Returns:
        Scalar covariance penalty (>= 0; 0 == perfectly decorrelated dims).
    """
    _require_torch()
    z = abstract.reshape(-1, abstract.shape[-1]).float()  # (N = B·N_c, D_c)
    n, d = z.shape
    if n < 2:
        return z.new_tensor(0.0)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (n - 1)  # (D, D)
    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag_sq / d


def slot_diversity_loss(abstract: Tensor, eps: float = 1e-8) -> Tensor:
    """Within-video slot-collapse penalty on the abstract latent `c_t`.

    The Run-A diagnosis showed `c_slot_diversity_rank ≈ 1.6/32` and nearly
    uniform cross-attention: all bottleneck query slots were reading the same
    average token. This loss directly penalizes that failure. For each video,
    L2-normalize the `N_c` slot vectors, form the `N_c×N_c` cosine-similarity
    matrix, and average the squared OFF-diagonal entries. Identical slots give
    ~1.0; mutually orthogonal slots give 0.0.

    Args:
        abstract: (B, N_c, D_c) online abstract latent `c_t`.
        eps: Numerical floor for slot-vector normalization.
    Returns:
        Scalar slot-diversity loss (>= 0; 0 == decorrelated slots).
    """
    _require_torch()
    x = abstract.float()
    b, n_c, _ = x.shape
    if n_c < 2:
        return x.new_tensor(0.0)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    sim = x @ x.transpose(1, 2)  # (B, N_c, N_c)
    diag_sq = sim.diagonal(dim1=1, dim2=2).pow(2).sum(dim=1)
    off_diag_sq = sim.pow(2).sum(dim=(1, 2)) - diag_sq
    return off_diag_sq.sum() / (b * n_c * (n_c - 1))
