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

import math

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


def residual_target(
    target_future: Tensor, target_present: Tensor, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """Temporal residual Δ = c_{t+k} - c_t and a noise-scale scalar (investigation_009).

    For residual flow matching, F_c predicts the temporal *change* Δ rather than the full
    future latent. Both inputs are EMA / stop-gradient bottleneck outputs (B_EMA on the
    future clip and on the present clip), so Δ is a PURELY temporal difference, not
    contaminated by the online-vs-EMA gap. The returned σ scales the flow noise so the
    rectified-flow velocity target (Δ - eps) is not dominated by unit-scale noise
    (‖Δ‖/‖c‖ ≈ 0.38): scaling eps to σ keeps the signal at ~50% of the target variance and
    leaves coarse_vs_copy_ratio identical to (and at the same scale as) the full-latent run.

    Args:
        target_future: (B, N_c, D_c) detached future latent c_{t+k} (B_EMA on e_{t+k}).
        target_present: (B, N_c, D_c) detached present latent c_t (B_EMA on e_t).
        eps: Numerical floor for the std scalar.
    Returns:
        (delta, sigma): the residual target (detached) and a 0-dim std for noise scaling.
    """
    _require_torch()
    delta = as_target(target_future) - as_target(target_present)
    sigma = delta.float().std().clamp_min(eps)
    return delta, sigma


def flow_matching_loss(u_hat: Tensor, u_target: Tensor) -> Tensor:
    """Mean-squared error for coarse, fine, and frame flow matching.

    Args:
        u_hat: (B, N, D) predicted velocity.
        u_target: (B, N, D) target velocity.
    Returns:
        Scalar loss averaged over all batch and token dimensions.
    """
    return (u_hat - u_target).pow(2).mean()


def reconstruction_loss(
    pred_detailed: Tensor,
    target_detailed: Tensor,
    mode: str = "cosine",
    eps: float = 1e-6,
) -> Tensor:
    """Reconstruction anchor loss for frozen detailed features.

    The reconstruction anchor: a small decoder `D` maps the abstract latent back to
    the frozen-encoder detailed features, forcing the bottleneck to keep `c`
    information-rich — directly attacking the ~13 effective-rank ceiling and the
    identical-`c` representational collapse the variance floor alone cannot stop.

    `mode="cosine"` is the current objective: each detailed token is L2-normalized
    along `D_e` before comparison, then scored as `mean(1 - cos(e_hat, e))`.
    Equivalently, this is one half of the squared distance between unit-normalized
    token vectors. There is intentionally NO `Var(e)` denominator: the objective removes
    magnitude as an escape route and leaves only angular alignment with the frozen
    target features.

    `mode="relative_mse"` is the legacy objective used in investigations 006-010:
    raw MSE divided by the frozen target variance. It is kept as an explicit
    ablation switch, not as the default.

    The target is the FROZEN encoder output, so it is detached via `as_target`: the
    anchor pins `c` to real per-video content but never lets reconstruction rewrite
    the target. In option 1 this scores the present (`D(c_t)` vs `e_t`) with the
    gradient flowing into `D` and `B` only; the same function scores the future
    readouts (`c_plus`, `c_hat`) under `no_grad` at diagnostic cadence.

    Args:
        pred_detailed: (B, N, D_e) decoder output `e_hat`.
        target_detailed: (B, N, D_e) frozen encoder features (context or future clip).
        mode: `"cosine"` for per-token cosine distance, or `"relative_mse"` for
            the legacy variance-normalized MSE.
        eps: Numerical floor for per-token L2 normalization.
    Returns:
        Scalar reconstruction loss averaged over batch and detailed tokens.
    """
    _require_torch()
    target = as_target(target_detailed)
    if mode == "cosine":
        pred_unit = torch.nn.functional.normalize(pred_detailed.float(), p=2.0, dim=-1, eps=eps)
        target_unit = torch.nn.functional.normalize(target.float(), p=2.0, dim=-1, eps=eps)
        return (1.0 - (pred_unit * target_unit).sum(dim=-1)).mean()
    if mode == "relative_mse":
        mse = (pred_detailed - target).pow(2).mean()
        denom = target.float().var(unbiased=False).clamp_min(1e-8)
        return mse / denom
    raise ValueError(f"Unknown reconstruction loss mode: {mode!r}")


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


def sigreg_loss(
    abstract: Tensor,
    n_projections: int = 128,
    max_rows: int = 512,
    beta: float = 1.0,
    eps: float = 1e-6,
    generator=None,
) -> Tensor:
    """SIGReg: push the pooled `c_t` distribution toward an isotropic unit Gaussian.

    The principled replacement for the variance floor's *active* anti-collapse role
    (LeJEPA, Balestriero & LeCun 2025). The risk-optimal embedding law is `N(0, I)`;
    isotropy makes every covariance eigenvalue equal, so it maximizes effective rank
    by construction — a direct attack on the `c_effective_rank ≈ 13/256` utilization
    ceiling that the one-sided variance floor (which only forbids *constant* dims)
    cannot move. See KANBAN investigation_008.

    A distribution is `N(0, I)` iff every 1-D projection is `N(0,1)` (Cramér–Wold), so
    we sketch many RANDOM unit directions of the `D_c` feature space, project, and
    penalize each projection's deviation from a standard normal with the closed-form
    BHEP / Epps–Pulley normality statistic. Testing against `N(0,1)` (not a rescaled
    normal) enforces unit variance + Gaussian shape; random directions enforce
    isotropy — jointly, in one term.

    Operates on the SAME pooled object `diagnostics.effective_rank` measures
    (`abstract` reshaped to `(N = B·N_c, D_c)`), so the loss and its target metric are
    the same distribution. Stochastic each step: rows are subsampled to `max_rows` and
    directions resampled, keeping the `O(P·S²)` pairwise term cheap.

    `T = mean_jk exp(-β²(y_j-y_k)²/2) - 2/√(1+β²)·mean_j exp(-β² y_j²/(2(1+β²))) + 1/√(1+2β²)`

    Args:
        abstract: (B, N_c, D_c) online abstract latent `c_t` (NOT the EMA target).
        n_projections: Number of random 1-D directions (P) sketched per call.
        max_rows: Cap on pooled rows used for the O(N²) pairwise term (subsampled).
        beta: BHEP smoothing bandwidth.
        eps: Numerical floor for direction normalization.
        generator: Optional `torch.Generator` for reproducible sketching.
    Returns:
        Scalar SIGReg loss (>= 0; 0 == pooled `c_t` is standard-normal isotropic).
    """
    _require_torch()
    z = abstract.reshape(-1, abstract.shape[-1]).float()  # (N, D_c)
    n, d = z.shape
    if n < 2:
        return z.new_tensor(0.0)
    # Compute the statistic in FP32 regardless of any outer autocast (WALK_FIXES F2):
    # the BHEP matmul/exp are precision-sensitive, and in inv008 (λ_sigreg>0) this
    # gradient trains B — bf16 here would silently degrade the regularizer.
    with torch.autocast(device_type=z.device.type, enabled=False):
        if n > max_rows:  # stochastic row subsample keeps the pairwise term cheap
            idx = torch.randperm(n, device=z.device, generator=generator)[:max_rows]
            z = z[idx]
            n = max_rows
        z = z - z.mean(dim=0, keepdim=True)  # center: the test reference is N(0, 1)
        v = torch.randn(d, n_projections, device=z.device, dtype=z.dtype, generator=generator)
        v = v / v.norm(dim=0, keepdim=True).clamp_min(eps)  # random UNIT directions
        y = (z @ v).t()  # (P, N) projected samples, one row per direction
        b2 = beta * beta
        sq = y * y  # (P, N)
        # (y_j - y_k)² = y_j² + y_k² - 2 y_j y_k, broadcast to (P, N, N)
        pair = sq.unsqueeze(2) + sq.unsqueeze(1) - 2.0 * y.unsqueeze(2) * y.unsqueeze(1)
        term1 = torch.exp(-0.5 * b2 * pair).mean(dim=(1, 2))  # (P,)
        term2 = (2.0 / math.sqrt(1.0 + b2)) * torch.exp(-0.5 * b2 / (1.0 + b2) * sq).mean(
            dim=1
        )  # (P,)
        term3 = 1.0 / math.sqrt(1.0 + 2.0 * b2)
        return (term1 - term2 + term3).clamp_min(0.0).mean()  # avg over directions


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
    average token. This loss directly penalizes that failure.

    IMPORTANT — the slots are CENTERED across the slot dimension before the
    cosine matrix is formed, matching the centered
    `c_slot_diversity_rank_centered` diagnostic. Without centering, the 32 slots
    share a large common-mean component (they all ≈ the same near-uniform-
    attention average token), the raw cosines pin at ~1.0, and the gradient
    mostly fights that shared mean — which the final LayerNorm / variance floor
    immediately restores. Centering removes the shared mean so the penalty
    operates on the RESIDUAL slot directions, i.e. the redundancy we actually
    want to break.

    Per video: subtract the across-slot mean, L2-normalize the `N_c` residual
    vectors, form the `N_c×N_c` cosine-similarity matrix, and average the squared
    OFF-diagonal entries. Identical residual directions give ~1.0; mutually
    orthogonal residuals give 0.0.

    Args:
        abstract: (B, N_c, D_c) online abstract latent `c_t`.
        eps: Numerical floor for slot-vector normalization.
    Returns:
        Scalar slot-diversity loss (>= 0; 0 == decorrelated slot residuals).
    """
    _require_torch()
    x = abstract.float()
    b, n_c, _ = x.shape
    if n_c < 2:
        return x.new_tensor(0.0)
    x = x - x.mean(dim=1, keepdim=True)  # match c_slot_diversity_rank_centered
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    sim = x @ x.transpose(1, 2)  # (B, N_c, N_c)
    diag_sq = sim.diagonal(dim1=1, dim2=2).pow(2).sum(dim=1)
    off_diag_sq = sim.pow(2).sum(dim=(1, 2)) - diag_sq
    return off_diag_sq.sum() / (b * n_c * (n_c - 1))
