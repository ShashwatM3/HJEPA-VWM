"""Unit tests for the SIGReg isotropic-Gaussian regularizer (investigation_008)."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def _losses():
    return importlib.import_module("losses")


def test_sigreg_near_zero_on_isotropic_gaussian():
    """A standard-normal isotropic latent is the target -> loss is small."""
    losses = _losses()
    g = torch.Generator().manual_seed(0)
    abstract = torch.randn(256, 8, 16, generator=g)  # pooled N=2048, D_c=16, ~N(0,I)

    val = float(losses.sigreg_loss(abstract, generator=torch.Generator().manual_seed(1)))

    assert val >= 0.0
    assert val < 0.05, f"SIGReg should be ~0 on N(0,I); got {val}"


def test_sigreg_penalizes_anisotropic_low_rank_more_than_isotropic():
    """A collapsed / anisotropic latent (the rank-13 failure) scores strictly higher."""
    losses = _losses()
    g = torch.Generator().manual_seed(2)
    iso = torch.randn(256, 8, 16, generator=g)

    # Low-rank: only 2 of 16 dims carry variance -> far from isotropic N(0, I).
    low_rank = torch.zeros(256, 8, 16)
    low_rank[..., :2] = torch.randn(256, 8, 2, generator=g) * 3.0

    # Same fixed sketch for both so the comparison isolates the distribution.
    iso_val = float(losses.sigreg_loss(iso, generator=torch.Generator().manual_seed(7)))
    low_val = float(losses.sigreg_loss(low_rank, generator=torch.Generator().manual_seed(7)))

    assert low_val > iso_val + 0.05, f"low-rank {low_val} not >> isotropic {iso_val}"


def test_sigreg_gradient_is_finite_and_flows():
    """The loss is differentiable w.r.t. the latent and produces a finite gradient."""
    losses = _losses()
    abstract = torch.randn(128, 8, 16, requires_grad=True)

    loss = losses.sigreg_loss(abstract, generator=torch.Generator().manual_seed(3))
    loss.backward()

    assert abstract.grad is not None
    assert torch.isfinite(abstract.grad).all()
    assert abstract.grad.abs().sum() > 0.0


def test_sigreg_returns_zero_for_degenerate_batch():
    """Fewer than 2 pooled rows -> 0 (matches the variance-floor guard)."""
    losses = _losses()
    abstract = torch.randn(1, 1, 16)  # pooled N=1

    assert float(losses.sigreg_loss(abstract)) == 0.0


def test_sigreg_is_reproducible_with_a_generator():
    """A fixed generator makes the stochastic sketch deterministic."""
    losses = _losses()
    abstract = torch.randn(256, 8, 16, generator=torch.Generator().manual_seed(4))

    a = float(losses.sigreg_loss(abstract, generator=torch.Generator().manual_seed(5)))
    b = float(losses.sigreg_loss(abstract, generator=torch.Generator().manual_seed(5)))

    assert a == b
