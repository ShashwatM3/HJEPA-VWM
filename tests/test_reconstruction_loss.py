"""Unit tests for the reconstruction anchor objective."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def _losses():
    return importlib.import_module("losses")


def test_reconstruction_loss_is_mean_cosine_distance_per_tubelet():
    """The loss is mean(1 - cos) over tubelets, not variance-normalized MSE."""
    losses = _losses()
    pred = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]]])
    target = torch.tensor([[[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]])

    loss = losses.reconstruction_loss(pred, target)

    # cosines are [1, 0, -1] -> losses [0, 1, 2] -> mean 1.
    assert loss.item() == pytest.approx(1.0)


def test_reconstruction_loss_can_use_legacy_relative_mse():
    """The legacy mode exactly restores MSE divided by frozen-target variance."""
    losses = _losses()
    pred = torch.tensor([[[1.0, 0.0], [3.0, 4.0]]])
    target = torch.tensor([[[2.0, 2.0], [1.0, 5.0]]])

    loss = losses.reconstruction_loss(pred, target, mode="relative_mse")
    expected = (pred - target).pow(2).mean() / target.var(unbiased=False).clamp_min(1e-8)

    assert loss.item() == pytest.approx(expected.item())


def test_reconstruction_loss_is_invariant_to_tubelet_norms():
    """Scaling either side of a tubelet leaves the angular objective unchanged."""
    losses = _losses()
    pred = torch.tensor([[[1.0, 2.0, 3.0], [2.0, -1.0, 0.5]]])
    target = torch.tensor([[[3.0, 0.0, 1.0], [-1.0, 4.0, 2.0]]])

    baseline = losses.reconstruction_loss(pred, target)
    scaled = losses.reconstruction_loss(pred * 37.0, target * 0.125)

    assert scaled.item() == pytest.approx(baseline.item(), rel=1e-6, abs=1e-6)


def test_reconstruction_loss_detaches_target_and_backprops_to_prediction():
    """The frozen detailed target is stop-gradient, while e_hat remains trainable."""
    losses = _losses()
    pred = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn(2, 3, 4, requires_grad=True)

    loss = losses.reconstruction_loss(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
    assert target.grad is None


def test_reconstruction_loss_rejects_unknown_mode():
    """A misspelled CLI/config mode should fail loudly before an expensive run."""
    losses = _losses()
    pred = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)

    with pytest.raises(ValueError, match="Unknown reconstruction loss mode"):
        losses.reconstruction_loss(pred, target, mode="not_a_mode")
