"""Unit tests for adaptive gradient clipping (AGC)."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


def test_adaptive_gradient_clip_scales_oversized_grad():
    diagnostics = importlib.import_module("diagnostics")
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    model.weight.grad = torch.full_like(model.weight, 10.0)

    metrics = diagnostics.adaptive_gradient_clip(model, clip_factor=0.10, eps=1e-3)

    assert metrics["agc_any_clipped"] == 1.0
    assert metrics["agc_clipped_tensors"] == 1.0
    grad_norm = model.weight.grad.norm().item()
    bound = 0.10 * (model.weight.norm().item() + 1e-3)
    assert grad_norm == pytest.approx(bound, rel=1e-4)


def test_adaptive_gradient_clip_leaves_small_grad():
    diagnostics = importlib.import_module("diagnostics")
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    model.weight.grad = torch.full_like(model.weight, 0.01)

    metrics = diagnostics.adaptive_gradient_clip(model, clip_factor=0.10, eps=1e-3)

    assert metrics["agc_any_clipped"] == 0.0
    assert model.weight.grad.norm().item() == pytest.approx(0.01 * (8**0.5), rel=1e-4)


def test_apply_trainable_agc_disabled():
    diagnostics = importlib.import_module("diagnostics")
    b = nn.Linear(2, 2)
    f = nn.Linear(2, 2)
    b.weight.grad = torch.ones_like(b.weight)
    out = diagnostics.apply_trainable_agc(
        b, f, enabled=False, lambda_bottleneck=0.2, lambda_coarse_flow=0.1, eps=1e-3
    )
    assert out == {"agc_active": 0.0}
    assert b.weight.grad.norm().item() == pytest.approx(2.0, rel=1e-4)
