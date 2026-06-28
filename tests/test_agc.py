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
    d = nn.Linear(2, 2)
    b.weight.grad = torch.ones_like(b.weight)
    out = diagnostics.apply_trainable_agc(
        b,
        f,
        d,
        enabled=False,
        lambda_bottleneck=0.2,
        lambda_coarse_flow=0.1,
        lambda_decoder=0.2,
        eps=1e-3,
    )
    assert out == {"agc_active": 0.0}
    assert b.weight.grad.norm().item() == pytest.approx(2.0, rel=1e-4)


def test_agc_skips_bias_norm_and_zero_init(monkeypatch):
    """AGC excludes 1-D params, named embeddings, and zero-init gate modules."""
    diagnostics = importlib.import_module("diagnostics")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_mat = nn.Linear(4, 4)  # weight (2-D) clipped; bias (1-D) skipped
            self.norm = nn.LayerNorm(4)  # 1-D scale/shift skipped
            self.queries = nn.Parameter(torch.ones(4, 4))  # named embedding skipped
            self.gate = nn.Linear(4, 4)  # zero-init gate skipped via marker
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            self.gate.is_zero_init = True

    m = Tiny()
    for p in m.parameters():
        p.grad = torch.full_like(p, 100.0)  # huge grads everywhere

    metrics = diagnostics.adaptive_gradient_clip(m, clip_factor=0.10, eps=1e-3)

    # Only weight_mat.weight is an eligible 2-D non-embedding non-gate tensor.
    assert metrics["agc_clipped_tensors"] == 1.0
    # Excluded tensors keep their (huge) gradient untouched: a (4,4) grad of 100 has
    # norm 100*sqrt(16)=400; a (4,) grad of 100 has norm 100*sqrt(4)=200.
    assert m.queries.grad.norm().item() == pytest.approx(400.0, rel=1e-4)
    assert m.gate.weight.grad.norm().item() == pytest.approx(400.0, rel=1e-4)
    assert m.norm.weight.grad.norm().item() == pytest.approx(200.0, rel=1e-4)


def test_partition_decay_params_groups():
    """partition_decay_params puts weights in decay, everything geometric in no-decay."""
    diagnostics = importlib.import_module("diagnostics")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)  # weight -> decay, bias -> no_decay
            self.norm = nn.LayerNorm(4)  # weight + bias -> no_decay (1-D)
            self.null_condition = nn.Parameter(torch.zeros(4, 4))  # name -> no_decay
            self.slot_pos = nn.Parameter(torch.zeros(4, 4))  # name -> no_decay
            self.gate = nn.Linear(4, 4)  # zero-init marker -> no_decay
            self.gate.is_zero_init = True

    m = Tiny()
    decay, no_decay = diagnostics.partition_decay_params(m)
    decay_ids = {id(p) for p in decay}
    assert id(m.lin.weight) in decay_ids
    # exactly one decayed tensor (lin.weight); everything else is held out
    assert len(decay) == 1
    no_decay_ids = {id(p) for p in no_decay}
    for p in (
        m.lin.bias,
        m.norm.weight,
        m.norm.bias,
        m.null_condition,
        m.slot_pos,
        m.gate.weight,
        m.gate.bias,
    ):
        assert id(p) in no_decay_ids
    # the partition is a clean cover: every param appears exactly once
    assert len(decay) + len(no_decay) == sum(1 for _ in m.parameters())
