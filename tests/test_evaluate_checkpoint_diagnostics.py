"""Tests for the offline paired checkpoint diagnostic entry point."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import evaluate_checkpoint_diagnostics as evaluate
from config import Config
from data import ClipBatch


class FakeEncoder(nn.Module):
    """Lightweight encoder recording evaluation and inference semantics."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.calls = 0
        self.grad_enabled: list[bool] = []
        self.spec = object()

    def forward(self, context):
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        assert not self.training
        return context.flatten(2).transpose(1, 2) * self.weight


class FakeBottleneck(nn.Module):
    """Lightweight bottleneck recording evaluation and inference semantics."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, detailed):
        self.calls += 1
        assert not self.training
        assert not torch.is_grad_enabled()
        return detailed[:, :2] + self.bias


class EmptyModule(nn.Module):
    """State-free stand-in for checkpoint modules unused by this diagnostic."""


def checkpoint_dict(cfg: Config) -> dict:
    """Return a minimal current-schema checkpoint carrying a serialized config."""
    return {
        "schema": "hjepa-phase1-checkpoint-v2",
        "next_step": 15_000,
        "config": {
            "model": vars(cfg.model),
            "encoder": vars(cfg.encoder),
            "train": vars(cfg.train),
            "data": vars(cfg.data),
            "debug_shapes": cfg.debug_shapes,
            "checkpoint_dir": cfg.checkpoint_dir,
            "hf_cache_dir": cfg.hf_cache_dir,
            "seed": cfg.seed,
        },
    }


def test_evaluate_once_uses_one_encoder_forward_for_both_metrics():
    encoder = FakeEncoder()
    bottleneck = FakeBottleneck()
    batch = ClipBatch(
        context=torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2),
        target=None,
        sample_ids=("validation/a_00000.mp4", "validation/b_00000.mp4"),
    )

    result = evaluate.evaluate_once(batch, encoder, bottleneck, torch.device("cpu"))

    assert encoder.calls == bottleneck.calls == 1
    assert encoder.grad_enabled == [False]
    assert "e_cross_video_cosine" in result
    assert "c_cross_video_cosine" in result


def test_checkpoint_evaluation_loads_live_modules_and_uses_fixed_batch(monkeypatch, tmp_path):
    cfg = Config()
    cfg.data.dataset = "ego4d"
    cfg.train.present_recon_only = True
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_dict(cfg), path)
    encoder = FakeEncoder()
    bottleneck = FakeBottleneck()
    modules = (encoder, bottleneck, EmptyModule(), EmptyModule(), EmptyModule())
    loaded: dict[str, object] = {}

    monkeypatch.setattr(evaluate, "_build_and_init", lambda cfg, device: modules)

    def fake_load(path, received, optimizer, **kwargs):
        loaded.update(path=path, modules=received, optimizer=optimizer, kwargs=kwargs)
        bottleneck.bias.data.fill_(3.0)
        return 15_000

    monkeypatch.setattr(evaluate, "load_checkpoint", fake_load)

    def fixed_batch(received_cfg, batch_size, *, needs_target):
        loaded.update(cfg=received_cfg, batch_size=batch_size, needs_target=needs_target)
        return ClipBatch(
            context=torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2),
            target=None,
            sample_ids=("validation/a_00000.mp4", "validation/b_00000.mp4"),
        )

    monkeypatch.setattr(evaluate, "build_fixed_diagnostic_batch", fixed_batch)
    report = evaluate.evaluate_checkpoint(path, requested_batch_size=2, device=torch.device("cpu"))

    assert loaded["modules"] is modules
    assert loaded["optimizer"] is None
    assert loaded["batch_size"] == 2
    assert loaded["needs_target"] is False
    assert encoder.calls == bottleneck.calls == 2
    assert report["weights"] == "live_online_bottleneck"
    assert report["repeat_consistent"] is True


def test_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        evaluate.evaluate_checkpoint(tmp_path / "missing.pt", device=torch.device("cpu"))


def test_mismatched_checkpoint_architecture_fails_without_partial_load(monkeypatch, tmp_path):
    cfg = Config()
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_dict(cfg), path)
    encoder = FakeEncoder()
    bottleneck = FakeBottleneck()
    modules = (encoder, bottleneck, EmptyModule(), EmptyModule(), EmptyModule())
    monkeypatch.setattr(evaluate, "_build_and_init", lambda cfg, device: modules)
    monkeypatch.setattr(
        evaluate,
        "load_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Checkpoint bottleneck tensor shapes do not match this architecture.")
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "build_fixed_diagnostic_batch",
        lambda *args, **kwargs: pytest.fail("batch must not load after architecture failure"),
    )

    with pytest.raises(RuntimeError, match="tensor shapes do not match"):
        evaluate.evaluate_checkpoint(path, device=torch.device("cpu"))

    assert bottleneck.bias.item() == 0.0
