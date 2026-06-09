"""SSv2 dataset and dataloader for HJEPA-VWM Phase 1 (v0.2).

Each item is a `(context_clip, target_clip)` pair: an 8-frame context window ending
at `t` and an 8-frame future window ending at `t+k` (`k = cfg.train.horizon_k`), both
at stride 2, at 256x256, normalized with the frozen encoder's expected statistics
(NOT [-1, 1]). The same geometric crop and color jitter are applied to both windows.
There is no tubelet dropout in v0.2 — the dataloader returns full clips and the
frozen encoder consumes all tokens.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np

try:  # Optional on local machines; required on RunPod.
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - lets docs/tests import without torch installed.
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    Dataset = object  # type: ignore[misc,assignment]
    DataLoader = object  # type: ignore[misc,assignment]
    F = None  # type: ignore[assignment]

from config import ENCODER_IMAGE_MEAN, ENCODER_IMAGE_STD, Config


def _require_torch() -> None:
    """Fail fast when dataloading is used without PyTorch installed."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for data loading. Install requirements.txt on RunPod."
        )


def _read_video_decord(path: Path) -> np.ndarray:
    """Decode all frames from one SSv2 .webm using decord on CPU.

    `num_threads=1` is required: SSv2 ships as VP9-encoded .webm and decord's
    threaded FFmpeg decoder fails with EAGAIN (-11, "Error sending packet") on
    some VP9 packets when threads > 1. See dmlc/decord#83, #145, #246. Each
    DataLoader worker still parallelises across videos.
    """
    try:
        from decord import VideoReader, cpu
    except ModuleNotFoundError as exc:  # pragma: no cover - RunPod dependency.
        raise RuntimeError("decord is required to read SSv2 videos") from exc
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    return reader.get_batch(list(range(len(reader)))).asnumpy()


def _resize_shorter_side(frames: Tensor, size: int) -> Tensor:
    """Resize frames so the shorter side equals `size`.

    Args:
        frames: (N, C, H, W) float tensor in [0, 1].
        size: Target shorter-side length.
    Returns:
        resized: (N, C, H', W') tensor where min(H', W') == size.
    """
    _require_torch()
    _, _, h, w = frames.shape
    scale = size / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    return F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _crop(frames: Tensor, size: int, split: Literal["train", "validation"]) -> Tensor:
    """Crop a resized clip without flips or rotations.

    Args:
        frames: (N, C, H, W) resized clip (context+target stacked share one crop).
        size: Spatial crop size.
        split: Training uses random crop; validation uses center crop.
    Returns:
        cropped: (N, C, size, size) tensor.
    """
    _, _, h, w = frames.shape
    if split == "train":
        top = random.randint(0, max(0, h - size))
        left = random.randint(0, max(0, w - size))
    else:
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
    return frames[:, :, top : top + size, left : left + size]


def _color_jitter(frames: Tensor) -> Tensor:
    """Apply brightness/contrast/saturation jitter without hue changes.

    One sampled set of params is applied to the whole stack so context and target
    windows stay photometrically consistent.

    Args:
        frames: (N, C, H, W) float tensor in [0, 1].
    Returns:
        jittered: (N, C, H, W) clipped to [0, 1].
    """
    brightness = 1.0 + random.uniform(-0.4, 0.4)
    contrast = 1.0 + random.uniform(-0.4, 0.4)
    saturation = 1.0 + random.uniform(-0.4, 0.4)
    out = frames * brightness
    mean = out.mean(dim=(-2, -1), keepdim=True)
    out = (out - mean) * contrast + mean
    gray = out.mean(dim=1, keepdim=True)
    out = (out - gray) * saturation + gray
    return out.clamp(0.0, 1.0)


def _normalize_encoder(frames: Tensor) -> Tensor:
    """Normalize [0, 1] frames with the frozen encoder's ImageNet statistics.

    Args:
        frames: (N, C=3, H, W) float tensor in [0, 1].
    Returns:
        normalized: (N, 3, H, W) tensor in the encoder's expected range (not [-1, 1]).
    """
    mean = torch.tensor(ENCODER_IMAGE_MEAN, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor(ENCODER_IMAGE_STD, dtype=frames.dtype).view(1, 3, 1, 1)
    return (frames - mean) / std


class SSV2Dataset(Dataset):
    """Load context/future clip pairs from SSv2 symlinks (v0.2).

    Args:
        root: Dataset root with `train/` and `validation/` symlink directories.
        split: `train` or `validation`.
        cfg: Global config with frame size, stride, and horizon constants.
    Returns:
        Each item is `(context_clip, target_clip)`, both
        (T=8, C=3, H=256, W=256) encoder-normalized float32 tensors. The context
        window ends at `t`; the target window ends at `t + horizon_k`.
    """

    def __init__(self, root: str | Path, split: Literal["train", "validation"], cfg: Config):
        """Index a split directory of SSv2 .webm symlinks."""
        _require_torch()
        self.root = Path(root)
        self.split = split
        self.cfg = cfg
        self.paths = sorted((self.root / split).glob("*.webm"))
        if not self.paths:
            raise FileNotFoundError(f"No .webm files found in {self.root / split}")

    def __len__(self) -> int:
        """Return the number of videos available in this split."""
        return len(self.paths)

    def _window_indices(self, num_frames: int) -> tuple[list[int], list[int]]:
        """Compute context and target frame indices within one video.

        The context window is `T` frames at `stride` ending at `t`; the target window
        is `T` frames at `stride` ending at `t + k`. Indices are clamped to the last
        frame for too-short videos (deterministic pad-by-repeat).

        Args:
            num_frames: Number of decoded frames in the video.
        Returns:
            (context_indices, target_indices), each of length `T`.
        """
        t_ctx = self.cfg.model.t_ctx
        stride = self.cfg.train.frame_stride
        k = self.cfg.train.horizon_k
        span = (t_ctx - 1) * stride + k  # first context frame -> last target frame
        max_start = num_frames - 1 - span
        if max_start < 0:
            start = 0
        elif self.split == "train":
            start = random.randint(0, max_start)
        else:
            start = max_start // 2
        last = num_frames - 1
        context = [min(last, start + i * stride) for i in range(t_ctx)]
        tgt_end = start + (t_ctx - 1) * stride + k
        target = [min(last, tgt_end - (t_ctx - 1 - i) * stride) for i in range(t_ctx)]
        return context, target

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return one encoder-normalized (context_clip, target_clip) pair."""
        frames_np = _read_video_decord(self.paths[index])
        context_idx, target_idx = self._window_indices(len(frames_np))
        t_ctx = self.cfg.model.t_ctx
        stacked = torch.from_numpy(frames_np[context_idx + target_idx])
        stacked = stacked.float().permute(0, 3, 1, 2) / 255.0
        stacked = _resize_shorter_side(stacked, self.cfg.model.h)
        stacked = _crop(stacked, self.cfg.model.h, self.split)
        if self.split == "train":
            stacked = _color_jitter(stacked)
        stacked = _normalize_encoder(stacked)
        return stacked[:t_ctx], stacked[t_ctx:]


def build_dataloader(
    cfg: Config,
    split: Literal["train", "validation"] = "train",
    batch_size: int | None = None,
) -> DataLoader:
    """Build a Phase 1 dataloader.

    Args:
        cfg: Global config with selected data root and worker settings.
        split: Dataset split to read.
        batch_size: Optional override for smoke tests.
    Returns:
        DataLoader yielding `(context_clip, target_clip)` batches, both
        (B, 8, 3, 256, 256).
    """
    _require_torch()
    dataset = SSV2Dataset(cfg.data.dataset_root(), split, cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.train.global_batch,
        shuffle=(split == "train"),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=(split == "train"),
    )


def smoke_test_dataloader() -> None:
    """Load two SSv2-tiny batches and assert Phase 1 v0.2 data contracts."""
    cfg = Config()
    loader = build_dataloader(cfg, "train", batch_size=2)
    expected = (cfg.model.t_ctx, 3, cfg.model.h, cfg.model.w)
    for i, (context_clip, target_clip) in enumerate(loader):
        print(
            context_clip.shape,
            target_clip.shape,
            float(context_clip.min()),
            float(context_clip.max()),
        )
        assert context_clip.shape[1:] == expected, context_clip.shape
        assert target_clip.shape[1:] == expected, target_clip.shape
        assert torch.isfinite(context_clip).all() and torch.isfinite(target_clip).all()
        # Encoder-normalized (ImageNet stats), so the range is NOT [-1, 1].
        assert context_clip.min() < -1.0 or context_clip.max() > 1.0
        if i == 1:
            break
