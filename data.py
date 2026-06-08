"""SSv2 dataset and dataloader for HJEPA-VWM Phase 1."""

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

from config import Config


def _require_torch() -> None:
    """Fail fast when dataloading is used without PyTorch installed."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for data loading. Install requirements.txt on RunPod."
        )


def _read_video_decord(path: Path) -> np.ndarray:
    """Decode all frames from one SSv2 .webm using decord on CPU."""
    try:
        from decord import VideoReader, cpu
    except ModuleNotFoundError as exc:  # pragma: no cover - RunPod dependency.
        raise RuntimeError("decord is required to read SSv2 videos") from exc
    reader = VideoReader(str(path), ctx=cpu(0))
    return reader.get_batch(list(range(len(reader)))).asnumpy()


def _resize_shorter_side(frames: Tensor, size: int) -> Tensor:
    """Resize frames so the shorter side equals `size`.

    Args:
        frames: (T, C, H, W) float tensor in [0, 1].
    Returns:
        resized: (T, C, H', W') tensor where min(H', W') == size.
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
        frames: (T, C, H, W) resized clip.
        size: Spatial crop size.
        split: Training uses random crop; validation uses center crop.
    Returns:
        cropped: (T, C, size, size) tensor.
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

    Args:
        frames: (T, C, H, W) float tensor in [0, 1].
    Returns:
        jittered: (T, C, H, W) clipped to [0, 1].
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


class SSV2Dataset(Dataset):
    """Load 4-frame context clips and one future frame from SSv2 symlinks.

    The dataloader returns complete clips. Tubelet dropout happens in the model
    patchifier path, never in `data.py`.

    Args:
        root: Dataset root with `train/` and `validation/` symlink directories.
        split: `train` or `validation`.
        cfg: Global config with frame size and stride constants.
    Returns:
        Each item is `(context_clip, future_frame)` where context is
        (T=4, C=3, H=128, W=128) and future is (C=3, H=128, W=128), both in [-1, 1].
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

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return one normalized context/future sample from a video."""
        frames_np = _read_video_decord(self.paths[index])
        total_needed = self.cfg.model.t_ctx + 1
        stride = self.cfg.train.frame_stride
        max_start = max(0, len(frames_np) - 1 - (total_needed - 1) * stride)
        start = random.randint(0, max_start) if self.split == "train" else max_start // 2
        indices = [min(len(frames_np) - 1, start + i * stride) for i in range(total_needed)]
        frames = torch.from_numpy(frames_np[indices]).float().permute(0, 3, 1, 2) / 255.0
        frames = _resize_shorter_side(frames, self.cfg.model.h)
        frames = _crop(frames, self.cfg.model.h, self.split)
        if self.split == "train":
            frames = _color_jitter(frames)
        frames = frames * 2.0 - 1.0
        context_clip = frames[: self.cfg.model.t_ctx]
        future_frame = frames[self.cfg.model.t_ctx]
        return context_clip, future_frame


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
        PyTorch DataLoader yielding `(context_clip, future_frame)` batches with shapes
        (B, 4, 3, 128, 128) and (B, 3, 128, 128).
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
    """Load two SSv2-tiny batches and assert Phase 1 data contracts."""
    cfg = Config()
    loader = build_dataloader(cfg, "train", batch_size=2)
    for i, (context_clip, future_frame) in enumerate(loader):
        print(
            context_clip.shape,
            future_frame.shape,
            context_clip.min().item(),
            context_clip.max().item(),
        )
        assert context_clip.shape[1:] == (4, 3, 128, 128)
        assert future_frame.shape[1:] == (3, 128, 128)
        assert torch.isfinite(context_clip).all()
        assert torch.isfinite(future_frame).all()
        assert context_clip.min() >= -1.2 and context_clip.max() <= 1.2
        if i == 1:
            break
