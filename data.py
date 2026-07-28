"""Canonical raw-video dataset and dataloader for HJEPA-VWM Phase 1.

Each item is a `(context_clip, target_clip)` pair: an 8-frame context window ending
at `t` and an 8-frame future window ending at `t+k` (`k = cfg.train.horizon_k`), both
at stride 2, at 256x256, in canonical raw `[0,1]` RGB. The same geometric crop and
color jitter are applied to both windows. Encoder-specific normalization belongs only
to :mod:`encoders`.
There is no detailed-token dropout in v0.2 — the dataloader returns full clips and
the selected frozen encoder consumes every frame.
"""

from __future__ import annotations

import hashlib
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
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

TRANSFORM_VERSION = "raw-rgb-resize-crop-jitter-v2"


@dataclass(frozen=True)
class ClipSample:
    """One deterministic dataset item before collation."""

    context: Tensor
    target: Tensor | None
    sample_id: str


@dataclass(frozen=True)
class ClipBatch:
    """One typed training batch with an optional future branch."""

    context: Tensor
    target: Tensor | None
    sample_ids: tuple[str, ...]


def source_video_id(sample_id: str, dataset_name: str) -> str:
    """Return the source-video identity represented by one dataset sample.

    EGO4D samples are deterministic chunks named `<video_uid>_<index:05d>.mp4`;
    SSv2 files are already one sample per source video.

    Args:
        sample_id: Dataset-relative path such as `validation/<clip name>`.
        dataset_name: Configured dataset alias.
    Returns:
        Stable source-video identity for diagnostic diversity checks.
    """
    if dataset_name in {"ssv2", "ssv2_tiny"}:
        return sample_id
    if dataset_name not in {"ego4d", "ego4d_tiny"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    path = Path(sample_id)
    source_uid, separator, chunk_index = path.stem.rpartition("_")
    if (
        path.suffix != ".mp4"
        or not separator
        or not source_uid
        or len(chunk_index) < 5
        or not chunk_index.isdecimal()
    ):
        raise ValueError(
            "EGO4D sample IDs must end in '<video_uid>_<index:05d>.mp4'; " f"got {sample_id!r}."
        )
    return source_uid


def select_unique_source_indices(
    sample_ids: Sequence[str],
    dataset_name: str,
    requested_size: int,
) -> list[int]:
    """Select the first deterministic sample from each distinct source video.

    Args:
        sample_ids: Samples in the existing deterministic validation order.
        dataset_name: Configured dataset alias.
        requested_size: Maximum number of distinct source videos to select.
    Returns:
        Dataset indices in their original order, with unique source identities.
    """
    if requested_size <= 0:
        raise ValueError("requested_size must be positive.")
    selected: list[int] = []
    seen: set[str] = set()
    for index, sample_id in enumerate(sample_ids):
        source_id = source_video_id(sample_id, dataset_name)
        if source_id in seen:
            continue
        seen.add(source_id)
        selected.append(index)
        if len(selected) == requested_size:
            break
    if len(selected) < requested_size:
        warnings.warn(
            "Fixed diagnostics requested "
            f"{requested_size} distinct source videos but only found {len(selected)}; "
            "using every available unique source without duplicate refill.",
            RuntimeWarning,
            stacklevel=2,
        )
    if len(selected) < 2:
        raise RuntimeError(
            "Fixed diagnostics require at least two distinct source videos; "
            f"found {len(selected)}."
        )
    return selected


def _require_torch() -> None:
    """Fail fast when dataloading is used without PyTorch installed."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for data loading. Install requirements.txt on RunPod."
        )


def _open_video_reader(path: Path):
    """Open a video file (SSv2 .webm or EGO4D .mp4) with decord at num_threads=1.

    Returning the open reader (rather than decoded frames) lets the caller
    read `len(reader)` cheaply and then decode only the indices it needs,
    instead of paying to decode every frame in the clip.

    `num_threads=1` is required for SSv2: it ships as VP9-encoded .webm and decord's
    threaded FFmpeg decoder fails with EAGAIN (-11, "Error sending packet")
    on some VP9 packets when threads > 1. See dmlc/decord#83, #145, #246.
    H.264 EGO4D chunks decode fine (and fast) single-threaded too, so the setting
    is shared. Each DataLoader worker still parallelises across videos.

    Args:
        path: Filesystem path to a single video clip.
    Returns:
        reader: an open decord `VideoReader` whose `len` is the clip's frame
            count and whose `get_batch(indices)` decodes only those indices.
    """
    try:
        from decord import VideoReader, cpu
    except ModuleNotFoundError as exc:  # pragma: no cover - RunPod dependency.
        raise RuntimeError("decord is required to read SSv2 videos") from exc
    return VideoReader(str(path), ctx=cpu(0), num_threads=1)


def _decode_frames(reader, indices: list[int]) -> np.ndarray:
    """Decode the requested frame indices from an open decord VideoReader.

    Random-access via decord `get_batch` so a 16-frame consumer does not pay
    for decoding the ~30-90 unused frames in each SSv2 clip.

    Args:
        reader: an open decord VideoReader (see `_open_video_reader`).
        indices: 1-D list of frame indices to decode, in the order returned.
    Returns:
        frames: (len(indices), H, W, 3) uint8 numpy array in the same order
            as `indices`.
    """
    return reader.get_batch(indices).asnumpy()


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


def _crop(
    frames: Tensor,
    size: int,
    split: Literal["train", "validation"],
    rng: random.Random | None = None,
) -> Tensor:
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
        source = rng or random
        top = source.randint(0, max(0, h - size))
        left = source.randint(0, max(0, w - size))
    else:
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
    return frames[:, :, top : top + size, left : left + size]


def _color_jitter(frames: Tensor, rng: random.Random | None = None) -> Tensor:
    """Apply brightness/contrast/saturation jitter without hue changes.

    One sampled set of params is applied to the whole stack so context and target
    windows stay photometrically consistent.

    Args:
        frames: (N, C, H, W) float tensor in [0, 1].
    Returns:
        jittered: (N, C, H, W) clipped to [0, 1].
    """
    source = rng or random
    brightness = 1.0 + source.uniform(-0.4, 0.4)
    contrast = 1.0 + source.uniform(-0.4, 0.4)
    saturation = 1.0 + source.uniform(-0.4, 0.4)
    out = frames * brightness
    mean = out.mean(dim=(-2, -1), keepdim=True)
    out = (out - mean) * contrast + mean
    gray = out.mean(dim=1, keepdim=True)
    out = (out - gray) * saturation + gray
    return out.clamp(0.0, 1.0)


class SSV2Dataset(Dataset):
    """Load context/future clip pairs from a clip-per-file video directory (v0.2).

    Named for the original SSv2 layout; the same contract serves any dataset that
    mirrors it (e.g. EGO4D 4-second chunks): a root with `train/` and `validation/`
    directories of short `.webm`/`.mp4` videos, one sample window pair per video.

    Args:
        root: Dataset root with `train/` and `validation/` video directories.
        split: `train` or `validation`.
        cfg: Global config with frame size, stride, and horizon constants.
    Returns:
        Each item is a :class:`ClipSample`. Context and optional target are
        `(T=8,C=3,H=256,W=256)` raw `[0,1]` float32 tensors.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "validation"],
        cfg: Config,
        *,
        needs_target: bool = True,
        epoch: int = 0,
    ) -> None:
        """Index a split directory of video files (SSv2 .webm symlinks or EGO4D .mp4 chunks)."""
        _require_torch()
        self.root = Path(root)
        self.split = split
        self.cfg = cfg
        self.needs_target = needs_target
        self.epoch = epoch
        if (cfg.encoder.input_height, cfg.encoder.input_width) != (256, 256):
            raise ValueError("The v1 raw-video transform supports only a square 256 crop.")
        split_dir = self.root / split
        # One combined sort over both extensions keeps indexing deterministic across
        # datasets: SSv2 ships VP9 .webm symlinks, EGO4D chunks are H.264 .mp4 files.
        self.paths = sorted([*split_dir.glob("*.webm"), *split_dir.glob("*.mp4")])
        if not self.paths:
            raise FileNotFoundError(f"No .webm or .mp4 files found in {split_dir}")

    def __len__(self) -> int:
        """Return the number of videos available in this split."""
        return len(self.paths)

    def sample_id_at(self, index: int) -> str:
        """Return one dataset-relative sample ID without decoding its video."""
        return str(self.paths[index].relative_to(self.root))

    def _rng_for(self, sample_id: str) -> random.Random:
        """Return the identity/epoch-scoped transform RNG for one clip."""
        identity = f"{self.cfg.seed}:{self.epoch}:{sample_id}:{TRANSFORM_VERSION}".encode()
        seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
        return random.Random(seed)

    def _window_indices(
        self, num_frames: int, rng: random.Random | None = None
    ) -> tuple[list[int], list[int]]:
        """Compute context and target frame indices within one video.

        The context window is `T` frames at `stride` ending at `t`; the target window
        is `T` frames at `stride` ending at `t + k`. Indices are clamped to the last
        frame for too-short videos (deterministic pad-by-repeat).

        Args:
            num_frames: Number of decoded frames in the video.
        Returns:
            (context_indices, target_indices), each of length `T`.
        """
        context, start = self._context_indices(num_frames, rng)
        t_ctx = self.cfg.encoder.input_frames
        stride = self.cfg.train.frame_stride
        k = self.cfg.train.horizon_k
        last = num_frames - 1
        tgt_end = start + (t_ctx - 1) * stride + k
        target = [min(last, tgt_end - (t_ctx - 1 - i) * stride) for i in range(t_ctx)]
        return context, target

    def _context_indices(
        self, num_frames: int, rng: random.Random | None = None
    ) -> tuple[list[int], int]:
        """Select context indices without constructing any future-index list.

        The admissible start range still reserves the configured future horizon,
        so context-only and full modes select identical contexts for a sample.
        """
        if num_frames <= 0:
            raise ValueError("A video must expose at least one frame.")
        t_ctx = self.cfg.encoder.input_frames
        stride = self.cfg.train.frame_stride
        span = (t_ctx - 1) * stride + self.cfg.train.horizon_k
        max_start = num_frames - 1 - span
        if max_start < 0:
            start = 0
        elif self.split == "train":
            start = (rng or random).randint(0, max_start)
        else:
            start = max_start // 2
        last = num_frames - 1
        context = [min(last, start + i * stride) for i in range(t_ctx)]
        return context, start

    def __getitem__(self, index: int) -> ClipSample:
        """Return one raw context clip and, only when requested, its future clip.

        Opens the video, reads frame count cheaply, computes either 8 context
        indices or 16 shared context/target indices, and decodes only those
        frames rather than the entire clip.
        """
        path = self.paths[index]
        sample_id = self.sample_id_at(index)
        rng = self._rng_for(sample_id)
        reader = _open_video_reader(path)
        if self.needs_target:
            context_idx, target_idx = self._window_indices(len(reader), rng)
        else:
            context_idx, _ = self._context_indices(len(reader), rng)
            target_idx = None
        t_ctx = self.cfg.encoder.input_frames
        decode_indices = context_idx + target_idx if target_idx is not None else context_idx
        frames_np = _decode_frames(reader, decode_indices)
        stacked = torch.from_numpy(frames_np)
        stacked = stacked.float().permute(0, 3, 1, 2) / 255.0
        stacked = _resize_shorter_side(stacked, self.cfg.encoder.input_height)
        stacked = _crop(stacked, self.cfg.encoder.input_height, self.split, rng)
        if self.split == "train":
            stacked = _color_jitter(stacked, rng)
        context = stacked[:t_ctx]
        target = stacked[t_ctx:] if self.needs_target else None
        return ClipSample(context=context, target=target, sample_id=sample_id)


def _collate_clip_samples(samples: list[ClipSample]) -> ClipBatch:
    """Stack one homogeneous context-only or context/target sample list."""
    _require_torch()
    if not samples:
        raise ValueError("Cannot collate an empty clip batch.")
    has_target = samples[0].target is not None
    if any((sample.target is not None) != has_target for sample in samples):
        raise ValueError("Clip samples in one batch must use the same target mode.")
    target = None
    if has_target:
        target = torch.stack([sample.target for sample in samples])
    return ClipBatch(
        context=torch.stack([sample.context for sample in samples]),
        target=target,
        sample_ids=tuple(sample.sample_id for sample in samples),
    )


def _worker_init(worker_id: int) -> None:
    """Seed Python/NumPy from PyTorch's explicit per-worker seed."""
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloader(
    cfg: Config,
    split: Literal["train", "validation"] = "train",
    batch_size: int | None = None,
    *,
    needs_target: bool | None = None,
    epoch: int = 0,
    start_offset: int = 0,
    drop_last: bool | None = None,
) -> DataLoader:
    """Build a Phase 1 dataloader.

    Args:
        cfg: Global config with selected data root and worker settings.
        split: Dataset split to read.
        batch_size: Optional override for smoke tests.
        needs_target: Whether to decode the future window.
        epoch: Deterministic shuffle/transform epoch.
        start_offset: Number of epoch-order samples already consumed on resume.
        drop_last: Override incomplete-batch handling. Training defaults to True;
            offline statistics pass False so every requested clip remains eligible.
    Returns:
        DataLoader yielding :class:`ClipBatch` values. Context and an optional
        target have shape `(B,8,3,256,256)` in raw `[0,1]` RGB.
    """
    _require_torch()
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    selected_target_mode = (
        not cfg.train.present_recon_only if needs_target is None else needs_target
    )
    dataset = SSV2Dataset(
        cfg.data.dataset_root(), split, cfg, needs_target=selected_target_mode, epoch=epoch
    )
    generator = torch.Generator().manual_seed(
        cfg.seed + (0 if split == "train" else 10_000) + epoch * 1_000_003
    )
    if split == "train":
        order = torch.randperm(len(dataset), generator=generator).tolist()
    else:
        order = list(range(len(dataset)))
    if not 0 <= start_offset <= len(order):
        raise ValueError(f"start_offset must be in [0,{len(order)}]; got {start_offset}.")
    sampler = order[start_offset:]
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.train.global_batch,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=(split == "train") if drop_last is None else drop_last,
        collate_fn=_collate_clip_samples,
        worker_init_fn=_worker_init,
        generator=generator,
    )


def build_fixed_diagnostic_batch(
    cfg: Config,
    batch_size: int,
    *,
    needs_target: bool,
) -> ClipBatch:
    """Build the deterministic source-diverse validation batch used by diagnostics.

    Args:
        cfg: Global config selecting the dataset and worker settings.
        batch_size: Requested number of distinct source videos.
        needs_target: Whether to decode the future window.
    Returns:
        One fixed validation batch with at most one clip per source video.
    """
    _require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    dataset = SSV2Dataset(
        cfg.data.dataset_root(),
        "validation",
        cfg,
        needs_target=needs_target,
        epoch=0,
    )
    sample_ids = [dataset.sample_id_at(index) for index in range(len(dataset))]
    selected = select_unique_source_indices(sample_ids, cfg.data.dataset, batch_size)
    generator = torch.Generator().manual_seed(cfg.seed + 10_000)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=selected,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
        collate_fn=_collate_clip_samples,
        worker_init_fn=_worker_init,
        generator=generator,
    )
    return next(iter(loader))


def smoke_test_dataloader() -> None:
    """Load two SSv2-tiny batches and assert Phase 1 v0.2 data contracts."""
    cfg = Config()
    loader = build_dataloader(cfg, "train", batch_size=2)
    expected = (
        cfg.encoder.input_frames,
        3,
        cfg.encoder.input_height,
        cfg.encoder.input_width,
    )
    for i, batch in enumerate(loader):
        context_clip, target_clip = batch.context, batch.target
        assert target_clip is not None
        print(
            context_clip.shape,
            target_clip.shape,
            float(context_clip.min()),
            float(context_clip.max()),
        )
        assert context_clip.shape[1:] == expected, context_clip.shape
        assert target_clip.shape[1:] == expected, target_clip.shape
        assert torch.isfinite(context_clip).all() and torch.isfinite(target_clip).all()
        assert 0.0 <= float(context_clip.min()) <= float(context_clip.max()) <= 1.0
        if i == 1:
            break
