"""Frozen vision-encoder seam for dense video features.

The four exported names are the whole common-pipeline interface. Backend models,
their output objects, and their token-selection rules stay private to this file.
Version 1 deliberately supports only raw 8-frame 256x256 RGB clips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, cast

import torch
from torch import Tensor, nn

from config import ENCODER_IMAGE_MEAN, ENCODER_IMAGE_STD, EncoderConfig

__all__ = [
    "FeatureLayout",
    "EncoderSpec",
    "FrozenEncoder",
    "build_frozen_encoder",
]

_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}")
_FINGERPRINT_SCHEMA = "hje-vwm-frozen-feature-v1"
_VJEPA2_VITL16_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
_SIGLIP2_VITB16_REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"


@dataclass(frozen=True)
class FeatureLayout:
    """Dense token-lattice geometry and canonical flattening order."""

    temporal: int
    height: int
    width: int
    order: Literal["time_y_x"]
    temporal_unit: Literal["frame", "tubelet"]
    temporal_stride_frames: int
    temporal_support_frames: int

    def __post_init__(self) -> None:
        """Reject ambiguous or non-positive lattice geometry at construction."""
        dimensions = (self.temporal, self.height, self.width)
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("FeatureLayout temporal/height/width must be positive integers.")
        if self.order != "time_y_x":
            raise ValueError("FeatureLayout order must be 'time_y_x'.")
        if self.temporal_unit not in {"frame", "tubelet"}:
            raise ValueError("FeatureLayout temporal_unit must be 'frame' or 'tubelet'.")
        if self.temporal_stride_frames <= 0 or self.temporal_support_frames <= 0:
            raise ValueError("FeatureLayout temporal stride/support must be positive.")

    @property
    def n_tokens(self) -> int:
        """Total dense tokens after flattening in time-major, then y/x order."""
        return self.temporal * self.height * self.width


@dataclass(frozen=True)
class EncoderSpec:
    """Resolved, immutable identity and dense-feature contract for one encoder."""

    family: str
    repo_id: str
    requested_revision: str
    resolved_revision: str
    input_frames: int
    input_height: int
    input_width: int
    feature_dim: int
    layout: FeatureLayout
    normalization_id: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    preprocess_version: str
    inference_precision: str
    frame_microbatch: int
    attention_implementation: str
    cache_dir: str
    parameter_count: int
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate the resolved contract and derive its canonical fingerprint."""
        for field_name in ("family", "repo_id", "normalization_id", "preprocess_version"):
            if not getattr(self, field_name):
                raise ValueError(f"EncoderSpec {field_name} must be non-empty.")
        for field_name in ("requested_revision", "resolved_revision"):
            revision = getattr(self, field_name)
            if _IMMUTABLE_REVISION.fullmatch(revision) is None:
                raise ValueError(
                    f"EncoderSpec {field_name} must be an immutable 40-character commit SHA."
                )
        if self.requested_revision != self.resolved_revision:
            raise ValueError("Requested and resolved encoder revisions must match exactly.")
        if (self.input_frames, self.input_height, self.input_width) != (8, 256, 256):
            raise ValueError("EncoderSpec v1 supports only 8 frames at 256x256.")
        if self.feature_dim <= 0:
            raise ValueError("EncoderSpec feature_dim must be positive.")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("EncoderSpec normalization mean/std must contain three values.")
        normalization_values = (*self.normalization_mean, *self.normalization_std)
        if not all(math.isfinite(value) for value in normalization_values):
            raise ValueError("EncoderSpec normalization mean/std must be finite.")
        if any(value <= 0 for value in self.normalization_std):
            raise ValueError("EncoderSpec normalization std values must be positive.")
        if self.inference_precision not in {"fp32", "bf16"}:
            raise ValueError("EncoderSpec inference_precision must be 'fp32' or 'bf16'.")
        if self.frame_microbatch <= 0:
            raise ValueError("EncoderSpec frame_microbatch must be positive.")
        if not self.attention_implementation:
            raise ValueError("EncoderSpec attention_implementation must be non-empty.")
        if not self.cache_dir:
            raise ValueError("EncoderSpec cache_dir must be non-empty.")
        if self.parameter_count < 0:
            raise ValueError("EncoderSpec parameter_count cannot be negative.")

        payload = {
            "schema": _FINGERPRINT_SCHEMA,
            "family": self.family,
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "input_frames": self.input_frames,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "feature_dim": self.feature_dim,
            "layout": asdict(self.layout),
            "normalization_id": self.normalization_id,
            "normalization_mean": self.normalization_mean,
            "normalization_std": self.normalization_std,
            "preprocess_version": self.preprocess_version,
            "inference_precision": self.inference_precision,
            "frame_microbatch": self.frame_microbatch,
            "attention_implementation": self.attention_implementation,
            "cache_dir": self.cache_dir,
            "parameter_count": self.parameter_count,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(self, "fingerprint", hashlib.sha256(canonical).hexdigest())


class _EncoderBackend(Protocol):
    """Structural contract shared by private backends and injected offline fakes."""

    frame_based: bool
    resolved_revision: str
    training: bool

    def parameters(self, recurse: bool = True):
        """Return every backend parameter for count/freeze validation."""
        ...

    def eval(self):
        """Enter backend evaluation mode and return the backend."""
        ...

    def forward_units(self, inputs: Tensor) -> Any:
        """Encode `(U,C,H,W)` frames or `(B,T,C,H,W)` clips into private output."""
        ...

    def select_dense_tokens(self, output: Any) -> Tensor:
        """Select dense `(U,S,D)` or `(B,N,D)` tokens from a private output."""
        ...


def _capture_resolved_revision(
    model: nn.Module, repo_id: str, requested_revision: str, cache_dir: str
) -> str:
    """Capture the actual Hub snapshot even when a nested vision config drops it.

    Some composite checkpoints (notably SigLIP) deserialize the vision sub-config
    without Transformers' private ``_commit_hash``. In that case the already
    resolved cached config path is authoritative and contains the snapshot SHA.
    """
    resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
    if isinstance(resolved, str) and _IMMUTABLE_REVISION.fullmatch(resolved):
        return resolved
    from transformers.utils.hub import cached_file, extract_commit_hash

    config_path = cached_file(
        repo_id,
        "config.json",
        revision=requested_revision,
        cache_dir=cache_dir,
    )
    resolved = extract_commit_hash(config_path, None)
    if not isinstance(resolved, str) or _IMMUTABLE_REVISION.fullmatch(resolved) is None:
        raise RuntimeError("Transformers did not expose an immutable resolved Hub revision.")
    return resolved


class FrozenEncoder(nn.Module):
    """Normalize raw clips once and return one validated dense token tensor.

    Production callers should use :func:`build_frozen_encoder`. The private
    ``_backend`` injection point exists so the full seam can be tested offline
    without downloading model weights.
    """

    def __init__(
        self,
        *,
        _backend: _EncoderBackend,
        spec: EncoderSpec,
    ) -> None:
        """Install one resolved private backend and freeze it immediately.

        Args:
            _backend: Injected private production adapter or offline test fake.
            spec: Immutable identity and output-layout contract for that backend.
        """
        super().__init__()
        if not isinstance(_backend, nn.Module):
            raise TypeError("FrozenEncoder _backend must be a torch.nn.Module.")
        if _backend.resolved_revision != spec.resolved_revision:
            raise ValueError("Backend and EncoderSpec resolved revisions do not match.")
        parameter_count = sum(parameter.numel() for parameter in _backend.parameters())
        if parameter_count != spec.parameter_count:
            raise ValueError(
                "Backend parameter count does not match EncoderSpec: "
                f"{parameter_count} != {spec.parameter_count}."
            )
        if _backend.frame_based:
            if spec.layout.temporal_unit != "frame" or spec.layout.temporal != spec.input_frames:
                raise ValueError(
                    "Frame backends require one time-major layout slot per input frame."
                )

        self._backend = cast(nn.Module, _backend)
        self._spec = spec
        mean = torch.tensor(spec.normalization_mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor(spec.normalization_std, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("_normalization_mean", mean, persistent=True)
        self.register_buffer("_normalization_std", std, persistent=True)
        self._pin_eval_and_freeze()

    @property
    def spec(self) -> EncoderSpec:
        """The resolved feature contract and immutable feature fingerprint."""
        return self._spec

    def _pin_eval_and_freeze(self) -> None:
        """Restore the safety invariant after any recursive module mode change."""
        self._backend.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.training = False

    def train(self, mode: bool = True) -> FrozenEncoder:
        """Remain in eval mode even when a parent recursively enters train mode."""
        super().train(False)
        self._pin_eval_and_freeze()
        return self

    def requires_grad_(self, requires_grad: bool = True) -> FrozenEncoder:
        """Remain frozen even when a parent recursively toggles gradient flags."""
        super().requires_grad_(False)
        return self

    def _validate_input(self, raw_clips: Tensor) -> None:
        """Validate the canonical raw range, dtype, device, and v1 clip shape."""
        if not isinstance(raw_clips, Tensor):
            raise TypeError("FrozenEncoder input must be a torch.Tensor.")
        if not raw_clips.is_floating_point():
            raise TypeError("FrozenEncoder input must use a floating point dtype.")
        expected = (
            raw_clips.shape[0] if raw_clips.ndim == 5 else "B",
            self.spec.input_frames,
            3,
            self.spec.input_height,
            self.spec.input_width,
        )
        if raw_clips.ndim != 5 or tuple(raw_clips.shape) != expected:
            raise ValueError(
                "FrozenEncoder input shape must be "
                f"(B,{self.spec.input_frames},3,{self.spec.input_height},{self.spec.input_width}); "
                f"received {tuple(raw_clips.shape)}."
            )
        if raw_clips.shape[0] <= 0:
            raise ValueError("FrozenEncoder input batch must be non-empty.")
        if raw_clips.device != self._normalization_mean.device:
            raise ValueError("FrozenEncoder and raw clips must be on the same device.")
        if not torch.isfinite(raw_clips).all():
            raise ValueError("FrozenEncoder raw clips must contain only finite values.")
        minimum, maximum = raw_clips.aminmax()
        if minimum.item() < 0.0 or maximum.item() > 1.0:
            raise ValueError(
                "FrozenEncoder raw clips must be in [0, 1]; "
                f"received range [{minimum.item():.6g}, {maximum.item():.6g}]."
            )

    def _inference_context(self, device: torch.device):
        """Own autocast state so caller contexts cannot change encoder precision."""
        if self.spec.inference_precision == "fp32":
            if device.type in {"cpu", "cuda", "mps"}:
                return torch.autocast(device_type=device.type, enabled=False)
            return nullcontext()
        if device.type != "cuda":
            raise RuntimeError(
                "bf16 encoder inference is supported only on CUDA; use fp32 on CPU/MPS."
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bf16 encoder inference.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _encode_frames(self, normalized: Tensor) -> Tensor:
        """Microbatch flattened frames and restore dense time-major token order.

        Args:
            normalized: Adapter-normalized fp32 clips, shape `(B,T,3,H,W)`.
        Returns:
            Dense time-major frame tokens, shape `(B,T*H_grid*W_grid,D_e)`.
        """
        batch, temporal, channels, height, width = normalized.shape
        frames = normalized.reshape(batch * temporal, channels, height, width)
        selected_chunks: list[Tensor] = []
        spatial_tokens = self.spec.layout.height * self.spec.layout.width
        backend = cast(_EncoderBackend, self._backend)
        for start in range(0, frames.shape[0], self.spec.frame_microbatch):
            units = frames[start : start + self.spec.frame_microbatch]
            selected = backend.select_dense_tokens(backend.forward_units(units))
            expected = (units.shape[0], spatial_tokens, self.spec.feature_dim)
            if not isinstance(selected, Tensor) or tuple(selected.shape) != expected:
                actual = tuple(selected.shape) if isinstance(selected, Tensor) else type(selected)
                raise ValueError(
                    f"Frame encoder output must have shape {expected}; received {actual}."
                )
            selected_chunks.append(selected)
        stacked = torch.cat(selected_chunks, dim=0)
        return stacked.reshape(batch, temporal * spatial_tokens, self.spec.feature_dim)

    @staticmethod
    def _require_valid_output(tokens: Tensor, expected: tuple[int, int, int]) -> None:
        """Reject backend outputs that violate the dense floating feature contract.

        Args:
            tokens: Candidate dense backend features, expected shape `(B,N_e,D_e)`.
            expected: Exact `(batch,tokens,feature_dim)` tuple required by the spec.
        """
        if not isinstance(tokens, Tensor):
            raise TypeError("FrozenEncoder output must be a torch.Tensor.")
        if not tokens.is_floating_point():
            raise TypeError("FrozenEncoder output must use a floating point dtype.")
        if tuple(tokens.shape) != expected:
            raise ValueError(
                f"FrozenEncoder output must have shape {expected}; received {tuple(tokens.shape)}."
            )
        if not torch.isfinite(tokens).all():
            raise ValueError("FrozenEncoder output must contain only finite values.")

    @torch.no_grad()
    def forward(self, raw_clips: Tensor) -> Tensor:
        """Normalize exactly once and encode one canonical raw video batch.

        Args:
            raw_clips: Floating RGB pixels in `[0,1]`, shape `(B,8,3,256,256)`.
        Returns:
            Dense finite features in time-major layout, shape
            `(B,spec.layout.n_tokens,spec.feature_dim)`.
        """
        self._validate_input(raw_clips)
        normalized = (raw_clips.float() - self._normalization_mean) / self._normalization_std
        if not torch.isfinite(normalized).all():
            raise ValueError("FrozenEncoder normalized input must contain only finite values.")

        backend = cast(_EncoderBackend, self._backend)
        self._pin_eval_and_freeze()
        with self._inference_context(raw_clips.device):
            if backend.frame_based:
                tokens = self._encode_frames(normalized)
            else:
                output = backend.forward_units(normalized)
                tokens = backend.select_dense_tokens(output)

        expected = (raw_clips.shape[0], self.spec.layout.n_tokens, self.spec.feature_dim)
        self._require_valid_output(tokens, expected)
        return tokens


class _VJEPA2Adapter(nn.Module):
    """Private Transformers adapter for the tested V-JEPA2 ViT-L/16 checkpoint."""

    frame_based = False

    def __init__(self, cfg: EncoderConfig, repo_id: str, revision: str) -> None:
        """Load one immutable Hub snapshot through the pinned Transformers API."""
        super().__init__()
        try:
            from transformers import AutoModel
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency installation error.
            raise RuntimeError(
                "transformers==4.57.6 is required for the V-JEPA2 encoder adapter."
            ) from exc

        self.model = AutoModel.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=cfg.hf_cache_dir,
            attn_implementation=cfg.attention_implementation,
        )
        self.resolved_revision = _capture_resolved_revision(
            self.model, repo_id, revision, cfg.hf_cache_dir
        )

    def forward_units(self, clips: Tensor) -> Tensor:
        """Return the encoder-only tubelet lattice, skipping V-JEPA's predictor.

        Args:
            clips: ImageNet-normalized video, shape `(B,8,3,256,256)`.
        Returns:
            Dense V-JEPA tubelet tokens, shape `(B,1024,1024)`.
        """
        return self.model.get_vision_features(clips)

    @staticmethod
    def select_dense_tokens(output: Tensor) -> Tensor:
        """Return V-JEPA's already-dense `(B,1024,1024)` tubelet tensor.

        Args:
            output: Direct `get_vision_features` tensor, shape `(B,1024,1024)`.
        Returns:
            The same dense tensor; V-JEPA has no special tokens to strip here.
        """
        return output


class _SigLIP2Adapter(nn.Module):
    """Private vision-only adapter for the fixed-resolution SigLIP 2 ViT-B/16."""

    frame_based = True

    def __init__(self, cfg: EncoderConfig, repo_id: str, revision: str) -> None:
        """Load only the immutable SigLIP vision tower through Transformers."""
        super().__init__()
        try:
            from transformers import SiglipVisionModel
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency installation error.
            raise RuntimeError(
                "transformers==4.57.6 is required for the SigLIP 2 encoder adapter."
            ) from exc

        self.model = SiglipVisionModel.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=cfg.hf_cache_dir,
            attn_implementation=cfg.attention_implementation,
        )
        vision_model = getattr(self.model, "vision_model", None)
        if not isinstance(vision_model, nn.Module) or not hasattr(vision_model, "head"):
            raise RuntimeError("SigLIP vision model no longer exposes its pooling-head seam.")
        # The experiment consumes patch tokens only. Remove the 7.1M-parameter
        # attention pooler after checkpoint loading so it is neither executed nor
        # retained as dead runtime weight; the 85.84M patch tower stays exact.
        vision_model.use_head = False
        vision_model.head = None
        self.resolved_revision = _capture_resolved_revision(
            self.model, repo_id, revision, cfg.hf_cache_dir
        )

    def forward_units(self, frames: Tensor) -> Any:
        """Encode independent normalized frames without constructing the text tower.

        Args:
            frames: SigLIP-normalized images, shape `(U,3,256,256)`.
        Returns:
            Transformers vision output containing `(U,256,768)` patch tokens.
        """
        return self.model(pixel_values=frames, return_dict=True)

    @staticmethod
    def select_dense_tokens(output: Any) -> Tensor:
        """Select the unpooled patch sequence; fixed-resolution SigLIP has no CLS token."""
        tokens = getattr(output, "last_hidden_state", None)
        if not isinstance(tokens, Tensor):
            raise TypeError("SigLIP vision output must expose tensor last_hidden_state patches.")
        return tokens


@dataclass(frozen=True)
class _AdapterRegistration:
    """Private stable-alias metadata and optional tested adapter factory."""

    family: str
    repo_id: str
    default_revision: str | None
    factory: Callable[[EncoderConfig, str, str], _EncoderBackend] | None
    feature_dim: int
    layout: FeatureLayout
    normalization_id: str
    preprocess_version: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]


_ADAPTER_REGISTRY = {
    "vjepa2_vitl16": _AdapterRegistration(
        family="vjepa2",
        repo_id="facebook/vjepa2-vitl-fpc64-256",
        default_revision=_VJEPA2_VITL16_REVISION,
        factory=_VJEPA2Adapter,
        feature_dim=1024,
        layout=FeatureLayout(4, 16, 16, "time_y_x", "tubelet", 2, 2),
        normalization_id="imagenet-mean-std",
        preprocess_version="raw-rgb-8x256-v1",
        normalization_mean=ENCODER_IMAGE_MEAN,
        normalization_std=ENCODER_IMAGE_STD,
    ),
    "dinov3_vitb16": _AdapterRegistration(
        family="dinov3",
        repo_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        default_revision=None,
        factory=None,
        feature_dim=768,
        layout=FeatureLayout(8, 16, 16, "time_y_x", "frame", 1, 1),
        normalization_id="imagenet-mean-std",
        preprocess_version="unresolved",
        normalization_mean=ENCODER_IMAGE_MEAN,
        normalization_std=ENCODER_IMAGE_STD,
    ),
    "siglip2_vitb16": _AdapterRegistration(
        family="siglip2",
        repo_id="google/siglip2-base-patch16-256",
        default_revision=_SIGLIP2_VITB16_REVISION,
        factory=_SigLIP2Adapter,
        feature_dim=768,
        layout=FeatureLayout(8, 16, 16, "time_y_x", "frame", 1, 1),
        normalization_id="siglip-minus-one-to-one",
        preprocess_version="siglip2-fixres256-last-hidden-state-v1",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
    ),
}


def _validate_config(cfg: EncoderConfig) -> None:
    """Enforce the deliberately narrow version-1 input and inference contract."""
    if cfg.input_frames != 8:
        raise ValueError("Encoder v1 requires exactly 8 frames.")
    if cfg.input_height != cfg.input_width:
        raise ValueError("Encoder v1 requires square inputs.")
    if (cfg.input_height, cfg.input_width) != (256, 256):
        raise ValueError("Encoder v1 supports only 256x256 inputs.")
    if cfg.precision not in {"fp32", "bf16"}:
        raise ValueError("Encoder precision must be 'fp32' or 'bf16'.")
    if not isinstance(cfg.frame_microbatch, int) or cfg.frame_microbatch <= 0:
        raise ValueError("Encoder frame_microbatch must be a positive integer.")
    if not cfg.attention_implementation:
        raise ValueError("Encoder attention_implementation must be non-empty.")
    if not cfg.hf_cache_dir:
        raise ValueError("Encoder hf_cache_dir must be non-empty.")


def build_frozen_encoder(cfg: EncoderConfig) -> FrozenEncoder:
    """Resolve a stable alias and build its frozen, validated feature wrapper.

    Args:
        cfg: Alias, immutable revision override, input contract, cache, and
            inference identity.
    Returns:
        Frozen raw-clip encoder exposing one resolved immutable `EncoderSpec`.
    Raises:
        ValueError: The alias/config/revision violates the strict v1 contract.
        RuntimeError: The alias has no tested adapter or resolution is mutable/mismatched.
    """
    if not isinstance(cfg, EncoderConfig):
        raise TypeError("build_frozen_encoder expects config.EncoderConfig.")
    registration = _ADAPTER_REGISTRY.get(cfg.alias)
    if registration is None:
        choices = ", ".join(sorted(_ADAPTER_REGISTRY))
        raise ValueError(f"Unknown encoder alias {cfg.alias!r}; expected one of: {choices}.")
    _validate_config(cfg)
    if registration.factory is None or registration.default_revision is None:
        raise RuntimeError(
            f"Encoder alias {cfg.alias!r} is reserved but not implemented with a tested "
            "private adapter and immutable default revision. It will not fall back to main."
        )

    requested_revision = cfg.revision or registration.default_revision
    if _IMMUTABLE_REVISION.fullmatch(requested_revision) is None:
        raise ValueError("Encoder revision must be an immutable 40-character Hub commit SHA.")
    backend = registration.factory(cfg, registration.repo_id, requested_revision)
    resolved_revision = backend.resolved_revision
    if _IMMUTABLE_REVISION.fullmatch(resolved_revision) is None:
        raise RuntimeError("Encoder backend did not resolve to a 40-character Hub commit SHA.")
    if resolved_revision != requested_revision:
        raise RuntimeError(
            "Encoder resolved revision differs from the requested immutable revision: "
            f"{resolved_revision} != {requested_revision}."
        )
    parameter_count = sum(parameter.numel() for parameter in backend.parameters())
    spec = EncoderSpec(
        family=registration.family,
        repo_id=registration.repo_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        input_frames=cfg.input_frames,
        input_height=cfg.input_height,
        input_width=cfg.input_width,
        feature_dim=registration.feature_dim,
        layout=registration.layout,
        normalization_id=registration.normalization_id,
        normalization_mean=registration.normalization_mean,
        normalization_std=registration.normalization_std,
        preprocess_version=registration.preprocess_version,
        inference_precision=cfg.precision,
        frame_microbatch=cfg.frame_microbatch,
        attention_implementation=cfg.attention_implementation,
        cache_dir=cfg.hf_cache_dir,
        parameter_count=parameter_count,
    )
    return FrozenEncoder(
        _backend=backend,
        spec=spec,
    )


def _smoke_cli() -> None:
    """Run a credential-safe real adapter forward and emit a JSON evidence report."""
    parser = argparse.ArgumentParser(description="Authenticated frozen-encoder smoke test")
    parser.add_argument("--smoke", action="store_true", help="run one real adapter forward")
    parser.add_argument("--encoder", default="vjepa2_vitl16", choices=sorted(_ADAPTER_REGISTRY))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--frame-microbatch", type=int, default=8)
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--hf-cache-dir", default="/workspace/hf_cache")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default=None)
    args = parser.parse_args()
    if not args.smoke:
        parser.error("pass --smoke to acknowledge real model loading")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device or default_device)
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    cfg = EncoderConfig(
        alias=args.encoder,
        revision=args.revision,
        precision=precision,
        frame_microbatch=args.frame_microbatch,
        attention_implementation=args.attention_implementation,
        hf_cache_dir=args.hf_cache_dir,
    )

    encoder = build_frozen_encoder(cfg).to(device)
    raw = torch.rand(
        args.batch_size,
        cfg.input_frames,
        3,
        cfg.input_height,
        cfg.input_width,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokens = encoder(raw)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    report = {
        "alias": cfg.alias,
        "repo_id": encoder.spec.repo_id,
        "requested_revision": encoder.spec.requested_revision,
        "resolved_revision": encoder.spec.resolved_revision,
        "feature_fingerprint": encoder.spec.fingerprint,
        "transformers_version": __import__("transformers").__version__,
        "parameter_count": encoder.spec.parameter_count,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "output_shape": list(tokens.shape),
        "output_dtype": str(tokens.dtype),
        "output_min": float(tokens.min().item()),
        "output_max": float(tokens.max().item()),
        "output_finite": bool(torch.isfinite(tokens).all().item()),
        "peak_encoder_memory_bytes": peak_memory,
        "device": str(device),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    _smoke_cli()
