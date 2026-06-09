"""Configuration for HJEPA-VWM Phase 1 (v0.2 — frozen encoder).

Naming map from AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md §3:
- x        -> context_clip
- E        -> encoder            (FROZEN pretrained V-JEPA 2 ViT-L/16; shared both branches)
- e_t      -> detailed           (frozen encoder output, no grad)
- B        -> bottleneck         (trainable)
- c_t      -> abstract
- x_{<=t+k}-> target_clip         (future clip ending at t+k)
- B_EMA    -> target_bottleneck   (EMA copy of B; the only EMA module)
- e_plus   -> target_detailed
- c_plus   -> target_abstract     (= c+_{t+k})
- F_c      -> coarse_flow
- F_e      -> fine_flow           (Phase 2)
- c_hat    -> pred_abstract
- e_hat    -> pred_detailed
- D        -> frame_generator     (Phase 3)
- A        -> vae_encoder         (Phase 3)

All numerical values come from AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md §2.6.
v0.2: encoder is frozen pretrained V-JEPA 2 ViT-L/16 (D_e=1024); EMA on bottleneck
only; collapse prevention is a variance floor on c_t (no SIGReg). See the encoder
note in §4 of AGENT_FILES/PHASES/PHASE_1.md for the repo-id verification fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# V-JEPA 2 / ViT-L processor normalization (ImageNet stats). Confirmed against
# transformers AutoVideoProcessor("facebook/vjepa2-vitl-fpc64-256"): the encoder
# expects these, NOT [-1, 1]. The VAE's [-1, 1] is a separate Stage-4 concern.
ENCODER_IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
ENCODER_IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Locked architecture constants from UNDERSTANDING.md §2.6 (v0.2)."""

    # Frozen encoder (V-JEPA 2 ViT-L/16). Native resolution 256 == our resolution.
    # The ViT-B/16 (D_e=768) checkpoint has no transformers repo (torch.hub only);
    # ViT-L via HF is the verified clean path. Verify the repo id at load.
    encoder_repo: str = "facebook/vjepa2-vitl-fpc64-256"
    encoder_frozen: bool = True
    encoder_patch: int = 16  # spatial patch (V-JEPA 2 native)
    encoder_tubelet: int = 2  # temporal tubelet (V-JEPA 2 native)

    t_ctx: int = 8  # context frames (tubelet-2 -> 4 temporal tokens)
    h: int = 256
    w: int = 256
    n_c: int = 32  # abstract tokens (bottleneck query slots)
    d_e: int = 1024  # frozen encoder embedding dim (ViT-L/16)
    d_c: int = 256  # abstract latent dim

    # Bottleneck B
    bottleneck_mixer_dim: int = 256  # input projection D_e -> mixer width (= d_c)
    bottleneck_convnext_blocks: int = 2
    bottleneck_cross_attn_heads: int = 8

    # Coarse flow F_c
    f_c_blocks: int = 6
    f_c_dim: int = 256
    f_c_heads: int = 8
    condition_dropout: float = 0.10

    # NOTE: no tubelet_dropout (removed in v0.2 — frozen encoder).
    # NOTE: no encoder_depth/heads — fixed by the pretrained checkpoint.

    @property
    def grid_spatial(self) -> int:
        """Patch-grid side length per frame, H / encoder_patch (256/16 = 16)."""
        return self.h // self.encoder_patch

    @property
    def n_temporal_tokens(self) -> int:
        """Temporal tokens after tubelet merging, T / tubelet (8/2 = 4)."""
        return self.t_ctx // self.encoder_tubelet

    @property
    def tokens_per_frame(self) -> int:
        """Spatial tokens per temporal slot, grid_spatial**2 (16*16 = 256)."""
        return self.grid_spatial * self.grid_spatial

    @property
    def n_ctx(self) -> int:
        """Context token count from encoder geometry, (T/2)*(H/16)**2 = 1024."""
        return self.n_temporal_tokens * self.tokens_per_frame

    @property
    def n_tgt(self) -> int:
        """Target token count — clip-level target, same geometry as context (1024)."""
        return self.n_ctx


@dataclass
class TrainConfig:
    """Phase 1 training and diagnostic constants from UNDERSTANDING.md §2.6."""

    global_batch: int = 64
    stage1_steps: int = 30_000
    max_steps: int = 30_000  # Phase 1 trains ONLY Stage 1
    # NOTE: no lr_encoder — encoder is frozen.
    lr_bottleneck: float = 2e-4
    lr_coarse_flow: float = 4e-4
    warmup_steps: int = 10_000
    total_latent_steps: int = 105_000  # cosine LR / EMA denominator (Phase 2+)
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    ema_m_start: float = 0.996
    ema_m_end: float = 0.9999
    ema_schedule_steps: int = 105_000
    lambda_var: float = 0.10  # variance-floor weight (replaces SIGReg)
    var_floor_std_target: float = 1.0  # hinge target in L_var
    horizon_k: int = 4  # single fixed horizon for Phases 1-3 (Phase 4: multi-horizon)
    frame_stride: int = 2
    precision: str = "bf16"
    log_every: int = 50
    diag_every: int = 500
    checkpoint_every: int = 5000


@dataclass
class DataConfig:
    """Dataset roots; only JEPA_DATA_ROOT overrides the /workspace path contract."""

    data_root: str = field(
        default_factory=lambda: os.environ.get("JEPA_DATA_ROOT", "/workspace/data")
    )
    dataset: str = "ssv2_tiny"  # CLI override: ssv2 | ssv2_tiny
    num_workers: int = 8
    pin_memory: bool = True

    @property
    def full_root(self) -> str:
        """Full SSv2 symlink root, `<data_root>/ssv2`."""
        return str(Path(self.data_root) / "ssv2")

    @property
    def tiny_root(self) -> str:
        """SSv2-tiny symlink root, `<data_root>/ssv2_tiny`."""
        return str(Path(self.data_root) / "ssv2_tiny")

    def dataset_root(self) -> str:
        """Return the selected dataset root for the configured dataset name."""
        if self.dataset == "ssv2":
            return self.full_root
        if self.dataset == "ssv2_tiny":
            return self.tiny_root
        raise ValueError(f"Unsupported dataset: {self.dataset}")


@dataclass
class Config:
    """Top-level Phase 1 configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    debug_shapes: bool = True
    checkpoint_dir: str = "/workspace/checkpoints"
    hf_cache_dir: str = "/workspace/hf_cache"
    seed: int = 42
