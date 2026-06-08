"""Configuration for HJEPA-VWM Phase 1.

Naming map from AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md §3:
- x -> context_clip
- E -> online_encoder
- e_t -> detailed
- B -> bottleneck
- c_t -> abstract
- y -> future_frame
- E_bar -> target_encoder
- B_bar -> target_bottleneck
- e_plus -> target_detailed
- c_plus -> target_abstract
- F_c -> coarse_flow
- F_e -> fine_flow
- c_hat -> pred_abstract
- e_hat -> pred_detailed
- D -> frame_generator
- A -> vae_encoder
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Locked architecture constants from UNDERSTANDING.md §2.6."""

    t_ctx: int = 4
    h: int = 128
    w: int = 128
    patch_t: int = 1
    patch_h: int = 16
    patch_w: int = 16
    n_ctx: int = 256
    n_tgt: int = 64
    n_c: int = 32
    d_e: int = 384
    d_c: int = 256
    encoder_depth: int = 12
    encoder_heads: int = 6
    encoder_mlp_ratio: int = 4
    bottleneck_convnext_blocks: int = 2
    bottleneck_cross_attn_heads: int = 8
    f_c_blocks: int = 6
    f_c_dim: int = 256
    f_c_heads: int = 8
    tubelet_dropout: float = 0.40
    condition_dropout: float = 0.10

    @property
    def grid_h(self) -> int:
        """Return the patch-grid height for 128×128 frames.

        This keeps geometry derived from locked config values instead of repeating
        bare literals throughout the model path.

        Returns:
            Number of vertical patch tokens, H / patch_h.
        """
        return self.h // self.patch_h

    @property
    def grid_w(self) -> int:
        """Return the patch-grid width for 128×128 frames.

        This mirrors `grid_h` so code can express patch geometry through config,
        not through scattered constants.

        Returns:
            Number of horizontal patch tokens, W / patch_w.
        """
        return self.w // self.patch_w

    @property
    def tokens_per_frame(self) -> int:
        """Return the number of detailed tokens per frame.

        The target side has one frame, so this value is also `N_tgt`; context
        tokens are `T * tokens_per_frame`.

        Returns:
            Number of spatial patch tokens in one 128×128 frame.
        """
        return self.grid_h * self.grid_w


@dataclass
class TrainConfig:
    """Phase 1 training and diagnostic constants from UNDERSTANDING.md §2.6."""

    global_batch: int = 64
    stage1_steps: int = 30_000
    max_steps: int = 30_000
    lr_encoder: float = 2e-4
    lr_bottleneck: float = 2e-4
    lr_coarse_flow: float = 4e-4
    warmup_steps: int = 10_000
    total_latent_steps: int = 105_000
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    ema_m_start: float = 0.996
    ema_m_end: float = 0.9999
    ema_schedule_steps: int = 105_000
    lambda_e_reg: float = 0.02
    lambda_c_reg: float = 0.10
    sigreg_m: int = 1024
    sigreg_knots: int = 17
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
    dataset: str = "ssv2_tiny"
    num_workers: int = 8
    pin_memory: bool = True

    @property
    def full_root(self) -> str:
        """Return the full SSv2 symlink root.

        Returns:
            Path string for `/workspace/data/ssv2` or the JEPA_DATA_ROOT equivalent.
        """
        return str(Path(self.data_root) / "ssv2")

    @property
    def tiny_root(self) -> str:
        """Return the SSv2-tiny symlink root.

        Returns:
            Path string for `/workspace/data/ssv2_tiny` or the JEPA_DATA_ROOT equivalent.
        """
        return str(Path(self.data_root) / "ssv2_tiny")

    def dataset_root(self) -> str:
        """Return the selected dataset root for CLI dataset names.

        Returns:
            Path string for `ssv2` or `ssv2_tiny` under `data_root`.
        """
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
