"""Configuration for HJEPA-VWM Phase 1 (frozen encoder).

Naming map from AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md §3.
Architecture reference: AGENT_FILES/AGENTS.md and GUIDES/latest_brief.md (narrative only).
Typed fallback defaults live in this file; the canonical editable recipe is
``configs/train.yaml``. Seven reviewed scientific fields retain direct CLI overrides.
"""

from __future__ import annotations

import hashlib
import math
import os
import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

# Shared ImageNet constants used by the tested V-JEPA2 and DINOv3 adapters.
# SigLIP owns its separate 0.5/0.5 mapping inside encoders.py.
ENCODER_IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
ENCODER_IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class EncoderConfig:
    """Encoder-seam inputs; the private registry resolves family and repository.

    ``revision=None`` means "use the alias's tested immutable default". It never
    means the mutable Hub ``main`` branch.
    """

    alias: Literal["vjepa2_vitl16", "siglip2_vitb16", "dinov3_vitb16"] = "vjepa2_vitl16"
    revision: str | None = None
    input_frames: int = 8
    input_height: int = 256
    input_width: int = 256
    precision: Literal["bf16", "fp32"] = "bf16"
    frame_microbatch: int | None = 8
    attention_implementation: Literal["sdpa", "eager"] = "sdpa"
    hf_cache_dir: str = "/workspace/hf_cache"


@dataclass
class ModelConfig:
    """Trainable Phase 1 architecture plus a narrow legacy V-JEPA shape bridge.

    Production code resolves all detailed-feature geometry from ``EncoderSpec``.
    The encoder fields/properties below remain only for historical checkpoint
    reconstruction and older small-geometry unit tests; they are not consulted
    when a real encoder is selected.
    """

    # Deprecated compatibility geometry for historical V-JEPA checkpoints/tests.
    # New runtime paths must use cfg.encoder plus the factory-resolved EncoderSpec.
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
    # Complete internal memory/slot width. The bottleneck stays at this width
    # through all latent blocks, then projects once to the external d_c code.
    bottleneck_mixer_dim: int = 256
    bottleneck_convnext_blocks: int = 2
    bottleneck_cross_attn_heads: int = 8
    # Perceiver-style latent processor depth (tmp/changes_bottleneck <1>): each block
    # runs s += CrossAttn(s, memory); s += SelfAttn(s); s += MLP(s) on the slot stream.
    # The old single-read bottleneck is depth-1 without slot self-attention; repeated
    # refinement + slot competition is the recommended 2-4 range. Every residual
    # branch is zero-init, so the internal slot stream equals the learned queries
    # at initialization at any depth; Bottleneck then applies its final projection/norm.
    bottleneck_latent_blocks: int = 3
    # Query/residual init is baked into models.py (Plan Phase 04, KANBAN/04):
    # orthogonal queries (Fix 1, fixes the ~0.02-scale near-uniform-attention
    # pathology) and zero-init residual branches in every latent block (Fix 2,
    # identity-residual start). These are fixes, not switches — earlier inits
    # live in git history if ever needed.

    # Coarse flow F_c
    f_c_blocks: int = 6
    f_c_dim: int = 256
    f_c_heads: int = 8
    condition_dropout: float = 0.10

    # Reconstruction decoder D (reconstruction anchor, option 1). A deliberately
    # small fixed-position cross-attention expander: c_t (N_c x D_c) -> e_hat
    # (N_e x D_e). Fixed lattice position codes tell D where to write; c-derived
    # values tell it what to write. There is no trainable per-output-token content
    # query table. It exists only to supply an information-richness gradient to B,
    # NOT as a showpiece generator (that is the separate Phase 3 frame generator on
    # pixels). Generic by design so the through-F_c "future anchor" (option 3) is a
    # caller change, not a rewrite. See KANBAN investigation_011.
    decoder_dim: int = 256
    decoder_blocks: int = 2
    decoder_heads: int = 8

    # NOTE: no detailed-token dropout (removed in v0.2 — frozen encoder).
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
    """Phase 1 training and diagnostic defaults (see AGENT_FILES/AGENTS.md §12)."""

    # Atomic trainability contract: joint updates B/F_c/D + B_EMA; fc_only updates only F_c.
    optimization_scope: Literal["joint", "fc_only"] = "joint"
    global_batch: int = 64
    # Step counts retuned 2026-06-10 after `peachy-terrain-5` crashed at step
    # 10750 (see AGENT_FILES/KANBAN/02-LAUNCH-FULL-PHASE-1-RUN/POSTMORTEM_RUN1.md).
    # The original 30k+10k-warmup spec was calibrated for the full SSv2 corpus
    # (~170k clips => ~11 epochs); on ssv2_tiny (~4k clips) 30k = ~480 epochs,
    # extreme overtraining. Sized down to a 15k+1.5k run that takes ~5 h on
    # ssv2_tiny and is still ~240 epochs of the subset.
    stage1_steps: int = 15_000
    max_steps: int = 15_000  # Phase 1 trains ONLY Stage 1
    # NOTE: no lr_encoder — encoder is frozen.
    # Peak LRs halved after the run-1 explosion at step ~10550 (peak lr_mult).
    # With bf16 + AdamW(beta2=0.95) + var-floor reg, the old 4e-4 peak was on
    # the edge of stable; one large-grad batch at peak destroyed precision.
    lr_bottleneck: float = 1e-4
    lr_coarse_flow: float = 2e-4
    warmup_steps: int = 1_500
    total_latent_steps: int = 105_000  # cosine LR / EMA denominator (Phase 2+)
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    # Tighter clip + skip-step guard added 2026-06-10. The clip-by-norm at 1.0
    # divides the gradient by `grad_norm / 1.0` to clip; in bf16 that division
    # destroys direction precision once the divisor is more than a few hundred.
    # AGC (below) clips moderate per-tensor spikes first; `grad_skip_threshold`
    # is a tail guard on the post-AGC global norm (see train.train_step).
    grad_clip: float = 0.5
    # Raised from 50 after elated-snowflake-15: healthy pre-break max was ~31;
    # first skip at 64.7 with L_flow≈1.9. AGC handles the 30–100 band; skip
    # only true blowups (8500 spike ~170).
    grad_skip_threshold: float = 150.0
    # Adaptive Gradient Clipping (AGC): per-tensor ||g|| <= λ(||w|| + eps).
    # Calibrated from elated healthy phase (grad p95≈3); tune λ via logged
    # agc_*_max_ratio (target ~5–10% of steps with agc_*_clipped > 0).
    agc_enabled: bool = True
    agc_lambda_bottleneck: float = 0.20
    agc_lambda_coarse_flow: float = 0.10
    agc_eps: float = 1e-3
    # Log-only early warning (elated: step 8400 had grad≈31, L_flow≈1.35).
    instability_warn_grad_norm: float = 30.0
    instability_warn_l_flow: float = 1.0
    ema_m_start: float = 0.996
    ema_m_end: float = 0.9999
    ema_schedule_steps: int = 105_000
    lambda_var: float = 0.5  # variance-floor weight (VICReg V); default ON
    var_floor_std_target: float = 1.0  # hinge target in L_var
    # SIGReg (LeJEPA isotropic-Gaussian regularizer) on c_t — investigation_008.
    # Pushes the pooled c_t distribution toward N(0, I), the provably risk-optimal
    # embedding law; isotropy maximizes effective rank by construction, attacking the
    # c_effective_rank≈13/256 utilization ceiling the one-sided variance floor can't
    # move. Default 0.0 -> L_sigreg is computed for logging but NOT added to the loss,
    # so the baseline is byte-identical. The var floor stays on (λ_var=0.5) as a
    # non-interfering safety net (its hinge is inactive once std≥1, where SIGReg lands).
    lambda_sigreg: float = 0.0
    # SIGReg warmup (investigation_010, Issue 7). SIGReg hits the ONLINE bottleneck
    # immediately, so a strong λ_sigreg can move c_t's coordinate system faster than
    # the EMA target B_EMA (and hence the flow target c_plus) can follow — giving F_c a
    # moving input/output geometry (online condition vs. lagged EMA target) and the
    # rank-improves-but-flow-plateaus pattern. Ramp the SIGReg weight linearly over
    # this many steps (same idea as recon_warmup_steps) so the embedding law is shaped
    # gradually and the EMA keeps up. Only active when lambda_sigreg > 0; the diag
    # readout logs effective rank/std for BOTH online c_t and EMA target c_plus so a
    # lagging target can't masquerade as a healthy online rank.
    sigreg_warmup_steps: int = 2_000
    # VICReg-C off-diagonal covariance penalty on c_t (Plan Phase 04, anti-collapse).
    # Default 0.01 -> term is active in the loss. Set 0.0 to compute L_cov for logging
    # only. Calibrate against L_cov's magnitude; watch copy-ratio.
    lambda_cov: float = 0.01
    # Within-video slot-diversity penalty on c_t (Plan Phase 04, anti slot-collapse).
    # Default 0.0 -> term computed for logging (L_slot) but NOT added to loss.
    # Nonzero is the primary knob after Run A showed c_slot_diversity_rank≈1.6/32.
    lambda_slot: float = 0.0
    # Reconstruction anchor (option 1: gradient through B only, NOT through F_c).
    # Decode c_t back to the frozen detailed features e_t and penalize per-token
    # cosine distance, forcing c_t to stay information-rich without letting D game
    # the feature norm. Default 0.0 -> the decoder is NOT run in the train step
    # (avoids the heavy N_ctx x D_e forward) so the baseline is byte-identical;
    # L_recon_* readouts are still logged at diag cadence. Nonzero ramps linearly
    # over recon_warmup_steps because early training is the fragile phase
    # (royal-cherry-17 8600 cliff). The through-F_c future anchor (option 3) is
    # intentionally NOT wired; the split diag readouts (present / c_plus / c_hat)
    # scope whether it is warranted.
    lambda_recon: float = 0.0
    # Reconstruction loss formula switch. "cosine" is the current norm-invariant
    # per-token objective; "relative_mse" restores the legacy MSE / Var(e)
    # objective used before the norm-cheating fix.
    recon_loss_mode: Literal["cosine", "relative_mse"] = "cosine"
    # Present-only reconstruction bottleneck test. When True, Stage 1 trains
    # D(B(e_t)) -> e_t and skips F_c / future prediction losses entirely. Optional
    # non-prediction regularizers (variance, SIGReg, cov, slot) still follow their
    # weights, so a pure reconstruction run should set those weights explicitly.
    present_recon_only: bool = False
    recon_warmup_steps: int = 2_000
    lr_decoder: float = 1e-4  # peak LR for the reconstruction decoder D
    agc_lambda_decoder: float = 0.20  # AGC λ for D (mirrors the bottleneck)
    # Prediction-side reconstruction anchor (option 3, the VITA-style joint objective).
    # Decode the PREDICTED future latent c_hat back to the future detailed features
    # e_{t+k} and penalize per-token cosine distance, with the gradient flowing
    # THROUGH F_c — and into B via the F_c conditioning on c_t (not detached) — so
    # the objective rewards a c that is PREDICTABLE, not merely reconstructable.
    # Runs ALONGSIDE the present anchor (lambda_recon), reusing the same decoder D
    # and the same recon_warmup ramp.
    # fanciful-lake-18 motivated this: option 1 removed the cliff but left the copy gate
    # failing (F_c loses to copy) while L_recon_chat ≈ L_recon_cplus showed present-anchor
    # recon is blind to prediction error. Default 0.0 => prediction branch not run
    # (byte-identical to the option-1 baseline); with it on, the diag readout
    # L_recon_chat should DROP. See KANBAN investigation_006/fanciful-lake-18/NEXT_STEPS.
    lambda_recon_pred: float = 0.0
    # Residual reconstruction target (investigation_013, the run-052 fix). When True,
    # the reconstruction anchors train D against the PER-POSITION RESIDUAL e - e_mean
    # instead of the absolute frozen features e, where e_mean is an EMA per-lattice-
    # position mean of e_t tracked across training batches (models.FeatureMeanTracker).
    # Run 052 proved the absolute cosine objective is ~85% satisfiable by a video-
    # independent "template" (decode the average feature field), so reconstruction
    # pressure never had to route video-specific information through c_t and the
    # representation collapsed (rank 13, cross-video cosine 0.91). Subtracting the
    # per-position mean makes the template worth exactly zero: every unit of recon
    # improvement must come from video-specific content. Applies to BOTH anchors
    # (present e_t and prediction-side e_{t+k}; same tracker — the two clips share
    # the encoder feature distribution). Default False -> byte-identical baseline
    # (tracker constructed but never updated or used). Orthogonal to predict_residual
    # (that is the TEMPORAL residual for F_c; this is a PER-POSITION feature residual
    # for D). Requires lambda_recon > 0 or lambda_recon_pred > 0.
    recon_residual_target: bool = False
    # EMA momentum for the per-position feature mean. First batch initializes the
    # mean directly; afterwards mean <- m * mean + (1 - m) * batch_mean. At 0.99 the
    # mean is ~99% converged in ~460 steps, well inside recon_warmup_steps.
    recon_mean_momentum: float = 0.99
    # Residual prediction (investigation_009). When True, F_c predicts the TEMPORAL
    # residual Δ = c_{t+k} - c_t (both from B_EMA -> a purely temporal target) instead of
    # the full future latent c_{t+k}, and the option-3 recon add-back becomes ĉ = c_t + Δ̂.
    # The flow noise eps_c is scaled to Δ's std so the rectified-flow velocity target
    # (Δ - eps) isn't noise-dominated (‖Δ‖/‖c‖ ≈ 0.38). The copy baseline becomes "predict
    # zero residual" (copy_loss = ‖Δ‖²), so coarse_vs_copy_ratio stays directly comparable
    # to the full-latent runs. Default False -> full-latent prediction, byte-identical.
    predict_residual: bool = False
    # Rectified-flow source endpoint. ``noise`` is the historical, unchanged default.
    # ``present`` uses the detached present target-bottleneck latent as x_0 and the
    # detached future target-bottleneck latent as x_1. It is intentionally incompatible
    # with predict_residual, which already changes the target semantics.
    flow_source: str = "noise"
    # Optional experimental fixed-coordinate contract. Loads only the saved ONLINE
    # bottleneck and freezes it for condition and future target. Empty preserves the
    # complete historical noise-path behavior.
    flow_bottleneck_checkpoint: str = ""
    # Fixed offline whitening of selected frozen features (investigation_014 /
    # tmp/changes_bottleneck <2>). The original V-JEPA rank probe showed `e` is
    # strongly anisotropic (pooled entropy rank ~193/1024 with a long low-energy tail), so the bottleneck
    # and decoder otherwise compress dominant-direction energy AND tail noise with the
    # same weighting. When True, every encoder output (context AND future clip) is
    # mapped e_w = (e - mu) @ W with W = U (Lambda + eps I)^{-1/2} U^T computed ONCE
    # offline over the training set (whiten_stats.py) — B, B_EMA, F_c targets, and D
    # all live in whitened space; unwhitening exists only for readouts that need raw
    # raw encoder space. NEVER per-batch whitening: mu/U/Lambda are fixed training-set
    # statistics reused for train/val/inference. Default False -> byte-identical
    # baseline (whitener never built).
    whiten_features: bool = False
    # Path to the stats file written by whiten_stats.py ({"mean","eigvals","eigvecs"}).
    # Required (and validated in finalize_training_config) when whiten_features=True.
    whiten_stats_path: str = ""
    # Exact number of clips that the selected whitening artifact must represent.
    # This is identity-bearing: a tiny smoke artifact may never satisfy a paid run
    # simply because its feature dimension and filename happen to match.
    whiten_expected_clips: int = 12_800
    # Eigenvalue floor added to Lambda before the inverse square root so tiny-variance
    # tail directions do not explode to huge whitened coordinates. Applied at whitener
    # build time, so eps can be swept without recomputing the offline stats.
    whiten_eps: float = 1e-4
    horizon_k: int = 4  # single fixed horizon for Phases 1-3 (Phase 4: multi-horizon)
    frame_stride: int = 2
    precision: Literal["bf16", "fp32"] = "bf16"
    log_every: int = 50
    diag_every: int = 500
    checkpoint_every: int = 2_500  # finer checkpoints on the shorter 15k run


@dataclass
class DataConfig:
    """Dataset roots; only JEPA_DATA_ROOT overrides the /workspace path contract."""

    data_root: str = field(
        default_factory=lambda: os.environ.get("JEPA_DATA_ROOT", "/workspace/data")
    )
    dataset: Literal["ssv2", "ssv2_tiny", "ego4d", "ego4d_tiny"] = (
        "ssv2_tiny"  # YAML field with a hot CLI override.
    )
    num_workers: int = 8
    pin_memory: bool = True
    # Retained acquisition manifests are provenance inputs, not discovery hints.
    # Keep these explicit so a pod relocation cannot silently bind a different
    # EGO4D selection or authoritative download tier.
    ego4d_selection_manifest: str = "/workspace/ego4d_raw/manifests/selection_manifest.json"
    ego4d_download_manifest: str = "/workspace/ego4d_raw/video_540ss_manifest.csv"

    @property
    def full_root(self) -> str:
        """Full SSv2 symlink root, `<data_root>/ssv2`."""
        return str(Path(self.data_root) / "ssv2")

    @property
    def tiny_root(self) -> str:
        """SSv2-tiny symlink root, `<data_root>/ssv2_tiny`."""
        return str(Path(self.data_root) / "ssv2_tiny")

    @property
    def ego4d_root(self) -> str:
        """EGO4D chunk root, `<data_root>/ego4d` (real files from chunk_ego4d.py)."""
        return str(Path(self.data_root) / "ego4d")

    @property
    def ego4d_tiny_root(self) -> str:
        """EGO4D-tiny symlink root, `<data_root>/ego4d_tiny` (make_ego4d_subset.py)."""
        return str(Path(self.data_root) / "ego4d_tiny")

    def dataset_root(self) -> str:
        """Return the selected dataset root for the configured dataset name."""
        if self.dataset == "ssv2":
            return self.full_root
        if self.dataset == "ssv2_tiny":
            return self.tiny_root
        if self.dataset == "ego4d":
            return self.ego4d_root
        if self.dataset == "ego4d_tiny":
            return self.ego4d_tiny_root
        raise ValueError(f"Unsupported dataset: {self.dataset}")


@dataclass
class Config:
    """Top-level Phase 1 configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    debug_shapes: bool = True
    checkpoint_dir: str = "/workspace/checkpoints"
    # Kept as a top-level constructor field for current callers/checkpoint shape.
    # __post_init__/__setattr__ synchronize it with EncoderConfig's identity field.
    hf_cache_dir: str | None = None
    seed: int = 42
    experiment_config_path: str = ""
    experiment_config_sha256: str = ""

    def __post_init__(self) -> None:
        """Resolve the legacy top-level cache field to one synchronized value."""
        selected = self.encoder.hf_cache_dir if self.hf_cache_dir is None else self.hf_cache_dir
        self.hf_cache_dir = selected

    def __setattr__(self, name: str, value: object) -> None:
        """Keep post-construction top-level cache assignments synchronized."""
        object.__setattr__(self, name, value)
        if name == "hf_cache_dir" and value is not None:
            encoder = self.__dict__.get("encoder")
            if encoder is not None:
                encoder.hf_cache_dir = str(value)


@dataclass
class RuntimeConfig:
    """Operator settings loaded from the experiment YAML rather than scientific CLI flags."""

    mode: Literal["train", "stage0", "preflight", "resource_preflight"] = "train"
    resume: str | None = None
    warm_start_from: str | None = None
    provenance_out: str | None = None
    compare_provenance: tuple[str, str] | None = None
    require_wandb: bool = False
    reset_optimizer: bool = False
    allow_dataset_transfer: bool = False
    allow_legacy_checkpoint: bool = False


@dataclass
class WandbConfig:
    """W&B identity associated with one YAML-defined experiment recipe."""

    entity: str | None = None
    project: str = "hjepa-vwm"
    group: str | None = None
    name: str | None = None
    run_id: str | None = None


@dataclass
class ExperimentConfig:
    """Resolved training, runtime, and tracking configuration loaded from one YAML file."""

    config: Config
    runtime: RuntimeConfig
    wandb: WandbConfig
    source_path: str


_NON_RECIPE_FIELDS: dict[str, frozenset[str]] = {
    "config": frozenset(
        {
            "experiment_config_path",
            "experiment_config_sha256",
            "hf_cache_dir",
        }
    ),
    "config.model": frozenset(
        {
            "encoder_repo",
            "encoder_frozen",
            "encoder_patch",
            "encoder_tubelet",
            "f_c_dim",
            "t_ctx",
            "h",
            "w",
            "d_e",
        }
    ),
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    """Construct one mapping without PyYAML's last-duplicate-wins behavior."""
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise TypeError("Experiment YAML mapping keys must be scalar values") from exc
        if duplicate:
            line = key_node.start_mark.line + 1
            raise ValueError(f"Duplicate YAML key {key!r} at line {line}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _coerce_config_value(value: Any, annotation: Any, location: str) -> Any:
    """Validate one YAML value against its dataclass annotation."""
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in {types.UnionType, Union}:
        if value is None and type(None) in arguments:
            return None
        candidates = [candidate for candidate in arguments if candidate is not type(None)]
        errors = []
        for candidate in candidates:
            try:
                return _coerce_config_value(value, candidate, location)
            except TypeError as exc:
                errors.append(str(exc))
        raise TypeError(errors[0] if errors else f"{location} has an unsupported value")
    if value is None:
        raise TypeError(f"{location} may not be null")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{location} must be a list")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_coerce_config_value(item, arguments[0], location) for item in value)
        if len(value) != len(arguments):
            raise TypeError(f"{location} must contain exactly {len(arguments)} values")
        return tuple(
            _coerce_config_value(item, expected, f"{location}[{index}]")
            for index, (item, expected) in enumerate(zip(value, arguments, strict=True))
        )
    if origin is Literal:
        if value not in arguments:
            choices = ", ".join(repr(choice) for choice in arguments)
            raise ValueError(f"{location} must be one of: {choices}; got {value!r}")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{location} must be a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{location} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{location} must be a number")
        coerced = float(value)
        if not math.isfinite(coerced):
            raise ValueError(f"{location} must be finite")
        return coerced
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"{location} must be a string")
        return value
    return value


def _apply_config_mapping(target: object, values: dict[str, Any], section: str) -> None:
    """Apply a strict YAML mapping recursively to one dataclass instance.

    Unknown keys fail before training so a misspelled hyperparameter cannot silently waste a run.

    Args:
        target: Dataclass instance whose existing defaults are being overridden.
        values: YAML mapping for this section.
        section: Dotted section name used in validation errors.
    Returns:
        None.
    """
    forbidden = sorted(set(values) & _NON_RECIPE_FIELDS.get(section, frozenset()))
    if forbidden:
        raise ValueError(
            f"{section}.{forbidden[0]} is not configurable from experiment YAML; "
            "it is a legacy, derived, or audit-owned field"
        )
    allowed = {item.name for item in fields(target)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown experiment config key(s) in {section}: {', '.join(unknown)}")
    annotations = get_type_hints(type(target))
    for name, value in values.items():
        current = getattr(target, name)
        location = f"{section}.{name}"
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{location} must be a mapping")
            _apply_config_mapping(current, value, location)
        else:
            setattr(target, name, _coerce_config_value(value, annotations[name], location))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load one strict YAML recipe over the shipped dataclass defaults.

    The YAML may be partial, but every supplied key must name a real dataclass field. Scientific
    command-line overrides are applied later by ``train.py`` only for the seven active sweep axes.

    Args:
        path: YAML file containing Config sections plus optional ``runtime`` and ``wandb`` maps.
    Returns:
        Resolved experiment configuration and its source path.
    """
    config_path = Path(path)
    payload = yaml.load(
        config_path.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError("Experiment config root must be a mapping")
    runtime_values = payload.pop("runtime", {})
    wandb_values = payload.pop("wandb", {})
    if not isinstance(runtime_values, dict):
        raise TypeError("runtime must be a mapping")
    if not isinstance(wandb_values, dict):
        raise TypeError("wandb must be a mapping")
    config = Config()
    runtime = RuntimeConfig()
    wandb_config = WandbConfig()
    _apply_config_mapping(config, payload, "config")
    config.hf_cache_dir = config.encoder.hf_cache_dir
    _apply_config_mapping(runtime, runtime_values, "runtime")
    _apply_config_mapping(wandb_config, wandb_values, "wandb")
    valid_modes = {"train", "stage0", "preflight", "resource_preflight"}
    if runtime.mode not in valid_modes:
        choices = ", ".join(sorted(valid_modes))
        raise ValueError(f"runtime.mode must be one of {choices}; got {runtime.mode!r}")
    config.experiment_config_path = str(config_path.resolve())
    config.experiment_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return ExperimentConfig(
        config=config,
        runtime=runtime,
        wandb=wandb_config,
        source_path=str(config_path.resolve()),
    )
