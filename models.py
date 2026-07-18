"""Encoder-independent Phase 1 trainable model modules.

The frozen detailed encoder lives exclusively in :mod:`encoders`. This module
constructs the trainable bottleneck `B`, EMA bottleneck `B_EMA`, coarse flow `F_c`,
and reconstruction decoder `D` from a resolved :class:`encoders.EncoderSpec`.
Fine flow, the VAE, and the frame generator remain outside Phase 1.
"""

from __future__ import annotations

import math
from copy import deepcopy

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - local docs-only environments.
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from config import Config, ModelConfig
from encoders import EncoderSpec, FeatureLayout, build_frozen_encoder
from losses import as_target


def _require_torch() -> None:
    """Fail fast when Phase 1 model code is used without PyTorch installed."""
    if torch is None:
        raise RuntimeError("PyTorch is required for models.py. Install requirements.txt on RunPod.")


def _legacy_encoder_spec(cfg: ModelConfig) -> EncoderSpec:
    """Build the narrow V-JEPA-shaped spec used only by legacy/offline callers.

    Production construction resolves a real spec from :func:`build_frozen_encoder`.
    This bridge keeps old checkpoints and tiny synthetic tests readable while they
    migrate; no adapter selection or preprocessing decision depends on it.
    """
    return EncoderSpec(
        family="legacy-vjepa2",
        repo_id=cfg.encoder_repo,
        requested_revision="b3c1679b7c34d3255ef3547f27c7b226aefab26f",
        resolved_revision="b3c1679b7c34d3255ef3547f27c7b226aefab26f",
        input_frames=8,
        input_height=256,
        input_width=256,
        feature_dim=cfg.d_e,
        layout=FeatureLayout(
            temporal=cfg.n_temporal_tokens,
            height=cfg.grid_spatial,
            width=cfg.grid_spatial,
            order="time_y_x",
            temporal_unit="tubelet",
            temporal_stride_frames=cfg.encoder_tubelet,
            temporal_support_frames=cfg.encoder_tubelet,
        ),
        normalization_id="legacy-pre-normalized",
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225),
        preprocess_version="legacy-model-config-v1",
        inference_precision="fp32",
        frame_microbatch=8,
        attention_implementation="sdpa",
        cache_dir="/workspace/hf_cache",
        parameter_count=0,
    )


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style 2D mixer for one temporal-slot token grid.

    Args:
        grid: (B*, C, H_grid, W_grid) per-temporal-slot spatial token grid.
    Returns:
        mixed: (B*, C, H_grid, W_grid) grid after depthwise + pointwise mixing.
    """

    def __init__(self, dim: int, mlp_ratio: int = 4):
        """Initialize depthwise and pointwise layers for one ConvNeXt-style block."""
        _require_torch()
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * mlp_ratio)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, grid: Tensor) -> Tensor:
        """Mix local spatial detail on a per-temporal-slot token grid.

        Local spatial context before query compression helps `c_t` retain
        future-relevant structure rather than raw per-token noise.

        Args:
            grid: (B*, dim, H_grid, W_grid) per-temporal-slot grid.
        Returns:
            mixed: (B*, dim, H_grid, W_grid) grid after ConvNeXt-style mixing.
        """
        residual = grid
        x = self.dwconv(grid).permute(0, 2, 3, 1)
        x = self.pw2(self.act(self.pw1(self.norm(x))))
        return residual + x.permute(0, 3, 1, 2)


class SharpCrossAttention(nn.Module):
    """Cosine cross-attention with a learned sharpness temperature.

    Normalizing queries and memory keys makes slot-token matching directional
    rather than norm-driven. The learned logit scale starts at CLIP-like
    temperature 1/0.07, high enough that each slot can select among a large
    detailed-token memory instead of averaging it uniformly.

    Args:
        q: (B, N_q, D) query slots.
        kv: (B, N_kv, D) memory tokens.
    Returns:
        Tuple `(out, attn)`, where `out` is (B, N_q, D) and `attn` is either
        (B, num_heads, N_q, N_kv) when requested or `None`.
    """

    def __init__(self, dim: int, num_heads: int):
        """Initialize projections and the learned cosine-attention temperature."""
        _require_torch()
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.h = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        # The attention branch begins as an exact no-op, so Bottleneck starts as
        # normalized learned slot identities and input-dependent evidence grows in.
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        self.o_proj.is_zero_init = True

    def forward(self, q: Tensor, kv: Tensor, need_weights: bool = False):
        """Attend from query slots to memory tokens with per-head cosine logits.

        Args:
            q: (B, N_q, D) query slots.
            kv: (B, N_kv, D) memory tokens used for both keys and values.
            need_weights: When True, return per-head attention maps for diagnostics.
        Returns:
            `(out, attn)` where `out` is (B, N_q, D) and `attn` is
            (B, num_heads, N_q, N_kv) if requested, else `None`.
        """
        b, nq, d = q.shape
        head_dim = d // self.h
        q = self.q_proj(q).reshape(b, nq, self.h, head_dim).transpose(1, 2)
        k = self.k_proj(kv).reshape(b, kv.shape[1], self.h, head_dim).transpose(1, 2)
        v = self.v_proj(kv).reshape(b, kv.shape[1], self.h, head_dim).transpose(1, 2)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        attn = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, nq, d)
        return self.o_proj(out), (attn if need_weights else None)


class BottleneckLatentBlock(nn.Module):
    """One Perceiver-style latent refinement block for the bottleneck slot stream.

    Implements the recommended latent-processor step (tmp/changes_bottleneck <1>):

    ```text
    s = s + CrossAttn(s, memory)   # read new evidence from detailed tokens
    s = s + SelfAttn(s)            # slot competition / de-duplication
    s = s + MLP(s)                 # per-slot refinement
    ```

    Every residual branch starts as an exact no-op — the sharp cross-attention
    zero-inits its `o_proj`, and this block zero-inits the self-attention output
    projection and the MLP's last layer (all tagged `is_zero_init` so AGC and
    weight decay hold them out, matching the adaLN-Zero convention). Stacking any
    number of blocks therefore preserves the bottleneck's identity-at-init
    contract: `c == LayerNorm(queries)` for every input at step 0.

    Args:
        slots: (B, N_c, D) current slot stream.
        memory: (B, N_kv, D) mixed/position-tagged detailed tokens.
    Returns:
        Tuple `(slots, attn)` — updated slots (B, N_c, D) and the per-head
        cross-attention weights (B, num_heads, N_c, N_kv) or `None`.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        """Initialize the cross-attention read, slot self-attention, and MLP."""
        _require_torch()
        super().__init__()
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = SharpCrossAttention(dim, num_heads)  # o_proj zero-init inside
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # Identity start for the slot-competition branch (mirrors SharpCrossAttention's
        # zero o_proj): only out_proj is zeroed/tagged — the in-projections stay live so
        # gradient can open the branch, and they remain ordinary decayed/AGC'd weights.
        nn.init.zeros_(self.self_attn.out_proj.weight)
        nn.init.zeros_(self.self_attn.out_proj.bias)
        self.self_attn.out_proj.is_zero_init = True
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # Identity start for the refinement MLP (Plan Phase 04 Fix 2 convention).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].is_zero_init = True

    def forward(self, slots: Tensor, memory: Tensor, *, need_weights: bool = False):
        """Run one read/compete/refine slot update.

        Args:
            slots: (B, N_c, D) slot stream entering the block.
            memory: (B, N_kv, D) detailed memory tokens (keys/values).
            need_weights: Return this block's per-head cross-attention map.
        Returns:
            `(slots, attn)` with attn (B, num_heads, N_c, N_kv) when requested.
        """
        attended, attn = self.cross_attn(self.norm_cross(slots), memory, need_weights=need_weights)
        slots = slots + attended
        h = self.norm_self(slots)
        competed, _ = self.self_attn(h, h, h, need_weights=False)
        slots = slots + competed
        slots = slots + self.mlp(slots)
        return slots, attn


class Bottleneck(nn.Module):
    """Compress detailed tokens into the low-bandwidth abstract latent `c_t`.

    Projects the frozen `e_t` from its resolved `D_e` to the configured internal
    width, mixes each resolved temporal/spatial grid with shared ConvNeXt blocks,
    adds learned memory position tags, then keeps memory and slots at that width
    through a small Perceiver-style latent processor. Its orthogonal slot identities
    are repeatedly updated by sharpened cosine
    cross-attention reads from the detailed-token memory, slot self-attention
    (competition), and a per-slot MLP (`bottleneck_latent_blocks` rounds,
    tmp/changes_bottleneck <1>). All updates are zero-init residuals on the slot
    identities. One final learned projection maps the refined slots to external
    `D_c` when the internal width differs; the default equal-width path uses an
    exact identity for strict checkpoint compatibility. The same module (and its EMA
    copy) is applied identically to `e_t` (-> `c_t`) and `e_plus` (-> `c_plus`);
    both clips share the selected encoder's resolved time-major geometry, so there
    is no separate target path and no kept-mask in v0.2.
    """

    def __init__(self, cfg: ModelConfig, encoder_spec: EncoderSpec | None = None):
        """Initialize wide memory/slots, latent refinement, and final code projection."""
        _require_torch()
        super().__init__()
        self.cfg = cfg
        self.encoder_spec = encoder_spec or _legacy_encoder_spec(cfg)
        layout = self.encoder_spec.layout
        if layout.order != "time_y_x":
            raise ValueError("Bottleneck requires time_y_x detailed-token order.")
        mix = cfg.bottleneck_mixer_dim
        self.in_proj = nn.Linear(self.encoder_spec.feature_dim, mix)
        self.mixers = nn.Sequential(
            *[ConvNeXtBlock(mix) for _ in range(cfg.bottleneck_convnext_blocks)]
        )
        self.pos_emb = nn.Parameter(torch.empty(1, layout.n_tokens, mix))
        nn.init.trunc_normal_(self.pos_emb, std=0.5)
        self.to_kv = nn.Linear(mix, mix)
        # Fix 1 (Plan Phase 04): orthogonal queries (unit-norm rows). The old
        # v0.2 `randn * 0.02` made the q.k logits tiny, so softmax was near-
        # uniform for every slot and all 32 slots read out ~the mean token.
        # Unit-norm rows fix the scale and start the slots decorrelated.
        q = torch.empty(cfg.n_c, mix)
        nn.init.orthogonal_(q)
        self.queries = nn.Parameter(q)
        if cfg.bottleneck_latent_blocks < 1:
            raise ValueError(
                f"bottleneck_latent_blocks must be >= 1; got {cfg.bottleneck_latent_blocks}"
            )
        # Keep the complete read/compete/refine stream at the configured internal
        # width. Compressing to D_c before these input-dependent updates would make
        # bottleneck_mixer_dim a ConvNeXt-only width rather than a true memory width.
        self.latent_blocks = nn.ModuleList(
            [
                BottleneckLatentBlock(mix, cfg.bottleneck_cross_attn_heads)
                for _ in range(cfg.bottleneck_latent_blocks)
            ]
        )
        if mix == cfg.d_c:
            # No parameters/state keys on the shipped 256-wide path: historical
            # checkpoints remain strict-load compatible and numerically unchanged.
            self.abstract_proj = nn.Identity()
        else:
            self.abstract_proj = nn.Linear(mix, cfg.d_c)
            nn.init.orthogonal_(self.abstract_proj.weight)
            nn.init.zeros_(self.abstract_proj.bias)
        self.norm = nn.LayerNorm(cfg.d_c)

    def forward(self, detailed: Tensor, *, return_attn: bool = False):
        """Compress detailed tokens to abstract tokens.

        Args:
            detailed: `(B,spec.layout.n_tokens,spec.feature_dim)` frozen-encoder
                tokens for a context or future clip.
            return_attn: When True, also return the PER-HEAD cross-attention
                weights (B, num_heads, N_c, N_ctx) of the FINAL latent block's
                read (the most refined attention pattern) for diagnostics.
                Per-head (not head-averaged) matters: 8 sharp-but-different heads
                average out to look uniform, so head-averaged entropy masks real
                selectivity. The training path leaves this False so the fast
                (no-weights) attention kernel is used.
        Returns:
            abstract: (B, N_c, D_c) abstract latent. When `return_attn`,
            a tuple `(abstract, attn_weights)` with attn (B, num_heads, N_c, N_ctx).
        """
        b, n, d = detailed.shape
        cfg = self.cfg
        layout = self.encoder_spec.layout
        if (n, d) != (layout.n_tokens, self.encoder_spec.feature_dim):
            raise ValueError(
                "Detailed features do not match EncoderSpec: expected "
                f"(N,D)=({layout.n_tokens},{self.encoder_spec.feature_dim}), got ({n},{d})."
            )
        mix = cfg.bottleneck_mixer_dim
        tokens = self.in_proj(detailed)
        t, h, w = layout.temporal, layout.height, layout.width
        # Canonical time-major order: index = t*(h*w) + y*w + x.
        grid = tokens.reshape(b * t, h, w, mix).permute(0, 3, 1, 2)
        mixed = self.mixers(grid).permute(0, 2, 3, 1).reshape(b, layout.n_tokens, mix)
        mixed = mixed + self.pos_emb
        memory = self.to_kv(mixed)
        slots = self.queries[None].expand(b, -1, -1)
        attn = None
        last = len(self.latent_blocks) - 1
        for i, block in enumerate(self.latent_blocks):
            slots, block_attn = block(slots, memory, need_weights=(return_attn and i == last))
            if block_attn is not None:
                attn = block_attn
        abstract = self.norm(self.abstract_proj(slots))
        if return_attn:
            return abstract, attn
        return abstract


class TargetBottleneck(nn.Module):
    """EMA copy of the bottleneck producing the detached target abstract latent.

    The encoder is frozen and shared, so v0.2 has NO target encoder — only the
    bottleneck has an EMA copy. This module never receives gradients; it is kept
    aligned with `B` by the EMA rule in `train.py` and produces `c_plus` from the
    frozen `e_plus`.

    Args:
        target_detailed: `(B,spec.layout.n_tokens,spec.feature_dim)` frozen
            encoder output on the future clip.
    Returns:
        target_abstract: (B, N_c=32, D_c=256), detached.
    """

    def __init__(self, bottleneck: Bottleneck):
        """Clone the online bottleneck into a frozen EMA module."""
        _require_torch()
        super().__init__()
        self.bottleneck = deepcopy(bottleneck)
        self.freeze()

    def freeze(self) -> None:
        """Disable backprop through the EMA bottleneck and pin it to eval mode."""
        for param in self.parameters():
            param.requires_grad = False
        self.bottleneck.eval()

    def train(self, mode: bool = True) -> TargetBottleneck:
        """Keep the EMA bottleneck in eval mode regardless of parent train()."""
        super().train(mode)
        self.bottleneck.eval()
        return self

    def copy_weights_from(self, bottleneck: Bottleneck) -> None:
        """Initialize the EMA bottleneck from the online bottleneck at Stage 0."""
        self.bottleneck.load_state_dict(bottleneck.state_dict())
        self.freeze()

    def forward(self, target_detailed: Tensor) -> Tensor:
        """Produce the detached target abstract latent `c_plus`.

        Args:
            target_detailed: (B, N_tgt, D_e) frozen encoder output on the future clip.
        Returns:
            target_abstract: (B, N_c, D_c) abstract latent, stop-gradient.
        """
        with torch.no_grad():
            abstract = self.bottleneck(target_detailed)
        return as_target(abstract)


class AdaLNBlock(nn.Module):
    """DiT-style adaLN-Zero transformer block for the coarse flow.

    Args:
        x: (B, 2*N_c, D_c) concatenated noised and conditioning abstract tokens.
        time_emb: (B, D_c) flow-time embedding.
    Returns:
        hidden: (B, 2*N_c, D_c) updated sequence.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        """Initialize attention, MLP, and zeroed time-modulation layers."""
        _require_torch()
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        # Issue 1/9: flag the zero-init adaLN-Zero modulation projection so it is held
        # out of AGC and weight decay (diagnostics.is_geometry_or_gate_param). These are
        # the gates that must "wake up" from identity; clipping/decaying them at ||w||≈0
        # is exactly what slows the flow predictor.
        self.mod[-1].is_zero_init = True

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        """Apply one adaLN-Zero self-attention/MLP block.

        Zero-initialized modulation keeps the flow block near identity at step 0,
        which stabilizes the rectified-flow velocity predictor.

        Args:
            x: (B, 2*N_c, D_c) concatenated noised and conditioning abstract tokens.
            time_emb: (B, D_c) processed flow-time embedding.
        Returns:
            hidden: (B, 2*N_c, D_c) updated sequence.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mod(time_emb).chunk(
            6, -1
        )
        h = self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa[:, None] * attn
        h = self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * self.mlp(h)
        return x


def _timestep_embedding(tau: Tensor, dim: int) -> Tensor:
    """Embed flow time using sinusoidal Fourier features.

    Args:
        tau: (B,) flow time values in [0, 1].
        dim: Embedding dimension.
    Returns:
        emb: (B, dim) time embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=tau.device) / max(1, half)
    )
    args = tau[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim]


class CoarseFlow(nn.Module):
    """Predict the rectified-flow velocity for future abstract latents.

    Args:
        z_c: (B, N_c=32, D_c=256) noised future abstract latent.
        tau_c: (B,) flow time.
        abstract: (B, N_c=32, D_c=256) current abstract conditioning latent c_t.
    Returns:
        u_c_hat: (B, N_c=32, D_c=256) predicted coarse velocity.
    """

    def __init__(self, cfg: ModelConfig):
        """Initialize the Stage 1 coarse abstract-flow predictor."""
        _require_torch()
        super().__init__()
        self.cfg = cfg
        self.null_condition = nn.Parameter(torch.zeros(cfg.n_c, cfg.d_c))
        # Issue 2: explicit token-role and slot-identity embeddings. The flow
        # transformer otherwise sees 2*N_c undifferentiated tokens and must infer
        # from content alone which N_c are the noised future and which N_c are the
        # c_t conditioning, and which slot index is which. Because the bottleneck
        # slots are LEARNED query slots, slot index carries meaning, so we hand the
        # flow that meaning explicitly:
        #   * slot_pos — a SHARED per-slot code added to both streams, so slot i of
        #     z_c binds to slot i of the conditioning (small-random: identity from
        #     the start, like the bottleneck queries);
        #   * z_type / cond_type — segment codes marking the noised-future vs.
        #     conditioning streams (zero-init: no effect at step 0, an identity start
        #     consistent with the adaLN-Zero gates).
        # All three are learned coordinate systems, so they are held out of weight
        # decay and AGC via _GEOMETRY_LEAF_NAMES (Issues 1 & 9). A horizon embedding
        # would slot in here for the Phase 4 multi-horizon extension.
        self.slot_pos = nn.Parameter(torch.randn(cfg.n_c, cfg.d_c) * 0.02)
        self.z_type = nn.Parameter(torch.zeros(1, 1, cfg.d_c))
        self.cond_type = nn.Parameter(torch.zeros(1, 1, cfg.d_c))
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_c, cfg.d_c * 4), nn.SiLU(), nn.Linear(cfg.d_c * 4, cfg.d_c)
        )
        self.blocks = nn.ModuleList(
            [AdaLNBlock(cfg.d_c, cfg.f_c_heads) for _ in range(cfg.f_c_blocks)]
        )
        self.norm = nn.LayerNorm(cfg.d_c)

    def forward(
        self, z_c: Tensor, tau_c: Tensor, abstract: Tensor, *, condition_drop: Tensor | None = None
    ) -> Tensor:
        """Predict the coarse flow velocity conditioned on the current abstract state.

        Condition dropout replaces `abstract` with a learned null token set so the
        coarse flow stays robust without changing the gradient path through `c_t`.

        Args:
            z_c: (B, N_c, D_c) noised future abstract latent.
            tau_c: (B,) flow time.
            abstract: (B, N_c, D_c) current abstract latent c_t.
            condition_drop: Optional (B,) boolean override for dropout tests.
        Returns:
            u_c_hat: (B, N_c, D_c) predicted rectified-flow velocity.
        """
        if self.training or condition_drop is not None:
            if condition_drop is None:
                condition_drop = (
                    torch.rand(abstract.shape[0], device=abstract.device)
                    < self.cfg.condition_dropout
                )
            null = self.null_condition[None].expand_as(abstract)
            abstract = torch.where(condition_drop[:, None, None], null, abstract)
        # Issue 2: stamp slot identity (shared across both streams) and token-type
        # before fusing the two N_c-token streams. Done on FUNCTION-LOCAL copies — the
        # caller's z_c is untouched, so train_step's rectified-flow endpoint
        # (z_c + (1-tau)*u_c_hat) still uses the raw noised latent. Condition dropout
        # ran first, so a dropped condition still carries its slot/type code.
        z_c = z_c + self.slot_pos[None] + self.z_type
        abstract = abstract + self.slot_pos[None] + self.cond_type
        x = torch.cat([z_c, abstract], dim=1)
        time_emb = self.time_mlp(_timestep_embedding(tau_c, self.cfg.d_c).to(dtype=x.dtype))
        for block in self.blocks:
            x = block(x, time_emb)
        return self.norm(x[:, : self.cfg.n_c])


def _axis_sincos_position_code(position: Tensor, dim: int) -> Tensor:
    """Encode one lattice-coordinate axis with deterministic sin/cos features.

    Args:
        position: (N,) lattice coordinates for one axis.
        dim: Number of decoder channels assigned to this axis.
    Returns:
        code: (N_ctx, dim) non-trainable positional features.
    """
    if dim <= 0:
        return position.new_zeros(position.shape[0], 0)
    half = max(1, math.ceil(dim / 2))
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=position.dtype, device=position.device)
        / max(1, half)
    )
    phase = position[:, None] * freqs[None]
    code = torch.cat([phase.sin(), phase.cos()], dim=-1)
    return code[:, :dim]


def _fixed_detailed_position_codes(layout: FeatureLayout, dim: int) -> Tensor:
    """Build fixed 3D codes for a canonical temporal-major detailed-token grid.

    The decoder may know *where* each detailed token lives, but the position code
    must not become a learned content template. The returned tensor is registered
    as a buffer by `Decoder`, never as an `nn.Parameter`.

    Args:
        layout: Resolved `(temporal,height,width)` feature geometry.
        dim: Decoder hidden width.
    Returns:
        fixed_pos: (N, dim) deterministic detailed-lattice position codes.
    """
    temporal = torch.arange(layout.temporal, dtype=torch.float32)
    row = torch.arange(layout.height, dtype=torch.float32)
    col = torch.arange(layout.width, dtype=torch.float32)
    tt, yy, xx = torch.meshgrid(temporal, row, col, indexing="ij")
    t_pos = tt.reshape(-1)
    y_pos = yy.reshape(-1)
    x_pos = xx.reshape(-1)

    t_dim = dim // 3
    y_dim = (dim - t_dim) // 2
    x_dim = dim - t_dim - y_dim
    fixed_pos = torch.cat(
        [
            _axis_sincos_position_code(t_pos, t_dim),
            _axis_sincos_position_code(y_pos, y_dim),
            _axis_sincos_position_code(x_pos, x_dim),
        ],
        dim=-1,
    )
    if fixed_pos.shape != (layout.n_tokens, dim):
        raise RuntimeError(f"Bad decoder position-code shape: {tuple(fixed_pos.shape)}")
    return fixed_pos


class DecoderBlock(nn.Module):
    """One fixed-position cross-attention + MLP block for the reconstruction decoder.

    Args:
        hidden: (B, N_ctx, dim) output-token content derived from `c`.
        memory: (B, N_c, dim) projected abstract latent (keys/values).
        fixed_pos: (N, dim) non-trainable detailed-lattice position codes.
    Returns:
        hidden: (B, N_ctx, dim) updated c-derived output-token content.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        """Initialize the cross-attention and per-token MLP sublayers."""
        _require_torch()
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, hidden: Tensor, memory: Tensor, fixed_pos: Tensor) -> Tensor:
        """Read from `c` using position-aware queries without adding position as content.

        Args:
            hidden: (B, N_ctx, dim) current output-token content, already derived from `c`.
            memory: (B, N_c, dim) projected abstract latent used as keys/values.
            fixed_pos: (N, dim) non-trainable detailed-lattice position codes.
        Returns:
            hidden: (B, N_ctx, dim) updated output-token content.
        """
        position_query = self.norm_q(hidden) + fixed_pos[None].to(dtype=hidden.dtype)
        attn, _ = self.cross_attn(position_query, memory, memory, need_weights=False)
        hidden = hidden + attn
        return hidden + self.mlp(self.norm_mlp(hidden))


class Decoder(nn.Module):
    """Reconstruct frozen detailed features from the abstract latent (richness anchor).

    A deliberately small cross-attention expander: fixed 3D lattice-position codes
    query the `N_c` abstract slots and project the resulting c-derived content to
    `D_e`, giving `e_hat` (B, N_ctx, D_e). Position tells the decoder where to
    write; the abstract latent `c` tells it what to write. There is no learned
    per-output-token content table.

    GENERIC BY DESIGN: `forward` takes any (B, N_c, D_c) latent. Option 1 feeds the
    online `c_t`, so the gradient flows into `D` and `B` only (`F_c` is untouched
    because we decode `c_t`, not `c_hat`). The through-`F_c` "future anchor"
    (option 3) would feed `c_hat` instead with no change here — only the train.py
    caller changes. This is NOT the Phase 3 frame generator (pixels); it
    reconstructs encoder features and exists only to supply a training signal.

    Args:
        latent: (B, N_c, D_c) abstract latent (`c_t` in option 1).
    Returns:
        pred_detailed: (B, N_ctx, D_e) reconstructed detailed features `e_hat`.
    """

    def __init__(self, cfg: ModelConfig, encoder_spec: EncoderSpec | None = None):
        """Initialize the latent projection, fixed position buffer, blocks, and head."""
        _require_torch()
        super().__init__()
        self.cfg = cfg
        self.encoder_spec = encoder_spec or _legacy_encoder_spec(cfg)
        dim = cfg.decoder_dim
        self.kv_proj = nn.Linear(cfg.d_c, dim)
        self.initial_pos_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.initial_cross_attn = nn.MultiheadAttention(dim, cfg.decoder_heads, batch_first=True)
        self.register_buffer(
            "fixed_pos", _fixed_detailed_position_codes(self.encoder_spec.layout, dim)
        )
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, cfg.decoder_heads) for _ in range(cfg.decoder_blocks)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, self.encoder_spec.feature_dim)

    def forward(self, latent: Tensor) -> Tensor:
        """Expand the abstract latent back to detailed features `e_hat`.

        Args:
            latent: (B, N_c, D_c) abstract latent (online `c_t` in option 1).
        Returns:
            pred_detailed: (B, N_ctx, D_e) reconstructed detailed features.
        """
        b = latent.shape[0]
        memory = self.kv_proj(latent)
        fixed_pos = self.fixed_pos.to(device=memory.device, dtype=memory.dtype)
        # First read: hidden is a weighted sum of c-derived values. The fixed
        # position code controls attention weights only; it is not residual content.
        initial_query = self.initial_pos_norm(fixed_pos)[None].expand(b, -1, -1)
        hidden, _ = self.initial_cross_attn(initial_query, memory, memory, need_weights=False)
        for block in self.blocks:
            hidden = block(hidden, memory, fixed_pos)
        return self.out_proj(self.out_norm(hidden))


class FeatureMeanTracker(nn.Module):
    """EMA per-lattice-position mean of the frozen detailed features (no parameters).

    Supports the residual reconstruction target (cfg.train.recon_residual_target,
    the run-052 fix): the reconstruction anchor trains `D` against `e - mean`
    instead of the absolute `e`, so the video-independent "template" component of
    the selected encoder's mean feature field earns zero loss, so reconstruction pressure routes
    video-specific information through `c_t`.

    Contracts:

    - Buffers only, NO trainable parameters — it can never enter the optimizer,
      AGC, weight decay, or the EMA schedule.
    - `mean` is kept in fp32 regardless of autocast so the running estimate does
      not drift in bf16.
    - `update` is called from the TRAIN step only (never from the diagnostic
      validation batch, which must not leak into the mean), and only while the
      residual target mode is active.
    - The first `update` copies the batch mean directly (no zero-initialized
      cold start); later updates apply `mean <- m * mean + (1 - m) * batch_mean`.

    Args:
        n_ctx: Number of detailed lattice positions.
        d_e: Frozen encoder feature dimension.
        momentum: EMA momentum `m` for the running mean.
    """

    def __init__(self, n_ctx: int, d_e: int, momentum: float):
        """Register the fp32 mean buffer and the initialized flag."""
        _require_torch()
        super().__init__()
        if not 0.0 < momentum < 1.0:
            raise ValueError(f"recon_mean_momentum must be in (0, 1); got {momentum}")
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(n_ctx, d_e, dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, features: Tensor) -> None:
        """Fold one training batch of frozen features into the running mean.

        Args:
            features: (B, N_ctx, D_e) frozen encoder output for the context clip
                (already no-grad; converted to fp32 for the running estimate).
        """
        if features.ndim != 3 or features.shape[1:] != self.mean.shape:
            raise ValueError(
                f"Expected (B, {self.mean.shape[0]}, {self.mean.shape[1]}) features; "
                f"got {tuple(features.shape)}"
            )
        batch_mean = features.detach().float().mean(dim=0)
        if bool(self.initialized):
            self.mean.lerp_(batch_mean, 1.0 - self.momentum)
        else:
            self.mean.copy_(batch_mean)
            self.initialized.fill_(True)

    def subtract(self, features: Tensor) -> Tensor:
        """Return the per-position residual `features - mean` in the input dtype.

        Used on TARGET tensors only (the frozen `e_t` / `e_{t+k}`), which are
        already detached; the mean is a buffer, so no new gradient path exists.

        Args:
            features: (B, N_ctx, D_e) frozen detailed features.
        Returns:
            residual: (B, N_ctx, D_e) features minus the running per-position mean.
        """
        return features - self.mean.to(device=features.device, dtype=features.dtype)


class FeatureWhitener(nn.Module):
    """Fixed offline ZCA whitening of selected frozen detailed features (no parameters).

    Implements tmp/changes_bottleneck <2>, originally motivated by the V-JEPA
    investigation_014 result: its frozen `e` token cloud was strongly anisotropic
    (pooled entropy rank ~193/1024 with a long low-energy tail), so `e -> c` compression otherwise weighs
    dominant-direction energy and tail noise with the same metric. Whitening with
    FIXED training-set statistics equalizes the retained directions:

    ```text
    e_w  = (e - mu) @ W        W     = U (Lambda + eps I)^{-1/2} U^T
    e    = e_w @ W_inv + mu    W_inv = U (Lambda + eps I)^{1/2}  U^T
    ```

    Contracts:

    - Buffers only, NO trainable parameters — it can never enter the optimizer,
      AGC, weight decay, or the EMA schedule.
    - Statistics are computed ONCE offline over the training set
      (`whiten_stats.py`) and reused unchanged for train/val/inference. Never
      per-batch whitening.
    - `whiten`/`unwhiten` compute in fp32 regardless of autocast (the `D_e x D_e`
      matmul is precision-sensitive in bf16) and return the input dtype.
    - The eigenvalue floor `eps` is applied at `configure` time, so it can be
      swept from config without recomputing the offline stats.
    - Applied to frozen no-grad encoder outputs only; no new gradient path exists.

    Args:
        d_e: Frozen encoder feature dimension.
    """

    def __init__(self, d_e: int):
        """Register fp32 mean/whiten/unwhiten buffers and the initialized flag."""
        _require_torch()
        super().__init__()
        self.register_buffer("mean", torch.zeros(d_e, dtype=torch.float32))
        self.register_buffer("whiten_mat", torch.eye(d_e, dtype=torch.float32))
        self.register_buffer("unwhiten_mat", torch.eye(d_e, dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def configure(self, mean: Tensor, eigvals: Tensor, eigvecs: Tensor, eps: float) -> None:
        """Build the whitening pair from offline covariance eigen-statistics.

        Args:
            mean: (D_e,) training-set feature mean `mu`.
            eigvals: (D_e,) covariance eigenvalues `Lambda` (ascending or any order).
            eigvecs: (D_e, D_e) matching orthonormal eigenvectors `U` (columns).
            eps: Eigenvalue floor added before the +/-1/2 powers.
        """
        if eps <= 0.0:
            raise ValueError(f"whiten eps must be > 0; got {eps}")
        d = self.mean.shape[0]
        if mean.shape != (d,) or eigvals.shape != (d,) or eigvecs.shape != (d, d):
            raise ValueError(
                f"Whitening stats shapes {tuple(mean.shape)}/{tuple(eigvals.shape)}/"
                f"{tuple(eigvecs.shape)} do not match d_e={d}."
            )
        lam = eigvals.double().clamp_min(0.0) + eps
        u = eigvecs.double()
        self.mean.copy_(mean.float())
        self.whiten_mat.copy_((u @ torch.diag(lam.pow(-0.5)) @ u.t()).float())
        self.unwhiten_mat.copy_((u @ torch.diag(lam.pow(0.5)) @ u.t()).float())
        self.initialized.fill_(True)

    def _require_initialized(self) -> None:
        """Fail fast if the whitener is used before stats are loaded."""
        if not bool(self.initialized):
            raise RuntimeError(
                "FeatureWhitener used before configure(); load offline stats "
                "(whiten_stats.py output) via train._build_whitener or a checkpoint."
            )

    def whiten(self, features: Tensor) -> Tensor:
        """Map raw frozen features into whitened space, `(e - mu) @ W`.

        Args:
            features: (..., D_e) frozen detailed features (no-grad).
        Returns:
            whitened: (..., D_e) whitened features in the input dtype.
        """
        self._require_initialized()
        # Autocast would downcast even an fp32 @ fp32 matmul to bf16 (the SIGReg
        # WALK_FIXES F2 trap); the D_e x D_e whitening product must stay fp32.
        with torch.autocast(device_type=features.device.type, enabled=False):
            out = (features.float() - self.mean) @ self.whiten_mat
        return out.to(dtype=features.dtype)

    def unwhiten(self, features: Tensor) -> Tensor:
        """Map whitened-space features back to raw encoder space, `e_w @ W^-1 + mu`.

        Args:
            features: (..., D_e) whitened-space features (e.g. a decoded e_hat_w).
        Returns:
            raw: (..., D_e) features in the original encoder space, input dtype.
        """
        self._require_initialized()
        with torch.autocast(device_type=features.device.type, enabled=False):
            out = features.float() @ self.unwhiten_mat + self.mean
        return out.to(dtype=features.dtype)


def build_phase1_modules(
    cfg: Config, *, load_encoder: bool = True, encoder_spec: EncoderSpec | None = None
) -> tuple[nn.Module | None, Bottleneck, TargetBottleneck, CoarseFlow, Decoder]:
    """Construct all Phase 1 modules in data-path order.

    Keeping construction in one place gives Stage 0 a single canonical module
    bundle and a consistent online/EMA bottleneck pairing.

    Args:
        cfg: Global config containing Phase 1 architecture constants.
        load_encoder: When False, skip loading the (large) frozen encoder — used by
            shape/gradient smoke tests that synthesize `e_t` directly.
    Returns:
        Modules `(encoder, bottleneck, target_bottleneck, coarse_flow, decoder)`;
        `encoder` is None when `load_encoder=False`.
    """
    encoder = build_frozen_encoder(cfg.encoder) if load_encoder else None
    resolved_spec = encoder.spec if encoder is not None else encoder_spec
    resolved_spec = resolved_spec or _legacy_encoder_spec(cfg.model)
    bottleneck = Bottleneck(cfg.model, resolved_spec)
    target_bottleneck = TargetBottleneck(bottleneck)
    coarse_flow = CoarseFlow(cfg.model)
    decoder = Decoder(cfg.model, resolved_spec)
    return encoder, bottleneck, target_bottleneck, coarse_flow, decoder


def smoke_test_models() -> None:
    """Run synthetic Phase 1 shape and gradient checks (no encoder download).

    Verifies the bottleneck / EMA bottleneck / coarse flow shapes and the critical
    gradient boundary using a synthetic `e_t`. The real frozen encoder is exercised
    separately by `smoke_test_encoder()` and by Stage 0 in `train.py`.
    """
    from losses import (
        covariance_floor,
        flow_matching_loss,
        interpolate,
        reconstruction_loss,
        residual_target,
        slot_diversity_loss,
        variance_floor,
        velocity_target,
    )

    cfg = Config()
    _, bottleneck, target_bottleneck, coarse_flow, decoder = build_phase1_modules(
        cfg, load_encoder=False
    )
    target_bottleneck.copy_weights_from(bottleneck)
    detailed = torch.randn(2, cfg.model.n_ctx, cfg.model.d_e)
    target_detailed = torch.randn(2, cfg.model.n_tgt, cfg.model.d_e)
    abstract = bottleneck(detailed)
    assert abstract.shape == (2, cfg.model.n_c, cfg.model.d_c), abstract.shape
    target_abstract = target_bottleneck(target_detailed)
    assert target_abstract.requires_grad is False
    eps = torch.randn_like(target_abstract)
    tau = torch.rand(2)
    z_c = interpolate(target_abstract, eps, tau)
    u_c = velocity_target(target_abstract, eps)
    u_c_hat = coarse_flow(z_c, tau, abstract)
    assert u_c_hat.shape == (2, cfg.model.n_c, cfg.model.d_c), u_c_hat.shape
    loss = flow_matching_loss(u_c_hat, u_c) + cfg.train.lambda_var * variance_floor(abstract)
    loss.backward()
    assert any(p.grad is not None for p in bottleneck.parameters())
    assert any(p.grad is not None for p in coarse_flow.parameters())
    assert all(p.grad is None for p in target_bottleneck.parameters())
    # VICReg-C (Plan Phase 04): finite scalar and gradients reach the bottleneck.
    bottleneck.zero_grad(set_to_none=True)
    cov = covariance_floor(bottleneck(detailed))
    assert cov.requires_grad and torch.isfinite(cov), cov
    cov.backward()
    assert any(p.grad is not None for p in bottleneck.parameters())
    # Slot-diversity loss: finite scalar and gradients reach the bottleneck.
    bottleneck.zero_grad(set_to_none=True)
    slot_loss = slot_diversity_loss(bottleneck(detailed))
    assert slot_loss.requires_grad and torch.isfinite(slot_loss), slot_loss
    slot_loss.backward()
    assert any(p.grad is not None for p in bottleneck.parameters())
    # Reconstruction anchor (option 1) gradient contract: D(c_t) vs e_t reaches the
    # decoder and B, but NEVER F_c (we decode c_t, not c_hat) or the EMA bottleneck.
    bottleneck.zero_grad(set_to_none=True)
    coarse_flow.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)
    e_hat = decoder(bottleneck(detailed))
    assert e_hat.shape == (2, cfg.model.n_ctx, cfg.model.d_e), e_hat.shape
    recon = reconstruction_loss(e_hat, detailed)
    assert recon.requires_grad and torch.isfinite(recon), recon
    recon.backward()
    assert any(p.grad is not None for p in decoder.parameters())
    assert any(p.grad is not None for p in bottleneck.parameters())
    assert all(p.grad is None for p in coarse_flow.parameters())
    assert all(p.grad is None for p in target_bottleneck.parameters())
    # Reconstruction anchor (option 3) gradient contract — the INVERSE of option 1:
    # D(c_hat) vs e_{t+k} must reach the decoder, F_c, AND B (via the F_c conditioning
    # on c_t, intentionally not detached), but still NEVER the EMA bottleneck. c_hat is
    # the rectified-flow one-step endpoint estimate from the trained predictor.
    bottleneck.zero_grad(set_to_none=True)
    coarse_flow.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)
    abstract_p = bottleneck(detailed)
    u_c_hat_p = coarse_flow(z_c, tau, abstract_p)
    c_hat_p = z_c + (1.0 - tau.reshape(-1, 1, 1)) * u_c_hat_p
    recon_pred = reconstruction_loss(decoder(c_hat_p), target_detailed)
    assert recon_pred.requires_grad and torch.isfinite(recon_pred), recon_pred
    recon_pred.backward()
    assert any(p.grad is not None for p in decoder.parameters())
    assert any(p.grad is not None for p in coarse_flow.parameters())  # inverse of option 1
    assert any(p.grad is not None for p in bottleneck.parameters())
    assert all(p.grad is None for p in target_bottleneck.parameters())
    # Residual prediction (investigation_009) gradient contract: predicting the temporal
    # residual Δ and decoding ĉ = c_t + Δ̂ must reach D, F_c, AND B (via the add-back and the
    # F_c conditioning), never the EMA bottleneck. Exercises the residual flow path end to
    # end so a launch can't first discover a crash on the pod.
    bottleneck.zero_grad(set_to_none=True)
    coarse_flow.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)
    target_present = target_bottleneck(detailed)
    delta, sigma = residual_target(target_abstract, target_present)
    eps_r = sigma.to(delta.dtype) * torch.randn_like(delta)
    z_r = interpolate(delta, eps_r, tau)
    u_r = velocity_target(delta, eps_r)
    abstract_r = bottleneck(detailed)
    u_r_hat = coarse_flow(z_r, tau, abstract_r)
    c_hat_r = abstract_r + z_r + (1.0 - tau.reshape(-1, 1, 1)) * u_r_hat
    loss_r = flow_matching_loss(u_r_hat, u_r) + reconstruction_loss(
        decoder(c_hat_r), target_detailed
    )
    loss_r.backward()
    assert any(p.grad is not None for p in decoder.parameters())
    assert any(p.grad is not None for p in coarse_flow.parameters())
    assert any(p.grad is not None for p in bottleneck.parameters())
    assert all(p.grad is None for p in target_bottleneck.parameters())
    # Residual reconstruction target (investigation_013) gradient contract: subtracting
    # the tracked per-position mean from the frozen target must leave the gradient
    # routing identical to option 1 (D and B train; F_c and the EMA bottleneck never
    # do), and the tracker itself must stay parameter-free so it cannot be optimized.
    tracker = FeatureMeanTracker(cfg.model.n_ctx, cfg.model.d_e, cfg.train.recon_mean_momentum)
    assert sum(1 for _ in tracker.parameters()) == 0
    tracker.update(detailed)
    assert bool(tracker.initialized)
    bottleneck.zero_grad(set_to_none=True)
    coarse_flow.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)
    residual_recon = reconstruction_loss(decoder(bottleneck(detailed)), tracker.subtract(detailed))
    assert residual_recon.requires_grad and torch.isfinite(residual_recon), residual_recon
    residual_recon.backward()
    assert any(p.grad is not None for p in decoder.parameters())
    assert any(p.grad is not None for p in bottleneck.parameters())
    assert all(p.grad is None for p in coarse_flow.parameters())
    assert all(p.grad is None for p in target_bottleneck.parameters())
    # Fixed offline whitening (tmp/changes_bottleneck <2>): parameter-free, exact
    # whiten/unwhiten round-trip, and the whitened features feed the bottleneck with
    # unchanged shapes. Whitening applies to no-grad encoder outputs only, so it can
    # never add a gradient path.
    whitener = FeatureWhitener(cfg.model.d_e)
    assert sum(1 for _ in whitener.parameters()) == 0
    stats_rows = torch.randn(4096, cfg.model.d_e)
    eigvals_w, eigvecs_w = torch.linalg.eigh(torch.cov(stats_rows.t()))
    whitener.configure(stats_rows.mean(dim=0), eigvals_w, eigvecs_w, eps=1e-4)
    whitened = whitener.whiten(detailed)
    assert whitened.shape == detailed.shape
    assert torch.allclose(whitener.unwhiten(whitened), detailed, atol=1e-3)
    assert bottleneck(whitened).shape == (2, cfg.model.n_c, cfg.model.d_c)
    from diagnostics import attention_entropy, slot_diversity_rank

    attn = attention_entropy(bottleneck, detailed)
    slot = slot_diversity_rank(abstract)
    print(
        f"Phase 1 model smoke test passed (synthetic e_t) | "
        f"queries=orthogonal latent_blocks={cfg.model.bottleneck_latent_blocks} "
        f"(zero-init residuals) | {attn} {slot}"
    )


def smoke_test_encoder() -> None:
    """Load the real frozen encoder and verify it is frozen and shaped correctly.

    Downloads the configured encoder checkpoint on first run; intended for RunPod
    or a machine willing to fetch the weights. Confirms 0 trainable params and the
    `(B, N_ctx, D_e)` output contract.
    """
    cfg = Config()
    if not torch.cuda.is_available():
        cfg.encoder.precision = "fp32"
    encoder = build_frozen_encoder(cfg.encoder)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert trainable == 0, trainable
    clip = torch.rand(
        1,
        cfg.encoder.input_frames,
        3,
        cfg.encoder.input_height,
        cfg.encoder.input_width,
    )
    out = encoder(clip)
    assert out.shape == (1, encoder.spec.layout.n_tokens, encoder.spec.feature_dim), out.shape
    print(f"Frozen encoder smoke test passed: output {tuple(out.shape)}")
