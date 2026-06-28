"""Phase 1 model modules for HJEPA-VWM (v0.2 — frozen encoder).

Implements the frozen pretrained encoder `E` (V-JEPA 2 ViT-L/16), the trainable
bottleneck `B`, the EMA bottleneck `B_EMA`, and the coarse flow `F_c`. Fine flow,
VAE, and frame generator are out of scope for Phase 1.

v0.2 changes from v0.1:
- `E` is a frozen pretrained ViT (no from-scratch encoder, no PatchEmbed) — the
  encoder owns tokenization, position encoding, and tubelet projection internally.
- EMA is on the bottleneck only (`B_EMA`); there is no target encoder.
- No tubelet dropout: all `N_ctx=1024` tokens are real, so the bottleneck no longer
  scatters a kept-mask back to a full grid.
"""

from __future__ import annotations

import math
from copy import deepcopy

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover - local docs-only environments.
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]

from config import Config, ModelConfig
from losses import as_target


def _require_torch() -> None:
    """Fail fast when Phase 1 model code is used without PyTorch installed."""
    if torch is None:
        raise RuntimeError("PyTorch is required for models.py. Install requirements.txt on RunPod.")


class FrozenEncoder(nn.Module):
    """Frozen pretrained V-JEPA 2 ViT-L/16 encoder, shared by both branches.

    A thin wrapper around the HF `transformers` V-JEPA 2 model. We do NOT implement
    a patchifier or position embeddings — tokenization, 3D-RoPE, and the tubelet
    projection are internal to the encoder. The same frozen instance encodes the
    context clip (`e_t`) and the future clip (`e_plus`).

    Args:
        clip: (B, T=8, C=3, H=256, W=256) pixel clip, encoder-normalized.
    Returns:
        detailed: (B, N_ctx=1024, D_e=1024) per-tubelet features (last_hidden_state).
    """

    def __init__(self, cfg: ModelConfig):
        """Load the pretrained encoder and freeze every parameter."""
        _require_torch()
        super().__init__()
        try:
            from transformers import AutoModel
        except ModuleNotFoundError as exc:  # pragma: no cover - RunPod dependency.
            raise RuntimeError(
                "transformers>=4.53,<5 (with vjepa2 support) is required for FrozenEncoder."
            ) from exc
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.encoder_repo, attn_implementation="sdpa")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable != 0:
            raise RuntimeError(f"Frozen encoder has {trainable} trainable params; expected 0.")

    def train(self, mode: bool = True) -> FrozenEncoder:
        """Keep the encoder in eval mode regardless of the parent's train() call."""
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, clip: Tensor) -> Tensor:
        """Encode a pixel clip into detailed tokens (no grad).

        Args:
            clip: (B, T, C, H, W) encoder-normalized pixel clip.
        Returns:
            detailed: (B, N_ctx, D_e) per-tubelet features.
        """
        features = self.model.get_vision_features(clip)
        return features


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


class Bottleneck(nn.Module):
    """Compress detailed tokens into the low-bandwidth abstract latent `c_t`.

    Projects the frozen `e_t` from `D_e` to the mixer width, mixes each of the 4
    temporal-slot 16x16 grids with shared ConvNeXt blocks, then cross-attends from
    32 learned query slots into the mixed tokens. The same module (and its EMA copy)
    is applied identically to `e_t` (-> `c_t`) and `e_plus` (-> `c_plus`); both
    clips share the encoder's `N_ctx=1024` temporal-major geometry, so there is no
    separate target path and no kept-mask in v0.2.
    """

    def __init__(self, cfg: ModelConfig):
        """Initialize input projection, ConvNeXt mixing, queries, and cross-attention."""
        _require_torch()
        super().__init__()
        self.cfg = cfg
        mix = cfg.bottleneck_mixer_dim
        self.in_proj = nn.Linear(cfg.d_e, mix)
        self.mixers = nn.Sequential(
            *[ConvNeXtBlock(mix) for _ in range(cfg.bottleneck_convnext_blocks)]
        )
        self.to_kv = nn.Linear(mix, cfg.d_c)
        # Fix 1 (Plan Phase 04): orthogonal queries (unit-norm rows). The old
        # v0.2 `randn * 0.02` made the q.k logits tiny, so softmax was near-
        # uniform for every slot and all 32 slots read out ~the mean token.
        # Unit-norm rows fix the scale and start the slots decorrelated.
        q = torch.empty(cfg.n_c, cfg.d_c)
        nn.init.orthogonal_(q)
        self.queries = nn.Parameter(q)
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_c, cfg.bottleneck_cross_attn_heads, batch_first=True
        )
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(cfg.d_c),
            nn.Linear(cfg.d_c, cfg.d_c * 4),
            nn.GELU(),
            nn.Linear(cfg.d_c * 4, cfg.d_c),
        )
        # Fix 2 (Plan Phase 04): adaLN-Zero-style identity start for the residual
        # MLP — the block begins as a pass-through so early training isn't
        # destabilized by random residual contributions.
        nn.init.zeros_(self.out_mlp[-1].weight)
        nn.init.zeros_(self.out_mlp[-1].bias)
        # Issue 1/9: flag this zero-init residual-output projection so it is held out
        # of AGC and weight decay (diagnostics.is_geometry_or_gate_param). While ||w||≈0
        # its AGC bound collapses to ~clip_factor·eps, which would throttle the gradients
        # that must open the residual branch; decaying it just re-pins identity.
        self.out_mlp[-1].is_zero_init = True
        self.norm = nn.LayerNorm(cfg.d_c)

    def forward(self, detailed: Tensor, *, return_attn: bool = False):
        """Compress detailed tokens to abstract tokens.

        Args:
            detailed: (B, N_ctx=1024, D_e=1024) frozen-encoder tokens (context or
                future clip — identical geometry).
            return_attn: When True, also return the PER-HEAD cross-attention
                weights (B, num_heads, N_c, N_ctx) for diagnostics. Per-head
                (not head-averaged) matters: 8 sharp-but-different heads average
                out to look uniform, so head-averaged entropy masks real
                selectivity. The training path leaves this False so the fast
                (no-weights) attention kernel is used.
        Returns:
            abstract: (B, N_c=32, D_c=256) abstract latent. When `return_attn`,
            a tuple `(abstract, attn_weights)` with attn (B, num_heads, N_c, N_ctx).
        """
        b, n, _ = detailed.shape
        cfg = self.cfg
        if n != cfg.n_ctx:
            raise ValueError(f"Expected {cfg.n_ctx} tokens, got {n}")
        mix = cfg.bottleneck_mixer_dim
        tokens = self.in_proj(detailed)
        t, g = cfg.n_temporal_tokens, cfg.grid_spatial
        # Temporal-major token order from the encoder: index = t*(g*g) + h*g + w.
        grid = tokens.reshape(b * t, g, g, mix).permute(0, 3, 1, 2)
        mixed = self.mixers(grid).permute(0, 2, 3, 1).reshape(b, t * g * g, mix)
        memory = self.to_kv(mixed)
        queries = self.queries[None].expand(b, -1, -1)
        attended, attn = self.cross_attn(
            queries,
            memory,
            memory,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        abstract = self.norm(attended + self.out_mlp(attended))
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
        target_detailed: (B, N_tgt=1024, D_e=1024) frozen encoder output on the
            future clip.
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


class DecoderBlock(nn.Module):
    """One cross-attention + MLP block for the reconstruction decoder.

    Args:
        queries: (B, N_ctx, dim) output-token queries.
        memory: (B, N_c, dim) projected abstract latent (keys/values).
    Returns:
        queries: (B, N_ctx, dim) updated output-token representations.
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

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        """Cross-attend output queries into the abstract latent, then MLP-mix."""
        attn, _ = self.cross_attn(self.norm_q(queries), memory, memory, need_weights=False)
        queries = queries + attn
        return queries + self.mlp(self.norm_mlp(queries))


class Decoder(nn.Module):
    """Reconstruct frozen detailed features from the abstract latent (richness anchor).

    A deliberately small cross-attention expander: `N_ctx` learned output-token
    queries cross-attend into the `N_c` abstract slots and project to `D_e`, giving
    `e_hat` (B, N_ctx, D_e). The reconstruction MSE against the frozen detailed
    features forces `c` to stay information-rich, attacking the ~13 effective-rank
    ceiling and the identical-`c` representational collapse.

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

    def __init__(self, cfg: ModelConfig):
        """Initialize the latent projection, output queries, blocks, and head."""
        _require_torch()
        super().__init__()
        self.cfg = cfg
        dim = cfg.decoder_dim
        self.kv_proj = nn.Linear(cfg.d_c, dim)
        # Orthogonal output-token queries: decorrelated start (mirrors Bottleneck).
        q = torch.empty(cfg.n_ctx, dim)
        nn.init.orthogonal_(q)
        self.queries = nn.Parameter(q)
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, cfg.decoder_heads) for _ in range(cfg.decoder_blocks)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, cfg.d_e)

    def forward(self, latent: Tensor) -> Tensor:
        """Expand the abstract latent back to detailed features `e_hat`.

        Args:
            latent: (B, N_c, D_c) abstract latent (online `c_t` in option 1).
        Returns:
            pred_detailed: (B, N_ctx, D_e) reconstructed detailed features.
        """
        b = latent.shape[0]
        memory = self.kv_proj(latent)
        queries = self.queries[None].expand(b, -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        return self.out_proj(self.out_norm(queries))


def build_phase1_modules(
      cfg: Config, *, load_encoder: bool = True
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
    encoder = FrozenEncoder(cfg.model) if load_encoder else None
    bottleneck = Bottleneck(cfg.model)
    target_bottleneck = TargetBottleneck(bottleneck)
    coarse_flow = CoarseFlow(cfg.model)
    decoder = Decoder(cfg.model)
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
    loss_r = flow_matching_loss(u_r_hat, u_r) + reconstruction_loss(decoder(c_hat_r), target_detailed)
    loss_r.backward()
    assert any(p.grad is not None for p in decoder.parameters())
    assert any(p.grad is not None for p in coarse_flow.parameters())
    assert any(p.grad is not None for p in bottleneck.parameters())
    assert all(p.grad is None for p in target_bottleneck.parameters())
    from diagnostics import attention_entropy, slot_diversity_rank

    attn = attention_entropy(bottleneck, detailed)
    slot = slot_diversity_rank(abstract)
    print(
        f"Phase 1 model smoke test passed (synthetic e_t) | "
        f"queries=orthogonal out_mlp=zero-init | {attn} {slot}"
    )


def smoke_test_encoder() -> None:
    """Load the real frozen encoder and verify it is frozen and shaped correctly.

    Downloads the V-JEPA 2 ViT-L checkpoint on first run; intended for RunPod or a
    machine willing to fetch the weights. Confirms 0 trainable params and the
    `(B, N_ctx, D_e)` output contract.
    """
    cfg = Config()
    encoder = FrozenEncoder(cfg.model)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert trainable == 0, trainable
    clip = torch.randn(1, cfg.model.t_ctx, 3, cfg.model.h, cfg.model.w)
    out = encoder(clip)
    assert out.shape == (1, cfg.model.n_ctx, cfg.model.d_e), out.shape
    print(f"Frozen encoder smoke test passed: output {tuple(out.shape)}")
