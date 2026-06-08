"""Phase 1 model modules for HJEPA-VWM.

Implements the online encoder, bottleneck, EMA target branch, and coarse flow.
Fine flow, VAE, and frame generator are intentionally out of scope for Phase 1.
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


def _sincos_1d(length: int, dim: int, device: torch.device | None = None) -> Tensor:
    """Create one axis of fixed sinusoidal position embeddings."""
    half = dim // 2
    omega = torch.arange(half, dtype=torch.float32, device=device) / max(1, half)
    omega = 1.0 / (10000**omega)
    pos = torch.arange(length, dtype=torch.float32, device=device)[:, None]
    table = pos * omega[None]
    emb = torch.cat([table.sin(), table.cos()], dim=1)
    if emb.shape[1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


def _factorized_3d_sincos(t: int, h: int, w: int, dim: int, device: torch.device) -> Tensor:
    """Build factorized T+H+W sinusoidal position embeddings.

    Args:
        t: Number of temporal positions to materialize.
        h: Grid height in patch tokens.
        w: Grid width in patch tokens.
        dim: Embedding dimension D_e.
        device: Device for returned tensor.
    Returns:
        pos: (t, h, w, dim) summed factorized 3D sin-cos embedding.
    """
    pe_t = _sincos_1d(t, dim, device)[:, None, None, :]
    pe_h = _sincos_1d(h, dim, device)[None, :, None, :]
    pe_w = _sincos_1d(w, dim, device)[None, None, :, :]
    return pe_t + pe_h + pe_w


class PatchEmbed(nn.Module):
    """Patchify context clips or target frames into detailed tokens.

    Adds fixed 3D sin-cos position embeddings before context-side tubelet
    dropout, which preserves positional identity for surviving tokens.
    """

    def __init__(self, cfg: ModelConfig):
        """Initialize the shared 3D convolutional patch projector."""
        _require_torch()
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv3d(
            3,
            cfg.d_e,
            kernel_size=(cfg.patch_t, cfg.patch_h, cfg.patch_w),
            stride=(cfg.patch_t, cfg.patch_h, cfg.patch_w),
        )

    def forward_context(self, context_clip: Tensor) -> tuple[Tensor, Tensor]:
        """Patchify a 4-frame context clip and apply tubelet dropout in training.

        Args:
            context_clip: (B, T=4, C=3, H=128, W=128) context frames in [-1, 1].
        Returns:
            tokens: (B, N_ctx_post, D_e) detailed input tokens.
            kept_mask: (B, N_ctx=256) boolean mask identifying kept tubelets.
        """
        b, t, c, h, w = context_clip.shape
        if c != 3:
            raise ValueError(f"Expected RGB context, got {context_clip.shape}")
        x = context_clip.permute(0, 2, 1, 3, 4)
        tokens = self.proj(x).permute(0, 2, 3, 4, 1)
        grid_t, grid_h, grid_w = tokens.shape[1:4]
        pos = _factorized_3d_sincos(grid_t, grid_h, grid_w, self.cfg.d_e, tokens.device)
        tokens = (tokens + pos[None]).reshape(b, grid_t * grid_h * grid_w, self.cfg.d_e)
        if not self.training or self.cfg.tubelet_dropout <= 0:
            kept_mask = torch.ones(b, self.cfg.n_ctx, dtype=torch.bool, device=tokens.device)
            return tokens, kept_mask
        keep = int(round(self.cfg.n_ctx * (1.0 - self.cfg.tubelet_dropout)))
        scores = torch.rand(b, self.cfg.n_ctx, device=tokens.device)
        idx = scores.topk(keep, dim=1).indices.sort(dim=1).values
        kept_mask = torch.zeros(b, self.cfg.n_ctx, dtype=torch.bool, device=tokens.device)
        kept_mask.scatter_(1, idx, True)
        gather_idx = idx[:, :, None].expand(-1, -1, self.cfg.d_e)
        return tokens.gather(1, gather_idx), kept_mask

    def forward_target(self, future_frame: Tensor) -> tuple[Tensor, None]:
        """Patchify a single future frame without tubelet dropout.

        Args:
            future_frame: (B, C=3, H=128, W=128) future frame in [-1, 1].
        Returns:
            tokens: (B, N_tgt=64, D_e) target tokens at temporal index 4.
            mask: Always None for the target path.
        """
        x = future_frame[:, :, None, :, :]
        tokens = self.proj(x).permute(0, 2, 3, 4, 1)
        b, _, grid_h, grid_w, _ = tokens.shape
        pos = _factorized_3d_sincos(self.cfg.t_ctx + 1, grid_h, grid_w, self.cfg.d_e, tokens.device)
        target_pos = pos[self.cfg.t_ctx : self.cfg.t_ctx + 1]
        tokens = (tokens + target_pos[None]).reshape(b, grid_h * grid_w, self.cfg.d_e)
        return tokens, None


class OnlineEncoder(nn.Module):
    """VideoViT-Small encoder over detailed tokens.

    Args:
        tokens: (B, N_ctx_post, D_e) or (B, N_tgt, D_e) detailed tokens.
    Returns:
        detailed: Same shape as `tokens`; one encoded detailed token per input token.
    """

    def __init__(self, cfg: ModelConfig):
        """Initialize a ViT-Small transformer encoder with locked Phase 1 dimensions."""
        _require_torch()
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_e,
            nhead=cfg.encoder_heads,
            dim_feedforward=cfg.d_e * cfg.encoder_mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.encoder_depth)
        self.norm = nn.LayerNorm(cfg.d_e)

    def forward(self, tokens: Tensor) -> Tensor:
        """Encode detailed tokens without changing token count.

        The encoder must preserve one output token per surviving input token so
        the bottleneck can reconstruct the spatial grid and keep shape contracts auditable.

        Args:
            tokens: (B, N_ctx_post, D_e) context tokens or (B, N_tgt, D_e) target tokens.
        Returns:
            detailed: Same shape as `tokens`, encoded detailed latent tokens.
        """
        return self.norm(self.blocks(tokens))


class ConvNeXtBlock(nn.Module):
    """Small ConvNeXt-style 2D mixer for bottleneck pre-attention tokens.

    Args:
        grid: (B, C=D_e, H=8, W=8) per-frame spatial token grid.
    Returns:
        mixed: (B, C=D_e, H=8, W=8) grid after depthwise + pointwise mixing.
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
        """Mix local spatial detail on a per-frame 8×8 token grid.

        This gives the bottleneck local spatial context before query compression,
        helping `c_t` retain future-relevant structure instead of raw token noise.

        Args:
            grid: (B, D_e, H_grid, W_grid) per-frame detailed-token grid.
        Returns:
            mixed: (B, D_e, H_grid, W_grid) grid after ConvNeXt-style mixing.
        """
        residual = grid
        x = self.dwconv(grid).permute(0, 2, 3, 1)
        x = self.pw2(self.act(self.pw1(self.norm(x))))
        return residual + x.permute(0, 3, 1, 2)


class Bottleneck(nn.Module):
    """Compress detailed tokens into the low-bandwidth abstract latent.

    Reconstructs spatial grids for ConvNeXt mixing, then cross-attends from 32
    learned query slots into projected detailed tokens.
    """

    def __init__(self, cfg: ModelConfig):
        """Initialize ConvNeXt mixing, learned query slots, and cross-attention."""
        _require_torch()
        super().__init__()
        self.cfg = cfg
        self.mixers = nn.Sequential(
            *[ConvNeXtBlock(cfg.d_e) for _ in range(cfg.bottleneck_convnext_blocks)]
        )
        self.to_kv = nn.Linear(cfg.d_e, cfg.d_c)
        self.queries = nn.Parameter(torch.randn(cfg.n_c, cfg.d_c) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_c, cfg.bottleneck_cross_attn_heads, batch_first=True
        )
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(cfg.d_c),
            nn.Linear(cfg.d_c, cfg.d_c * 4),
            nn.GELU(),
            nn.Linear(cfg.d_c * 4, cfg.d_c),
        )
        self.norm = nn.LayerNorm(cfg.d_c)

    def _scatter_context(self, detailed: Tensor, kept_mask: Tensor) -> Tensor:
        """Scatter post-dropout context tokens back to the full 4×8×8 grid."""
        b, _, d = detailed.shape
        full = detailed.new_zeros(b, self.cfg.n_ctx, d)
        for i in range(b):
            full[i, kept_mask[i]] = detailed[i, : kept_mask[i].sum()]
        return full.reshape(b, self.cfg.t_ctx, self.cfg.grid_h, self.cfg.grid_w, d)

    def forward(self, detailed: Tensor, kept_mask: Tensor | None = None) -> Tensor:
        """Compress detailed tokens to abstract tokens.

        Args:
            detailed: Context `(B, N_ctx_post, D_e)` or target `(B, N_tgt=64, D_e)` tokens.
            kept_mask: Context `(B, N_ctx=256)` tubelet mask, or None for target tokens.
        Returns:
            abstract: (B, N_c=32, D_c=256) abstract latent.
        """
        b, n, d = detailed.shape
        if kept_mask is None:
            frames = 1
            grid = detailed.reshape(b, 1, self.cfg.grid_h, self.cfg.grid_w, d)
            real_tokens = None
        else:
            frames = self.cfg.t_ctx
            grid = self._scatter_context(detailed, kept_mask)
            real_tokens = kept_mask
        grid2d = grid.reshape(b * frames, self.cfg.grid_h, self.cfg.grid_w, d).permute(0, 3, 1, 2)
        mixed = (
            self.mixers(grid2d)
            .permute(0, 2, 3, 1)
            .reshape(b, frames * self.cfg.tokens_per_frame, d)
        )
        if real_tokens is not None:
            kept = []
            for i in range(b):
                kept.append(mixed[i, real_tokens[i]])
            mixed = torch.stack(kept, dim=0)
        memory = self.to_kv(mixed)
        queries = self.queries[None].expand(b, -1, -1)
        attended, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        abstract = attended + self.out_mlp(attended)
        return self.norm(abstract)


class AdaLNBlock(nn.Module):
    """DiT-style adaLN-Zero transformer block for the coarse flow.

    Args:
        x: (B, 64, D_c) concatenated noised and conditioning abstract tokens.
        time_emb: (B, D_c) flow-time embedding.
    Returns:
        hidden: (B, 64, D_c) updated sequence.
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
        abstract: (B, N_c=32, D_c=256) current abstract conditioning latent.
    Returns:
        u_c_hat: (B, N_c=32, D_c=256) predicted coarse velocity.
    """

    def __init__(self, cfg: ModelConfig):
        """Initialize the Stage 1 coarse abstract-flow predictor."""
        _require_torch()
        super().__init__()
        self.cfg = cfg
        self.null_condition = nn.Parameter(torch.zeros(cfg.n_c, cfg.d_c))
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
        """Predict the coarse flow velocity conditioned on current abstract state.

        Condition dropout replaces `abstract` with a learned null token set so the
        coarse flow remains robust without changing the gradient path through `c_t`.

        Args:
            z_c: (B, N_c, D_c) noised future abstract latent.
            tau_c: (B,) flow time.
            abstract: (B, N_c, D_c) current abstract latent c_t.
            condition_drop: Optional (B,) boolean override for dropout tests.
        Returns:
            u_c_hat: (B, N_c, D_c) predicted rectified-flow velocity.
        """
        if self.training:
            if condition_drop is None:
                condition_drop = (
                    torch.rand(abstract.shape[0], device=abstract.device)
                    < self.cfg.condition_dropout
                )
            null = self.null_condition[None].expand_as(abstract)
            abstract = torch.where(condition_drop[:, None, None], null, abstract)
        x = torch.cat([z_c, abstract], dim=1)
        time_emb = self.time_mlp(_timestep_embedding(tau_c, self.cfg.d_c).to(dtype=x.dtype))
        for block in self.blocks:
            x = block(x, time_emb)
        return self.norm(x[:, : self.cfg.n_c])


class TargetBranch(nn.Module):
    """EMA target branch producing detached future detailed and abstract latents.

    Args:
        future_frame: (B, C=3, H=128, W=128) future frame in [-1, 1].
    Returns:
        target_detailed: (B, N_tgt=64, D_e=384), detached.
        target_abstract: (B, N_c=32, D_c=256), detached.
    """

    def __init__(
        self, patch_embed: PatchEmbed, online_encoder: OnlineEncoder, bottleneck: Bottleneck
    ):
        """Clone online modules into the frozen EMA target branch."""
        _require_torch()
        super().__init__()
        self.patch_embed = patch_embed
        self.target_encoder = deepcopy(online_encoder)
        self.target_bottleneck = deepcopy(bottleneck)
        self.freeze()

    def freeze(self) -> None:
        """Disable backprop through the EMA target branch.

        The target branch exists only to produce stable stop-gradient future
        latents; optimizer updates would collapse the JEPA target contract.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def copy_weights_from_online(
        self, online_encoder: OnlineEncoder, bottleneck: Bottleneck
    ) -> None:
        """Initialize EMA branch from online modules at Stage 0."""
        self.target_encoder.load_state_dict(online_encoder.state_dict())
        self.target_bottleneck.load_state_dict(bottleneck.state_dict())
        self.freeze()

    def forward(self, future_frame: Tensor) -> tuple[Tensor, Tensor]:
        """Encode the future frame into detached target latents.

        The target outputs are always wrapped with `as_target()` so Stage 1 predicts
        them without sending gradients into `E_bar` or `B_bar`.

        Args:
            future_frame: (B, C=3, H=128, W=128) target frame in [-1, 1].
        Returns:
            target_detailed: (B, N_tgt, D_e) detached detailed target latent.
            target_abstract: (B, N_c, D_c) detached abstract target latent.
        """
        with torch.no_grad():
            tokens, _ = self.patch_embed.forward_target(future_frame)
            target_detailed = self.target_encoder(tokens)
            target_abstract = self.target_bottleneck(target_detailed, None)
        return as_target(target_detailed), as_target(target_abstract)


def _build_phase1_modules(
    cfg: Config,
) -> tuple[PatchEmbed, OnlineEncoder, Bottleneck, TargetBranch, CoarseFlow]:
    """Construct all Phase 1 modules in data-path order.

    Keeping construction in one place prevents mismatched online/target module
    instances and gives Stage 0 a single canonical module bundle.

    Args:
        cfg: Global config containing Phase 1 architecture constants.
    Returns:
        Modules `(patch_embed, online_encoder, bottleneck, target_branch, coarse_flow)`.
    """
    patch_embed = PatchEmbed(cfg.model)
    online_encoder = OnlineEncoder(cfg.model)
    bottleneck = Bottleneck(cfg.model)
    target_branch = TargetBranch(patch_embed, online_encoder, bottleneck)
    coarse_flow = CoarseFlow(cfg.model)
    return patch_embed, online_encoder, bottleneck, target_branch, coarse_flow


def smoke_test_models() -> None:
    """Run synthetic Phase 1 model shape and gradient checks.

    This is the Phase 1 verification hook requested by PHASE_1.md. It confirms
    shapes and the critical gradient boundary before real SSv2 data is available.
    """
    from losses import flow_matching_loss, interpolate, velocity_target

    cfg = Config()
    modules = _build_phase1_modules(cfg)
    patch_embed, online_encoder, bottleneck, target_branch, coarse_flow = modules
    context_clip = torch.randn(2, 4, 3, 128, 128)
    future_frame = torch.randn(2, 3, 128, 128)
    tokens, kept_mask = patch_embed.forward_context(context_clip)
    detailed = online_encoder(tokens)
    abstract = bottleneck(detailed, kept_mask)
    target_detailed, target_abstract = target_branch(future_frame)
    assert target_detailed.requires_grad is False
    assert target_abstract.requires_grad is False
    eps = torch.randn_like(target_abstract)
    tau = torch.rand(2)
    z_c = interpolate(target_abstract, eps, tau)
    u_c = velocity_target(target_abstract, eps)
    u_c_hat = coarse_flow(z_c, tau, abstract)
    loss = flow_matching_loss(u_c_hat, u_c)
    loss.backward()
    assert any(p.grad is not None for p in online_encoder.parameters())
    assert any(p.grad is not None for p in bottleneck.parameters())
    assert any(p.grad is not None for p in coarse_flow.parameters())
    assert all(p.grad is None for p in target_branch.target_encoder.parameters())
    print("Phase 1 model smoke test passed")
