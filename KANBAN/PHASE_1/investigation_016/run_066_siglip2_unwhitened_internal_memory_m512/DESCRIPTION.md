# Run 66 — Investigation 16 · Encoder substrate · SigLIP 2 unwhitened memory M=512

## Status

PLANNED — not yet launched. This folder is the pre-launch record (triad + `PLAN.md` + `GUIDE.md`).

## W&B

- Display name: `Investigation 16 · Encoder substrate · SigLIP 2 unwhitened memory M=512`
- Group: `inv016_encoder_substrate_unwhitened_memory`
- ID / URL: to be filled after launch.

## Question

Does swapping only the frozen encoder — from the video transformer **V-JEPA 2 ViT-L/16**
(`vjepa2_vitl16`) to the standard image transformer **SigLIP 2 ViT-B/16** (`siglip2_vitb16`) —
change present-only reconstruction behaviour when everything else is held at the
investigation-016 unwhitened internal-memory M=512 recipe?

This is a **direct encoder-substrate comparison** against the V-JEPA M=512 arm
[`run_064_unwhitened_internal_memory_m512`](../run_064_unwhitened_internal_memory_m512/)
(W&B [`4biwq87o`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4biwq87o)). The single
intended delta versus run 064 is the encoder alias + revision. Loss weights, seed, steps, batch,
horizon, bottleneck width `M=512`, latent blocks, decoder capacity, learning rates, and dataset
stay identical.

## Why run this

The purpose is narrow and diagnostic: **measure the difference between the two encoders in terms
of present-only reconstruction performance**, at a settled architecture, with no confounding
regularizer or whitening changes. There is deliberately **no architectural change** — the
bottleneck's `in_proj` (`D_e -> M`) and the decoder's output projection (`decoder_dim -> D_e`)
already adapt to whatever `EncoderSpec` the selected backend resolves, so the swap is a pure
configuration change with no edits to `models.py`, `losses.py`, or `train.py`.

It is worth documenting despite the low architectural significance because:

1. It is the first **same-commit** V-JEPA-vs-SigLIP comparison at the unwhitened M=512 operating
   point. Run 059 (SigLIP, whitened, cov+var, `lambda_recon=0.05`) predates this recipe and uses a
   different objective, so it is not a controlled encoder comparison for run 064.
2. The two encoders produce **different feature contracts**, so the raw cosine reconstruction
   numbers are **not directly comparable across encoders** (see caveat below). Recording the exact
   config, feature identity, and provenance now prevents a future invalid apples-to-oranges read.

## Config delta versus run 064 (the only intended change)

| Field | Run 064 (V-JEPA) | Run 66 (SigLIP) |
|---|---|---|
| `--encoder` | `vjepa2_vitl16` | `siglip2_vitb16` |
| `--encoder-revision` | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` |
| resolved feature layout | `4x16x16` tubelets → `N_e=1024`, `D_e=1024` | `8x16x16` frame patches → `N_e=2048`, `D_e=768` |
| `--encoder-frame-microbatch` | `8` | settled by resource preflight (candidate `32`, ladder `16/8/4/2/1`) |

Everything else is byte-identical to run 064:

```text
data                = ego4d
steps               = 15000
seed                = 42
encoder-precision   = bf16
encoder-attention   = sdpa
batch-size          = 64
horizon-k           = 12
present-recon-only  = true
lambda-recon        = 1.0
lambda-recon-pred   = 0
recon-loss-mode     = cosine
recon-warmup-steps  = 2000
lambda-var          = 0
lambda-cov          = 0
lambda-sigreg       = 0
lambda-slot         = 0
whitening           = none
bottleneck-mixer-dim (M) = 512
bottleneck-latent-blocks = 3
decoder-dim         = 512
decoder-blocks      = 4
n-c                 = 32   (external D_c = 256)
lr-bottleneck / lr-coarse-flow / lr-decoder = 1e-4 / 1e-4 / 1e-4
log-every / diag-every = 50 / 500
```

**No regularizers and no whitening**, exactly as in the internal-memory-width sweep: variance,
covariance, SIGReg, and slot-loss weights are all zero, and there is no `--whiten-*` flag. Any
whitening `.pt` artifacts already on the volume are ignored.

## Caveat — cross-encoder loss is not directly comparable

V-JEPA 2 and SigLIP 2 emit different `D_e`/`N_e`/layout feature spaces, so the reconstruction
target differs. Do **not** read a lower raw cosine loss as "the better encoder". The defensible
comparisons are *within-run* diagnostics: correct-vs-shuffled gap, exact-chunk conditioned share,
stability, and geometry trajectory — plus a qualitative note on how the two substrates behave under
the identical objective. The fixed EGO4D validation batch is within-source/exact-chunk, so any
"cross-video"/shuffled metric is an exact-chunk claim, not a global cross-source claim (same scope
limit recorded for runs 058–065).

## Context

- Parent investigation: [`../DESCRIPTION.md`](../DESCRIPTION.md) (Investigation 016).
- V-JEPA control at the same operating point:
  [`../run_064_unwhitened_internal_memory_m512`](../run_064_unwhitened_internal_memory_m512/).
- Prior (non-controlled) SigLIP run:
  [`../run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d`](../run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d/)
  — whitened, cov+var, `lambda_recon=0.05`; different recipe, not a control for run 064.
- SigLIP 2 adapter is implemented and pinned (`encoders.py`,
  `siglip2_vitb16` @ `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`, `85,843,200` params, 0 trainable).
- Exact launch: [`GUIDE.md`](GUIDE.md). Implementation/execution notes: [`PLAN.md`](PLAN.md).
