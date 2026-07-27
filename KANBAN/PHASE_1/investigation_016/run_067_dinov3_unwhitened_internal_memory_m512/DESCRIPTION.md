# Run 67 — Investigation 16 · Encoder substrate · DINOv3 unwhitened memory M=512

> **Local-label correction (2026-07-27):** W&B `it7sq8nz` is canonical local
> [Run 069](../run_069_unwhitened_dinov3_m512_no_geometry_regularizers/), not local Run 067.
> This directory preserves source-branch history; use the canonical folder for the current
> analysis and verdict.

## Status

COMPLETE — 15,000/15,000 steps; W&B finished; final checkpoint verified.

## W&B

- Display name: `Investigation 16 · Encoder substrate · DINOv3 unwhitened memory M=512`
- Group: `inv016_encoder_substrate_unwhitened_memory`
- Entity / project: `smahalanobis-uc-davis/hjepa-vwm`
- ID: `it7sq8nz`
- URL: <https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/it7sq8nz>

## Question

Does swapping only the frozen encoder to **DINOv3 ViT-B/16** (`dinov3_vitb16`) change
present-only reconstruction behaviour when everything else is held at the investigation-016
unwhitened internal-memory M=512 recipe?

This is the completed DINOv3 counterpart to the planned SigLIP 2 arm
[`../run_066_siglip2_unwhitened_internal_memory_m512/`](../run_066_siglip2_unwhitened_internal_memory_m512/)
and the V-JEPA M=512 control
[`../run_064_unwhitened_internal_memory_m512/`](../run_064_unwhitened_internal_memory_m512/)
(W&B [`4biwq87o`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4biwq87o)). The intended
scientific delta is the encoder alias + revision. The encoder-only frame microbatch was settled by
resource preflight at 32.

## Runtime identity

| Field | Verified value |
|---|---|
| Git branch | `codex/task3-dino-run066` |
| Git commit | `083cf8a6e87168702efe46ac6bfe485756dcb439` |
| Encoder | `dinov3_vitb16` |
| Encoder revision | `5931719e67bbdb9737e363e781fb0c67687896bc` |
| Steps / batch / horizon | `15000 / 64 / 12` |
| Encoder frame microbatch | `32` |
| Present reconstruction only | `true` |
| Whitening | `false` |
| Bottleneck | `M=512`, three latent blocks, `N_c=32`, external `D_c=256` |
| Decoder | width `512`, four blocks |

Everything else matches the run-066 recipe: full EGO4D, seed 42, bf16, SDPA, absolute cosine
reconstruction at weight 1, 2,000-step reconstruction warmup, all variance/covariance/SIGReg/slot
weights zero, all three learning rates `1e-4`, logging every 50, diagnostics every 500, and no
resume.

## Completion artifacts

```text
checkpoint   /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/phase1_step15000.pt
sha256       f4513c9ce01e18f4db6cc6c3434c5a5f53749019f36992ed332fc14e67dd33db
provenance   /workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/run_provenance.json
W&B artifact it7sq8nz-provenance:v0 (type run-provenance, state COMMITTED)
```

The checkpoint itself remains only on the RunPod persistent volume. It was not uploaded as a W&B
artifact; W&B contains the run metrics and committed provenance artifact.

## Caveat — cross-encoder loss is not directly comparable

DINOv3 and V-JEPA emit different feature spaces, so a lower raw cosine reconstruction loss is not
an encoder-winner claim. Compare within-run correct-vs-shuffled separation, conditioned share,
geometry trajectory, and stability. The fixed EGO4D validation batch is within-source/exact-chunk,
so shuffled metrics do not establish global cross-source conditioning.

Exact reproduction command: [`GUIDE.md`](GUIDE.md). Result and comparison placeholders:
[`OBSERVATIONS.md`](OBSERVATIONS.md).
