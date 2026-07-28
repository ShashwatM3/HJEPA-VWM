# Run 072 — V-JEPA2 unwhitened M=512 with SIGReg only

## Status

PLANNED — not yet launched.

## W&B

- Display name: `Investigation 18 · Regularizer replacement · V-JEPA2 isotropy 10 only`
- Group: `inv018_vjepa2_sigreg_only`
- Entity / project: `smahalanobis-uc-davis/hjepa-vwm`
- W&B ID / URL: pending launch.

## Question

Can SIGReg alone preserve a decodable, high-rank V-JEPA2 present representation on the current
raw/unwhitened bottleneck when both the variance floor and covariance penalty are disabled?

## Exact recipe

- full EGO4D, seed 42, physical batch 64, 15,000 updates;
- frozen `vjepa2_vitl16` at registry-pinned revision
  `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, bf16, SDPA, frame microbatch 8;
- native detailed lattice `(B,1024,1024)`;
- raw/unwhitened present-only absolute cosine reconstruction;
- `M=512`, three latent blocks, `N_c=32`, `D_c=256`;
- decoder `512×4`;
- `lambda_recon=1`, `lambda_recon_pred=0`;
- `lambda_sigreg=10` with a 2,000-step linear ramp;
- `lambda_var=lambda_cov=lambda_slot=0`;
- all three learning rates `1e-4`, 1,500-step LR warmup, checkpoint every 2,500 updates.

## Delta from the latest scientific run

The latest scientific run is DINOv3 W&B
[`fiactcw6`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fiactcw6).

| Field | `fiactcw6` | This run |
|---|---|---|
| frozen encoder | DINOv3 ViT-B/16 | V-JEPA2 ViT-L/16 |
| resolved feature lattice | `8×16×16`, width 768 | `4×16×16`, width 1024 |
| frame microbatch | 32 | 8 |
| `lambda_var` | 0.5 | 0 |
| `lambda_cov` | 0.01 | 0 |
| `lambda_sigreg` | 0 | 10 |
| all other scientific fields | latest raw `M=512`, `32×256`, decoder `512×4` recipe | unchanged |

The frame-microbatch change is a mechanical adapter default. The encoder and regularizer changes
are both scientific deltas, so this run does not isolate either one against `fiactcw6`. Raw
reconstruction losses are not numerically comparable across the two encoder feature spaces.

## Within-V-JEPA2 comparisons

- Covariance plus variance: W&B `guiduvjp`, stable intentional partial endpoint.
- No geometry pressure: W&B `4biwq87o`, complete low-rank decodable control.

These are the defensible references for whether pure SIGReg changes V-JEPA2 geometry and
correct-versus-rolled code dependence under the current bottleneck.
