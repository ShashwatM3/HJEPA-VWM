# Run 68 — Investigation 16 · Encoder substrate · DINOv3 whitened memory M=512 covariance plus variance

## Status

LAUNCHED — qualitative observations were supplied after launch, but completion, exact W&B identity, checkpoint hashes, and final metrics are not repository-verified.

## W&B

- Display name: `Investigation 16 · Encoder substrate · DINOv3 whitened memory M=512 covariance plus variance`
- Group: `inv016_encoder_substrate_whitened_memory`
- Entity / project: `smahalanobis-uc-davis/hjepa-vwm`
- ID / URL: to be filled after launch.

## Question

Does fixed offline feature whitening plus the canonical covariance-and-variance geometry pressure
improve the DINOv3 M=512 present representation relative to the completed unwhitened DINOv3 arm?

The baseline is
[`../run_067_dinov3_unwhitened_internal_memory_m512/`](../run_067_dinov3_unwhitened_internal_memory_m512/),
W&B [`it7sq8nz`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/it7sq8nz). Preserve its full
scientific configuration and change only the whitening bundle:

| Field | Run 67 | Run 68 |
|---|---:|---:|
| `--lambda-var` | `0` | `0.5` |
| `--lambda-cov` | `0` | `0.01` |
| `--lambda-sigreg` | `0` | `0` |
| `--lambda-slot` | `0` | `0` |
| feature whitening | off | fixed offline ZCA on |
| `--whiten-eps` | inactive (`1e-4` default) | `1e-4` |
| `--whiten-expected-clips` | inactive | `12800` |

The whitening envelope must be newly fitted for DINOv3/EGO4D at the final settled frame
microbatch. V-JEPA and SigLIP whitening statistics are invalid for this run.

## Locked configuration

Full EGO4D; `dinov3_vitb16` @ `5931719e67bbdb9737e363e781fb0c67687896bc`; bf16; SDPA; seed 42;
physical batch 64; 15,000 steps; horizon 12; present-only absolute cosine reconstruction;
`lambda_recon=1`; `lambda_recon_pred=0`; 2,000-step reconstruction warmup; `M=512`; three latent
blocks; `N_c=32`; external `D_c=256`; 512-by-4 decoder; all three learning rates `1e-4`; logging
every 50; diagnostics every 500; no residual target and no resume. Frame microbatch 32 is the
baseline candidate and must pass whitening-active resource preflight.

Exact gated execution: [`GUIDE.md`](GUIDE.md). Implementation/execution plan: [`PLAN.md`](PLAN.md).

## Preliminary qualitative report

Whitening was reported active. Reconstruction worsened to approximately .3, while effective rank improved and cross-video cosine decreased. These are preliminary qualitative observations, not a final metric readout. Attribution remains confounded across the broader sequence because whitening and the late-projection architecture were both present. Run 69 isolates whitening while preserving the projection placement and lambda_var=0.5 / lambda_cov=0.01.
