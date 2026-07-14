# Investigation 016 — EGO4D transfer of the strongest present-only recipe

## Question

Does the strongest present-only autoencoder recipe found on SSv2 (investigation_015
run 057, `ae_latent_stack_whiten_abs_recon_cov_var`) reproduce its healthy
representation geometry and honest, video-specific reconstruction when the **only** thing
that changes is the dataset — SSv2 → EGO4D — plus the whitening statistics that dataset
requires?

## Status

OPEN. First run complete and analyzed:
[`run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/`](run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/)
(W&B `mvbx96nv`, group `inv016_ego4d_transfer`, finished 15,000 steps).

## Why this investigation exists

EGO4D was added to the codebase as an **additive sibling dataset** (see
[`AGENT_FILES/KNOWLEDGE/ego4d/`](../../../AGENT_FILES/KNOWLEDGE/ego4d/)). The migration's
stated purpose is to break the `coarse_vs_copy_ratio` failure that SSv2's static-camera,
near-identical-consecutive-frame footage causes for the full-prediction path. Before
spending a ~6 h full-prediction A/B on EGO4D, this investigation asks a cheaper, prior
question: **does the abstract representation itself even stay healthy on EGO4D under the
recipe that made it healthiest on SSv2?** A present-only autoencoder isolates that question
— it exercises the whole data → frozen encoder → whitening → bottleneck → decoder → loss
pipeline without the future-prediction branch, so any change is attributable to the
substrate, not to `F_c`.

## Design — clean single-variable dataset A/B

The run is byte-identical to SSv2 run 057 except the dataset and its whitening-stats file
(W&B `compare_runs` confirms: the ONLY config deltas are `data.dataset` ego4d↔ssv2,
`whiten_stats_path` ego4d↔ssv2, and the cosmetic `checkpoint_dir`). Shared recipe:

```text
present-only reconstruction, cosine loss
ABSOLUTE (clean) whitened-feature target   (recon_residual_target = FALSE)
FIXED offline feature whitening            (EGO4D-specific stats, whiten_eps=1e-4)
Perceiver latent-stack bottleneck          (bottleneck_latent_blocks = 3)
lambda_var=0.5, lambda_cov=0.01, lambda_sigreg=0, lambda_slot=0
lambda_recon=0.05, recon_warmup_steps=2000
no prediction (F_c inactive), decoder 512x4, n_c=32, seed 42, 15,000 steps
horizon_k=12, frame_stride=2  (EGO4D chunks are re-encoded to 12 FPS, Option A, so the
                               SSv2-calibrated config carries over unchanged)
```

Baseline for every overlay: run 057 (`cdvp6hou`, SSv2), the SSv2 arm of the same recipe.

## Verdict (one line)

**Collapsed rep / template shortcut.** The recipe does NOT transfer: on EGO4D the code
collapses to a near-video-independent template (cross-video cosine 0.863, effective rank
52.9, std 0.419, video-conditioned reconstruction share ~5.6%), where the identical recipe
on SSv2 held rank ~208, cosine ~0.059, std ~1.12, and ~79% video-conditioned. Pipeline and
optimization are perfectly healthy; the representation is not. Full reading:
[`run_058.../METRIC_READOUT.md`](run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/METRIC_READOUT.md)
and [`run_058.../ANALYSIS.md`](run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md).
