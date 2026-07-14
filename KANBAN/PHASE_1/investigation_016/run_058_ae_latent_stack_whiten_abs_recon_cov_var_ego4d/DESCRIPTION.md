# Run 058 — `ae_latent_stack_whiten_abs_recon_cov_var_ego4d` (inv016)

## Status

FINISHED and analyzed. W&B `smahalanobis-uc-davis/hjepa-vwm/mvbx96nv`, group
`inv016_ego4d_transfer`, present-only autoencoder (Reading Cycle B). Created
2026-07-13, ran the full 15,000-step schedule cleanly (last diag step 14,500;
`_runtime` ≈ 20,008 s ≈ 5.6 h).

## What this run is

The **first EGO4D run** in the project. It transfers investigation_015 run 057's
recipe — the strongest present-only autoencoder the program has produced — onto the
newly added EGO4D sibling dataset, changing nothing but the dataset and the
whitening statistics that dataset requires. It is a clean, single-variable dataset
A/B against run 057 (`cdvp6hou`, SSv2).

## Recipe (identical to run 057 except dataset + whitening stats)

```text
--data ego4d                          (was ssv2)
--whiten-stats-path logs/whiten/whiten_stats_ego4d_train_seed42.pt   (was ssv2 stats)
--present-recon-only                  --whiten-features
--recon-loss-mode cosine              recon_residual_target = FALSE (clean/absolute target)
--lambda-var 0.5  --lambda-cov 0.01   --lambda-sigreg 0  --lambda-slot 0
--lambda-recon 0.05  --recon-warmup-steps 2000
--decoder-dim 512  --decoder-blocks 4  --n-c 32
--horizon-k 12  --seed 42  --steps 15000   (frame_stride 2; EGO4D chunks re-encoded to 12 FPS)
```

W&B `compare_runs(mvbx96nv, cdvp6hou)` confirms the config diff is exactly
`{data.dataset, train.whiten_stats_path, checkpoint_dir}` and nothing else.

## Hypothesis under test

H1 (transfer): the run-057 geometry+honesty win is a property of the recipe, so it
should reproduce on EGO4D — `c_effective_rank` well above the ~40 slot-structure
ceiling, `c_cross_video_cosine` well below 0.5, `c_std_mean` near 1.0, a large and
growing `L_recon_video_gap`, and a high video-conditioned reconstruction share.

H0 (substrate-specific): the win depends on SSv2's feature statistics; on EGO4D the
clean/absolute reconstruction target reopens the video-independent template shortcut
(the run-052 failure mode that the residual target fixed in run 054), and the geometry
regularizers — satisfied on the training batch — fail to hold the held-out
representation.

## Outcome

**H0 confirmed, emphatically.** Verdict: **Collapsed rep / template shortcut**. See
[`METRIC_READOUT.md`](METRIC_READOUT.md) (deliverable 1: metric/graph interpretation and
EGO4D pipeline performance) and [`ANALYSIS.md`](ANALYSIS.md) (deliverable 2: in-depth
analysis, mechanism, and next steps). Belief update and follow-ups:
[`OBSERVATIONS.md`](OBSERVATIONS.md), [`NEXT_STEPS.md`](NEXT_STEPS.md).
