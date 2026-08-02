# Observations - run 036 `upbeat-frog-36`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `b4lf89if`  
**State:** `killed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=killed; step=250; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.6437 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.4952; dead_dim=0; cross_video_cosine=0.7239 |
| Q3 | Rich latent? | FAIL | c_effective_rank=9.4729; c_plus_effective_rank=9.2304 |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=flat; ratio=22.9366; story=insufficient temporal evidence or early run |
| Q5 | F_c beats copy? | FAIL | model_loss=1.2232; copy_loss=0.0533; ratio=22.9366; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=24.3867; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=1.0386; cplus=1.0378; chat=1.0373; chat-cplus=-5.323e-04; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Smoke / inconclusive | The run is too short for Phase 1 learning gates; use it as infrastructure evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 1.3752 @ 0 | 1.2561 @ 250 | -0.1191 | 1.2561 / 1.3522 / 1.4764 | n/a |
| `L_flow` | 1.1691 @ 0 | 1.2176 @ 250 | 0.0485 | 1.1691 / 1.2676 / 1.4416 | n/a |
| `L_var` | 0.4121 @ 0 | 0.0437 @ 250 | -0.3684 | 0.0416 / 0.1521 / 0.4121 | n/a |
| `L_sigreg` | 0.0444 @ 0 | 0.0095 @ 250 | -0.0349 | 0.0073 / 0.0168 / 0.0444 | n/a |
| `sigreg_scale` | 0 @ 0 | 0.125 @ 250 | 0.125 | 0 / 0.0625 / 0.125 | n/a |
| `L_recon` | 1.0389 @ 0 | 0.8496 @ 250 | -0.1893 | 0.8496 / 0.9368 / 1.0389 | n/a |
| `L_recon_pred` | 1.0385 @ 0 | 0.8494 @ 250 | -0.1891 | 0.8494 / 0.9365 / 1.0385 | n/a |
| `recon_scale` | 0 @ 0 | 0.125 @ 250 | 0.125 | 0 / 0.0625 / 0.125 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 0.4952 @ 0 | 0 | 0.4952 / 0.4952 / 0.4952 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.7239 @ 0 | 0 | 0.7239 / 0.7239 / 0.7239 | n/a |
| `c_effective_rank` | 9.4729 @ 0 | 9.4729 @ 0 | 0 | 9.4729 / 9.4729 / 9.4729 | n/a |
| `c_plus_effective_rank` | 9.2304 @ 0 | 9.2304 @ 0 | 0 | 9.2304 / 9.2304 / 9.2304 | n/a |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 16.7111 @ 0 | 0 | 16.7111 / 16.7111 / 16.7111 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9999 @ 0 | 0 | 0.9999 / 0.9999 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.9998 @ 0 | 0 | 0.9998 / 0.9998 / 0.9998 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.0533 @ 0 | 0 | 0.0533 / 0.0533 / 0.0533 | n/a |
| `coarse_model_loss` | 1.2232 @ 0 | 1.2232 @ 0 | 0 | 1.2232 / 1.2232 / 1.2232 | n/a |
| `coarse_batch_mean_loss` | 0.0502 @ 0 | 0.0502 @ 0 | 0 | 0.0502 / 0.0502 / 0.0502 | n/a |
| `coarse_vs_copy_ratio` | 22.9366 @ 0 | 22.9366 @ 0 | 0 | 22.9366 / 22.9366 / 22.9366 | n/a |
| `coarse_vs_batch_mean_ratio` | 24.3867 @ 0 | 24.3867 @ 0 | 0 | 24.3867 / 24.3867 / 24.3867 | n/a |
| `L_recon_present` | 1.0386 @ 0 | 1.0386 @ 0 | 0 | 1.0386 / 1.0386 / 1.0386 | n/a |
| `L_recon_cplus` | 1.0378 @ 0 | 1.0378 @ 0 | 0 | 1.0378 / 1.0378 / 1.0378 | n/a |
| `L_recon_chat` | 1.0373 @ 0 | 1.0373 @ 0 | 0 | 1.0373 / 1.0373 / 1.0373 | n/a |
| `grad_norm` | 1.5719 @ 0 | 0.6437 @ 250 | -0.9282 | 0.5025 / 0.9516 / 1.7302 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 250 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.1673 @ 250 | 0.1667 | 6.667e-04 / 0.084 / 0.1673 | n/a |

## Last Key Metrics

`c_effective_rank`=9.4729 @ 0; `c_cross_video_cosine`=0.7239 @ 0; `c_std_mean`=0.4952 @ 0; `coarse_vs_copy_ratio`=22.9366 @ 0; `coarse_vs_batch_mean_ratio`=24.3867 @ 0; `L_recon_present`=1.0386 @ 0

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 22.9366 and the latest batch-mean ratio is 24.3867; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 9.4729, cross-video cosine is 0.7239, and copy loss is 0.0533. The reading-cycle verdict is **Smoke / inconclusive** because The run is too short for Phase 1 learning gates; use it as infrastructure evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The research moved to reconstruction geometry and decoder honesty, because prediction failure could no longer be blamed only on rank collapse.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (W&B `b4lf89if`):** reached only ~step 250, so all diagnostics are step-0
init values (rank 9.47, cross-video cosine 0.72, std 0.50, copy ratio 22.9). No learning signal —
this is an infrastructure record confirming the residual + plumbing launch is well-formed
(predict_residual in config, sigreg_scale ramping, finite losses), not a result. Relaunched
as run 037.
