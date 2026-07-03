# Observations - run 015 `elated-snowflake-15`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `jhodg49x`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Invalid**  
**Verdict note:** Training-health signals failed; later metrics should not be treated as reliable evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | FAIL | state=crashed; step=13850; grad_skipped max=1; grad_has_nan max=0; grad_norm last=248.692 |
| Q2 | c_t alive / video-specific? | PASS | std=1.1022; dead_dim=0; cross_video_cosine=0.1655 |
| Q3 | Rich latent? | FAIL | c_effective_rank=13.6879; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.301 (0.0533 @ 0 -> 0.3543 @ 13500); ratio=3.4732; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.2307; copy_loss=0.3543; ratio=3.4732; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.0147; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Invalid | Training-health signals failed; later metrics should not be treated as reliable evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0821 @ 0 | 1.5282 @ 13850 | -1.5539 | 0.2351 / 1.0169 / 3.5248 | 1.3623 / 1.6302 / 1.8728 (n=78) |
| `L_flow` | 2.8663 @ 0 | 1.5267 @ 13850 | -1.3396 | 0.2319 / 1.0102 / 3.3741 | 1.3615 / 1.6289 / 1.8716 (n=78) |
| `L_var` | 0.4316 @ 0 | 0.0029 @ 13850 | -0.4286 | 7.148e-04 / 0.0133 / 0.4316 | 7.148e-04 / 0.0027 / 0.0057 (n=78) |
| `c_std_mean` | 0.4953 @ 0 | 1.1022 @ 13500 | 0.6069 | 0.4953 / 1.0021 / 1.1022 | 1.097 / 1.0999 / 1.1022 (n=8) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 13500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=8) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.1655 @ 13500 | -0.5584 | 0.1655 / 0.2191 / 0.7239 | 0.1655 / 0.1688 / 0.1733 (n=8) |
| `c_effective_rank` | 9.4728 @ 0 | 13.6879 @ 13500 | 4.2151 | 6.889 / 11.9523 / 13.7279 | 13.6253 / 13.6619 / 13.6879 (n=8) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 18.9882 @ 13500 | 2.2772 | 16.7111 / 18.9938 / 20.4312 | 18.9803 / 18.9875 / 19.0045 (n=8) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.918 @ 13500 | -0.082 | 0.9168 / 0.9446 / 1 | 0.9179 / 0.9179 / 0.918 (n=8) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.7895 @ 13500 | -0.2103 | 0.787 / 0.8588 / 0.9999 | 0.7885 / 0.789 / 0.7895 (n=8) |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.3543 @ 13500 | 0.301 | 0.0533 / 0.2889 / 0.3841 | 0.3543 / 0.3717 / 0.3841 (n=8) |
| `coarse_model_loss` | 3.3106 @ 0 | 1.2307 @ 13500 | -2.0799 | 0.2142 / 1.0109 / 3.3106 | 1.2307 / 1.5521 / 1.8876 (n=8) |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.2128 @ 13500 | 0.9501 | 0.2627 / 1.0419 / 1.2128 | 1.2041 / 1.2082 / 1.2128 (n=8) |
| `coarse_vs_copy_ratio` | 62.0666 @ 0 | 3.4732 @ 13500 | -58.5934 | 0.8316 / 5.019 / 62.0666 | 3.4732 / 4.1677 / 4.9144 (n=8) |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 1.0147 @ 13500 | -11.5864 | 0.186 / 1.274 / 12.6011 | 1.0147 / 1.285 / 1.5663 (n=8) |
| `grad_norm` | 1.5844 @ 0 | 248.692 @ 13850 | 247.1077 | 0.5877 / 58.1266 / 467.1484 | 60.014 / 156.5097 / 467.1484 (n=78) |
| `grad_skipped` | 0 @ 0 | 1 @ 13850 | 1 | 0 / 0.3921 / 1 | 1 / 1 / 1 (n=78) |
| `grad_has_nan` | 0 @ 0 | 0 @ 13500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=8) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0178 @ 13850 | 0.0171 | 6.667e-04 / 0.5392 / 1 | 0.0178 / 0.1355 / 0.302 (n=78) |

## Last Key Metrics

`c_effective_rank`=13.6879 @ 13500; `c_cross_video_cosine`=0.1655 @ 13500; `c_std_mean`=1.1022 @ 13500; `coarse_vs_copy_ratio`=3.4732 @ 13500; `coarse_vs_batch_mean_ratio`=1.0147 @ 13500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 3.4732 and the latest batch-mean ratio is 1.0147; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 13.6879, cross-video cosine is 0.1655, and copy loss is 0.3543. The reading-cycle verdict is **Invalid** because Training-health signals failed; later metrics should not be treated as reliable evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next pivot added reconstruction anchors to test whether c_t lacked usable information rather than just variance.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — elated-snowflake-15

## Outcome

**Did not complete 15k usefully.** Crashed ~step 13850 after **~5300 steps of zero learning**
post step 8500.

## Key numbers

| When | Step | Highlights |
|---|---|---|
| Best window | 4500–8000 | copy ratio **0.83–0.96**, rank ~13.7, std ~1.04, no skips |
| Break | 8500 | `grad_norm` **170**, `grad_skipped=1`, copy ratio **5.7** |
| Frozen | 8500–13850 | skip every step, copy ratio **3.5–5.7** |
| Final | 13850 | `grad_norm` ~249, copy ratio **3.47** |

Latent health post-8500: cosine ~0.17, rank ~13.7, std ~1.10 — **misleading** (forward-only).

## What worked

Pre-8500 behavior matched `cerulean-snow-13` expectations: collapse fixed, beats copy,
stable grads.

## What broke

Sudden pre-clip grad spike → skip guard engaged → weights frozen in a region where every
subsequent batch still produced huge grads → no recovery.

## Interpretation

**Config is not the failure mode; optimizer instability late in run is.** Final checkpoint
is garbage; resume from **phase1_step7500** (or 6500–8000 window) with lower flow LR.

## Report

W&B run report: `elated-snowflake-15 — Run Report` (structured narrative + charts).
