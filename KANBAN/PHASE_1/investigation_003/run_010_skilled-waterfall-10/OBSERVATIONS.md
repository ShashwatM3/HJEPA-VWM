# Observations - run 010 `skilled-waterfall-10`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `27i1r9qi`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Invalid**  
**Verdict note:** Training-health signals failed; later metrics should not be treated as reliable evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | FAIL | state=crashed; step=2550; grad_skipped max=1; grad_has_nan max=0; grad_norm last=17.0764 |
| Q2 | c_t alive / video-specific? | PARTIAL | std=0.6035; dead_dim=0; cross_video_cosine=0.5756 |
| Q3 | Rich latent? | FAIL | c_effective_rank=7.2924; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.6041 (0.0533 @ 0 -> 0.6575 @ 2500); ratio=1.8968; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.2471; copy_loss=0.6575; ratio=1.8968; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=2.3786; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Invalid | Training-health signals failed; later metrics should not be treated as reliable evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 2.9639 @ 0 | 1.3408 @ 2550 | -1.6231 | 1.2714 / 1.5117 / 3.463 | n/a |
| `L_flow` | 2.8663 @ 0 | 1.2435 @ 2550 | -1.6228 | 1.1733 / 1.4225 / 3.3739 | n/a |
| `L_var` | 0.4316 @ 0 | 0.394 @ 2550 | -0.0375 | 0.0833 / 0.2956 / 0.5071 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 0.6035 @ 2500 | 0.1082 | 0.4952 / 0.5944 / 0.7766 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 2500 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.5756 @ 2500 | -0.1482 | 0.3403 / 0.5948 / 0.7239 | n/a |
| `c_effective_rank` | 9.4733 @ 0 | 7.2924 @ 2500 | -2.1809 | 7.2924 / 10.3131 / 12.6847 | n/a |
| `c_slot_diversity_rank` | 16.711 @ 0 | 3.8182 @ 2500 | -12.8928 | 3.8182 / 8.1077 / 16.711 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9993 @ 2500 | -6.166e-04 | 0.9993 / 0.9997 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.977 @ 2500 | -0.0228 | 0.977 / 0.9948 / 0.9998 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.6575 @ 2500 | 0.6041 | 0.0533 / 0.5418 / 1.1519 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 1.2471 @ 2500 | -2.0635 | 1.1949 / 1.6036 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 0.5243 @ 2500 | 0.2616 | 0.2627 / 0.5182 / 0.6736 | n/a |
| `coarse_vs_copy_ratio` | 62.066 @ 0 | 1.8968 @ 2500 | -60.1692 | 1.1149 / 12.2978 / 62.066 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 2.3786 @ 2500 | -10.2224 | 1.9983 / 3.958 / 12.6011 | n/a |
| `grad_norm` | 0.2672 @ 0 | 17.0764 @ 2550 | 16.8092 | 0.1843 / 29.2439 / 203.907 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 2550 | 0 | 0 / 0.2115 / 1 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 2500 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.9851 @ 2550 | 0.9845 | 6.667e-04 / 0.7002 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=7.2924 @ 2500; `c_cross_video_cosine`=0.5756 @ 2500; `c_std_mean`=0.6035 @ 2500; `coarse_vs_copy_ratio`=1.8968 @ 2500; `coarse_vs_batch_mean_ratio`=2.3786 @ 2500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.8968 and the latest batch-mean ratio is 2.3786; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 7.2924, cross-video cosine is 0.5756, and copy loss is 0.6575. The reading-cycle verdict is **Invalid** because Training-health signals failed; later metrics should not be treated as reliable evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — skilled-waterfall-10

## Headline finding

**`L_slot` glued at ~0.99–1.00 for the entire run** — the gradient did not move the
slot objective at all, even with `lambda_slot=0.05` active. Meanwhile the diagnostic
`c_slot_diversity_rank` slid 16.7 → 3.8, exactly the collapse the loss was supposed to
prevent. The loss was inert.

## Key numbers (verified vs W&B `27i1r9qi`, k=4, raw slot loss)

| Step | `L_slot` | `c_slot_diversity_rank` | `c_effective_rank` | `c_cross_video_cosine` | `coarse_vs_copy_ratio` | `grad_norm` | `grad_skipped` |
|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 16.7 | 9.47 | 0.72 | 62.1 | 0.27 | 0 |
| 500 | 0.998 | 8.1 | 12.68 | 0.34 | 3.19 | 2.99 | 0 |
| 1000 | 0.998 | 7.3 | 12.39 | 0.51 | 3.60 | 8.86 | 0 |
| 1500 | 1.000 | 6.9 | 10.43 | 0.70 | 1.11 | 55.1 | **1** |
| 2000 | 0.999 | 5.8 | 9.60 | 0.72 | 1.92 | 50.8 | **1** |
| 2500 | 0.993 | 3.8 | 7.29 | 0.58 | 1.90 | 204 | **1** |

Run crashed at `_step=2550`. (`coarse_vs_copy_ratio` near 1 here is **not** progress —
it's the moving-baseline artifact: `copy_loss` rises as the latent drifts, so the ratio
flatters a collapsing model. See the same caution in
[`cerulean-snow-13`](../run_013_cerulean-snow-13/) where the ratio became trustworthy only once
the representation stabilized.)

## Root cause (code) — the slot loss/metric mismatch

`slot_diversity_loss` in `losses.py` computed cosine similarity on the **raw** slot
vectors, while the `slot_diversity_rank` diagnostic first **mean-centered** the slots.
A constant DC component shared across all 32 slots makes raw pairwise cosine ≈ 1
regardless of how the slots actually differ in their residual directions — so the loss
sat pinned near 1.0 and its gradient carried almost no information about the metric the
team was trying to move. Fixed by centering the slots in the loss to match the
diagnostic (`commit ffc33ed`, 06-17).

## Interpretation

This is **not** a verdict on slot loss in general — it is a verdict on a buggy
implementation. No `lambda_slot` experiment is interpretable until loss and metric
agree, which is why the centering fix was made **mandatory** before the next slot run
([`copper-sky-12`](../run_012_copper-sky-12/), the post-fix rerun). It is also the canonical
"don't just add a CLI flag — read the code" realization the user had pushed for.

Note also the early grad-skips (step 1500+) and the SSH drop ~step 950 in chat: between
the inert loss, the k=4/k=12 launch drift, and the early crash, this run produced no
clean signal on either the horizon or the slot lever.
