# Observations - run 014 `jolly-forest-14`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `8bkeeuio`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=3900; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.6833 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9446; dead_dim=0; cross_video_cosine=0.2281 |
| Q3 | Rich latent? | FAIL | c_effective_rank=9.392; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.165 (0.0533 @ 0 -> 0.2183 @ 3500); ratio=1.5639; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.3414; copy_loss=0.2183; ratio=1.5639; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.3607; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0821 @ 0 | 0.3037 @ 3900 | -2.7783 | 0.3037 / 0.8897 / 3.5248 | n/a |
| `L_flow` | 2.8663 @ 0 | 0.2991 @ 3900 | -2.5672 | 0.2991 / 0.8729 / 3.3741 | n/a |
| `L_var` | 0.4316 @ 0 | 0.0093 @ 3900 | -0.4223 | 0.0093 / 0.0337 / 0.4316 | n/a |
| `c_std_mean` | 0.4953 @ 0 | 0.9446 @ 3500 | 0.4494 | 0.4953 / 0.8492 / 0.9446 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 3500 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.2281 @ 3500 | -0.4958 | 0.1779 / 0.278 / 0.7239 | n/a |
| `c_effective_rank` | 9.4728 @ 0 | 9.392 @ 3500 | -0.0808 | 6.6944 / 8.3028 / 10.1252 | n/a |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 18.9801 @ 3500 | 2.2691 | 16.7111 / 18.7525 / 20.4704 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9666 @ 3500 | -0.0334 | 0.9666 / 0.9902 / 1 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.9244 @ 3500 | -0.0754 | 0.9244 / 0.9784 / 0.9999 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.2183 @ 3500 | 0.165 | 0.0533 / 0.2089 / 0.2662 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 0.3414 @ 3500 | -2.9692 | 0.3414 / 1.0433 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 0.9464 @ 3500 | 0.6837 | 0.2627 / 0.7691 / 0.9464 | n/a |
| `coarse_vs_copy_ratio` | 62.0666 @ 0 | 1.5639 @ 3500 | -60.5027 | 1.5639 / 10.4145 / 62.0666 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.3607 @ 3500 | -12.2404 | 0.3607 / 2.355 / 12.6011 | n/a |
| `grad_norm` | 1.5844 @ 0 | 2.6833 @ 3900 | 1.0989 | 0.5868 / 2.6106 / 3.7896 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 3900 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 3500 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.924 @ 3900 | 0.9234 | 6.667e-04 / 0.788 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=9.392 @ 3500; `c_cross_video_cosine`=0.2281 @ 3500; `c_std_mean`=0.9446 @ 3500; `coarse_vs_copy_ratio`=1.5639 @ 3500; `coarse_vs_batch_mean_ratio`=0.3607 @ 3500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 1.5639 and the latest batch-mean ratio is 0.3607; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 9.392, cross-video cosine is 0.2281, and copy loss is 0.2183. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — jolly-forest-14

## Outcome

**Winning config reproduced — then crashed early at step 3900.** Over its short life it
re-traced cerulean-snow-13's healthy trajectory (variance climbing to the floor,
cross-video cosine low, model beating batch-mean), independently confirming the
`lambda_var=0.5` result before the crash ended it.

## Key numbers (verified vs W&B `8bkeeuio`, var=0.5, k=12, no slot/cov)

| Step | `c_std_mean` | `c_cross_video_cosine` | `c_effective_rank` | `coarse_vs_copy_ratio` | `coarse_vs_batch_mean_ratio` | `grad_skipped` |
|---|---|---|---|---|---|---|
| 0 | 0.50 | 0.72 | 9.47 | 62.1 | 12.6 | 0 |
| 500 | 0.87 | **0.18** | 10.13 | 4.78 | 1.73 | 0 |
| 1500 | 0.87 | 0.23 | 7.55 | 3.69 | 1.06 | 0 |
| 2500 | 0.92 | 0.21 | 6.69 | 2.03 | **0.50** | 0 |
| 3500 | 0.94 | 0.23 | 9.39 | 1.56 | **0.36** | 0 |

These tracks are nearly identical to [`cerulean-snow-13`](../run_013_cerulean-snow-13/) over the
same step range (e.g. cerulean at 3500: std 0.93, cosine 0.24, rank 9.27, batch-mean
0.37) — strong reproducibility of the winning config.

## Interpretation

- **Not exploratory or off-config.** Earlier KANBAN treated this as an uncertain
  "intermediate" run; W&B confirms it is the winning `lambda_var=0.5` k=12 config,
  reproducing cerulean. It belongs to the winning-config lineage, bridging
  investigation_003 → investigation_005.
- `c_std_mean` is already climbing toward 1.0 and the model already **beats batch-mean**
  (ratio < 1 from step 2500) — the same reversal cerulean showed.
- It crashed at 3900 (state: crashed), so it never reached 15k. The clean 15k push moved
  to [`elated-snowflake-15`](../../investigation_005/run_015_elated-snowflake-15/).

## Chronology note

Created 06-20, after cerulean-snow-13 (06-19) and before elated-snowflake-15 (06-20).
Treat cerulean as the canonical "Run 6 win" and this as its short confirmation; neither
is the completed acceptance run.
