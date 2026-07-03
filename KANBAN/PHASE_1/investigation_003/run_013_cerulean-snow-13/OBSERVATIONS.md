# Observations - run 013 `cerulean-snow-13`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `4lo4j7qb`  
**State:** `killed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=killed; step=6900; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.6807 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0378; dead_dim=0; cross_video_cosine=0.2356 |
| Q3 | Rich latent? | FAIL | c_effective_rank=13.6717; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | PARTIAL | copy_loss trend=up 0.1932 (0.0533 @ 0 -> 0.2466 @ 6500); ratio=0.9493; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | PARTIAL | model_loss=0.2341; copy_loss=0.2466; ratio=0.9493; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.2046; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0821 @ 0 | 0.3659 @ 6900 | -2.7161 | 0.2284 / 0.6389 / 3.5248 | n/a |
| `L_flow` | 2.8663 @ 0 | 0.3625 @ 6900 | -2.5038 | 0.2257 / 0.627 / 3.3741 | n/a |
| `L_var` | 0.4316 @ 0 | 0.007 @ 6900 | -0.4246 | 0.0048 / 0.0238 / 0.4316 | n/a |
| `c_std_mean` | 0.4953 @ 0 | 1.0378 @ 6500 | 0.5426 | 0.4953 / 0.9148 / 1.0434 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 6500 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.2356 @ 6500 | -0.4882 | 0.1773 / 0.2583 / 0.7239 | n/a |
| `c_effective_rank` | 9.4728 @ 0 | 13.6717 @ 6500 | 4.1989 | 7.0215 / 10.207 / 13.6717 | n/a |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 19.5321 @ 6500 | 2.821 | 16.7111 / 18.8953 / 20.5189 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.93 @ 6500 | -0.07 | 0.93 / 0.9695 / 1 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.8147 @ 6500 | -0.1852 | 0.8147 / 0.9265 / 0.9999 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.2466 @ 6500 | 0.1932 | 0.0533 / 0.231 / 0.2762 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 0.2341 @ 6500 | -3.0765 | 0.2341 / 0.7187 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.1442 @ 6500 | 0.8814 | 0.2627 / 0.8944 / 1.1442 | n/a |
| `coarse_vs_copy_ratio` | 62.0666 @ 0 | 0.9493 @ 6500 | -61.1172 | 0.8556 / 6.4053 / 62.0666 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.2046 @ 6500 | -12.3965 | 0.2046 / 1.4619 / 12.6011 | n/a |
| `grad_norm` | 1.5844 @ 0 | 2.6807 @ 6900 | 1.0963 | 0.5876 / 2.6007 / 4.8327 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 6900 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 6500 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.6545 @ 6900 | 0.6538 | 6.667e-04 / 0.7929 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=13.6717 @ 6500; `c_cross_video_cosine`=0.2356 @ 6500; `c_std_mean`=1.0378 @ 6500; `coarse_vs_copy_ratio`=0.9493 @ 6500; `coarse_vs_batch_mean_ratio`=0.2046 @ 6500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 0.9493 and the latest batch-mean ratio is 0.2046; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 13.6717, cross-video cosine is 0.2356, and copy loss is 0.2466. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — cerulean-snow-13

## Key numbers (from local log + BRIEF_V0_3)

| Step | `c_std_mean` | `c_cross_video_cosine` | `c_effective_rank` | `coarse_vs_copy_ratio` | `grad_skipped` |
|---|---|---|---|---|---|
| 0 | 0.50 | 0.72 | 9.47 | 62.1 | 0 |
| 4500 | 0.97 | 0.26 | 12.11 | **0.97** | 0 |
| 5000 | 1.00 | 0.22 | 12.64 | 1.37 | 0 |
| 6500 | 1.04 | 0.24 | 13.67 | **0.95** | 0 |
| 6900 | — | — | — | — | 0 (log ends) |

Throughout: `grad_norm` ~2–3, no sustained skips, `c_dead_dim_frac=0`.

## What worked

- Variance floor at 0.5 **wins over** `L_flow` collapse pressure
- Video-specific codes (cosine ~0.20–0.26)
- Rank rising to **~13.7+**
- **Beats copy baseline** (ratio < 1 in best windows)
- Training stable under post-Run-1 hyperparams

## What surprised us

Copy ratio can wobble batch-to-batch (e.g. 1.37 at 5000) — use diagnostic cadence
and trends, not single steps.

## Interpretation

**Validates investigation 003 conclusion.** This is the Phase 1 Stage 1 config to
carry forward. Open question: rank plateau level over full 15k (feeds investigation 005).

## Note on log length

Workspace `output.log` ends at step 6900; chat/W&B record run as healthy further.
Treat as **best known checkpoint window**, not necessarily final step 15000.
