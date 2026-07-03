# Observations - run 026 `pious-mountain-28`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `7u5zkw6t`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=200; grad_skipped max=0; grad_has_nan max=0; grad_norm last=1.2448 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.4799; dead_dim=0; cross_video_cosine=0.7359 |
| Q3 | Rich latent? | FAIL | c_effective_rank=8.8219; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=flat; ratio=62.9974; story=insufficient temporal evidence or early run |
| Q5 | F_c beats copy? | FAIL | model_loss=3.3072; copy_loss=0.0525; ratio=62.9974; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=12.9475; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=1.0299; cplus=1.0289; chat=1.0284; chat-cplus=-5.516e-04; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Smoke / inconclusive | The run is too short for Phase 1 learning gates; use it as infrastructure evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0952 @ 0 | 2.6372 @ 200 | -0.4581 | 2.6372 / 3.0381 / 3.5318 | n/a |
| `L_flow` | 2.8763 @ 0 | 2.5855 @ 200 | -0.2908 | 2.5855 / 2.9415 / 3.3901 | n/a |
| `L_var` | 0.4378 @ 0 | 0.0939 @ 200 | -0.3439 | 0.0496 / 0.1882 / 0.4378 | n/a |
| `L_recon` | 1.0346 @ 0 | 0.9372 @ 200 | -0.0974 | 0.9372 / 0.9974 / 1.0346 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 0.1 @ 200 | 0.1 | 0 / 0.05 / 0.1 | n/a |
| `c_std_mean` | 0.4799 @ 0 | 0.4799 @ 0 | 0 | 0.4799 / 0.4799 / 0.4799 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7359 @ 0 | 0.7359 @ 0 | 0 | 0.7359 / 0.7359 / 0.7359 | n/a |
| `c_effective_rank` | 8.8219 @ 0 | 8.8219 @ 0 | 0 | 8.8219 / 8.8219 / 8.8219 | n/a |
| `c_slot_diversity_rank` | 24.144 @ 0 | 24.144 @ 0 | 0 | 24.144 / 24.144 / 24.144 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9999 @ 0 | 0 | 0.9999 / 0.9999 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.9998 @ 0 | 0 | 0.9998 / 0.9998 / 0.9998 | n/a |
| `coarse_copy_loss` | 0.0525 @ 0 | 0.0525 @ 0 | 0 | 0.0525 / 0.0525 / 0.0525 | n/a |
| `coarse_model_loss` | 3.3072 @ 0 | 3.3072 @ 0 | 0 | 3.3072 / 3.3072 / 3.3072 | n/a |
| `coarse_batch_mean_loss` | 0.2554 @ 0 | 0.2554 @ 0 | 0 | 0.2554 / 0.2554 / 0.2554 | n/a |
| `coarse_vs_copy_ratio` | 62.9974 @ 0 | 62.9974 @ 0 | 0 | 62.9974 / 62.9974 / 62.9974 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.9475 @ 0 | 12.9475 @ 0 | 0 | 12.9475 / 12.9475 / 12.9475 | n/a |
| `L_recon_present` | 1.0299 @ 0 | 1.0299 @ 0 | 0 | 1.0299 / 1.0299 / 1.0299 | n/a |
| `L_recon_cplus` | 1.0289 @ 0 | 1.0289 @ 0 | 0 | 1.0289 / 1.0289 / 1.0289 | n/a |
| `L_recon_chat` | 1.0284 @ 0 | 1.0284 @ 0 | 0 | 1.0284 / 1.0284 / 1.0284 | n/a |
| `grad_norm` | 1.624 @ 0 | 1.2448 @ 200 | -0.3793 | 0.471 / 1.1163 / 1.624 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.134 @ 200 | 0.1333 | 6.667e-04 / 0.0673 / 0.134 | n/a |

## Last Key Metrics

`c_effective_rank`=8.8219 @ 0; `c_cross_video_cosine`=0.7359 @ 0; `c_std_mean`=0.4799 @ 0; `coarse_vs_copy_ratio`=62.9974 @ 0; `coarse_vs_batch_mean_ratio`=12.9475 @ 0; `L_recon_present`=1.0299 @ 0

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 62.9974 and the latest batch-mean ratio is 12.9475; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 8.8219, cross-video cosine is 0.7359, and copy loss is 0.0525. The reading-cycle verdict is **Smoke / inconclusive** because The run is too short for Phase 1 learning gates; use it as infrastructure evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — pious-mountain-28 (n_c=64)

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **350 s**, last heartbeat **03:47:43Z** — the same instant as all four
siblings (synchronized whole-pod death; see [Wave 2 OBSERVATIONS](../OBSERVATIONS.md) for the
forensics). Diagnostics log every 500 steps, so the only logged row is **step 0 (initialization)**.

### The step-0 values are initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `L_recon_present` | 1.030 | ≈ 1.0 = untrained (as bad as predicting the mean) |
| `coarse_vs_copy_ratio` | 63.0 | random-init value (every run starts here) |
| `c_effective_rank` | 8.82 | pre-training |
| `c_cross_video_cosine` | 0.736 | collapsed random init (drops to ~0.2 only after training) |

These are indistinguishable from every Wave-1 run's step 0 and say **nothing** about the n_c=64
hypothesis. The recon warmup runs to 2000 steps; this died at 10% of warmup.

## What we still don't know

Whether doubling `n_c` moves the 0.585 floor — the central open question of investigation_007. The
[pre-registered prediction](../../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md) (floor flat or cosmetic,
copy-ratio still >1 → pivot) is **untested**. This run is a Tier-0 re-run priority — see
[NEXT_STEPS.md](NEXT_STEPS.md).
