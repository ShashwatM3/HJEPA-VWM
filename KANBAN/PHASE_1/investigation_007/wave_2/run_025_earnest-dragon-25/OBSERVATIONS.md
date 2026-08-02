# Observations - run 025 `earnest-dragon-25`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `2xsd5jwr`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=200; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.9493 |
| Q2 | c_t alive / video-specific? | PARTIAL | std=0.5243; dead_dim=0; cross_video_cosine=0.6885 |
| Q3 | Rich latent? | FAIL | c_effective_rank=9.4052; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=flat; ratio=47.1775; story=insufficient temporal evidence or early run |
| Q5 | F_c beats copy? | FAIL | model_loss=2.9459; copy_loss=0.0624; ratio=47.1775; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=9.839; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=1.0324; cplus=1.0322; chat=1.0319; chat-cplus=-3.101e-04; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Smoke / inconclusive | The run is too short for Phase 1 learning gates; use it as infrastructure evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.2323 @ 0 | 2.4913 @ 200 | -0.741 | 2.4913 / 3.0058 / 3.2323 | n/a |
| `L_flow` | 3.0218 @ 0 | 2.4447 @ 200 | -0.577 | 2.4447 / 2.9152 / 3.162 | n/a |
| `L_var` | 0.421 @ 0 | 0.0837 @ 200 | -0.3373 | 0.0441 / 0.1762 / 0.421 | n/a |
| `L_recon` | 1.0331 @ 0 | 0.9415 @ 200 | -0.0916 | 0.9415 / 0.9996 / 1.0331 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 0.1 @ 200 | 0.1 | 0 / 0.05 / 0.1 | n/a |
| `c_std_mean` | 0.5243 @ 0 | 0.5243 @ 0 | 0 | 0.5243 / 0.5243 / 0.5243 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.6885 @ 0 | 0.6885 @ 0 | 0 | 0.6885 / 0.6885 / 0.6885 | n/a |
| `c_effective_rank` | 9.4052 @ 0 | 9.4052 @ 0 | 0 | 9.4052 / 9.4052 / 9.4052 | n/a |
| `c_slot_diversity_rank` | 31.7811 @ 0 | 31.7811 @ 0 | 0 | 31.7811 / 31.7811 / 31.7811 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9999 @ 0 | 0 | 0.9999 / 0.9999 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9997 @ 0 | 0.9997 @ 0 | 0 | 0.9997 / 0.9997 / 0.9997 | n/a |
| `coarse_copy_loss` | 0.0624 @ 0 | 0.0624 @ 0 | 0 | 0.0624 / 0.0624 / 0.0624 | n/a |
| `coarse_model_loss` | 2.9459 @ 0 | 2.9459 @ 0 | 0 | 2.9459 / 2.9459 / 2.9459 | n/a |
| `coarse_batch_mean_loss` | 0.2994 @ 0 | 0.2994 @ 0 | 0 | 0.2994 / 0.2994 / 0.2994 | n/a |
| `coarse_vs_copy_ratio` | 47.1775 @ 0 | 47.1775 @ 0 | 0 | 47.1775 / 47.1775 / 47.1775 | n/a |
| `coarse_vs_batch_mean_ratio` | 9.839 @ 0 | 9.839 @ 0 | 0 | 9.839 / 9.839 / 9.839 | n/a |
| `L_recon_present` | 1.0324 @ 0 | 1.0324 @ 0 | 0 | 1.0324 / 1.0324 / 1.0324 | n/a |
| `L_recon_cplus` | 1.0322 @ 0 | 1.0322 @ 0 | 0 | 1.0322 / 1.0322 / 1.0322 | n/a |
| `L_recon_chat` | 1.0319 @ 0 | 1.0319 @ 0 | 0 | 1.0319 / 1.0319 / 1.0319 | n/a |
| `grad_norm` | 1.673 @ 0 | 0.9493 @ 200 | -0.7238 | 0.4397 / 1.0451 / 1.673 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.134 @ 200 | 0.1333 | 6.667e-04 / 0.0673 / 0.134 | n/a |

## Last Key Metrics

`c_effective_rank`=9.4052 @ 0; `c_cross_video_cosine`=0.6885 @ 0; `c_std_mean`=0.5243 @ 0; `coarse_vs_copy_ratio`=47.1775 @ 0; `coarse_vs_batch_mean_ratio`=9.839 @ 0; `L_recon_present`=1.0324 @ 0

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 47.1775 and the latest batch-mean ratio is 9.839; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 9.4052, cross-video cosine is 0.6885, and copy loss is 0.0624. The reading-cycle verdict is **Smoke / inconclusive** because The run is too short for Phase 1 learning gates; use it as infrastructure evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — earnest-dragon-25 (n_c=256)

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **352 s**, last heartbeat **03:47:43Z** — same instant as all siblings
(synchronized whole-pod death; forensics in [Wave 2 OBSERVATIONS](../OBSERVATIONS.md)). Only **step 0
(initialization)** was logged.

### Step-0 values = initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `L_recon_present` | 1.032 | untrained |
| `coarse_vs_copy_ratio` | 47.2 | random init (slightly lower than siblings — init variance, not signal) |
| `c_effective_rank` | 9.41 | pre-training |
| `c_slot_diversity_rank` | 31.8 | trivially highest (256 slots) — not a result |
| `c_cross_video_cosine` | 0.688 | collapsed random init |

The lower init copy-ratio (47 vs ~63) and higher slot-diversity are trivial consequences of having
8× the slots at random init — **not** evidence of anything.

## What we still don't know — and why this run matters most

This was the **highest-value run of the wave**: the decisive test of the latent-capacity hypothesis
at 8× bandwidth. Its result (floor + `c_effective_rank` + `coarse_vs_copy_ratio` together) is what
would either confirm "latent capacity is the lever" (→ Stage 2) or, far more likely per
[Wave 1](../../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md), confirm the pivot. **Untested.** Tier-0 re-run
priority — see [NEXT_STEPS.md](NEXT_STEPS.md).
