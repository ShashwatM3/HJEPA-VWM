# Observations - run 028 `classic-yogurt-29`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `ryuh8cpr`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=200; grad_skipped max=0; grad_has_nan max=0; grad_norm last=1.2718 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.4843; dead_dim=0; cross_video_cosine=0.7349 |
| Q3 | Rich latent? | FAIL | c_effective_rank=9.6446; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=flat; ratio=66.8338; story=insufficient temporal evidence or early run |
| Q5 | F_c beats copy? | FAIL | model_loss=3.5841; copy_loss=0.0536; ratio=66.8338; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=14.055; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=1.028; cplus=1.0287; chat=1.0304; chat-cplus=0.0017; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Smoke / inconclusive | The run is too short for Phase 1 learning gates; use it as infrastructure evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 2.958 @ 0 | 2.6361 @ 200 | -0.3218 | 2.5012 / 2.8788 / 3.207 | n/a |
| `L_flow` | 2.7347 @ 0 | 2.5872 @ 200 | -0.1475 | 2.474 / 2.78 / 3.0535 | n/a |
| `L_var` | 0.4465 @ 0 | 0.0885 @ 200 | -0.3581 | 0.047 / 0.1927 / 0.4465 | n/a |
| `L_recon` | 1.0322 @ 0 | 0.9489 @ 200 | -0.0833 | 0.9489 / 1.0015 / 1.0322 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 0.1 @ 200 | 0.1 | 0 / 0.05 / 0.1 | n/a |
| `c_std_mean` | 0.4843 @ 0 | 0.4843 @ 0 | 0 | 0.4843 / 0.4843 / 0.4843 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7349 @ 0 | 0.7349 @ 0 | 0 | 0.7349 / 0.7349 / 0.7349 | n/a |
| `c_effective_rank` | 9.6446 @ 0 | 9.6446 @ 0 | 0 | 9.6446 / 9.6446 / 9.6446 | n/a |
| `c_slot_diversity_rank` | 29.586 @ 0 | 29.586 @ 0 | 0 | 29.586 / 29.586 / 29.586 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9999 @ 0 | 0 | 0.9999 / 0.9999 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.9998 @ 0 | 0 | 0.9998 / 0.9998 / 0.9998 | n/a |
| `coarse_copy_loss` | 0.0536 @ 0 | 0.0536 @ 0 | 0 | 0.0536 / 0.0536 / 0.0536 | n/a |
| `coarse_model_loss` | 3.5841 @ 0 | 3.5841 @ 0 | 0 | 3.5841 / 3.5841 / 3.5841 | n/a |
| `coarse_batch_mean_loss` | 0.255 @ 0 | 0.255 @ 0 | 0 | 0.255 / 0.255 / 0.255 | n/a |
| `coarse_vs_copy_ratio` | 66.8338 @ 0 | 66.8338 @ 0 | 0 | 66.8338 / 66.8338 / 66.8338 | n/a |
| `coarse_vs_batch_mean_ratio` | 14.055 @ 0 | 14.055 @ 0 | 0 | 14.055 / 14.055 / 14.055 | n/a |
| `L_recon_present` | 1.028 @ 0 | 1.028 @ 0 | 0 | 1.028 / 1.028 / 1.028 | n/a |
| `L_recon_cplus` | 1.0287 @ 0 | 1.0287 @ 0 | 0 | 1.0287 / 1.0287 / 1.0287 | n/a |
| `L_recon_chat` | 1.0304 @ 0 | 1.0304 @ 0 | 0 | 1.0304 / 1.0304 / 1.0304 | n/a |
| `grad_norm` | 1.5308 @ 0 | 1.2718 @ 200 | -0.259 | 0.5088 / 1.1114 / 1.6717 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.134 @ 200 | 0.1333 | 6.667e-04 / 0.0673 / 0.134 | n/a |

## Last Key Metrics

`c_effective_rank`=9.6446 @ 0; `c_cross_video_cosine`=0.7349 @ 0; `c_std_mean`=0.4843 @ 0; `coarse_vs_copy_ratio`=66.8338 @ 0; `coarse_vs_batch_mean_ratio`=14.055 @ 0; `L_recon_present`=1.028 @ 0

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 66.8338 and the latest batch-mean ratio is 14.055; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 9.6446, cross-video cosine is 0.7349, and copy loss is 0.0536. The reading-cycle verdict is **Smoke / inconclusive** because The run is too short for Phase 1 learning gates; use it as infrastructure evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — classic-yogurt-29 (n_c=128)

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **381 s** (the longest-surviving of the five, by seconds), last heartbeat
**03:47:43Z** — same instant as all siblings (synchronized whole-pod death; forensics in
[Wave 2 OBSERVATIONS](../OBSERVATIONS.md)). Only **step 0 (initialization)** was logged.

### Step-0 values = initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `L_recon_present` | 1.028 | untrained |
| `coarse_vs_copy_ratio` | 66.8 | random init |
| `c_effective_rank` | 9.64 | pre-training |
| `c_slot_diversity_rank` | 29.6 | trivially higher (more slots) — not a result |
| `c_cross_video_cosine` | 0.735 | collapsed random init |

Indistinguishable from any run's step 0; says nothing about n_c=128.

## What we still don't know

Whether the floor responds monotonically across the latent ladder, and whether n_c=128 trips
slot-collapse. Untested. As an *interpolation* point between the two decisive bookends (n_c=64,
n_c=256), it is lower re-run priority — see [NEXT_STEPS.md](NEXT_STEPS.md).
