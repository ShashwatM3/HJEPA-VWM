# Observations - run 029 `helpful-snow-25`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `tw685b5g`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=200; grad_skipped max=0; grad_has_nan max=0; grad_norm last=1.1527 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.4952; dead_dim=0; cross_video_cosine=0.7239 |
| Q3 | Rich latent? | FAIL | c_effective_rank=9.4729; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=flat; ratio=62.0671; story=insufficient temporal evidence or early run |
| Q5 | F_c beats copy? | FAIL | model_loss=3.3106; copy_loss=0.0533; ratio=62.0671; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=12.6011; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=1.0246; cplus=1.0251; chat=1.0266; chat-cplus=0.0015; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Smoke / inconclusive | The run is too short for Phase 1 learning gates; use it as infrastructure evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0873 @ 0 | 2.6983 @ 200 | -0.389 | 2.6983 / 3.0744 / 3.5594 | n/a |
| `L_flow` | 2.8724 @ 0 | 2.5652 @ 200 | -0.3072 | 2.5652 / 2.9346 / 3.3888 | n/a |
| `L_var` | 0.4299 @ 0 | 0.0792 @ 200 | -0.3506 | 0.0442 / 0.1829 / 0.4299 | n/a |
| `L_recon` | 1.0264 @ 0 | 0.9348 @ 200 | -0.0916 | 0.9348 / 0.9887 / 1.0264 | n/a |
| `L_recon_pred` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `recon_scale` | 0 @ 0 | 0.1 @ 200 | 0.1 | 0 / 0.05 / 0.1 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 0.4952 @ 0 | 0 | 0.4952 / 0.4952 / 0.4952 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.7239 @ 0 | 0 | 0.7239 / 0.7239 / 0.7239 | n/a |
| `c_effective_rank` | 9.4729 @ 0 | 9.4729 @ 0 | 0 | 9.4729 / 9.4729 / 9.4729 | n/a |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 16.7111 @ 0 | 0 | 16.7111 / 16.7111 / 16.7111 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9999 @ 0 | 0 | 0.9999 / 0.9999 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.9998 @ 0 | 0 | 0.9998 / 0.9998 / 0.9998 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.0533 @ 0 | 0 | 0.0533 / 0.0533 / 0.0533 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 3.3106 @ 0 | 0 | 3.3106 / 3.3106 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 0.2627 @ 0 | 0 | 0.2627 / 0.2627 / 0.2627 | n/a |
| `coarse_vs_copy_ratio` | 62.0671 @ 0 | 62.0671 @ 0 | 0 | 62.0671 / 62.0671 / 62.0671 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 12.6011 @ 0 | 0 | 12.6011 / 12.6011 / 12.6011 | n/a |
| `L_recon_present` | 1.0246 @ 0 | 1.0246 @ 0 | 0 | 1.0246 / 1.0246 / 1.0246 | n/a |
| `L_recon_cplus` | 1.0251 @ 0 | 1.0251 @ 0 | 0 | 1.0251 / 1.0251 / 1.0251 | n/a |
| `L_recon_chat` | 1.0266 @ 0 | 1.0266 @ 0 | 0 | 1.0266 / 1.0266 / 1.0266 | n/a |
| `grad_norm` | 1.4785 @ 0 | 1.1527 @ 200 | -0.3258 | 0.5103 / 1.0544 / 1.5371 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 200 | 0 | 0 / 0 / 0 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.134 @ 200 | 0.1333 | 6.667e-04 / 0.0673 / 0.134 | n/a |

## Last Key Metrics

`c_effective_rank`=9.4729 @ 0; `c_cross_video_cosine`=0.7239 @ 0; `c_std_mean`=0.4952 @ 0; `coarse_vs_copy_ratio`=62.0671 @ 0; `coarse_vs_batch_mean_ratio`=12.6011 @ 0; `L_recon_present`=1.0246 @ 0

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 62.0671 and the latest batch-mean ratio is 12.6011; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 9.4729, cross-video cosine is 0.7239, and copy loss is 0.0533. The reading-cycle verdict is **Smoke / inconclusive** because The run is too short for Phase 1 learning gates; use it as infrastructure evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — helpful-snow-25 (λ_recon=1.0)

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **346 s**, last heartbeat **03:47:43Z** — same instant as all siblings
(synchronized whole-pod death; forensics in [Wave 2 OBSERVATIONS](../OBSERVATIONS.md)). Only **step 0
(initialization)** was logged.

### Step-0 values = initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `loss` | 3.087 | healthy init |
| `grad_norm` | 1.478 | healthy, no blow-up |
| `grad_has_nan` / `grad_skipped` | 0 / 0 | clean |
| `L_recon_present` | 1.025 | untrained |
| `c_effective_rank` | 9.47 | pre-training |

### This run's forensic value

Because it is **architecturally identical to Wave 1** (n_c=32, decoder 256×2) and still died at the
same step 200 as the n_c=256 run, it is the control that **rules out an `n_c` shape bug** as the cause
of the Wave-2 failure. The healthy step-0 `grad_norm`/`loss`/no-NaN also rule out divergence. The
failure was external (whole-pod / session / volume) — see [Wave 2 OBSERVATIONS](../OBSERVATIONS.md).

## What we still don't know

Whether λ=1.0 confirms the weight axis stays flat at the extreme (predicted ~0.581, no break) and
whether `L_flow` degrades. Untested — but **predictable**, hence low re-run priority
([NEXT_STEPS.md](NEXT_STEPS.md)).
