# Observations - run 011 `olive-terrain-11`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `q40nq0l3`  
**State:** `killed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Invalid**  
**Verdict note:** Training-health signals failed; later metrics should not be treated as reliable evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | FAIL | state=killed; step=3900; grad_skipped max=1; grad_has_nan max=0; grad_norm last=23.9852 |
| Q2 | c_t alive / video-specific? | PARTIAL | std=0.5552; dead_dim=0; cross_video_cosine=0.6538 |
| Q3 | Rich latent? | FAIL | c_effective_rank=9.0648; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.3374 (0.0533 @ 0 -> 0.3907 @ 3500); ratio=2.8482; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.1128; copy_loss=0.3907; ratio=2.8482; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=2.6544; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Invalid | Training-health signals failed; later metrics should not be treated as reliable evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 2.9158 @ 0 | 1.1792 @ 3900 | -1.7366 | 1.1693 / 1.3736 / 3.4147 | n/a |
| `L_flow` | 2.8663 @ 0 | 1.1342 @ 3900 | -1.7321 | 1.1254 / 1.3339 / 3.3739 | n/a |
| `L_var` | 0.4316 @ 0 | 0.3273 @ 3900 | -0.1043 | 0.0832 / 0.2785 / 0.4887 | n/a |
| `c_std_mean` | 0.4952 @ 0 | 0.5552 @ 3500 | 0.06 | 0.4952 / 0.6126 / 0.779 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 3500 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.6538 @ 3500 | -0.0701 | 0.337 / 0.5744 / 0.7239 | n/a |
| `c_effective_rank` | 9.4733 @ 0 | 9.0648 @ 3500 | -0.4085 | 8.4699 / 10.0075 / 12.5106 | n/a |
| `c_slot_diversity_rank` | 16.7119 @ 0 | 4.6187 @ 3500 | -12.0932 | 4.6187 / 15.0683 / 25.949 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9997 @ 3500 | -2.330e-04 | 0.9997 / 0.9999 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.9826 @ 3500 | -0.0172 | 0.9826 / 0.9959 / 0.9998 | n/a |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.3907 @ 3500 | 0.3374 | 0.0533 / 0.3707 / 0.5265 | n/a |
| `coarse_model_loss` | 3.3106 @ 0 | 1.1128 @ 3500 | -2.1977 | 1.1128 / 1.4663 / 3.3106 | n/a |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 0.4192 @ 3500 | 0.1565 | 0.2627 / 0.4883 / 0.671 | n/a |
| `coarse_vs_copy_ratio` | 62.0662 @ 0 | 2.8482 @ 3500 | -59.218 | 2.3413 / 10.3765 / 62.0662 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 2.6544 @ 3500 | -9.9467 | 2.0031 / 3.6238 / 12.6011 | n/a |
| `grad_norm` | 0.268 @ 0 | 23.9852 @ 3900 | 23.7172 | 0.1845 / 73.2486 / 2.259e+03 | n/a |
| `grad_skipped` | 0 @ 0 | 0 @ 3900 | 0 | 0 / 0.2152 / 1 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 3500 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.924 @ 3900 | 0.9234 | 6.667e-04 / 0.788 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=9.0648 @ 3500; `c_cross_video_cosine`=0.6538 @ 3500; `c_std_mean`=0.5552 @ 3500; `coarse_vs_copy_ratio`=2.8482 @ 3500; `coarse_vs_batch_mean_ratio`=2.6544 @ 3500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 2.8482 and the latest batch-mean ratio is 2.6544; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 9.0648, cross-video cosine is 0.6538, and copy loss is 0.3907. The reading-cycle verdict is **Invalid** because Training-health signals failed; later metrics should not be treated as reliable evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — olive-terrain-11

## Outcome

**First clean look at the centered slot loss — and the first sign of Goodhart.** The
centered penalty now *works mechanically* (it holds slot diversity high for a while),
but rank still collapses and cross-video cosine still climbs. Killed at step 3900;
[`copper-sky-12`](../run_012_copper-sky-12/) re-ran the identical config and carried the story
to its conclusion.

## Key numbers (verified vs W&B `q40nq0l3`, centered slot, k=12)

| Step | `L_slot` | `c_slot_diversity_rank` | `c_effective_rank` | `c_cross_video_cosine` | `c_std_mean` | `coarse_vs_copy_ratio` | `grad_skipped` |
|---|---|---|---|---|---|---|---|
| 0 | 0.038 | 16.7 | 9.47 | 0.72 | 0.50 | 62.1 | 0 |
| 500 | 0.011 | **25.9** | 12.51 | 0.34 | 0.78 | 3.25 | 0 |
| 1000 | 0.011 | 24.0 | 11.54 | 0.48 | 0.69 | 4.28 | 0 |
| 1500 | 0.011 | 17.0 | 10.74 | 0.67 | 0.55 | 2.85 | **1** |
| 2500 | 0.013 | 10.1 | 9.35 | 0.61 | 0.59 | 2.35 | 0 |
| 3500 | 0.016 | **4.6** | 9.06 | 0.65 | 0.56 | 2.85 | 0 |

## Interpretation

- **The centering fix did its job at the loss level:** `L_slot` now starts ~0.01–0.04
  (vs the inert ~1.0 in [`skilled-waterfall-10`](../run_010_skilled-waterfall-10/)) and slot
  diversity *spikes to ~26/32* in the first 500 steps — the penalty genuinely pushes
  slots apart early.
- **But it does not hold, and it does not transfer to the axes we care about.** Slot
  diversity decays 26 → 4.6 over the run despite the active penalty, `c_effective_rank`
  stays ~9–12 (no climb), `c_cross_video_cosine` drifts back up toward 0.65, and
  `c_std_mean` sits at ~0.55 (well under the 1.0 floor — the λ_var=0.1 dose is too weak).
  Forcing slot diversity buys a transient that the rest of the objective erodes.
- This is the **first** Goodhart signal (a working slot loss that still leaves the
  representation collapsed); copper-sky-12 confirms it with a longer run and adds the
  grad-spike failure.

## Mapping note

Resolved from "intermediate / config TBD" via MCP. It is **not** a horizon-only ablation
and **not** a pre-centering run — it is the first centered-slot k=12 slot=0.05 run, the
opening half of the combined "Run 5" slot evidence.
