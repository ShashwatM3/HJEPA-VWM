# Observations - run 008 `serene-cloud-8`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `dhp1i3fk`  
**State:** `killed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Invalid**  
**Verdict note:** Training-health signals failed; later metrics should not be treated as reliable evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | FAIL | state=killed; step=4300; grad_skipped max=1; grad_has_nan max=0; grad_norm last=232.2137 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.45; dead_dim=0; cross_video_cosine=0.7435 |
| Q3 | Rich latent? | FAIL | c_effective_rank=14.4777; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.0999 (0.0183 @ 0 -> 0.1183 @ 4250); ratio=11.4562; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.3548; copy_loss=0.1183; ratio=11.4562; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=5.9294; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Invalid | Training-health signals failed; later metrics should not be treated as reliable evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.1651 @ 0 | 1.5071 @ 4300 | -1.658 | 1.4639 / 1.779 / 3.6621 | n/a |
| `L_flow` | 2.8667 @ 0 | 1.3807 @ 4300 | -1.486 | 1.3367 / 1.6354 / 3.3731 | n/a |
| `L_var` | 0.4419 @ 0 | 0.4609 @ 4300 | 0.019 | 0.0832 / 0.4471 / 0.5703 | n/a |
| `c_std_mean` | 0.4958 @ 0 | 0.45 @ 4250 | -0.0458 | 0.4206 / 0.4993 / 0.8251 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 4250 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7217 @ 0 | 0.7435 @ 4250 | 0.0218 | 0.2497 / 0.6954 / 0.7833 | n/a |
| `c_effective_rank` | 9.2474 @ 0 | 14.4777 @ 4250 | 5.2303 | 9.2474 / 14.7275 / 17.9007 | n/a |
| `c_slot_diversity_rank` | 16.6139 @ 0 | 7.9933 @ 4250 | -8.6206 | 6.6213 / 8.5087 / 16.6139 | n/a |
| `c_attn_entropy` | 0.9999 @ 0 | 0.9802 @ 4250 | -0.0197 | 0.9802 / 0.9888 / 0.9999 | n/a |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.8663 @ 4250 | -0.1335 | 0.8663 / 0.9323 / 0.9998 | n/a |
| `coarse_copy_loss` | 0.0183 @ 0 | 0.1183 @ 4250 | 0.0999 | 0.0183 / 0.2445 / 0.6751 | n/a |
| `coarse_model_loss` | 3.3121 @ 0 | 1.3548 @ 4250 | -1.9574 | 1.2778 / 1.628 / 3.3121 | n/a |
| `coarse_batch_mean_loss` | 0.259 @ 0 | 0.2285 @ 4250 | -0.0305 | 0.2165 / 0.3096 / 0.6955 | n/a |
| `coarse_vs_copy_ratio` | 180.5868 @ 0 | 11.4562 @ 4250 | -169.1307 | 1.8929 / 17.0324 / 180.5868 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.7899 @ 0 | 5.9294 @ 4250 | -6.8606 | 1.8371 / 5.8887 / 12.7899 | n/a |
| `grad_norm` | 0.2715 @ 0 | 232.2137 @ 4300 | 231.9422 | 0.1743 / 63.1467 / 2.445e+03 | n/a |
| `grad_skipped` | 0 @ 0 | 1 @ 4300 | 1 | 0 / 0.1264 / 1 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 4250 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 6.667e-04 @ 0 | 0.8976 @ 4300 | 0.8969 | 6.667e-04 / 0.7992 / 1 | n/a |

## Last Key Metrics

`c_effective_rank`=14.4777 @ 4250; `c_cross_video_cosine`=0.7435 @ 4250; `c_std_mean`=0.45 @ 4250; `coarse_vs_copy_ratio`=11.4562 @ 4250; `coarse_vs_batch_mean_ratio`=5.9294 @ 4250

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 11.4562 and the latest batch-mean ratio is 5.9294; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 14.4777, cross-video cosine is 0.7435, and copy loss is 0.1183. The reading-cycle verdict is **Invalid** because Training-health signals failed; later metrics should not be treated as reliable evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — serene-cloud-8

## Outcome

**First slot-loss run — early Goodhart, on the raw (pre-centering) loss.** Aggressive
`lambda_slot=0.25` did *not* lift slot diversity (the raw loss couldn't, see below);
instead `c_cross_video_cosine` climbed back toward video-independence and the model
stayed far worse than copy. Killed at `_step=4300`. Aggressive slot weight rejected.

## Key numbers (verified vs W&B `dhp1i3fk`, k=4, raw slot loss, lambda_slot=0.25)

| Step | `L_slot` | `c_slot_diversity_rank` | `c_effective_rank` | `c_cross_video_cosine` | `c_std_mean` | `coarse_vs_copy_ratio` | `grad_norm` |
|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 16.6 | 9.25 | 0.72 | 0.50 | 180.6 | 0.27 |
| 250 | 0.9996 | 10.7 | 12.36 | 0.25 | 0.83 | 2.86 | 0.17 |
| 500 | 0.242 | 6.6 | **15.9** | 0.62 | 0.58 | 1.89 | 2.5 |
| 1000 | 0.195 | 8.3 | 17.9 | 0.66 | 0.55 | 15.2 | 20.5 |
| 2000 | 0.213 | 7.7 | 13.7 | **0.78** | 0.42 | 5.3 | 8.9 |
| 3000 | 0.158 | 7.8 | 13.6 | 0.76 | 0.44 | 9.2 | 24.2 |
| 4250 | 0.188 | 8.0 | 14.5 | 0.74 | 0.45 | 11.5 | 5.0 |

## Interpretation (and corrections)

- **`c_effective_rank` was NOT "worse than run-2."** It actually *rose* to ~16–18 and
  settled ~13–15 — higher than [`sleek-leaf-7`](../run_007_sleek-leaf-7/)'s 8.7. The earlier
  KANBAN claim ("rank worse than run-2") is wrong. What's damning is not the rank but
  that **`c_cross_video_cosine` stayed ~0.70–0.78** (not the ~0.84 the old note cited,
  though it does reach ~0.84 later in [`copper-sky-12`](../run_012_copper-sky-12/)) and the
  model stayed 5–15× worse than copy. High rank with high cross-video cosine and bad
  copy ratio = the metric moving without the representation improving = **Goodhart**.
- **The slot metric did not actually improve** (`c_slot_diversity_rank` fell 16.6 → ~8).
  On the **raw** loss, `L_slot` does drop from 1.0 to ~0.2 but that is the loss gaming
  its own (uncentered) cosine, not real slot diversity — the same raw/centered mismatch
  later isolated in [`skilled-waterfall-10`](../run_010_skilled-waterfall-10/). So serene's
  rejection of `lambda_slot=0.25` was made on a partially-broken objective; the cleaner
  rejection comes after centering (copper-sky-12).
- Training was unstable post-warmup (grad spikes 20–24, `c_std_mean` falling to ~0.45)
  — the same collapse↔instability coupling seen throughout the slot arc.

## What worked

Nothing for production config — but the negative result (aggressive slot weight pushes
the latent toward video-independence) set up the dose-reduction (0.25 → 0.05) and the
horizon discussion that followed. Source: W&B `dhp1i3fk`; chat ~6900–7012.
