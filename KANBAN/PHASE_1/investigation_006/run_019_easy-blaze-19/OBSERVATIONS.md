# Observations - run 019 `easy-blaze-19`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `3syv6wp2`  
**State:** `finished`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | PASS | state=finished; step=14950; grad_skipped max=0; grad_has_nan max=0; grad_norm last=3.7063 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9456; dead_dim=0; cross_video_cosine=0.4141 |
| Q3 | Rich latent? | FAIL | c_effective_rank=12.5658; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.0741 (0.0533 @ 0 -> 0.1275 @ 14500); ratio=2.9533; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.3765; copy_loss=0.1275; ratio=2.9533; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.3898; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5989; cplus=0.5948; chat=0.5986; chat-cplus=0.0038; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0842 @ 0 | 0.4988 @ 14950 | -2.5854 | 0.4027 / 0.7167 / 3.5278 | 0.4027 / 0.4594 / 0.5565 (n=100) |
| `L_flow` | 2.8692 @ 0 | 0.4215 @ 14950 | -2.4476 | 0.3301 / 0.6486 / 3.3798 | 0.3301 / 0.3894 / 0.4861 (n=100) |
| `L_var` | 0.4301 @ 0 | 0.0258 @ 14950 | -0.4043 | 0.0021 / 0.0149 / 0.4301 | 0.0041 / 0.0113 / 0.0279 (n=100) |
| `L_recon` | 1.0264 @ 0 | 0.6403 @ 14950 | -0.3861 | 0.6293 / 0.6628 / 1.0264 | 0.6323 / 0.6416 / 0.6506 (n=100) |
| `L_recon_pred` | 1.0274 @ 0 | 0.6467 @ 14950 | -0.3807 | 0.633 / 0.6657 / 1.0274 | 0.633 / 0.6448 / 0.6598 (n=100) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 0.9456 @ 14500 | 0.4504 | 0.4952 / 0.9542 / 1.0541 | 0.9456 / 0.9992 / 1.0447 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.4141 @ 14500 | -0.3097 | 0.1842 / 0.2894 / 0.7239 | 0.2537 / 0.3325 / 0.4141 (n=10) |
| `c_effective_rank` | 9.473 @ 0 | 12.5658 @ 14500 | 3.0929 | 7.9447 / 10.9979 / 13.0839 | 12.5658 / 12.8561 / 13.0839 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 18.4046 @ 14500 | 1.6935 | 15.7198 / 17.4996 / 20.3134 | 18.013 / 18.2612 / 18.4046 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.8789 @ 14500 | -0.1211 | 0.8789 / 0.9303 / 1 | 0.8789 / 0.89 / 0.8977 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.7655 @ 14500 | -0.2343 | 0.7655 / 0.857 / 0.9999 | 0.7655 / 0.7809 / 0.7933 (n=10) |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.1275 @ 14500 | 0.0741 | 0.0533 / 0.2305 / 0.3315 | 0.1275 / 0.1813 / 0.232 (n=10) |
| `coarse_model_loss` | 3.3106 @ 0 | 0.3765 @ 14500 | -2.9341 | 0.309 / 0.6786 / 3.3106 | 0.309 / 0.3774 / 0.4432 (n=10) |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 0.9657 @ 14500 | 0.703 | 0.2627 / 0.9738 / 1.1627 | 0.9657 / 1.0705 / 1.1627 (n=10) |
| `coarse_vs_copy_ratio` | 62.0669 @ 0 | 2.9533 @ 14500 | -59.1136 | 1.4273 / 4.4638 / 62.0669 | 1.4273 / 2.1465 / 2.9533 (n=10) |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.3898 @ 14500 | -12.2113 | 0.2736 / 1.0315 / 12.6011 | 0.2736 / 0.3534 / 0.405 (n=10) |
| `L_recon_present` | 1.0246 @ 0 | 0.5989 @ 14500 | -0.4256 | 0.5976 / 0.6271 / 1.0246 | 0.5976 / 0.5984 / 0.5989 (n=10) |
| `L_recon_cplus` | 1.0251 @ 0 | 0.5948 @ 14500 | -0.4303 | 0.5932 / 0.6231 / 1.0251 | 0.5932 / 0.594 / 0.5948 (n=10) |
| `L_recon_chat` | 1.0266 @ 0 | 0.5986 @ 14500 | -0.428 | 0.5961 / 0.6271 / 1.0266 | 0.5962 / 0.5976 / 0.5999 (n=10) |
| `grad_norm` | 1.4788 @ 0 | 3.7063 @ 14950 | 2.2275 | 0.5002 / 2.359 / 4.7504 | 2.5551 / 3.0862 / 4.7504 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=12.5658 @ 14500; `c_cross_video_cosine`=0.4141 @ 14500; `c_std_mean`=0.9456 @ 14500; `coarse_vs_copy_ratio`=2.9533 @ 14500; `coarse_vs_batch_mean_ratio`=0.3898 @ 14500; `L_recon_present`=0.5989 @ 14500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 2.9533 and the latest batch-mean ratio is 0.3898; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 12.5658, cross-video cosine is 0.4141, and copy loss is 0.1275. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The project moved into decoder capacity and latent-utilization sweeps to determine whether the reconstruction floor was architectural capacity or c-space utilization.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — easy-blaze-19

**W&B:** [`3syv6wp2`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3syv6wp2)
**State:** finished (full 15k) · **Runtime:** ~6h 10m · steps **0–14950**

Data: W&B sampled history (`get_run_history_tool`, 30 diag points) + summary. Clean A/B vs
[`fanciful-lake-18`](../run_018_fanciful-lake-18/OBSERVATIONS.md) (same seed/config, `lambda_recon_pred`
0 → 0.05). **Pulled, not invented.**

---

## Outcome (one sentence)

**Option 3 is a negative result: the predicted-latent anchor did NOT improve prediction
(copy gate unchanged/slightly worse, `L_recon_chat` did not drop) and mildly degraded the
representation — because the reconstruction channel is capacity-saturated at ~0.60 and cannot
transmit prediction-quality information to `F_c`.**

## Investigation 006 gates

| Question | Result |
|---|---|
| Did option 3 fix the copy gate (`F_c` beats copy)? | **No.** `coarse_vs_copy_ratio` tracks fanciful, ends slightly worse (2.95 vs 2.59). |
| Did `L_recon_chat` drop (gradient acting on the prediction)? | **No.** ~0.60, unchanged from fanciful. `L_recon_pred` stuck at ~0.64 all run. |
| Did stability hold (the option-3 risk)? | **Yes.** No cliff; `agc_Fc` 9–18 (never spiked); 0 skips; 0 NaN. |
| Did rank / representation improve? | **No — mildly worse** (rank 12.6 vs 13.1; cross-video cosine 0.41 vs 0.32). |

---

## 1. The copy gate did not move (A/B at matched steps)

`coarse_vs_copy_ratio` (gate ≤0.70; lower=better):

| Step | fanciful (opt 1) | easy-blaze (opt 3) |
|---|---|---|
| 8500 | 1.43 | 1.51 |
| 10500 | 1.47 | 1.43 |
| 12000 | 1.93 | 1.89 |
| 14000 | 2.59 | 2.73 |
| 14500 | — | **2.95** |

The two curves sit on top of each other. Adding the prediction anchor made `F_c` **no better**
at beating copy — marginally worse at the end. `coarse_model_loss` ~0.38 (≈ fanciful);
`coarse_copy_loss` fell further (0.156 → 0.127), i.e. `c` got *more static*, so the copy
baseline strengthened.

## 2. `L_recon_chat` did not drop — the mechanism didn't engage

We added gradient specifically to drive `decode(c_hat) → e_{t+k}`. Result:

| Readout | fanciful (untrained) | easy-blaze (**trained**) |
|---|---|---|
| `L_recon_chat` (end) | ~0.603 | ~0.599 |
| `L_recon_pred` (training value) | n/a | **~0.64–0.65, flat all run** |

`L_recon_pred` never descends — the optimizer pays the 0.05 cost but the loss is pinned. The
training-through-`F_c` moved `L_recon_chat` by ~0.004. Essentially nothing.

## 3. WHY — the capacity floor (the real finding)

In **both** runs every reconstruction readout sits at the same wall:
`L_recon_present ≈ L_recon_cplus ≈ L_recon_chat ≈ L_recon_pred ≈ 0.60`. That is a **capacity
floor**: `c` (8,192 numbers) can rebuild only ~40% of `e`'s (~1.05M-number) variance; the
other ~60% is structurally unreachable at 128:1 compression.

Consequence: a *perfect* and a *noisy* prediction both reconstruct to ~0.60 (they share the
static content `c` can represent; the part `c` can't is invisible). So the reconstruction
objective is **blind to prediction quality at every τ**, there is no "predict better" gradient
direction, and `F_c` receives no useful corrective signal. This **confirms and sharpens** the
`fanciful-lake-18` finding (recon "blind to prediction error") — the blindness is a capacity
wall that training-through-`F_c` cannot overcome.

**Important corollary:** the cheap-vs-clean `c_hat` choice is **moot** — even genuine low-τ
predictions reconstruct to the same floor, so the clean from-noise variant would not help. The
floor is the killer, not the τ-leakage.

## 4. Mild but consistent adverse drift (option 3 slightly hurt the representation)

Same seed/config, so these divergences are attributable to `lambda_recon_pred`:

| Metric (end) | fanciful (opt 1) | easy-blaze (opt 3) | direction |
|---|---|---|---|
| `c_effective_rank` | 13.10 (stable) | 12.57 (declining from 13.08) | worse |
| `c_cross_video_cosine` | 0.319 | **0.414** (rising faster) | worse (toward Mode A) |
| `c_std_mean` | 1.007 | 0.946 | worse |
| `L_var` | 0.011 | 0.024 (floor re-engaging) | worse |

Not collapse, but a coherent, late-accelerating slide toward representational collapse. Likely
because the prediction-recon gradient enters `B` via the conditioning, but since reconstruction
can't improve (floor), it just nudges `B` toward a more generic/shareable `c` with no useful
objective. `agc_B_max_ratio` ran slightly hotter (peak 22.6; 11.7 @13500 vs fanciful's calmer
profile), consistent with the extra `B` gradient.

## 5. What did NOT break — stability

No cliff. `grad_skipped=0`, `grad_has_nan=0`, `agc_Fc_max_ratio` 9–18 (never spiked), `agc_D`
~0.02, `L_flow` ~0.35–0.42 (healthy). **The pre-run risk — routing gradient through `F_c`
destabilizes it — did not materialize.** The recon-into-`B` stabilization carried over.

---

## Conclusion

Option 3, implemented faithfully (joint objective, gradient through `F_c`, shared decoder),
**ran cleanly and revealed that the reconstruction channel is too low-capacity to carry the
prediction signal** VITA/Arbab counts on. It did not improve prediction and mildly hurt the
representation. The honest next fork is **not** cheap-vs-clean; it is:

- **(a) Break the capacity floor first** — bigger `c` (`n_c`/`d_c`) or higher `lambda_recon` —
  so reconstruction becomes unsaturated and *can* see prediction error. The decoder size is
  NOT the limit (`D` ≈ 2.17M params; the 8,192-number latent is the bottleneck).
- **(b) Question whether the copy gate is reconstruction-fixable at all.** Over horizon-12, `c`
  moves only ~38–47% of its magnitude and shrinks over training (`coarse_copy_loss` 0.23 →
  0.13) — the target barely moves, so this may be a **task/horizon** property, not an `F_c`
  failure. Lever would be the prediction horizon/target, not reconstruction.

This is a citable scientific finding, not an implementation miss. See `NEXT_STEPS.md`.
