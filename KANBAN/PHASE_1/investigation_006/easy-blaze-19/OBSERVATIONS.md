# Observations — easy-blaze-19

**W&B:** [`3syv6wp2`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3syv6wp2)
**State:** finished (full 15k) · **Runtime:** ~6h 10m · steps **0–14950**

Data: W&B sampled history (`get_run_history_tool`, 30 diag points) + summary. Clean A/B vs
[`fanciful-lake-18`](../fanciful-lake-18/OBSERVATIONS.md) (same seed/config, `lambda_recon_pred`
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
