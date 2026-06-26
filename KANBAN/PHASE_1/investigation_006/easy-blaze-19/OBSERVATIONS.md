# Observations — easy-blaze-19

**W&B:** [`3syv6wp2`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3syv6wp2)
**State:** finished · steps **0-15000** (target completed)

Data source: W&B MCP-derived trajectory and matched-step comparison supplied for this
KANBAN update, plus the uploaded early launch log (`logs/1.json`, steps 0-199). The
local JSON is not a full-run trajectory; the full-run conclusions below come from the
W&B pull.

---

## Outcome (one sentence)

The predicted-latent reconstruction branch **did not fix the copy gate**, **did not
make reconstruction sensitive to prediction quality**, and **mildly degraded the
representation**, while the stabilized option-1 training path remained numerically
healthy through the full 15k steps.

## Investigation 006 gates

| Hypothesis / guardrail | Result |
|---|---|
| Primary — predicted recon improves `coarse_vs_copy_ratio` | **NOT met.** The curve stayed on top of `fanciful-lake-18`; end value was slightly worse. |
| Success signal — `L_recon_pred` descends | **NOT met.** It stayed around ~0.64-0.65 all run. |
| Guardrail — no `F_c` instability / Mode-B cliff | **Met.** No cliff, zero grad skips / NaNs, AGC stayed controlled. |
| Secondary watch — representation should not drift worse | **Mildly failed.** Rank, cross-video cosine, `c_std_mean`, and `L_var` all moved in the wrong direction late. |

## PHASE_1 §12 acceptance gates

| Gate | threshold | run result | verdict |
|---|---:|---|---|
| `coarse_vs_copy_ratio` | <= 0.70 | best matched read ~1.43; end ~2.95 | **FAIL** |
| `coarse_vs_batch_mean_ratio` | <= 0.50 | not the limiting failure in this analysis | not headline |
| `c_effective_rank` | > 60 | ended ~12.57, down from ~13.08 | **FAIL** |

---

## Copy gate — option 3 did not move the target metric

`coarse_vs_copy_ratio` (gate <=0.70; lower is better) at matched steps:

| Step | fanciful-lake-18 (option 1) | easy-blaze-19 (option 3) |
|---:|---:|---:|
| 8500 | 1.43 | 1.51 |
| 10500 | 1.47 | 1.43 |
| 12000 | 1.93 | 1.89 |
| 14000 | 2.59 | 2.73 |
| 14500 | — | 2.95 |

The option-3 branch did not create a visible separation from the option-1 curve. Both
runs bottomed around ~1.43 and then drifted upward into the ~2.6-3.0 range. The exact
failure option 3 was meant to address — `F_c` losing to copy-forward — remained.

Interpretation: the additional objective did not supply a usable prediction-quality
gradient to the coarse flow. At the end of the run, easy-blaze was marginally worse
than fanciful on the copied-target comparison.

---

## `L_recon_pred` — the intended success signal stayed flat

The predicted-latent reconstruction branch was supposed to drive
`decode(c_hat) -> e_{t+k}`. It did not.

| Metric | fanciful-lake-18 | easy-blaze-19 |
|---|---:|---:|
| `L_recon_chat` at end | ~0.603 | ~0.599 |
| `L_recon_pred` training loss | n/a | ~0.64-0.65 all run |

Training through the predicted latent moved the evaluation readout by only about
0.004 versus the untrained `fanciful-lake-18` readout. The optimizer paid the
additional `0.05 * L_recon_pred` cost, but the metric did not meaningfully descend.

This is the run's most diagnostic negative result: the branch was wired and stable,
but the reconstruction channel did not become a useful error signal for prediction.

---

## Mechanism — the ~0.60 reconstruction floor is the real finding

Across the option-1 and option-3 runs, the reconstruction readouts all pin near the
same wall:

- `L_recon_present`
- `L_recon_cplus`
- `L_recon_chat`
- `L_recon_pred`

All sit around **~0.60** once the decoder reaches its floor. At this bottleneck,
`c` has 32 x 256 = 8,192 scalar slots trying to summarize `e` at roughly 1024 x
1024 ~= 1M scalars, a ~128:1 compression. The decoder can reconstruct the shared /
static part of `e` and misses the rest.

Consequences:

1. A good prediction and a bad prediction can both decode to the same ~0.60 floor,
   because the decoder only sees the portion of `e` that the bottleneck can carry.
2. The reconstruction objective is therefore **blind to prediction quality** at the
   current capacity point.
3. Option 3's gradient has no clear direction for "predict the future better"; it
   mostly says "stay on the reconstructable manifold."
4. The cheap-vs-clean implementation question is not the limiting issue here. Even a
   cleaner from-noise prediction target would still decode into the same saturated
   floor unless the reconstruction channel becomes unsaturated first.

This sharpens the `fanciful-lake-18` result. Fanciful showed present recon was blind
to prediction error. Easy-blaze shows that **training through that blindness does not
break it**.

---

## Representation drift — small, coherent, and adverse

The run did not collapse, but the option-3 delta nudged representation metrics in a
consistent worse direction relative to the same-seed option-1 run.

| Metric (end of run) | fanciful-lake-18 | easy-blaze-19 | Direction |
|---|---:|---:|---|
| `c_effective_rank` | 13.10 stable | 12.57, declining from 13.08 | worse |
| `c_cross_video_cosine` | 0.319 | 0.414, rising faster | worse / more Mode-A-like |
| `c_std_mean` | 1.007 | 0.946 | worse |
| `L_var` | 0.011 | 0.024 | worse / variance floor re-engaging |

The likely mechanism is that `L_recon_pred` sends gradient into `B` through the
conditioning path, but the reconstruction loss is saturated and cannot reward better
prediction. That pressure can still make `B` more generic / shareable across videos,
which explains the higher cross-video cosine and lower rank without producing any
copy-gate benefit.

---

## Stability — the feared failure did not happen

The main pre-run risk was destabilizing `F_c` by routing the new reconstruction
gradient through it. That did **not** materialize:

- `grad_skipped=0`
- `grad_has_nan=0`
- `agc_Fc_max_ratio` stayed in the ~9-18 range, with no royal-style spike
- `agc_D` stayed tiny (~0.02 scale)
- `L_flow` stayed in the healthy ~0.35-0.42 range late

This matters because it separates two conclusions: option 3 is not rejected because
it blew up. It is rejected because it ran cleanly and still supplied no useful
prediction signal.

---

## Uploaded early-step JSON check

The uploaded `logs/1.json` covers 200 parsed terminal-log steps, not the completed
15k W&B history. It confirms early launch behavior only:

| Step | `loss` | `L_flow` | `L_var` | `lr_mult` | notes |
|---:|---:|---:|---:|---:|---|
| 0 | 2.915 | 2.873 | 0.427 | 0.0001 | only early diagnostic row with representation metrics |
| 100 | 2.946 | 2.919 | 0.270 | 0.0101 | warmup still early |
| 199 | 3.150 | 3.136 | 0.141 | 0.0200 | no final-run signal |

Because diagnostic fields are mostly null after step 0 in this file, it is not used
for the copy-gate, reconstruction-floor, or rank conclusions.

---

## Conclusion

Option 3 is a **clean negative result**. The joint objective was implemented and ran
stably, but reconstruction through the current decoder / bottleneck is already
capacity-saturated around relative MSE ~0.60. That saturated channel cannot transmit
prediction-quality information to `F_c`, so the copy gate does not improve.

The run also slightly worsened representation health, likely because the extra
gradient into `B` encourages more generic latents without receiving a meaningful
future-prediction reward. The next decision should not be cheap-vs-clean option 3;
the run says the reconstruction channel itself must be made informative first, or
the project should question whether reconstruction is the right lever for a horizon
where copy-forward remains so strong.
