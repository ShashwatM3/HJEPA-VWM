# Observations — fanciful-lake-18

**W&B:** [`yd5958s6`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/yd5958s6)
**State:** killed (operator) · **Runtime:** 23326s (~6h 29m) · steps **0–14400** (target 15000)

Data source: W&B sampled history (`get_run_history_tool`, 29 diag points @ `diag_every=500`)
+ run summary. **Never invented — pulled.**

---

## Outcome (one sentence)

The reconstruction anchor **completely eliminated the Mode-B optimization cliff** —
`F_c` stayed stable through 14400 steps where every prior run died at ~8600 — but it
**did not break the rank ceiling** (`c_effective_rank` plateaued at ~13.3, not >60) and
**did not beat the copy baseline** (`coarse_vs_copy_ratio` floor 1.43, end 2.59), giving
us the first sustained, honest read of the prediction gate in a non-collapsed run.

## Investigation 006 gates

| Hypothesis | Result |
|---|---|
| Primary — lift `c_effective_rank` past ~13 toward >60 | **NOT met.** Plateaued ~13.1–13.3. |
| Secondary — soften / remove the step-8600 cliff | **Met decisively.** No cliff; `agc_Fc` peaked 17.9 vs royal's 1284. |
| Guardrail — no `L_flow` / copy-ratio regression | `L_flow` healthy (~0.36–0.40); copy gate **persistently failed** (see below). |

## PHASE_1 §12 acceptance gates

| Gate | threshold | best in run | @14000 | verdict |
|---|---|---|---|---|
| `coarse_vs_copy_ratio` | ≤ 0.70 | **1.43** @8500 | 2.59 | **FAIL** (never < 1) |
| `coarse_vs_batch_mean_ratio` | ≤ 0.50 | **0.28** @10500 | 0.37 | **PASS** from ~5500 on |
| `c_effective_rank` | > 60 | **13.31** @11500 | 13.10 | **FAIL** (ceiling intact) |

---

## Phase timeline

| Phase | Steps | What happened |
|---|---|---|
| Warmup + recon ramp | 0–2000 | `L_flow` 2.87→1.08; `recon_scale` 0→1; rank dips 9.5→8.1 then recovers; `L_recon` 1.02→0.65 |
| Rank climb | 2000–9000 | `L_flow` 1.08→0.40; rank 8.6→**13.0**; copy ratio 4.3→1.5; `agc_Fc` flat ~10–16 |
| **8600 window** | 8000–9000 | **No cliff.** `L_flow` 0.47→0.38→0.40; rank 12.7→12.9→13.0; `agc_Fc` 15→14→16 |
| Plateau | 9000–14400 | rank pinned **13.1–13.3**; `L_flow` ~0.36–0.45; copy ratio drifts **1.5→2.6**; everything stable |

---

## The cliff that did not happen (8000–9000, the royal-cherry-17 kill zone)

| Step | `L_flow` | `agc_Fc_max_ratio` | `agc_B_max_ratio` | `c_effective_rank` | `coarse_vs_copy_ratio` |
|---|---|---|---|---|---|
| 8000 | 0.467 | 15.3 | 4.2 | 12.74 | 1.62 |
| 8500 | **0.381** | 14.0 | 5.9 | 12.85 | 1.43 |
| 9000 | 0.402 | 15.6 | 5.7 | **13.03** | 1.51 |

**Direct contrast — same global steps, `royal-cherry-17`:** at 8600 `L_flow` 0.29→1.11,
`agc_Fc` →19→**1284** by 8750, rank 13.9→5.8 over the next 2500 steps. Here the same
window is a non-event. `grad_skipped=0` and `grad_has_nan=0` for all 14400 steps;
`agc_Fc_max_ratio` never exceeded **17.9** in the entire run.

This is the strongest result of the run: **the recon-into-`B` gradient regularizes the
bottleneck enough that `F_c`'s forward prediction never enters the sharp-blow-up regime.**
The secondary hypothesis is confirmed.

---

## The rank ceiling held (primary hypothesis refuted)

| Step | 0 | 1500 (min) | 4500 | 6500 | 8500 | 11500 (max) | 14000 |
|---|---|---|---|---|---|---|---|
| `c_effective_rank` | 9.47 | 8.12 | 9.05 | 11.51 | 12.85 | **13.31** | 13.10 |

Rank climbed out of the warmup dip and **plateaued at the same ~13/256 ceiling every
healthy run hits** (royal's pre-cliff peak was 13.9). The anchor **stabilized** rank — it
did not **enrich** it. `L_recon_present` itself flatlines at **~0.599** from step ~5000
onward (started 1.02): the decoder rebuilds ~40% of `e`'s variance and then stops
improving. Reconstructing `e` to that floor evidently requires only ~13 dims of `c`, so
`lambda_recon=0.05` supplies no pressure to use more. **The ceiling looks structural at
this decoder capacity / recon weight, not like a collapse the anchor can lift.**

---

## The copy gate — first honest sustained read

`coarse_vs_copy_ratio` (model endpoint loss ÷ copy-forward `c_t` loss):

| Step | 500 | 3000 | 5500 | **8500 (min)** | 10000 | 12000 | 14000 |
|---|---|---|---|---|---|---|---|
| ratio | 4.13 | 2.78 | 1.65 | **1.43** | 1.89 | 1.93 | **2.59** |
| `coarse_copy_loss` | 0.33 | 0.32 | 0.32 | 0.26 | 0.23 | 0.20 | **0.156** |
| `coarse_model_loss` | 1.37 | 0.88 | 0.53 | 0.38 | 0.44 | 0.39 | 0.40 |

The model **never beats copy** and drifts worse late. The mechanism is visible in the
columns: `coarse_copy_loss` keeps **falling** (0.33→0.156) — i.e. `c_t` becomes *more
static over the horizon*, so naive copy-forward becomes a stronger and stronger
baseline — while `coarse_model_loss` parks at ~0.40. `royal-cherry-17` is not a fair
comparison (its only sub-1 ratio, 0.85 @8500, was a single transient point right before
it collapsed and was never sustained). **fanciful-lake-18 is the first run to hold stable
training past 8600 at all, so this is the first trustworthy measurement of the gate — and
it says the prediction problem is real and persistent, not a collapse artifact.**

Interpretation: the **present-anchor pulls `c` toward a good per-frame autoencoder code,
which is exactly a code that changes little across the horizon** — helping reconstruction
and stability while making the copy baseline harder to beat. Option 1 regularizes the
*representation*; it has no term that rewards *predictability*.

---

## The option-3 decision metric (split recon readouts)

The whole point of logging `L_recon_present` / `_cplus` / `_chat` was to decide whether to
route reconstruction through `F_c` (option 3). Final values, and they are **flat and nearly
identical for the entire run**:

| Readout | meaning | value (stable from ~step 4000) |
|---|---|---|
| `L_recon_present` | decode online `c_t` | ~0.599 |
| `L_recon_cplus` | decode true future `c_plus` | ~0.596 |
| `L_recon_chat` | decode **predicted** `c_hat` | ~0.603 |

**`L_recon_chat − L_recon_cplus ≈ 0.007`** — tiny. The pre-registered gate
([NEXT_STEPS](../NEXT_STEPS.md)) said "promote to option 3 only if this gap is large
(F_c's prediction lands where the representation reconstructs poorly)." Taken literally,
the gate says **don't**.

**But the gate's premise is contradicted by the copy result, and that is the important
finding.** `F_c` is demonstrably *bad* at prediction (loses to copy by 1.4–2.6×), yet its
predicted latent `c_hat` reconstructs `e_{t+k}` *as well as* the true `c_plus`. The only
way both are true: **the present-anchored recon objective is blind to prediction quality.**
It saturates on the static, shared-across-horizon content of `e` (the ~0.6 compression
floor), which `c_hat` and `c_plus` share, so it literally cannot see where `F_c` errs. A
small `chat−cplus` gap therefore does **not** mean "F_c is fine / option 3 unwarranted"; it
means **the recon signal is sitting in the wrong place to help prediction at all.** To make
reconstruction improve prediction, the objective has to be *on* `c_hat` with gradient
*through* `F_c` (option 3 / the tech-lead's prediction-side branch). See NEXT_STEPS.

---

## Decoder & gradient health (the new module behaved)

- `agc_D_max_ratio` stayed **0.002–0.11** the whole run; `agc_D_clipped=0` at the end. The
  decoder never destabilized and never needed clipping — `lr_decoder=1e-4` + `agc_λ_D=0.20`
  were comfortably conservative.
- `grad_skipped=0`, `grad_has_nan=0` for 14400 steps. `grad_norm` 1.5→~3 (one blip 4.4
  @13000, coincident with an `agc_B` spike to 27.6 — isolated, self-corrected).
- `grad_global_norm` pinned at the 0.5 clip ceiling (expected).
- `recon_scale` reached 1.0 at step 2000 and held — every post-2k result is the anchor at
  full `lambda_recon=0.05`.

## Representation health (no collapse of any kind)

| Metric | trajectory | read |
|---|---|---|
| `c_std_mean` | 0.50→0.86→**~1.0–1.06** | variance floor holding; healthy |
| `c_dead_dim_frac` | 0 throughout | no dead dims |
| `c_cross_video_cosine` | 0.72→0.20, drifts **0.24→0.32** over 12k–14k | low (no Mode A); mild late rise worth watching |
| `c_attn_entropy` | 0.9999→**0.884** | attention sharpening (slots specializing), not uniform-collapse |
| `c_slot_diversity_rank` | 16.7→15.2(min)→**18.1** | recovered and rising |
| `L_var` | 0.43→~0.005–0.01 | floor satisfied with almost no hinge |
| `L_cov` (logged only) | 1.4→41(max@3500)→**10.7** | decorrelation improved on its own as rank rose |

None of the investigation-003 (Mode A) or investigation-005 (Mode B) collapse signatures
appeared. This run is healthy end-to-end; its limitations are *ceilings*, not *failures*.

---

## Mechanism (best current explanation)

1. **Recon-into-`B` is a strong stabilizer.** Pinning `c_t` to reconstruct `e_t` keeps the
   bottleneck output well-conditioned, which keeps `F_c`'s *input* well-conditioned, which
   removes the sharp-landscape blow-up that was Mode B. Stability — not rank — is what the
   anchor actually bought.
2. **The present-anchor and the prediction objective are decoupled.** `D(c_t)→e_t` rewards
   `c` for being a faithful *present* code; nothing rewards `c` for being *predictable* by
   `F_c`. So `c` drifts toward a static per-frame autoencoder code (copy baseline ↓) while
   `F_c` is left to predict a target it can't beat copy on.
3. **The rank ceiling is a capacity/weight floor, not a collapse.** Recon to ~0.6 relative
   MSE needs ~13 dims; `lambda_recon=0.05` won't pay the `L_flow` cost to use more. Breaking
   >60 likely needs a different lever (stronger/decoded recon, bigger `D`, or — the open
   question — whether >60 is even the right target for a 128:1 bottleneck).

## What worked
- **Mode-B cliff eliminated** — first run to train stably past 8600 (to 14400).
- Decoder integrated cleanly: zero NaNs, zero skips, negligible AGC on `D`.
- Variance floor + low cross-video cosine held; no representational collapse.
- The split readouts did their job: they *diagnosed* (even if not the way the gate framed it).

## What didn't
- **Rank ceiling unbroken** (13.3, target >60) — primary hypothesis refuted.
- **Copy gate failed and worsened** (1.43→2.59) — prediction is the unsolved problem.
- Mild late rise in `c_cross_video_cosine` (0.24→0.32) — watch if a longer run is launched.

---

## Conclusion

Option 1 is a **stability win and a rank-enrichment null result.** It does exactly what a
representation regularizer can do (kills the cliff, keeps `c` healthy) and nothing a
prediction objective would do (it left the copy gate failing and the rank ceiling intact).
The split readouts, read against the copy result, show **present-anchored reconstruction is
structurally blind to prediction quality** — which is the data-grounded case for moving the
reconstruction objective onto the predicted latent through `F_c`. That next step coincides
with the tech-lead's (Arbab) VITA-based proposal; it is now supported by *this run's
evidence*, not just prior art. Caveats carried forward: option 3 is unlikely to break the
rank ceiling on its own, and it adds gradient at the `F_c` site — though fanciful shows
recon-into-`B` greatly de-risks that site.

See `NEXT_STEPS.md`.
