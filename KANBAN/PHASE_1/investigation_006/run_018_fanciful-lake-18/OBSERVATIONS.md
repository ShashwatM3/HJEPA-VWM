# Observations - run 018 `fanciful-lake-18`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `yd5958s6`  
**State:** `killed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=killed; step=14400; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.7618 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0074; dead_dim=0; cross_video_cosine=0.3192 |
| Q3 | Rich latent? | FAIL | c_effective_rank=13.1001; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1028 (0.0533 @ 0 -> 0.1561 @ 14000); ratio=2.5897; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.4043; copy_loss=0.1561; ratio=2.5897; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.3722; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5992; cplus=0.5959; chat=0.6026; chat-cplus=0.0067; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.0842 @ 0 | 0.3953 @ 14400 | -2.6889 | 0.3786 / 0.7084 / 3.5265 | 0.3786 / 0.4418 / 0.5099 (n=89) |
| `L_flow` | 2.8692 @ 0 | 0.3554 @ 14400 | -2.5138 | 0.3431 / 0.6715 / 3.3798 | 0.3431 / 0.4063 / 0.4755 (n=89) |
| `L_var` | 0.4301 @ 0 | 0.0155 @ 14400 | -0.4145 | 0.0024 / 0.0133 / 0.4301 | 0.0024 / 0.0069 / 0.0179 (n=89) |
| `L_recon` | 1.0264 @ 0 | 0.6422 @ 14400 | -0.3842 | 0.6276 / 0.663 / 1.0264 | 0.6319 / 0.6412 / 0.6503 (n=89) |
| `recon_scale` | 0 @ 0 | 1 @ 14400 | 1 | 0 / 0.9291 / 1 | 1 / 1 / 1 (n=89) |
| `c_std_mean` | 0.4952 @ 0 | 1.0074 @ 14000 | 0.5122 | 0.4952 / 0.9731 / 1.0634 | 1.0074 / 1.0425 / 1.0634 (n=9) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.3192 @ 14000 | -0.4047 | 0.1722 / 0.2534 / 0.7239 | 0.2331 / 0.2615 / 0.3192 (n=9) |
| `c_effective_rank` | 9.473 @ 0 | 13.1001 @ 14000 | 3.6272 | 8.1233 / 11.2133 / 13.3064 | 13.1001 / 13.2152 / 13.3064 (n=9) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 18.1081 @ 14000 | 1.397 | 15.2409 / 17.2638 / 20.3112 | 17.7711 / 17.9833 / 18.1081 (n=9) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.8845 @ 14000 | -0.1155 | 0.8845 / 0.9312 / 1 | 0.8845 / 0.8916 / 0.8966 (n=9) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.7919 @ 14000 | -0.208 | 0.7919 / 0.8697 / 0.9999 | 0.7919 / 0.8029 / 0.8104 (n=9) |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.1561 @ 14000 | 0.1028 | 0.0533 / 0.2427 / 0.3469 | 0.1561 / 0.1956 / 0.2333 (n=9) |
| `coarse_model_loss` | 3.3106 @ 0 | 0.4043 @ 14000 | -2.9063 | 0.3206 / 0.7028 / 3.3106 | 0.3206 / 0.3939 / 0.4566 (n=9) |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.0864 @ 14000 | 0.8237 | 0.2627 / 1.0028 / 1.1719 | 1.0864 / 1.1438 / 1.1719 (n=9) |
| `coarse_vs_copy_ratio` | 62.0669 @ 0 | 2.5897 @ 14000 | -59.4772 | 1.4315 / 4.4981 / 62.0669 | 1.4725 / 2.0378 / 2.5897 (n=9) |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.3722 @ 14000 | -12.2289 | 0.2813 / 1.0537 / 12.6011 | 0.2813 / 0.3444 / 0.3963 (n=9) |
| `L_recon_present` | 1.0246 @ 0 | 0.5992 @ 14000 | -0.4254 | 0.598 / 0.6282 / 1.0246 | 0.598 / 0.5988 / 0.5992 (n=9) |
| `L_recon_cplus` | 1.0251 @ 0 | 0.5959 @ 14000 | -0.4292 | 0.5953 / 0.6304 / 1.0251 | 0.5953 / 0.596 / 0.597 (n=9) |
| `L_recon_chat` | 1.0266 @ 0 | 0.6026 @ 14000 | -0.424 | 0.5999 / 0.6364 / 1.0266 | 0.5999 / 0.6011 / 0.6026 (n=9) |
| `grad_norm` | 1.4788 @ 0 | 2.7618 @ 14400 | 1.283 | 0.5031 / 2.3629 / 4.4382 | 2.7618 / 3.1406 / 4.4382 (n=89) |
| `grad_skipped` | 0 @ 0 | 0 @ 14400 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=89) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0049 @ 14400 | 0.0042 | 6.667e-04 / 0.519 / 1 | 0.0049 / 0.12 / 0.302 (n=89) |

## Last Key Metrics

`c_effective_rank`=13.1001 @ 14000; `c_cross_video_cosine`=0.3192 @ 14000; `c_std_mean`=1.0074 @ 14000; `coarse_vs_copy_ratio`=2.5897 @ 14000; `coarse_vs_batch_mean_ratio`=0.3722 @ 14000; `L_recon_present`=0.5992 @ 14000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 2.5897 and the latest batch-mean ratio is 0.3722; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 13.1001, cross-video cosine is 0.3192, and copy loss is 0.1561. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The project moved into decoder capacity and latent-utilization sweeps to determine whether the reconstruction floor was architectural capacity or c-space utilization.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

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
