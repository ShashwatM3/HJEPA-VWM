# Run 058 metric readout — `ae_latent_stack_whiten_abs_recon_cov_var_ego4d` (`mvbx96nv`)

> Deliverable (1): a full end-to-end interpretation of every graph and metric relevant to
> the EGO4D dataset-change test, and a plain statement of how the pipeline and the model
> perform on EGO4D. Read with Reading Cycle B (present-reconstruction-only) from
> [`GUIDES/READING_EXPERIMENTS.md`](../../../../GUIDES/READING_EXPERIMENTS.md). Every number
> below was pulled from W&B (run `mvbx96nv`, unsampled 30-point diagnostic history at
> `diag_every=500`) and overlaid against the SSv2 arm of the identical recipe, run 057
> `cdvp6hou`. The two runs differ ONLY in `data.dataset` (ego4d vs ssv2) and
> `whiten_stats_path` (verified by W&B `compare_runs`), so every difference below is
> attributable to the dataset substrate alone.

## Headline

The pipeline runs flawlessly on EGO4D, but the model's representation **collapses**. The
identical recipe that produced the program's healthiest present-only representation on SSv2
produces a near-video-independent template on EGO4D. Side-by-side, at the final diagnostic
step (14,500):

| Metric (final) | EGO4D (run 058, `mvbx96nv`) | SSv2 (run 057, `cdvp6hou`) | Verdict axis |
|---|---|---|---|
| `c_effective_rank` (max 256) | **52.9** | 208.2 | rank collapsed (−155) |
| `c_cross_video_cosine` (lower better) | **0.863** | 0.059 | directional collapse (+0.80) |
| `c_std_mean` (target ~1.0) | **0.419** | 1.117 | spread collapsed (−0.70) |
| `c_slot_diversity_rank_centered` (max 32) | 30.67 | 30.71 | identical / healthy |
| `c_dead_dim_frac` | 0.000 | 0.000 | identical / healthy |
| `L_recon_present` (lower better) | **0.678** | 0.713 | EGO4D raw recon *better* |
| `L_recon_shuffled_c` (wrong-video decode) | **0.696** | 0.940 | template shortcut |
| `L_recon_video_gap` (correct−wrong) | **0.018** | 0.227 | honesty collapsed (−0.21) |
| video-conditioned share of decoder gain | **~5.6%** | ~79% | honesty collapsed |
| `loss`, `grad_skipped`, `grad_has_nan` | 0.037, 0, 0 | 0.037, 0, 0 | identical / healthy |

The single most important line is `L_recon_video_gap` and its share: on EGO4D the decoder
reconstructs a video's features **almost as well from the wrong video's code as from the
right one** (0.696 vs 0.678). Only ~5.6% of the decoder's improvement over the no-information
baseline is actually conditioned on the specific video, versus ~79% on SSv2.

---

## Q1 — Did the run train, and in the right mode? (PASS)

Optimization and wiring are spotless for the entire schedule.

| Metric | Value across the run | Reading |
|---|---|---|
| `grad_skipped` | 0 at every step | no skipped optimizer steps |
| `grad_has_nan` | 0 at every step | no NaN gradients |
| `instability_warn` | 0 at every step | never tripped |
| `grad_norm` | 0.42 → transient 5.4 @500 (recon warmup) → settles 0.04–0.06, ends 0.046 | calm, lower than SSv2's 0.076 |
| `agc_B_max_ratio` / `agc_D_max_ratio` | peak 0.22 @500, then ≤0.015 / ~0.0004 | AGC essentially never bites after warmup |
| `present_recon_only` / `prediction_active` | 1 / 0 all run | correct mode |
| `L_flow` / `L_recon_pred` | 0 / 0 all run | prediction branch fully off, as intended |
| `whiten_active` | 1 all run | whitening engaged |
| `recon_target_residual` / `recon_mean_norm` | 0 / 0 all run | clean/absolute target (mean tracker inactive) |
| `recon_scale` | ramps 0 → 1.0 by step 2,000 | reconstruction warmup as designed |

**Pipeline verdict:** the EGO4D data path (chunked 12-FPS `.mp4` → decord → resize/crop/
normalize → frozen V-JEPA → EGO4D-specific ZCA whitening → latent-stack bottleneck → decoder)
is wired and numerically healthy end to end. The EGO4D-specific whitening stats file loaded
and engaged (`whiten_active=1`), the clean-target arm is confirmed (`recon_mean_norm=0`), and
training is if anything *calmer* than the SSv2 run (lower grad norm, quieter AGC). Nothing here
is invalid; the failure is representational, not operational.

---

## Q2 — Is `c_t` alive and video-specific, or collapsed? (FAIL — the core result)

This is where EGO4D and SSv2 diverge completely. All three collapse flags are computed on the
same fixed 16-example validation batch in both runs.

| step | EGO4D cosine | SSv2 cosine | EGO4D std | SSv2 std |
|---|---|---|---|---|
| 0 | 1.000* | 1.000* | ~0* | ~0* |
| 500 | 0.596 | 0.617 | 0.573 | 0.592 |
| 1,000 | **0.369** | **−0.004** | 0.746 | 0.981 |
| 2,000 | 0.515 | 0.019 | 0.705 | 1.043 |
| 3,000 | 0.693 | 0.103 | 0.577 | 1.060 |
| 5,000 | 0.793 | 0.078 | 0.514 | 1.113 |
| 8,000 | **0.893** (peak) | 0.082 | **0.373** (min) | 1.103 |
| 10,000 | 0.833 | 0.027 | 0.464 | 1.137 |
| 14,500 | **0.863** | 0.059 | **0.419** | 1.117 |

(*Step-0 cosine 1.0, std ~0, rank 31 are mechanical — the 32 fixed slot identity vectors
before any information routes through them.)

Two facts define the run:

1. **EGO4D briefly specialized, then re-collapsed.** Cross-video cosine fell to a *healthy*
   0.369 at step 1,000 — the codes were becoming video-specific — and then **reversed** and
   climbed monotonically to ~0.86–0.89 as reconstruction reached full weight (`recon_scale`
   hit 1.0 at step 2,000). This is not a static artifact of a homogeneous validation batch;
   the representation actively *de-specialized* under the objective. On SSv2 the same
   reconstruction ramp drove cosine to ≈0 and it stayed there.
2. **Spread never reached the floor on validation.** `c_std_mean` peaked at ~0.75 (step 1,000)
   and then fell to a 0.37–0.46 band, ending 0.419 — far below the 1.0 variance-floor target,
   where SSv2 sat at 1.12. `c_dead_dim_frac` is 0 throughout, so this is a uniform contraction
   of every dimension's spread, not a set of dead units.

**Important cross-check (why `L_var≈0` yet `c_std_mean=0.42`).** `L_var` and `L_cov` are
training-step losses on the 64-example *training* batch; `c_std_mean`/`c_effective_rank`/
`c_cross_video_cosine` are diagnostics on the fixed 16-example *validation* batch (train.py:
`run_diagnostics` on `val_batch`). `L_var` fell to ~0.0005 and `L_cov` to ~0.2, i.e. the
geometry regularizers are *satisfied on the training distribution* — yet the held-out
representation is contracted and directionally collapsed. **The regularizers do not
generalize on EGO4D.** On SSv2 they did (val `c_std_mean` 1.12 matches the satisfied floor).

Q2 verdict: **FAIL** — cross-video cosine 0.863 ≫ 0.5 and std 0.419 < 0.8. The EGO4D code is
video-independent and under-spread on held-out data.

---

## Q3 — Is `c_t` rich, or low-rank? (FAIL)

| step | EGO4D `c_effective_rank` | SSv2 `c_effective_rank` |
|---|---|---|
| 1,000 | 65.4 | 76.4 |
| 2,000 | **80.7** (EGO4D peak) | 146.3 |
| 5,000 | 63.1 | 193.5 |
| 8,000 | 47.7 (EGO4D min) | 192.3 |
| 14,500 | **52.9** | 208.2 |

EGO4D rank rose during the recon warmup to a peak of ~80.7 at step 2,000, then **contracted**
to a ~48–57 plateau as reconstruction took over, ending 52.9. SSv2 climbed monotonically to
~208 and held. A rank of 52.9 is above the ~32–40 that slot structure alone can produce, so
the EGO4D code is not *maximally* cramped — but it uses roughly a quarter of the feature
directions SSv2 does, and the direction of drift after step 2,000 is contraction, not growth.
The `c_effective_rank` and `c_cross_video_cosine` curves are mirror images: rank falls exactly
as cosine rises, the signature of energy concentrating into a shared direction.

Q3 verdict: **FAIL** relative to the recipe's SSv2 behavior — the code did not stay rich; it
contracted while reconstruction improved (the run-053 "geometry pays with no content benefit"
pattern, here with directional collapse on top).

---

## Q4 — Is present reconstruction learning useful content? (PASS in isolation — but misleading)

| step | EGO4D `L_recon_present` | SSv2 `L_recon_present` |
|---|---|---|
| 0 | 1.013 | 1.004 |
| 2,000 | 0.819 | 0.827 |
| 5,000 | 0.709 | 0.746 |
| 10,000 | 0.682 | 0.719 |
| 14,500 | **0.678** | 0.713 |

Raw reconstruction falls monotonically and plateaus after ~step 10,000, reaching a *lower*
(nominally better) value than SSv2 (0.678 vs 0.713). Taken alone this looks like a success —
and it is the trap Reading Cycle B warns about. On EGO4D the decoder achieves lower raw loss
**precisely because** it learned a near-universal template it can apply to any clip (Q7), not
because `c_t` carries more video-specific content. `recon_scale=1.0` after step 2,000 and
`agc_D` clipping is negligible, so the reconstruction objective was fully and cleanly active.

Q4 verdict: reconstruction *loss* improved (PASS in isolation), but Q7 shows the improvement
is not video-conditioned, so this is not a content win.

---

## Q5 — Are geometry and reconstruction cooperating or fighting? (FIGHTING)

Reading Cycle B's cooperation table, read on the EGO4D curves:

| `L_recon_present` trend | `c_effective_rank` | `c_cross_video_cosine` | Story |
|---|---|---|---|
| falling | falling (80.7→52.9 post-peak) | rising to 0.86 | **video-independent collapse** |

The decisive overlay is the post-warmup window (steps 2,000–14,500): `L_recon_present`
improved from 0.819 to 0.678 (−0.14) while `c_effective_rank` fell 80.7→52.9, `c_std_mean`
fell 0.705→0.419, and `c_cross_video_cosine` rose 0.515→0.863. Reconstruction bought its
improvement by *contracting and de-specializing* the code. Supporting terms: `L_cov` fell
6.7→0.21 and `L_var` → ~0.0005 (both satisfied on train), confirming the regularizers were
not the bottleneck — they were satisfied while the held-out geometry collapsed anyway.

Q5 verdict: **FIGHTING** — the clean-target reconstruction objective actively degraded the
held-out representation, and the geometry regularizers could not hold it.

---

## Q6 — Verdict (Reading Cycle B)

**Collapsed rep** (with the reconstruction-honesty evidence identifying the specific mode as a
video-independent **template shortcut**). Q1 passes (valid, correct mode, stable); Q2 fails
(cosine 0.863, std 0.419); Q3 fails (rank contracts to 52.9); Q4's loss improvement is real
but Q5/Q7 show it is not video-conditioned.

---

## Q7 detail — the honesty readouts (the least-confounded evidence)

`L_recon_shuffled_c` decodes each target from a deliberately wrong (rolled) video's code;
`L_recon_video_gap = L_recon_shuffled_c − L_recon_present` measures how much the *correct*
code helps.

| step | EGO4D present / shuffled / gap | SSv2 present / shuffled / gap |
|---|---|---|
| 1,000 | 0.908 / 0.912 / 0.003 | 0.899 / 0.946 / 0.047 |
| 5,000 | 0.709 / 0.723 / 0.013 | 0.746 / 0.935 / 0.188 |
| 10,000 | 0.682 / 0.700 / 0.018 | 0.719 / 0.940 / 0.222 |
| 14,500 | 0.678 / 0.696 / **0.018** | 0.713 / 0.940 / **0.227** |

On EGO4D the wrong-video loss (0.696) tracks the correct-video loss (0.678) within 0.018 the
entire run; on SSv2 the wrong-video loss stays pinned near 0.94 while the correct-video loss
falls to 0.71, a 0.227 gap. The **video-conditioned share** — (correct-code gain − wrong-code
gain) / correct-code gain, baseline 1.0 — is:

- EGO4D: (0.322 − 0.304) / 0.322 ≈ **5.6%** (roughly flat 3.6% → 5.6% all run).
- SSv2: (0.287 − 0.060) / 0.287 ≈ **79%**.

This is the run-052 template-collapse signature: the decoder learned a video-independent
reconstruction template. It is the least-confounded metric here — even if the small 16-example
validation batch contains some sibling chunks (which could inflate `c_cross_video_cosine`), a
roll-by-one pairing still mostly pairs unrelated clips, and the gap is still ~0.018.

Attention context (`c_attn_entropy` 0.55, `c_attn_entropy_min` 0.0027) shows the bottleneck did
sharpen one head almost to a delta — reads are not uniform-collapsed — yet the slots stay
diverse (centered slot rank 30.67) and the code still collapses across videos. So the collapse
is **directional in feature space**, not a slot merge or an attention-uniformity failure.

---

## How the pipeline and model perform on EGO4D — summary

- **Pipeline: healthy.** Data loading, frozen-encoder features, EGO4D-specific whitening, the
  latent-stack bottleneck, the decoder, the loss terms, the optimizer/EMA/AGC plumbing, and all
  wiring flags behave exactly as on SSv2 — in fact with lower gradient norms and quieter
  clipping. The EGO4D sibling-dataset integration and the one-time EGO4D whitening-stats step
  are working correctly.
- **Model / representation: fails to transfer.** Under the exact recipe that was the program's
  best present-only result on SSv2, EGO4D drives the abstract code into a near-video-independent
  template: cross-video cosine 0.863, effective rank 52.9, per-dimension spread 0.419, and only
  ~5.6% of the decoder's improvement conditioned on the actual video. The geometry regularizers
  are satisfied on the training batch but do not generalize to held-out data.

The mechanism, cross-run attribution, and the concrete levers to try next are in
[`ANALYSIS.md`](ANALYSIS.md).
