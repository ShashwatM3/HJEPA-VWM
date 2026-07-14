# Run 058 analysis — `ae_latent_stack_whiten_abs_recon_cov_var_ego4d` (`mvbx96nv`, inv016)

> Deliverable (2): in-depth analysis of the EGO4D dataset-change test. Builds on the metric
> narration in [`METRIC_READOUT.md`](METRIC_READOUT.md). All numbers are W&B-verified from
> the unsampled diagnostic history of run `mvbx96nv` and its byte-identical SSv2 twin run 057
> (`cdvp6hou`). Analyzed 2026-07-14 with Reading Cycle B and the investigation_012–015 arc as
> the interpretive frame.

## 1. What was actually tested, and how clean the test is

This is the project's first EGO4D run and the first empirical answer to the question the whole
EGO4D migration raises: *is the representation machinery dataset-portable?* The migration
knowledge base ([`AGENT_FILES/KNOWLEDGE/ego4d/`](../../../../AGENT_FILES/KNOWLEDGE/ego4d/))
argued for EGO4D as an additive sibling dataset launched as a **single-variable A/B** so any
metric change is attributable to the dataset alone. That discipline held here: W&B
`compare_runs(mvbx96nv, cdvp6hou)` reports the entire config diff as
`{data.dataset: ego4d↔ssv2, train.whiten_stats_path: ego4d↔ssv2, checkpoint_dir}` — every
lambda, learning rate, decoder dimension, seed, mode flag, and step count is identical, and
both runs share the *same initialization path* (seed 42, train-from-scratch, mechanical step-0
rank 31). So the ~4× rank collapse, the 0.80 cross-video-cosine swing, and the honesty collapse
from ~79% to ~5.6% are caused by the substrate (EGO4D features + EGO4D whitening), not by any
knob. That is exactly the clean attribution the migration plan was designed to buy, and it is
why a null-looking present-only run is still highly informative.

## 2. The mechanism: the clean-target template shortcut, reopened on EGO4D

The collapse is not random; it is the **run-052 failure mode returning**. The causal chain is
visible in the curves:

1. **Early specialization (steps 0–1,000).** Before reconstruction has weight (`recon_scale`
   ramps 0→1 over steps 0–2,000), the geometry regularizers dominate and the code specializes:
   cross-video cosine falls 1.0 → 0.596 → **0.369**, rank rises to ~65, std to ~0.75. EGO4D
   *can* be represented video-specifically. This is real and important — it rules out "the
   bottleneck simply cannot separate EGO4D clips."
2. **Reconstruction takes over → re-collapse (steps 2,000–8,000).** As `recon_scale` reaches
   1.0, the objective becomes dominated by the cosine reconstruction of the **absolute
   (clean)** whitened features. Cross-video cosine reverses and climbs 0.515 → 0.79 → **0.893**;
   rank contracts 80.7 → 47.7; std contracts 0.705 → 0.373. The timing is the smoking gun: the
   representation de-specializes exactly when and because the clean-reconstruction term becomes
   the dominant gradient.
3. **Template equilibrium (steps 8,000–15,000).** The system settles at cosine ~0.86, rank ~52,
   std ~0.42, with `L_recon_present` (0.678) and `L_recon_shuffled_c` (0.696) nearly equal — a
   decoder that reconstructs any clip from a video-independent template plus a tiny
   video-specific correction (`L_recon_video_gap` 0.018, ~5.6% conditioned).

Why the clean target enables this is settled project knowledge: cosine reconstruction of the
*absolute* features is satisfiable by reproducing the features' large shared/common component,
which is video-independent, so the decoder can learn a per-position template and route almost
nothing through `c_t`. Investigation_013 (run 052 → 053) diagnosed exactly this and showed the
**residual reconstruction target** (`recon_residual_target=TRUE`, reconstruct `e − mean`)
removes the template component and forces video-specific content through `c_t` — run 054
measured that at ~92% video-conditioned. The inv015 "clean arm" (runs 055/056/057) deliberately
dropped the residual target and *still* held honesty on SSv2 (run 057 ~79%) because SSv2's
whitened features have a weak enough shared component that whitening + covariance were enough to
block the shortcut. **On EGO4D that is no longer true.**

## 3. Why EGO4D specifically — the shared-component hypothesis

The clean cosine target only collapses when the whitened features have a dominant shared
direction the decoder can exploit as a template. The evidence says EGO4D's do and SSv2's do not,
under identical whitening machinery:

- Egocentric footage is structurally self-similar in a way exocentric SSv2 is not: every clip is
  a head-mounted view with the same rough layout (hands/arms low, a floor/surface, a
  forward-facing field of view), so the frozen V-JEPA features of different EGO4D clips likely
  share a larger common component than SSv2's object-on-a-table clips. Ironically this is the
  flip side of the very property that motivated the migration — EGO4D's whole-frame motion is
  great for breaking temporal copy, but its *global appearance* across clips may be more
  homogeneous, which is what feeds a spatial-reconstruction template.
- Whitening is computed per-dataset (`whiten_stats_ego4d_train_seed42.pt`). ZCA whitening
  equalizes the *training-set* covariance, but it cannot remove a shared component that is
  common to individual clips yet still varies across the dataset; and with a fixed
  `whiten_eps=1e-4` it may be amplifying a low-energy EGO4D tail differently than on SSv2. The
  run does not isolate this, so it is a hypothesis, not a conclusion.

The **train/val generalization gap** is the other half of the mechanism. `L_var` (~0.0005) and
`L_cov` (~0.2) show the regularizers are satisfied on the 64-example *training* batch, yet the
16-example *validation* diagnostics show std 0.419 and rank 52.9. The regularizers are being met
by whatever per-coordinate variance the shifting training batches happen to present, without the
bottleneck learning a *generalizable* video-specific code — so held-out clips collapse onto the
template. This did not happen on SSv2 (val std 1.12 matched the satisfied floor), which is
further evidence that the EGO4D substrate, not the recipe, is the operative variable.

## 4. What did NOT fail — and why that sharpens the diagnosis

- **Optimization** is perfect (0 skips, 0 NaNs, grad norm ~0.05, AGC quiet) — cleaner than SSv2.
- **Slots** stay diverse: `c_slot_diversity_rank_centered` 30.67 ≈ SSv2's 30.71, `dead_dim_frac`
  0. The collapse is not a slot merge and not a dead-unit collapse.
- **Attention** sharpened one head near a delta (`c_attn_entropy_min` 0.0027) but the mean stays
  moderate (0.55) — not a uniform-attention failure.

So the failure is specifically a **directional (cross-video) collapse in feature space plus a
decoder template shortcut** — the exact pair the residual target was built to fix — and nothing
else. That specificity is what makes the next step obvious.

## 5. Cross-run placement

| | run 052 (SSv2, sharp-slot, clean, no geom) | run 057 (SSv2, whiten+cov+var, clean) | **run 058 (EGO4D, whiten+cov+var, clean)** |
|---|---|---|---|
| `c_effective_rank` | 13.4 | 208 | **52.9** |
| `c_cross_video_cosine` | 0.906 | 0.059 | **0.863** |
| `c_std_mean` | 0.295 | 1.12 | **0.419** |
| video-conditioned share | ~0% | ~79% | **~5.6%** |
| slot diversity (centered) | (declining) | 30.7 | **30.7** |

Run 058 sits much closer to the *pre-fix* run-052 template collapse than to its own SSv2 twin
run 057, despite carrying the full whitening + covariance + variance machinery that fixed run
052's descendants on SSv2. In one line: **the inv015 clean-arm result is SSv2-specific; on
EGO4D the clean absolute target is not safe.**

## 6. Implications for the EGO4D migration

- **The migration's actual goal (breaking `coarse_vs_copy_ratio`) is untouched by this run.**
  This is present-only; `F_c` never trained, no copy/batch-mean gate applies. This run does not
  confirm or deny that EGO4D fixes the copy-ratio problem.
- **But it is a genuine prerequisite warning.** A future full-prediction EGO4D run needs a
  video-specific abstract code to predict *into*; if the present representation collapses to a
  template, `F_c` would be predicting a near-constant target and the run would fail for a
  representational reason before the copy-ratio hypothesis could even be tested. The
  representation must be fixed on EGO4D first.
- **The fix is already on the shelf.** The residual reconstruction target is implemented,
  flag-gated (`--recon-residual-target`), and demonstrated (inv013/inv014) to remove exactly
  this template shortcut. It was dropped for the SSv2 clean arm; EGO4D is precisely the regime
  where it should be turned back on. This is the highest-value next run and it is a pure config
  delta.

## 7. What this run cannot tell us (honesty section)

- **One seed, one recipe, one dataset build.** No factorial isolation of "EGO4D features" vs
  "EGO4D whitening" vs "clean target." The residual-target run (next) is the direct control.
- **The 16-example validation diagnostic batch is small and may contain sibling chunks** (chunks
  of the same long source video are near-duplicates). That can inflate `c_cross_video_cosine`
  specifically. It does *not* explain the honesty collapse: `L_recon_video_gap` from a roll-by-one
  shuffle over 16 clips mostly pairs unrelated clips and is still only 0.018 (vs SSv2's 0.227 on
  the same-size batch). A larger, source-disjoint diagnostic batch is a cheap robustness check.
- **Says nothing about prediction, copy ratio, or fine flow.** Present-only by construction.
- **`whiten_eps` was not swept.** The floor tuned on SSv2 may not suit EGO4D's spectrum.

## 8. Recommended next steps (see NEXT_STEPS.md for the ranked plan)

1. **Residual-target EGO4D run** (`--recon-residual-target`, everything else identical). Direct
   test of the shared-component/template hypothesis; the single highest-value follow-up.
2. **Robustness check on the diagnostic batch**: rerun diagnostics with a larger, source-disjoint
   validation batch (or offline via the rank/drift probes) to confirm the collapse is not a
   16-sample sibling-chunk artifact.
3. **Offline EGO4D feature probes** (`rank_probe.py --data ego4d`, `drift_probe.py --data ego4d`,
   per the EGO4D GUIDE Stage 7B): quantify EGO4D `e`'s shared-component/anisotropy directly, the
   inv014 analog for the new substrate — this tells us *before* another training run whether the
   shared-template component is intrinsically larger on EGO4D.
4. Only after the representation holds on EGO4D: proceed to the full-prediction EGO4D A/B (the
   run-037 recipe) that the migration actually exists to run.

## 9. Verdict

**Collapsed rep / template shortcut.** The strongest SSv2 present-only recipe does not transfer
to EGO4D: with only the dataset (and its whitening stats) changed, the abstract code collapses
to a near-video-independent template (cross-video cosine 0.863, rank 52.9, std 0.419, ~5.6%
video-conditioned) while training stays perfectly stable and slots stay diverse. The clean
absolute reconstruction target — safe on SSv2 — is unsafe on EGO4D's more self-similar features,
and the covariance/variance regularizers, satisfied on the training batch, do not generalize to
held-out data. The residual reconstruction target is the pre-registered, on-the-shelf fix and the
mandatory next experiment before any EGO4D prediction run.
