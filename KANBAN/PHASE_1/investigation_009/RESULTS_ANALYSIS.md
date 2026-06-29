# Investigation 009 — Results & Analysis (residual prediction vs SIGReg substrate)

In-depth analysis of the two-run wave. Design + registered predictions:
[`DESCRIPTION.md`](DESCRIPTION.md) / [`OBSERVATIONS.md`](OBSERVATIONS.md). Execution:
[`GUIDE.md`](GUIDE.md). All numbers pulled live from W&B
(`smahalanobis-uc-davis/hjepa-vwm`) via `get_run_history` on 2026-06-28 (full trajectories,
29 diag points each, every 500 steps to step 14000).

---

## 0. Provenance

| | **Run 1 — SIGReg substrate** | **Run 2 — residual prediction** |
|---|---|---|
| W&B | `fine-meadow-34` (`xz3nabr9`) | `graceful-river-35` (`jsh6uo7p`) |
| commit | `bc77db6` | `bc77db6` |
| `predict_residual` | **false** (full latent) | **true** (Δ = c_{t+k} − c_t) |
| λ_sigreg · λ_var | 6.0 · 0.5 | 5.0 · 0.5 |
| λ_recon · λ_recon_pred | **0 · 0** (no recon) | 0.05 · 0.05 |
| decoder · n_c · k | 512×4 · 32 · 12 | 512×4 · 32 · 12 |
| runtime · last step | ~6h10m · 14050 | ~6h10m · 14050 |
| state | `crashed` = **manually ended** | `crashed` = **manually ended** |

Both ran flawlessly: `grad_skipped = instability_warn = grad_has_nan = c_dead_dim_frac = 0` at
**every** logged step. No cliff, no NaN. The `--predict-residual` path worked end to end on the
pod exactly as built; the wave is clean data.

---

## 1. TL;DR

**The residual run is the first time in the entire project that `c` became temporally *dynamic*
instead of static — and it did so while reaching the richest, healthiest `c` yet (rank ~58,
cross-video cosine 0.17, std 1.0).** The decision metric `coarse_vs_copy_ratio` fell from the
full-latent **~6** (Run 1) to **~1.0–1.1** (Run 2). **But** that ~1.0 is a *tie-by-predicting-zero*,
not skillful prediction: `F_c` predicts the residual as ≈ 0 (ratio sits *just above* 1, i.e.
marginally *worse* than the zero baseline by the end). The mechanism is a master variable — the
temporal autocorrelation **ρ(c_t, c_{t+k})**:

- **Run 1 (SIGReg substrate)** drove **ρ ↑ from ~0.72 to ~0.92**: `c` *froze* in time, the copy
  baseline became trivially strong, and `F_c` lost to it by **6×**. SIGReg pumps rank (→50) by
  filling `d_c` with **static appearance** — exactly the inv007/008 disease, reproduced cleanly
  *without* recon.
- **Run 2 (residual + recon)** drove **ρ ↓ from ~0.88 to ~0.23**: `c` *decorrelated* in time
  (the future latent became nearly independent of the present). This **reverses the static-`c`
  disease** — but it over-shot: at ρ ≈ 0.23 the future is barely predictable from the present, so
  `F_c` reverts to predicting ≈ 0 (ties copy). A genuine predictive window was glimpsed
  **transiently** at step ~3500 (ρ ≈ 0.77, ratio **0.915 < 1** — `F_c` actually beat copy) before
  ρ collapsed further.

**Net: the bottleneck moved.** It is no longer "`c` is static" (Run 2 fixed that); it is now
"the temporal change in `c` is not *predictable* by `F_c`." The reconstruction-with-residuals
hypothesis is **half-vindicated** (it made `c` move) and **half-refuted** (recon stayed blind to
prediction; `L_recon_chat ≈ L_recon_cplus`).

---

## 2. Master table (plateau = mean of last 3 diag points, step 13000–14000)

| metric | Run 1 (λ6, full, no recon) | Run 2 (λ5, residual, recon) | note |
|---|---|---|---|
| **coarse_vs_copy_ratio** | **6.06** (still rising) | **1.08** (osc ~1.05) | gate < 0.70 |
| ↳ min over run (excl. init) | 2.37 @ 4500 | **0.915 @ 3500** | Run 2 briefly **beat** copy |
| **c_effective_rank** | **50.4** (plateaued) | **57.9** (still climbing) | gate > 60 |
| coarse_copy_loss = ‖Δ‖² | 0.143 (↓ from 0.45) | **1.554 (↑ from 0.20)** | opposite signs! |
| coarse_model_loss | 0.867 | 1.682 | |
| **implied ρ(c_t, c_{t+k})** | **~0.92** (↑) | **~0.23** (↓) | derived, see §5 |
| coarse_vs_batch_mean_ratio | 0.96 | 1.17 | |
| L_flow | 0.878 | 1.65 | Δ-scale in Run 2 — not comparable |
| c_std_mean | 0.928 (floor active) | 1.005 (floor satisfied) | |
| c_cross_video_cosine | 0.388 (↑, rising) | **0.170** (low, stable) | Run 2 far healthier |
| c_slot_diversity_rank | 5.39 | 6.28 | both low (~5–6/32) |
| c_attn_entropy (mean) | 0.842 | 0.607 (sharp) | |
| c_attn_entropy_min | 0.118 | ~1e-8 (one slot fully sharp) | |
| L_recon_present | ~1.045 (untrained — n/a) | **0.569** (< 0.585 inv007 floor) | |
| L_recon_chat − L_recon_cplus | n/a | 0.582 − 0.567 = **0.015** | still blind |
| L_var / L_sigreg | 0.033 / 0.0023 | 0.016 / 0.0025 | |
| grad_skipped / instability | 0 / 0 | 0 / 0 | |

---

## 3. Run 1 — the SIGReg substrate (a clean, decisive *negative* on prediction)

Run 1 isolates SIGReg-as-rank-lever with the recon machinery stripped out, full-latent prediction.
It reproduces the inv008 law **without any recon confound**:

- **Rank rises, prediction worsens — monotonically and causally.** `c_effective_rank` climbs
  9.5 → 50.4. `coarse_vs_copy_ratio` *bottoms at 2.37 (step 4500, rank ~18)* and then **rises to
  6.1** as rank continues to 50. The "improvement" phase (steps 0–4500) is just early settling;
  once SIGReg's rank-pumping engages hard (rank > 20), **every additional rank point costs
  prediction.** This is the cleanest demonstration yet that, under SIGReg, *utilization and
  temporal informativeness are anti-correlated.*
- **`c` freezes in time.** `coarse_copy_loss = ‖Δ‖²` *falls* 0.45 → 0.14, i.e. ρ(c_t, c_{t+k})
  *rises* to ~0.92 — the future latent becomes a near-copy of the present. SIGReg fills the 256
  `d_c` dimensions with **per-video static appearance** (high pooled rank) that does not move over
  the horizon, so "predict no change" wins ever more easily.
- **Health caveats.** `c_cross_video_cosine` drifts **up** to 0.39 (videos becoming less distinct —
  the same wrong-direction signal inv008 flagged), and `c_slot_diversity_rank` collapses mid-run
  (18 → 1.3 around step 3500–4500) before partially recovering to 5.4/32. So the SIGReg-only `c`
  is high-pooled-rank but **slot-redundant and drifting toward cross-video similarity**.
- **Stability:** perfect (0 skips, grad_norm ~1.5–2.9, std held at ~0.92 by the load-bearing var
  floor — exactly as the pre-launch W&B check predicted).

**Verdict on Run 1:** SIGReg is a rank lever, not a prediction lever, and **not a clean substrate
to build prediction on** — its `c` is static, slot-redundant, and (mildly) collapsing cross-video.
The `λ_recon=0` decoder was never trained, so its `L_recon_*` readouts (~1.04) are meaningless, as
pre-registered.

---

## 4. Run 2 — residual prediction (the representation breakthrough)

Run 2 keeps the full stack and switches `F_c` to predict the temporal residual. The result is
qualitatively unlike anything prior in the project.

### 4.1 The decision metric collapsed toward 1.0 — but read *how*

`coarse_vs_copy_ratio`: 23 → 4.45 (500) → 2.22 (1500) → 1.17 (2500) → **0.915 (3500)** → then
oscillates **1.01–1.12** for the entire back half, ending ~1.08. Versus Run 1's 6.06 and the
inv008 full-latent 1.6–8.2, this is a **~6× improvement on the literal gate metric** — and it
*crossed below 1.0* at step 3500.

The catch is the **components**:
- `coarse_copy_loss = ‖Δ‖²` **rises** 0.20 → **1.554** (the *opposite* of Run 1 and every prior
  run). The copy baseline is no longer trivial: `c` now *moves* a lot over the horizon.
- `coarse_model_loss = ‖Δ̂ − Δ‖²` tracks it, 0.37 → **1.68**, and by the end is *slightly above*
  copy_loss. ratio = model/copy ≈ 1.08 > 1 means **`F_c`'s predicted residual `Δ̂` is, on average,
  marginally *worse* than predicting `Δ̂ = 0`.** `F_c` has effectively reverted to "predict no
  change" — the *tie-by-zero* outcome pre-registered as the likely risk.
- The transient ratio < 1 at step 3500 (model 0.371 < copy 0.406) is the one window where `F_c`
  extracted **real** predictable motion. It evaporated as the motion grew.

### 4.2 The representation is the best `c` the project has produced

- **`c_effective_rank` → 57.9 and still climbing** at the cut — *higher* than Run 1's 50 despite
  *lower* λ_sigreg (5 vs 6). The recon + residual task adds rank on top of SIGReg, and it is
  approaching the > 60 gate for the first time.
- **`c_cross_video_cosine` stays low at 0.17** (min 0.084 at step 2000), vs Run 1's rising 0.39.
  The recon anchor keeps videos genuinely distinct — no Mode-A drift.
- **`c_std_mean` reaches 1.0 and holds** (var floor satisfied, `L_var` ~0.016, barely active). The
  recon supplies the variance work, so unlike Run 1 the floor is *not* load-bearing here.
- Bottleneck attention sharpens hard (`c_attn_entropy` 1.0 → 0.61; `c_attn_entropy_min` → ~1e-8,
  i.e. at least one slot attends to essentially a single token). Slot diversity still dips to ~3.6
  then recovers to 6.3/32 — slot redundancy persists, as in Run 1.

### 4.3 Reconstruction: floor nudged, blindness intact

- `L_recon_present` falls to **0.569**, just *below* the inv007 0.585 floor (and still gently
  dropping) — the first time any lever moved that floor, plausibly because `c` is genuinely richer
  (rank 58). But it is a ~0.015 move (41.5% → 43% of variance), not a breakthrough.
- `L_recon_chat` (0.582) ≈ `L_recon_cplus` (0.567), gap **0.015**. Decoding the *predicted* future
  `ĉ = c_t + Δ̂` reconstructs `e_{t+k}` essentially as well as decoding the *true* future `c_plus`
  — **reconstruction is still blind to prediction quality**, exactly the inv007 finding. The
  "reconstruction-with-residuals" hypothesis does **not** rescue recon as a prediction signal.

---

## 5. The synthesis — ρ(c_t, c_{t+k}) is the master variable

The two runs are the two failure modes of one quantity, the **temporal autocorrelation** of the
latent. Estimating ρ from `1 − ρ ≈ coarse_copy_loss / (2 · c_std_mean²)` (assuming mean(Δ) ≈ 0;
this is a derived lens, not a logged metric):

| step | Run 1 ρ | Run 2 ρ |
|---|---|---|
| 500 | 0.72 | 0.88 |
| 3500 | 0.83 | **0.77** ← Run 2 beats copy here |
| 8000 | 0.89 | 0.33 |
| 14000 | **0.92** | **0.23** |

- **Too static (ρ → 1, Run 1):** copy is a wall, there is almost no motion to predict, `F_c` loses
  badly. SIGReg pushes *here*.
- **Too dynamic (ρ → 0.23, Run 2):** the future is nearly *independent* of the present, so it is
  **not predictable from c_t** — `F_c` can only tie zero. Residual + recon pushes *here*.
- **The predictive sweet spot is in between** (`c` moves *and* the motion is inferable from the
  present). Run 2 passed through it around step 3500 (ρ ≈ 0.77) and **briefly beat copy
  (ratio 0.915)** before the recon-driven decorrelation ran away.

Why does residual + recon decorrelate `c` in time? The two recon anchors pull in opposite temporal
directions — option 1 forces `D(c_t) → e_t` (present detail) and option 3 forces
`D(c_t + Δ̂) → e_{t+k}` (future detail). With k = 12 frames of real SSv2 motion between them, `c_t`
and `c_{t+k}` are driven to encode *different* per-clip appearance, collapsing their correlation.
That appearance change is high-dimensional and **not** what a coarse flow can predict from `c_t`
alone — which is why `F_c` ties zero and `L_recon_chat` stays at the floor. This is the inv007
thesis playing out one level up: **reconstruction inflates `c`'s motion with *appearance* change,
not *predictable dynamics*.**

---

## 6. Reads against the registered predictions (OBSERVATIONS §2026-06-28)

| # | Registered | Observed | Reads as |
|---|---|---|---|
| R1-P1 | Run 1 rank ~55 | 50.4 (plateaued) | Close (slightly low) |
| R1-P2 | Run 1 ratio > 1, worse than baseline | 2.37 → 6.06, monotone worsening | **Match** |
| R1-P4 | Run 1 `L_recon_*` meaningless | ~1.045 (untrained) | **Match** |
| R2-P1 | Run 2 rank ~45–50 | **57.9** | Exceeded |
| R2-P2 | Run 2 ratio "most likely ≥ 1, watch for dip below baseline" | osc ~1.05, **dip to 0.915 @3500** | **Match** (both clauses) |
| R2-P3 | `L_recon_chat` likely stays ~0.585 | chat ≈ cplus ≈ 0.57–0.58, gap 0.015 | **Match** (blindness holds) |
| R2-P4 | stability fine; watch `Δ̂ → 0` and cosine | 0 skips; **`Δ̂ → 0` materialized**; cosine *low* 0.17 | Match + the flagged collapse occurred |

The pre-registered "tie-by-zero" risk for Run 2 is exactly what happened — and the *new* finding
(not pre-registered) is that the residual objective **reversed the static-`c` disease at the
representation level** (ρ collapse, rank 58, healthy `c`), relocating the problem to predictability.

---

## 7. What this means / next steps (→ a probable investigation_010)

The decision triad says: **Run 2's ratio fell toward 1 but did not decisively beat copy, while the
representation became dynamic and healthy.** Per [`NEXT_STEPS.md`](NEXT_STEPS.md), this is the
"residual is a real lever on `c`, but prediction is the new bottleneck" branch. Prioritized:

1. **Anti-collapse on `Δ̂` (the deliberately-omitted change #7) — now strongly motivated.** `F_c`
   ties copy by predicting `Δ̂ ≈ 0`. A variance floor / norm-matching term on `Δ̂` (force
   `‖Δ̂‖ ≈ ‖Δ‖`) would forbid the zero shortcut and make `F_c` commit to a nonzero motion. This is
   the single most direct follow-up.
2. **Govern the decorrelation — find the ρ sweet spot.** The predictive window was at ρ ≈ 0.77
   (step ~3500). Reaching and *holding* it: lower λ_recon_pred (the option-3 anchor is the engine
   of the runaway decorrelation), drop recon entirely in the residual run (clean residual A/B), or
   shorten/anneal the horizon `k` so the induced motion stays inferable.
3. **Sweep horizon `k`** (END_OF_WAVE_2 Tier 1 #1) *with* residual prediction — `k` directly sets
   how far `c` must move and how predictable that move is; residual + a tuned `k` is the natural
   pairing.
4. **A dynamics-specific predictor signal** (inverse-dynamics / multi-step rollout) instead of
   recon-induced motion — recon inflates `Δ` with appearance, not dynamics; an objective that
   rewards *predictable* transition structure is the deeper fix.

**Decision metric going forward is unchanged:** `coarse_vs_copy_ratio` *decisively and stably* < 1
(ideally → 0.70), with `c_effective_rank` and a *moderate* ρ (motion that is predictable) as
corroborating signals. Run 2 is the first datapoint that makes < 1 look reachable rather than
hypothetical.

### TL;DR — §7
Residual prediction is a **real, novel lever on the representation** (first reversal of static-`c`;
richest healthy `c` yet, rank 58) and a **~6× improvement on the gate metric**, but `F_c` still
does not *skillfully* predict — it ties copy by going to zero once the recon-driven temporal
decorrelation overshoots. Next: forbid the `Δ̂ → 0` shortcut and govern ρ to the predictive band
(anti-collapse on `Δ̂`, tune recon/`k`), then judge purely on a stable `coarse_vs_copy_ratio < 1`.
