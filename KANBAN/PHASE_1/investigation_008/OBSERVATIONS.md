# Observations — investigation_008 (SIGReg sweep)

> **Scope of this file (rewritten 2026-06-28).** Measured W&B metrics only, plus how each
> reads **against the predictions this wave was registered on**. No mechanism hypotheses,
> verdicts, or recommendations — those are deliberately excluded so this record does not
> bias later analysis. Registered predictions + design rationale live in
> [`DESCRIPTION.md`](DESCRIPTION.md) and [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md) §6 (and in
> git history at `9f787a9`). "Reads as" lines state only match / contradiction / no-prediction
> relative to the registered statement.

---

## 1. Provenance

- **Source:** W&B `smahalanobis-uc-davis/hjepa-vwm`, group `inv008_sigreg`, pulled
  2026-06-28 via `run.scan_history()` (full, unsampled).
- **Plateau** = mean of the last 3 logged points of a metric. Diag-cadence metrics
  (`c_*`, `coarse_*`, `L_recon_*`) log every 500 steps; per-step metrics (`L_flow`,
  `L_sigreg`, `loss`, `grad_norm`, …) every 50.
- **Runs:**

| λ_sigreg | run name | id | commit | state | last step | runtime | λ_recon | λ_var |
|---|---|---|---|---|---|---|---|---|
| 0 (control) | eager-plant-22 | 591mt31k | 823eaa5 | crashed | 9100 | 3.96 h | 0.05 | 0.5 |
| 0.3 | lambda_sigreg_0.3 | x7z6e0ah | 35f24f1 | crashed | 13150 | 5.71 h | 0.05 | 0.5 |
| 1.0 | lambda_sigreg_1.0 | jk8kj7h7 | 35f24f1 | crashed | 13100 | 5.69 h | 0.05 | 0.5 |
| 3.0 | lambda_sigreg_3.0 | 9jxc8i1q | 35f24f1 | crashed | 13050 | 5.71 h | 0.05 | 0.5 |
| 10 | lambda_sigreg_10.0 | fbqgix1x | 35f24f1 | crashed | 13150 | 5.71 h | 0.05 | 0.5 |

- **Notes that affect reading the numbers:**
  - All 5 states are `crashed` = **runs terminated manually**, not a training failure
    (`grad_has_nan`, `grad_skipped`, `instability_warn` all 0 throughout — see §4.9).
  - The 4 sweep runs are on commit **`35f24f1`** (includes WALK_FIXES F1–F3: fp32 SIGReg,
    dedicated SIGReg RNG, `grad_global_norm`→`grad_global_norm_postclip`).
  - The control **eager-plant-22** is an inv007 Wave-1 run (commit `823eaa5`): it has **no
    `L_sigreg`**, logs the post-clip grad norm under the old name `grad_global_norm`, and
    ran to 9.1k vs the sweep's ~13.1k.
  - `recon_scale` = 1.0 at the plateau in every run (the 2000-step recon warmup is complete).

## 2. Registered predictions (neutral restatement; source: SIGREG_DESIGN §6 / DESCRIPTION)

- **P1 (primary):** `c_effective_rank` would rise above its ~13 prior level under SIGReg.
- **P2 (secondary/decision):** `coarse_vs_copy_ratio` would (most likely) remain > 1.
- **P3:** `L_recon_present` might drop modestly.
- **P4 (health):** at high λ, `L_flow` might rise; `c_cross_video_cosine` was expected to
  **fall** (more spread); a **rise** in it was flagged as a collapse-watch.
- Registered decision pair: `c_effective_rank` ∧ `coarse_vs_copy_ratio`.

## 3. Master table — plateau value per metric × run

| metric | ctrl λ0 | λ0.3 | λ1.0 | λ3.0 | λ10 | pattern in λ |
|---|---|---|---|---|---|---|
| **c_effective_rank** | 12.67 | 13.23 | 13.07 | 34.11 | 73.26 | ↑ monotone (steep ≥ λ3) |
| **coarse_vs_copy_ratio** | 1.575 | 2.180 | 2.566 | 4.721 | 8.153 | ↑ monotone |
| coarse_vs_batch_mean_ratio | 0.335 | 0.381 | 0.437 | 0.814 | 1.010 | ↑ monotone |
| coarse_model_loss | 0.376 | 0.428 | 0.482 | 0.785 | 0.880 | ↑ monotone |
| coarse_copy_loss | 0.239 | 0.197 | 0.188 | 0.167 | 0.108 | ↓ monotone |
| coarse_batch_mean_loss | 1.124 | 1.125 | 1.103 | 0.965 | 0.871 | ↓ (from λ1) |
| L_recon_present | 0.5847 | 0.5879 | 0.5860 | 0.5748 | 0.5768 | flat → −0.01 at λ≥3 |
| L_recon_cplus | 0.5839 | 0.5853 | 0.5836 | 0.5739 | 0.5742 | tracks present (±0.002) |
| L_recon_chat | 0.5918 | 0.5934 | 0.5921 | 0.6011 | 0.6110 | ↑ at λ≥3 |
| L_recon (train) | 0.629 | 0.625 | 0.625 | 0.614 | 0.615 | ~flat (−0.015) |
| L_sigreg | — | 0.0084 | 0.0065 | 0.0039 | 0.0014 | ↓ monotone |
| c_cross_video_cosine | 0.244 | 0.290 | 0.275 | 0.344 | 0.384 | ↑ (from λ1) |
| c_std_mean | 1.030 | 1.030 | 1.023 | 0.962 | 0.907 | ↓ at λ≥3 |
| c_std_median | 1.032 | 1.018 | 1.022 | 0.958 | 0.900 | ↓ at λ≥3 |
| c_dead_dim_frac | 0 | 0 | 0 | 0 | 0 | flat at 0 |
| c_slot_diversity_rank | 17.69 | 18.24 | 14.45 | 2.85 | 11.52 | non-monotone (min at λ3) |
| c_attn_entropy | 0.908 | 0.890 | 0.887 | 0.865 | 0.882 | ~flat (slight ↓) |
| c_attn_entropy_min | 0.799 | 0.785 | 0.740 | 0.278 | 0.497 | dip at λ3 |
| L_flow | 0.413 | 0.439 | 0.497 | 0.801 | 0.925 | ↑ monotone |
| L_var | 0.007 | 0.006 | 0.011 | 0.020 | 0.036 | ↑ at λ≥1 |
| loss (total) | 0.448 | 0.475 | 0.541 | 0.854 | 0.988 | ↑ monotone |
| grad_norm (pre-clip) | 2.98 | 3.08 | 3.40 | 2.69 | 2.28 | non-monotone, ~2–4 |
| grad_skipped / instability_warn / grad_has_nan | 0 | 0 | 0 | 0 | 0 | flat at 0 |

## 4. Metric-by-metric

### 4.1 `c_effective_rank` — primary
- **Trajectory** (step:value): ctrl 0:9.5 → 3k:7.8 → 6k:9.9 → 9k:12.9. λ0.3 → 13.2 by ~9k, flat to 13.2. λ1.0 → peak 13.7 (~7.5k), ends 13.0. λ3.0 0:9.5 → 4.5k:13.4 → 7.5k:22.9 → 13k:34.4. λ10 0:9.5 → 4.5k:30.4 → 7.5k:57.7 → 13k:73.8.
- **Range:** all start 9.47 (init). Plateau 12.67 / 13.23 / 13.07 / 34.11 / 73.26. Max 12.95 / 13.27 / 13.75 / 34.45 / 73.79.
- **Shape note:** λ3 and λ10 had **not plateaued** — both at their max at the final logged step (still increasing when terminated). λ0.3/λ1/ctrl were flat.
- **Reads as P1:** Matches for λ≥3 (12.7 → 34.1 → 73.3, monotone in λ). For λ≤1, within ~0.5 of the control (~13).

### 4.2 `coarse_vs_copy_ratio` — decision metric (with components)
- **Trajectory:** all start 62.1 (init) and drop fast. ctrl bottoms ~1.5 (5–9k). λ0.3 ~1.8–2.3. λ1.0 ~2.2–2.7. λ3.0 falls to 2.27 (~4.5k) then rises to 5.24 (13k). λ10 falls to 2.45 (~3k) then rises to 9.07 (13k).
- **Plateau / min-over-run:** plateau 1.575 / 2.180 / 2.566 / 4.721 / 8.153; min (excl. init) 1.528 / 1.771 / 2.093 / 2.273 / 2.450. Both monotone in λ; **all values > 1 at all logged steps in all runs.**
- **Components:** `coarse_model_loss` ↑ with λ (0.376 → 0.880); `coarse_copy_loss` ↓ with λ (0.239 → 0.108). (ratio = model / copy.)
- **Shape note:** λ3 and λ10 ratios were **still rising** at termination (not plateaued).
- **Reads as P2:** Matches — ratio stayed > 1 in every run. Relative to the registered ≤0.70 / <1 gate, it moved **further from** the gate as λ increased.

### 4.3 `coarse_vs_batch_mean_ratio` (supporting baseline)
- Plateau 0.335 (ctrl) → 1.010 (λ10), monotone ↑. `coarse_batch_mean_loss` fell 1.124 → 0.871.
- **Reads as:** no registered prediction. (Control beat the batch-mean baseline by ~3×; at λ10 the ratio is ≈1.0.)

### 4.4 `L_sigreg` — the swept regularizer
- All sweep runs start 0.0444 (init) and fall within ~2k steps. Plateau 0.0084 / 0.0065 / 0.0039 / 0.0014 for λ 0.3/1/3/10 — **monotone ↓ in λ**. Min 0.0063 / 0.0054 / 0.0028 / 0.0013.
- Control: not logged (pre-SIGReg run).
- **Reads as:** no separate registered prediction; confirms the swept term was active and decreased with training, more strongly at higher λ. (Step-0 value 0.0444 in all runs.)

### 4.5 Reconstruction — `L_recon_present` / `cplus` / `chat`, `L_recon`(train)
- **present:** all start 1.04 (init) → plateau 0.585 / 0.588 / 0.586 / 0.575 / 0.577. Full spread across runs 0.575–0.588 (0.013). Lowest 0.5747 at λ3. min-over-run ≈ plateau (flat tails).
- **cplus:** tracks present within ±0.002 in every run.
- **chat:** plateau 0.592 / 0.593 / 0.592 / 0.601 / 0.611, ↑ at λ≥3; last value at λ10 = 0.626. Gap `chat − cplus`: 0.008 / 0.008 / 0.009 / 0.027 / 0.037 — ↑ with λ.
- **L_recon (train batch):** plateau 0.629 / 0.625 / 0.625 / 0.614 / 0.615.
- **Reads as P3:** Matches — `L_recon_present` dropped modestly (~0.01) at λ≥3; near-flat at λ≤1. (No prediction was registered for `chat` or the gap.)

### 4.6 `c_cross_video_cosine` (collapse-watch)
- All start 0.724 (init), drop to a mid-training min of 0.11–0.13 (~1.5–3k), then rise. Plateau 0.244 / 0.290 / 0.275 / 0.344 / 0.384 — ↑ with λ (from λ1). All plateaus < 0.5 and below init.
- Co-logged at the same plateaus: `c_dead_dim_frac` = 0 in all runs; `c_std_mean` = 0.91–1.03.
- **Reads as P4:** Contradicts the directional expectation — it **rose** with λ rather than falling. The value flagged as a collapse-watch (a rise) occurred; the plateaus remained < 0.5 (vs init 0.724).

### 4.7 `c_std_mean` / `c_std_median` / `c_dead_dim_frac` / `L_var`
- `c_std_mean` plateau 1.030 / 1.030 / 1.023 / 0.962 / 0.907 (↓ below 1 at λ≥3). `c_std_median` similar (0.900 at λ10). All runs start 0.495.
- `c_dead_dim_frac` = 0 at every logged step in every run.
- `L_var` (variance-floor hinge loss) plateau 0.007 / 0.006 / 0.011 / 0.020 / 0.036 — ↑ with λ at λ≥1.
- **Reads as:** no registered prediction. (Per-dim std sits below the floor target of 1.0 at λ≥3; the hinge loss `L_var` is correspondingly larger.)

### 4.8 `c_slot_diversity_rank` / `c_attn_entropy(_min)`
- `c_slot_diversity_rank` plateau 17.69 / 18.24 / 14.45 / 2.85 / 11.52 — **non-monotone**, minimum at λ3 (min-over-run 1.15 at λ3, 3.82 at λ10). Control and λ0.3 are the highest (~18).
- `c_attn_entropy` ~flat (0.91 → 0.88). `c_attn_entropy_min` dips at λ3 (0.278) and λ10 (0.497) vs ctrl 0.799.
- **Reads as:** no registered prediction (not part of P1–P4). Reported for completeness; the λ3 run shows the lowest slot-rank and lowest min-attention-entropy of the set.

### 4.9 Flow / optimization — `L_flow`, `loss`, `grad_norm`, stability flags
- `L_flow` plateau 0.413 / 0.439 / 0.497 / 0.801 / 0.925 — monotone ↑ with λ. (`coarse_model_loss`, the same MSE measured in the diagnostic pass, shows the same ordering.)
- `loss` (total) plateau 0.448 → 0.988, monotone ↑.
- `grad_norm` (pre-clip; the F1 authoritative one) stayed ~2.3–4.1 across all runs and steps; max 4.31 (λ1). `grad_global_norm_postclip` is the post-clip value and is pinned near the 0.5 clip (per WALK_FIXES F1 — not a magnitude reading).
- `grad_skipped` = `instability_warn` = `grad_has_nan` = 0 at every logged step in every run.
- **Reads as P4:** `L_flow` rose at high λ — matches the P4 "might rise" note. No optimizer instability in any run.

## 5. Registered-prediction scorecard (neutral)

| # | Registered prediction | Observed | Reads as |
|---|---|---|---|
| P1 | `c_effective_rank` rises above ~13 | 12.7 (ctrl) → 13.2 / 13.1 / 34.1 / 73.3; monotone in λ; λ≥3 still rising at end | **Matches** (λ≥3); λ≤1 ≈ control |
| P2 | `coarse_vs_copy_ratio` stays > 1 | > 1 at all steps, all runs; plateau 1.6 → 8.2, monotone ↑; moved further from the <1 gate with λ | **Matches** |
| P3 | `L_recon_present` drops modestly | 0.585 → 0.575 at λ≥3 (~0.01); flat at λ≤1 | **Matches** |
| P4a | `L_flow` may rise at high λ | 0.41 → 0.93, monotone ↑ | **Matches** |
| P4b | `c_cross_video_cosine` expected to fall (rise = collapse-watch) | rose 0.24 → 0.38 with λ; stayed < 0.5; `c_dead_dim_frac`=0, `c_std`≈0.91–1.03 | **Contradicts direction**; collapse-watch value present, other collapse indicators flat |

## 6. Items flagged for measurement (not interpreted here)

These are open *measurement* questions raised during review, not conclusions:
- A `‖Δc‖/‖c‖` trajectory was referenced in prior discussion but is **not a logged metric**;
  `coarse_copy_loss` (= ‖c_t − c_plus‖²) is its closest logged proxy (plateau 0.239 → 0.108, ↓ with λ).
- The reconstruction floor's behavior under (a) a per-dimension R² normalization and (b) a
  c-ablation has **not been measured** (would require a checkpoint forward; see WALK_FIXES /
  prior session notes). The `L_recon_*` numbers above use the existing global-variance
  normalizer in `losses.reconstruction_loss`.

*(Per PROTOCOL: append dated sections for any new data; do not rewrite the above.)*
