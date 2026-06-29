# Observations — investigation_009 (residual prediction + SIGReg substrate)

No run data yet. This file registers the **pre-run predictions** the wave is launched on; metrics +
a neutral "reads as match/contradiction" pass are appended as dated sections once the runs land
(mirror the inv008 OBSERVATIONS discipline: measurements first, verdicts deferred). Design rationale
lives in [`DESCRIPTION.md`](DESCRIPTION.md) and [`GUIDE.md`](GUIDE.md).

---

## 2026-06-28 — Registered predictions

λ-extrapolations are log-linear from the inv008 SIGReg points (`fbqgix1x` λ10 → rank 73.3,
`9jxc8i1q` λ3 → rank 34.1).

### Run 1 — SIGReg substrate (λ_sigreg=6.0, λ_var=0.5, no recon, full-latent)

- **P1 `c_effective_rank` ~55** (between λ3=34 and λ10=73) — likely *just under* the 60 gate.
- **P2 `coarse_vs_copy_ratio` > 1, ~6–7** — worse than the no-SIGReg baseline (the inv008 rank↑⟺
  prediction↓ law). Run 1 characterizes a substrate; it is **not** expected to fix prediction.
- **P3 `L_flow` elevated ~0.85–0.90**; `c_std_mean` ~0.90–0.93 (floor active), `c_dead_dim_frac=0`,
  `c_cross_video_cosine` < 0.5 — no collapse.
- **P4 `L_recon_*` are meaningless** for Run 1 — the decoder is never trained (`λ_recon=0`), so the
  diag-cadence readouts decode an init decoder (~1.0). Expected and harmless; ignore them for Run 1.

### Run 2 — residual prediction (λ_sigreg=5.0, λ_var=0.5, recon 0.05+0.05, **residual**)

- **P1 `c_effective_rank` ~45–50** (λ5 substrate).
- **P2 (decision) `coarse_vs_copy_ratio`** — the whole point. Two outcomes:
  - falls **below the full-latent baseline** (toward <1) → residual prediction extracts predictable
    motion → a real lever → build on it (k sweep, anti-collapse on Δ, drop recon for a clean A/B).
  - stays **≥ 1 / tracks full-latent** → residual alone is insufficient (Δ is mostly
    unpredictable at k=12) → the horizon/inverse-dynamics/rollout pivot is the path.
  - Registered prior: **most likely ≥ 1** (residual does not weaken copy; it needs predictable
    motion), but watch for a dip below Run 1 / the inv008 full-latent ratio.
- **P3 `L_recon_chat`** with the residual decode `ĉ=c_t+Δ̂` — likely still pinned at the ~0.585
  recon floor (inv007 recon-blindness); a drop would be the surprising, hypothesis-supporting result.
- **P4 stability:** no cliff, `grad_norm` in band (~2–4), `grad_skipped=0`; `c_std_mean` ~0.92.
  Collapse-watch: `Δ̂ → 0` (ratio → 1 trivially) and `c_cross_video_cosine` climbing.

### Cross-run read

- `coarse_vs_copy_ratio` is **comparable across both runs and vs inv008** (the residual copy baseline
  is the same `‖Δ‖²`; σ-scaling cancels). The headline comparison is **Run 2 ratio vs Run 1 / inv008
  full-latent ratio at matched λ**.
- Absolute `L_flow` / `coarse_copy_loss` are **at natural Δ scale** in Run 2 (smaller than full-latent
  in absolute terms) — read the **ratio**, not the absolutes, across the two parametrizations.

*(Per PROTOCOL: append dated sections for any new data; do not rewrite the above.)*

---

## 2026-06-28 — Measured results (both runs complete, step 14050)

Runs: **`fine-meadow-34`** (`xz3nabr9`, Run 1) · **`graceful-river-35`** (`jsh6uo7p`, Run 2), both
commit `bc77db6`, ~6h10m, manually ended. Pulled via `get_run_history` (full, 29 diag points/run).
`grad_skipped = instability_warn = grad_has_nan = c_dead_dim_frac = 0` at every step in both.
**Full interpretation + the ρ(c_t,c_{t+k}) analysis: [`RESULTS_ANALYSIS.md`](RESULTS_ANALYSIS.md).**

| metric (plateau = last-3 mean) | Run 1 (λ6, full, no recon) | Run 2 (λ5, residual, recon) |
|---|---|---|
| coarse_vs_copy_ratio | 6.06 (rising) | 1.08 (osc ~1.05) |
| ↳ min over run (excl init) | 2.37 @4500 | 0.915 @3500 |
| c_effective_rank | 50.4 | 57.9 (climbing) |
| coarse_copy_loss (‖Δ‖²) | 0.143 (↓ from 0.45) | 1.554 (↑ from 0.20) |
| coarse_model_loss | 0.867 | 1.682 |
| coarse_vs_batch_mean_ratio | 0.96 | 1.17 |
| L_flow | 0.878 | 1.65 (Δ-scale) |
| c_std_mean | 0.928 | 1.005 |
| c_cross_video_cosine | 0.388 (↑) | 0.170 (low) |
| c_slot_diversity_rank | 5.39 | 6.28 |
| c_attn_entropy / _min | 0.842 / 0.118 | 0.607 / ~1e-8 |
| L_recon_present | ~1.045 (untrained; n/a) | 0.569 (< 0.585 floor) |
| L_recon_chat − cplus | n/a | 0.015 |
| L_var / L_sigreg | 0.033 / 0.0023 | 0.016 / 0.0025 |

**Scorecard vs registered predictions (§2026-06-28 above):** R1-P1 close (rank 50 vs ~55);
R1-P2 **match** (ratio worse, 6.06); R1-P4 **match** (recon readouts meaningless); R2-P1 exceeded
(rank 57.9 vs ~45–50); R2-P2 **match** (osc ~1.05 **and** the flagged dip < 1, 0.915@3500);
R2-P3 **match** (`L_recon_chat ≈ cplus`, blindness holds); R2-P4 stability fine, **the `Δ̂→0`
tie-by-zero materialized**, cosine stayed *low* (0.17, not the climb that was the collapse-watch).

**Neutral headline:** Run 2's decision metric fell ~6× vs Run 1 (1.08 vs 6.06) and its rank is
higher (57.9 vs 50.4), but `coarse_copy_loss` moved in *opposite* directions (Run 2 ↑ 0.2→1.55,
Run 1 ↓ 0.45→0.14) — i.e. the residual run's `c` became temporally dynamic while the substrate
run's `c` froze. Whether Run 2's ratio≈1 is a skillful tie or a predict-zero tie: see RESULTS_ANALYSIS §4–5.
