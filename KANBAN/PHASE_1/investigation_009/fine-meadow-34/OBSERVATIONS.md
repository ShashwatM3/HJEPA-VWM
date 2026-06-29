# Observations — fine-meadow-34 (Run 1: SIGReg substrate)

## 2026-06-28 — Final read (manually ended, step 14050; plateau = mean of last 3 diag points)

All values pulled from W&B (`xz3nabr9`, full history, 29 diag points). Cross-run synthesis and the
derived ρ(c_t, c_{t+k}) lens live in [../RESULTS_ANALYSIS.md](../RESULTS_ANALYSIS.md).

| Metric | Value | Trajectory |
|---|---|---|
| `coarse_vs_copy_ratio` | **6.06** | min **2.37 @4500** → rises monotonically to 6.06 |
| `c_effective_rank` | **50.4** | 9.5 → 50.4, plateaued by ~12k |
| `coarse_copy_loss` = ‖Δ‖² | **0.143** | **falls** 0.45 → 0.143 |
| `coarse_model_loss` | 0.867 | tracks above copy throughout |
| implied ρ(c_t, c_{t+k}) | **~0.92** | rises ~0.72 → ~0.92 (derived, see RESULTS_ANALYSIS §5) |
| `coarse_vs_batch_mean_ratio` | 0.96 | ~1 (beats batch-mean baseline only marginally) |
| `L_flow` | 0.878 | elevated, flat |
| `c_std_mean` | 0.928 | held at ~0.90–0.93 by the var floor all run |
| `c_cross_video_cosine` | 0.388 | **rises** toward 0.39 |
| `c_slot_diversity_rank` | 5.39 / 32 | dips to ~1.3 @3500–4500, partial recovery to 5.4 |
| `c_attn_entropy` / `_min` | 0.842 / 0.118 | |
| `L_recon_present` | ~1.045 | untrained decoder — **n/a** |
| `L_var` / `L_sigreg` | 0.033 / 0.0023 | floor continuously active |
| `grad_skipped` / `instability_warn` / `grad_has_nan` / `c_dead_dim_frac` | 0 / 0 / 0 / 0 | every step |

## Interpretation (metric-level)

- **The ratio is U-shaped, not monotone-improving.** `coarse_vs_copy_ratio` bottoms at 2.37 (step
  4500, rank ~18) and then climbs to 6.06 as rank continues to 50. The early descent is settling;
  once SIGReg's rank-pumping engages past rank ~20, the ratio rises with every additional rank point.
  Read together with `c_effective_rank`, rank and prediction move in opposite directions over the
  back two-thirds of the run.
- **`c` froze in time.** `coarse_copy_loss = ‖c_t − c_{t+k}‖²` *falls* 0.45 → 0.143 — the future
  latent becomes a near-copy of the present (implied ρ → ~0.92). With almost no motion over the
  horizon, the "predict no change" baseline strengthens, which is the direct cause of the rising
  ratio: `coarse_model_loss` (0.867) stays roughly flat while its denominator shrinks.
- **Health signals split.** `c_std_mean` is held at ~0.93 by the load-bearing var floor
  (`L_var ≈ 0.033` every step), and there is zero instability. But `c_cross_video_cosine` drifts
  *up* to 0.39 (videos becoming less distinct) and `c_slot_diversity_rank` collapses mid-run
  (~18 → ~1.3) before only partly recovering to 5.4/32. So the high pooled `c_effective_rank` is
  concentrated in static, cross-video-similar, slot-redundant content.

## Interpretation w.r.t. the registered hypothesis

- **R1-P1 (rank ~55):** reads as a close near-miss — measured **50.4**, slightly under the estimate
  and under the >60 gate.
- **R1-P2 (ratio > 1, ~6–7, worse than baseline):** reads as a **match** — 6.06, monotone worsening
  over the back of the run; the inv008 "rank↑ ⟺ prediction↓" law reproduced cleanly with recon
  removed, so the law is not a recon artifact.
- **R1-P3 (`L_flow` ~0.85–0.90, std floor-active, no collapse):** reads as a **match** on `L_flow`
  (0.878) and std (0.928, floor continuously active); "no collapse" holds for std/dead-dims but the
  *cross-video* cosine and slot-rank drift in the unhealthy direction.
- **R1-P4 (`L_recon_*` meaningless):** reads as a **match** — readouts decode an untrained decoder
  (~1.045); ignored for this run as pre-registered.

## Connection to the wave

This run answers the substrate question for [investigation_009](../DESCRIPTION.md): SIGReg at λ=6
moves `c_effective_rank` but does so by filling `d_c` with static appearance, leaving `c` temporally
frozen, slot-redundant, and drifting toward cross-video similarity. It is the **static-ρ pole** of
the master variable — the opposite failure mode to its pair, [graceful-river-35](../graceful-river-35/),
whose residual objective drives ρ the other way (→0.23). Read side by side, the two runs bracket the
predictive sweet spot rather than land in it.
