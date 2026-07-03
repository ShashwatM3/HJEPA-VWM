# Observations - run 034 `sigreg-only`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `xz3nabr9`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=14050; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.4745 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9241; dead_dim=0; cross_video_cosine=0.3934 |
| Q3 | Rich latent? | PARTIAL | c_effective_rank=50.5622; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.0859 (0.0533 @ 0 -> 0.1392 @ 14000); ratio=6.1219; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.8524; copy_loss=0.1392; ratio=6.1219; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=0.9538; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=1.0454; cplus=1.0459; chat=1.0464; chat-cplus=4.464e-04; no active reconstruction readout for this run |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.3389 @ 0 | 0.8895 @ 14050 | -2.4494 | 0.7637 / 0.9856 / 3.6867 | 0.7888 / 0.881 / 1.0094 (n=82) |
| `L_flow` | 2.8686 @ 0 | 0.8557 @ 14050 | -2.0129 | 0.7321 / 0.9447 / 3.382 | 0.7597 / 0.8502 / 0.9775 (n=82) |
| `L_var` | 0.4081 @ 0 | 0.0391 @ 14050 | -0.3689 | 0.0189 / 0.0324 / 0.4081 | 0.0225 / 0.0307 / 0.0403 (n=82) |
| `L_sigreg` | 0.0444 @ 0 | 0.0024 @ 14050 | -0.042 | 0.002 / 0.0041 / 0.0444 | 0.002 / 0.0026 / 0.0031 (n=82) |
| `L_recon` | 0 @ 0 | 0 @ 14050 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=82) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14050 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=82) |
| `recon_scale` | 0 @ 0 | 0 @ 14050 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=82) |
| `c_std_mean` | 0.4952 @ 0 | 0.9241 @ 14000 | 0.4288 | 0.4952 / 0.9221 / 0.9655 | 0.9241 / 0.9337 / 0.9392 (n=9) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.3934 @ 14000 | -0.3305 | 0.1124 / 0.3021 / 0.7239 | 0.3536 / 0.3726 / 0.3934 (n=9) |
| `c_effective_rank` | 9.4729 @ 0 | 50.5622 @ 14000 | 41.0893 | 9.4729 / 30.1172 / 50.5622 | 42.7919 / 48.1341 / 50.5622 (n=9) |
| `c_slot_diversity_rank` | 16.711 @ 0 | 5.4472 @ 14000 | -11.2638 | 1.2617 / 5.8197 / 18.3168 | 3.8339 / 4.8481 / 5.4472 (n=9) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.842 @ 14000 | -0.1579 | 0.8156 / 0.8829 / 0.9999 | 0.8383 / 0.8409 / 0.842 (n=9) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.1173 @ 14000 | -0.8825 | 0.1172 / 0.4505 / 0.9998 | 0.1172 / 0.1484 / 0.2216 (n=9) |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.1392 @ 14000 | 0.0859 | 0.0533 / 0.2315 / 0.4505 | 0.1392 / 0.1568 / 0.1723 (n=9) |
| `coarse_model_loss` | 3.3106 @ 0 | 0.8524 @ 14000 | -2.4582 | 0.671 / 0.9535 / 3.3106 | 0.7151 / 0.809 / 0.9087 (n=9) |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 0.8937 @ 14000 | 0.631 | 0.2627 / 0.8996 / 0.9812 | 0.8937 / 0.9138 / 0.9226 (n=9) |
| `coarse_vs_copy_ratio` | 62.0676 @ 0 | 6.1219 @ 14000 | -55.9457 | 2.3664 / 5.9263 / 62.0676 | 4.276 / 5.2074 / 6.1943 (n=9) |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.9538 @ 14000 | -11.6473 | 0.7136 / 1.3536 / 12.6011 | 0.7759 / 0.8858 / 1.0001 (n=9) |
| `L_recon_present` | 1.0398 @ 0 | 1.0454 @ 14000 | 0.0056 | 1.0262 / 1.0354 / 1.0454 | 1.0325 / 1.0401 / 1.0454 (n=9) |
| `L_recon_cplus` | 1.0407 @ 0 | 1.0459 @ 14000 | 0.0052 | 1.0275 / 1.0362 / 1.0459 | 1.0323 / 1.0396 / 1.0459 (n=9) |
| `L_recon_chat` | 1.04 @ 0 | 1.0464 @ 14000 | 0.0063 | 1.0276 / 1.0361 / 1.0464 | 1.0327 / 1.0397 / 1.0464 (n=9) |
| `grad_norm` | 3.2838 @ 0 | 2.4745 @ 14050 | -0.8093 | 0.7867 / 2.252 / 3.2838 | 2.2776 / 2.6886 / 3.0369 (n=82) |
| `grad_skipped` | 0 @ 0 | 0 @ 14050 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=82) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0122 @ 14050 | 0.0115 | 6.667e-04 / 0.5317 / 1 | 0.0122 / 0.1295 / 0.302 (n=82) |

## Last Key Metrics

`c_effective_rank`=50.5622 @ 14000; `c_cross_video_cosine`=0.3934 @ 14000; `c_std_mean`=0.9241 @ 14000; `coarse_vs_copy_ratio`=6.1219 @ 14000; `coarse_vs_batch_mean_ratio`=0.9538 @ 14000; `L_recon_present`=1.0454 @ 14000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 6.1219 and the latest batch-mean ratio is 0.9538; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 50.5622, cross-video cosine is 0.3934, and copy loss is 0.1392. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch cleaned optimizer and regularization settings around residual prediction to see whether the result survived a full clean run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

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
the master variable — the opposite failure mode to its pair, [graceful-river-35](../run_035_sigreg-recon-residual/),
whose residual objective drives ρ the other way (→0.23). Read side by side, the two runs bracket the
predictive sweet spot rather than land in it.
