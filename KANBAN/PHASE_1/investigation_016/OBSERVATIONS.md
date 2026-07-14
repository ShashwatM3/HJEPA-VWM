# Observations — investigation_016

## 2026-07-14 — Run 058, the first EGO4D run: recipe does NOT transfer

Run 058 (`ae_latent_stack_whiten_abs_recon_cov_var_ego4d`, W&B `mvbx96nv`) transferred
investigation_015 run 057's strongest present-only recipe onto EGO4D as a clean single-variable
A/B (W&B `compare_runs` confirms the only config deltas are `dataset`, `whiten_stats_path`, and
`checkpoint_dir`). Result: **Collapsed rep / template shortcut**.

Final EGO4D (run 058) vs SSv2 twin (run 057, `cdvp6hou`):

| Metric | EGO4D 058 | SSv2 057 |
|---|---|---|
| `c_effective_rank` | 52.9 | 208.2 |
| `c_cross_video_cosine` | 0.863 | 0.059 |
| `c_std_mean` | 0.419 | 1.117 |
| `L_recon_video_gap` | 0.018 | 0.227 |
| video-conditioned share | ~5.6% | ~79% |
| `c_slot_diversity_rank_centered` | 30.67 | 30.71 |
| stability (`grad_skipped`/`grad_has_nan`) | 0 / 0 | 0 / 0 |

Synthesis:

- **The inv015 clean-arm win is SSv2-specific.** The clean/absolute whitened cosine target,
  which whitening + covariance + variance held honest on SSv2, reopens the run-052
  video-independent template shortcut on EGO4D. The re-collapse is causally tied to the
  reconstruction warmup (`recon_scale`→1.0 at step 2,000): the code specialized early
  (cosine → 0.369 @1k) and then de-specialized as clean recon took over.
- **Pipeline vs representation split.** The EGO4D data/whitening/bottleneck/decoder/optimizer
  pipeline is fully healthy (calmer than SSv2 even); only the learned representation collapses.
  The EGO4D sibling-dataset integration and the one-time EGO4D whitening-stats step work.
- **Regularizers satisfied on train, not on val.** `L_var`≈0 / `L_cov`≈0.2 (train batch) coexist
  with collapsed val-batch geometry — a generalization gap that did not appear on SSv2.
- **On-the-shelf fix:** the residual reconstruction target (inv013/054) removes exactly this
  template shortcut and is a pure config delta; it is the pre-registered next run (059).

Open confound preserved: the 16-example validation diagnostic batch may contain sibling chunks
inflating `c_cross_video_cosine`; the honesty gap (0.018) is the least-confounded metric and is
still catastrophic. Queued as a robustness check.

Per-run triad + full analysis:
[`run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/`](run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/)
(DESCRIPTION / METRIC_READOUT / ANALYSIS / OBSERVATIONS / NEXT_STEPS).
