# Observations — run 058 (`mvbx96nv`)

Priors (before reading, from the run 057 result and the inv015 arc):

- P1 (transfer): the whiten+cov+var clean-arm win reproduces on EGO4D — rank ≫ 40, cosine
  < 0.5, std ~1.0, large growing video gap, high video-conditioned share.
- P0 (substrate-specific / null): the clean absolute target reopens the run-052 template
  shortcut on EGO4D's more self-similar features; geometry regularizers satisfied on train
  fail to hold the held-out code.

## 2026-07-14 — Final read (run finished, W&B `mvbx96nv`, last diag step 14,500)

**P0 confirmed; P1 rejected.** Verdict: **Collapsed rep / template shortcut** (Reading Cycle B).
Full detail in [`METRIC_READOUT.md`](METRIC_READOUT.md) and [`ANALYSIS.md`](ANALYSIS.md).

Measured, EGO4D (run 058) vs SSv2 twin (run 057, `cdvp6hou`), final step:

- `c_cross_video_cosine` **0.863** vs 0.059 — directional collapse (recovered to 0.369 @1k, then
  reversed and climbed as `recon_scale`→1.0).
- `c_effective_rank` **52.9** vs 208.2 — peaked ~80.7 @2k, then contracted.
- `c_std_mean` **0.419** vs 1.117 — never reached the 1.0 floor on validation.
- `L_recon_present` **0.678** vs 0.713 — EGO4D raw recon nominally *better* (the template trap).
- `L_recon_shuffled_c` **0.696** vs 0.940; `L_recon_video_gap` **0.018** vs 0.227.
- Video-conditioned share **~5.6%** vs ~79%.
- Slots healthy and identical (`c_slot_diversity_rank_centered` 30.67 vs 30.71; `dead_dim_frac`
  0). Stability spotless (0 skips/NaNs, grad norm ~0.05, AGC quiet). Wiring correct
  (`whiten_active=1`, `present_recon_only=1`, `recon_target_residual=0`, `recon_mean_norm=0`,
  `L_flow=0`).

Key facts:

- The config diff vs run 057 is exactly `{dataset, whiten_stats_path, checkpoint_dir}` (W&B
  `compare_runs`), so the collapse is attributable to the EGO4D substrate alone.
- `L_var`≈0.0005 / `L_cov`≈0.2 (satisfied on the 64-ex train batch) coexist with collapsed
  val-batch geometry — a train→val generalization gap; the regularizers do not generalize on
  EGO4D.
- Mechanism = the run-052 clean-target template shortcut, which the residual target fixed in
  inv013/054 and which the inv015 clean arm got away with on SSv2 only because SSv2 features have
  a weaker shared component.

Caveat preserved: the 16-example validation diagnostic batch may contain sibling chunks that
inflate `c_cross_video_cosine`; the honesty gap (0.018) is the least-confounded evidence and is
still catastrophic. A larger source-disjoint diagnostic batch is a queued robustness check.
