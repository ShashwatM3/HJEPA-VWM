# Next Steps — investigation_016

Detailed, ranked plan lives in the run folder:
[`run_058.../NEXT_STEPS.md`](run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/NEXT_STEPS.md).
Summary of the investigation-level thread:

1. **Run 059 — EGO4D + residual reconstruction target** (`--recon-residual-target`, else identical
   to run 058). The decisive test of the shared-component/template hypothesis and the highest-value
   next experiment. Keep this investigation OPEN until it lands.
2. **Robustness check** on the small (16-example) validation diagnostic batch (larger,
   source-disjoint) to close the sibling-chunk confound on `c_cross_video_cosine`.
3. **Offline EGO4D feature probes** (`rank_probe.py`/`drift_probe.py --data ego4d`, GUIDE Stage 7B)
   to measure EGO4D `e`'s shared-component/anisotropy directly (the inv014 analog for EGO4D).
4. Contingent `whiten_eps` sweep if the residual target only partially recovers geometry.
5. Gate: do **not** launch the full-prediction EGO4D copy-ratio A/B (run-037 recipe) until a
   present-only EGO4D run holds healthy, video-specific geometry — otherwise a copy-ratio result
   is un-attributable.

This investigation stays OPEN pending run 059.
