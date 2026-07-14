# Next Steps — run 058 (`mvbx96nv`)

Ranked, most informative first. All are pure config deltas or offline probes (no code change).

1. **Run 059 — residual-target EGO4D run (highest value).** Rerun 058's exact recipe with
   `--recon-residual-target` ON (reconstruct `e − mean`), everything else identical (EGO4D,
   whiten, cov+var, clean→residual, seed 42, 15k steps). This directly tests the
   shared-component/template hypothesis: if the residual target restores a large `L_recon_video_gap`
   and drives `c_cross_video_cosine` back below 0.5 on EGO4D as it did on SSv2 (inv013/054), the
   collapse was the clean-target template shortcut and the residual target is mandatory on EGO4D.
   If it does *not* recover, the problem is deeper (whitening or the EGO4D feature geometry
   itself) and points to step 3.

2. **Diagnostic-batch robustness check.** Confirm the collapse is not a 16-example sibling-chunk
   artifact: re-measure `c_cross_video_cosine` / `c_effective_rank` / video gap on a larger,
   source-disjoint EGO4D validation batch (raise the diag batch size, or measure offline from a
   checkpoint over a source-deduplicated set). The honesty gap already argues the collapse is
   real; this closes the one open confound.

3. **Offline EGO4D feature probes (inv014 analog for the new substrate).** Run
   `rank_probe.py --data ego4d` and `drift_probe.py --data ego4d` (EGO4D GUIDE Stage 7B) to
   quantify EGO4D `e`'s shared-component / anisotropy and within-video drift directly. This tells
   us *before* more training whether EGO4D features intrinsically carry a larger video-independent
   template than SSv2 — the root-cause measurement behind this run's collapse.

4. **(Contingent) `whiten_eps` sensitivity.** If the residual-target run only partially recovers,
   sweep `--whiten-eps` on EGO4D — the SSv2-tuned 1e-4 floor may under-equalize or amplify an
   EGO4D low-energy tail differently.

5. **Only after the representation holds on EGO4D:** launch the full-prediction EGO4D A/B (the
   run-037 recipe, `--data ego4d`, `predict_residual`, prediction-side recon) that the migration
   was undertaken to run, and read `coarse_vs_copy_ratio` / `coarse_vs_batch_mean_ratio` against
   run 037. Do not run this until a present-only EGO4D run holds healthy, video-specific geometry —
   otherwise a copy-ratio failure would be un-attributable (collapsed target vs weak predictor).
