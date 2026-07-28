# Next steps — Run 68, DINOv3 whitened internal-memory M=512 covariance plus variance

> **Local-label/status correction (2026-07-27):** completed follow-up is recorded under canonical
> local [Run 070](../run_070_whitened_dinov3_m512_cov_var/), W&B `qqozribu`. The pre-launch list
> below is retained source-branch history, not current work.

Before launch:

1. Execute [`GUIDE.md`](GUIDE.md) in order and stop at the first failed gate.
2. Fit and strictly validate a new DINOv3/EGO4D whitening envelope only after frame microbatch is
   settled by resource preflight.
3. Confirm the final no-step provenance differs from Run 67 only in the registered whitening
   bundle and collision-avoiding operational identity.

After the run completes:

1. Fill [`OBSERVATIONS.md`](OBSERVATIONS.md) with W&B identity, whitening identity, final checkpoint
   SHA-256, and the present-only Reading Cycle B table.
2. Compare within-run diagnostics against completed unwhitened DINOv3 Run 67 (`it7sq8nz`); do not
   directly compare raw whitened and unwhitened reconstruction magnitudes.
3. Update the parent investigation records and the Phase 1 run table with the verified state and
   verdict.
