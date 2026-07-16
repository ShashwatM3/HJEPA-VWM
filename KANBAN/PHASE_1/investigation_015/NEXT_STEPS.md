# Next Steps - investigation_015

## Current carry-forward (2026-07-16)

1. ~~Run and analyze the whitened latent-stack sequence 054–057.~~ Complete. Run 057 is the
   settled absolute-target SSv2 control and geometry bundle: whitened reconstruction plus
   covariance and the variance floor, without SIGReg. The residual-target-plus-geometry
   combination remains unrun.
2. Keep the bottleneck-only control (run 054 without `--whiten-*`) open only if causal
   whitening-versus-latent-stack attribution is still worth its GPU cost. It is not needed to
   choose the current recipe.
3. ~~Transfer run 057 to EGO4D.~~ Completed as investigation 016 run 058, followed by the
   run-060 reconstruction-weight arm. The EGO4D fixed diagnostic batch is single-source, so
   repair source-aware validation before choosing another objective or capacity arm.
4. Do not activate full prediction from a checkpoint merely because present-side rank is high.
   First require honest source-aware reconstruction; then a full-prediction run must separately
   pass both copy and batch-mean gates.

## Historical plan (superseded by the completed runs above)

## Immediate

1. Launch run 054 `ae_latent_stack_whiten_recon_only` per [`GUIDE.md`](GUIDE.md)
   (includes the one-time `whiten_stats.py` prerequisite on the pod).
2. Analyze with Reading Cycle B (present-recon-only) plus the per-change attribution
   matrix in [`README.md`](README.md).

## Contingent follow-ups (pick by the README's 2x2 outcome)

- Feature rank holds but slots still merge -> whitening is the working lever; consider
  keeping whitening and re-testing slot competition strength (more latent blocks) or a
  small `lambda_slot`.
- Slots differentiate but feature rank contracts -> sweep `--whiten-eps` (floor may be
  amplifying tail noise or under-equalizing) before adding geometry terms back.
- Both hold -> transfer the recipe into a full-prediction run (the standing Phase 1
  gate: `coarse_vs_copy_ratio` / `coarse_vs_batch_mean_ratio`).
- Both collapse -> rerun with inv011-style geometry terms (`lambda_var`, small
  `lambda_cov`) on top, keeping whitening + residual target; the ambiguity tiebreaker is
  one bottleneck-only control run (drop the two `--whiten-*` flags, ~6 h).

## 2026-07-05 — after run 054 ("neither delta sufficient" cell; see ANALYSIS_054.md)

1. **Run 055 (subsequently completed):**
   [`run_055_ae_latent_stack_whiten_abs_recon/`](run_055_ae_latent_stack_whiten_abs_recon/)
   — run 054's exact recipe minus the residual reconstruction target. Tests whether
   whitening alone blocks the run-052 template shortcut (fills the absolute/whitened
   cell of the target x space 2x2). Not the §4 bottleneck-only control.
2. Still queued from ANALYSIS_054: the bottleneck-only control
   (drop `--whiten-*`, keep residual target — load-bearing for 053->054 attribution),
   then the "neither sufficient" follow-up (whitening + residual + `lambda_var` +
   small `lambda_cov`).
