# Next Steps — run 060

## Closure / carry-forward status

Run 060 is complete. Do not relaunch this command or transfer its checkpoint into prediction. The
weight-extreme question is answered narrowly: `lambda_recon=1.0` is numerically stable and can
coexist with covariance/variance geometry, but it does not materially lower the whitened
reconstruction floor or establish strong content conditioning.

Carry forward these limits:

- this was present-only; it contains no evidence about `F_c`, temporal dynamics, copy beating, or
  batch-mean beating;
- the comparison with historical run 058 is not causal because the runtime/data/provenance stack
  changed;
- the current honesty and pair-cosine metrics are within one EGO4D source recording;
- a lower raw reconstruction value does not count without source-aware conditioning and healthy
  geometry.

## Ordered next work

### 1. Repair the fixed validation contract

Build a deterministic batch with one chunk per EGO4D source UID, assert uniqueness, and replace
the unconditional `torch.roll` with a verified cross-source derangement. Log separate same-source
and cross-source cosine/gap metrics and record paths plus parsed UIDs in provenance.

If the final run-060 checkpoint and compatible runtime remain available, re-evaluate it on the
repaired manifest before paying for retraining. This is the cheapest way to answer whether the code
is globally source-specific.

### 2. Add explicit template and capacity oracles

On cached source-diverse frozen features, measure:

1. zero-code, mean-code, `B(zeros)`, global-mean, and per-position-mean reconstruction;
2. rank-r/PCA reconstruction at 64, 128, 256, 512, and 1,024 dimensions in raw and whitened space;
3. a fixed-batch overfit ladder: free per-sample latent plus D, then B+D without geometry, then
   variance, covariance, augmentation, and the production schedule;
4. decoder output norms, per-module update/weight ratios, and per-loss B-gradient norms/cosines.

These probes distinguish decoder/latent bandwidth, the early 1,024-to-256 projection, bottleneck
content routing, regularizer conflict, schedule closure, and a constant-position template.

### 3. Run a real reconstruction-weight ablation only if still needed

Pair `lambda_recon=0.05` and `1.0` on one commit with the same initialization fingerprint, sample
order, augmentation identities, whitening payload, source-diverse validation manifest, encoder
fingerprint, and diagnostics. The historical run-058 comparison is insufficient for a causal
weight estimate.

### 4. Choose the model arm from the probes

- Residual target is still the leading honesty arm once cross-source diagnostics are valid.
- Increase input projection width, slot count, latent width, or decoder capacity only if the
  fixed-batch/PCA ladder identifies the corresponding bandwidth boundary.
- Test partial/no whitening only if the cached-feature oracle shows full ZCA is setting the target
  difficulty.
- Change optimizer/LR schedule only if the architecture can overfit but production training stalls.
- Compare SigLIP only against a same-commit V-JEPA companion; run 059 remains genuinely unlaunched.

## Chronological linkage

- Previous science run: run 058
  [`ae_latent_stack_whiten_abs_recon_cov_var_ego4d`](../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/)
  (`mvbx96nv`).
- No later W&B science run exists in the live project at this audit; run 060 (`2423b84g`) is the
  61st and newest entry.
- The analysis-only
  [`reconstruction_floor_architecture_audit`](../reconstruction_floor_architecture_audit/)
  supplies the corrected measurement boundary and the experiment ordering above.

## Acceptance gate before prediction

Do not activate prediction from this branch until a source-diverse batch shows a materially positive
cross-source reconstruction gap, the representation remains spread/high-rank under that measurement,
and the capacity probe explains the attainable reconstruction floor. A subsequent full-prediction
run must still pass `coarse_vs_copy_ratio <= 0.70` and
`coarse_vs_batch_mean_ratio <= 0.50`.
