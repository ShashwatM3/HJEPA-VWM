# Next Steps — run 056 `ae_latent_stack_whiten_abs_recon_geom`

## Closure / carry-forward status

Run 056 is complete. Its primary question is answered positively: the inv011 geometry bundle
prevents the run-055 contraction and can coexist with clearly code-conditioned whitened
reconstruction. The run should not be relaunched as-is.

Carry forward these conclusions:

1. explicit anti-collapse pressure is necessary in this architecture family;
2. covariance is the dominant feature-rank lever;
3. the variance floor is a cheap amplitude guard;
4. SIGReg is not necessary once covariance plus variance are active and imposes a small
   reconstruction/honesty tax;
5. no present-only result is evidence that `F_c` predicts the future.

## Immediate successor — completed

The next documented science run was run 057
[`ae_latent_stack_whiten_abs_recon_cov_var`](../run_057_ae_latent_stack_whiten_abs_recon_cov_var/)
([`cdvp6hou`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/cdvp6hou)). It removed SIGReg
and finished all 15,000 steps. It preserved or improved rank, slot diversity, reconstruction,
video gap, conditioned share, attention breadth, and gradient calmness. That run supersedes 056 as
the SSv2 present-only geometry recipe.

Do not restore SIGReg in this branch without a new measurement showing that its small reduction in
cross-video cosine is worth the decodability and attention-specialization cost.

## Research chain after the successor

- Run 057 was transferred to EGO4D as run 058. That transfer did not reproduce the SSv2 fixed-batch
  geometry/honesty pattern, although the later source-manifest audit showed the EGO4D “cross-video”
  and shuffled-code probes were actually within one source recording.
- The correct next work is therefore measurement repair and capacity/template oracles on EGO4D,
  not a revival of run 056's full SIGReg bundle.
- A residual-target plus covariance/variance arm remains scientifically meaningful as an honesty
  test, but it should be run only after the diagnostic batch is source-diverse so its shuffled-code
  gap has the intended interpretation.

## Reuse guardrails

- Use run 057, not run 056, when a later experiment needs the settled covariance-plus-variance
  present encoder recipe.
- Keep `lambda_slot=0`; the rejected slot-loss branch is not revived by this result.
- Do not compare absolute `L_recon_present` across raw and whitened target spaces.
- Before transferring a checkpoint into prediction, activate `F_c` in a fresh controlled run and
  require both copy and batch-mean gates; this run contains no forecast evidence.
