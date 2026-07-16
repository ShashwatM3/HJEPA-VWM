# Next Steps — run 057 `ae_latent_stack_whiten_abs_recon_cov_var`

## Closure

1. ~~Determine whether SIGReg is needed on top of covariance plus the variance floor.~~ Complete.
   It is not: removing SIGReg preserved rank and slot diversity while improving reconstruction,
   the video gap, attention breadth, and gradient behavior.
2. Carry run 057—not run 056—as the settled absolute-target SSv2 control for the geometry bundle:
   latent-stack bottleneck, fixed offline whitening, `lambda_cov=0.01`, `lambda_var=0.5`, and
   `lambda_sigreg=0`. Do not infer that the residual-target-plus-geometry combination has been
   tested; it has not.

## Successor chain

1. ~~Test the recipe on EGO4D.~~ Executed as investigation 016 run 058 (`mvbx96nv`). The observed
   geometry and conditioning were much weaker, but the later source audit showed that its fixed
   validation batch contained adjacent chunks from one source UID. Treat it as an historical
   transfer result, not a clean global-collapse or causal dataset verdict.
2. Run 060 (`2423b84g`) tested the absolute-target reconstruction-weight extreme on the newer
   runtime. Raising `lambda_recon` to `1.0` moved present reconstruction by only `0.0073`; simple
   scalar underweighting is not the leading explanation.
3. Repair source-aware EGO4D validation and run cached-feature/template oracles before selecting
   residual-target, capacity, whitening-strength, or optimizer arms. See
   [`../../investigation_016/reconstruction_floor_architecture_audit/NEXT_STEPS.md`](../../investigation_016/reconstruction_floor_architecture_audit/NEXT_STEPS.md).
4. Activate full prediction only after present reconstruction is demonstrably source-conditioned.
   A prediction run must then pass both copy (`<=0.70`) and batch-mean (`<=0.50`) gates; high
   present-side rank alone is insufficient.

The optional bottleneck-only control from investigation 015 remains the test for causal
whitening-versus-latent-stack attribution. It is not required to choose the current operational
recipe.
