# Observations — investigation_021

## Registered prior (2026-08-02)

1. Investigation-020 residual run `3y2hxj5t` established a healthy, dynamic representation but a
   failed predictor: late copy and batch-mean ratios were `1.726204` and `1.817973`.
2. In that run, `L_flow` updated both online `B` and `F_c`, while the residual target came from
   detached `B_EMA`. Healthy endpoint geometry does not prove that `F_c` learned in a stationary
   input/target coordinate system during the trajectory.
3. The fixed-coordinate run must begin from the same Investigation-019 checkpoint as the prior
   residual arm. Freezing Investigation-020's final `B` would change the starting representation
   and weaken the causal comparison.
4. Under `fc_only`, `B`, `B_EMA`, and `D` are immutable, `B_EMA=B`, EMA updates are disabled, and
   `L_flow` is the sole optimized objective. The configured variance, covariance, and present
   reconstruction coefficients remain part of the recorded recipe but have no training gradient
   because their modules are frozen and those terms are excluded from the optimized total.
5. With a fixed validation batch and fixed representation, `coarse_copy_loss`, rank, spread,
   cross-video cosine, and true-code reconstruction readouts should be stationary. Learning should
   appear only in `F_c`-dependent quantities such as model loss, baseline ratios, and predicted
   future reconstruction.

No run has launched and no W&B evidence exists yet. Append launch identity and measured results
only after the guide's proof gates pass.
