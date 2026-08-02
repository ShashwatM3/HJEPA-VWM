# Analysis — residual prediction (`3y2hxj5t`)

## Verdict

**Healthy rep, no predictor.** The run is operationally valid, retains a rich and strongly
video-specific DINO bottleneck, and makes present/future codes more—not less—separated. It still
fails both Phase-1 prediction gates at every diagnostic checkpoint. Residual parameterization
therefore avoided the static-code shortcut but did not make the current Fc learn the dynamics.

The factual Q1–Q8 record and all final-window values are in
[`METRIC_READOUT.md`](METRIC_READOUT.md).

## What happened, in temporal order

1. **The pair began from a genuinely strong common state.** At step 0, online/EMA ranks were
   `364.276/353.867` out of 512, latent cross-video cosine was `0.109707` against an encoder value
   of `0.402768`, and no dimensions were dead. This was not another run trying to learn prediction
   while first repairing a collapsed bottleneck.
2. **Warmup temporarily perturbed, but did not destroy, the representation.** Online rank reached
   its low of `335.644` at step 2,000 while spread stayed near one. It recovered above the initial
   level by step 5,000 and continued upward.
3. **The residual target remained meaningfully dynamic.** Copy loss—the cost of predicting zero
   residual—rose by about 35% from `0.962544` at step 0 to the late median `1.300293`. The late
   online and target rank fractions were `0.7362` and `0.7109`; their small gap does not show an EMA
   branch stranded in an old low-rank coordinate system.
4. **Fc improved from its random start but stopped far above both baselines.** The copy ratio fell
   from `3.138901` to a best value `1.574568` at step 6,500, then settled at a late median
   `1.726204`. The batch-mean ratio followed the same shape and settled at `1.817973`. The late
   model is 72.6% worse than zero residual and 81.8% worse than the batch mean.
5. **The learning-rate tail did not rescue the forecaster.** From step 12,000 through 14,500 every
   copy ratio lies in `1.684899–1.795698` and every batch ratio in `1.769730–1.887769`; these are
   stable failure plateaus, not one noisy terminal sample.

## Hypothesis update

The preregistered hypothesis had two parts:

- residual prediction should prevent Fc from wasting capacity on the large static component;
- that cleaner target should improve actual forecasting relative to the full-latent arm.

The first part is supported. Unlike the matched full-latent arm, this bottleneck does not erase
temporal separation: copy loss rises, rank rises, source separation remains strong, and true-future
reconstruction improves. The second and decisive part is falsified by the acceptance gates. The
residual ratio is better than the full-latent ratio, but `1.726204` is still a clear loss to the
zero-residual baseline, not a near pass.

This makes the run a stronger version of Investigation 009/010's canonical negative result: even
with a pretrained `64×512` DINO code at roughly 71% initial rank utilization, the current Fc and
training objective do not convert temporal separation into forecast skill.

## Representation versus prediction

The run cleanly separates two questions that earlier investigations often entangled.

**Representation passes.** Late spread is `1.118518`, dead fraction is zero, cross-video cosine is
`0.179245`, and online rank is `376.932`. The source-diverse latent is substantially less aligned
than its DINO input (`0.402768`). The EMA target tracks the online geometry closely enough to rule
out a gross EMA-rank lag.

**Prediction fails.** In residual mode the copy baseline is exactly zero temporal change:
`copy_velocity=-eps`, so its loss is the squared residual. A ratio above one means Fc is not merely
failing to explain all of the change; it is worse than predicting no change at all. The same model
also loses to the batch-mean residual, so it does not extract useful per-video future change from
the condition.

The important causal boundary is that `L_flow` trains both Fc and online B through the undetached
condition, while the target residual comes from detached B_EMA present/future codes. The healthy
geometry proves the joint system did not collapse. It does not prove Fc ever saw a stationary
regression problem.

## Reconstruction is informative here

This run does not have the old reconstruction-blindness signature. The late true-future decoder
loss is `0.100083`, whereas the predicted-future endpoint gives `0.181156`; the gap is `0.081073`.
Rolling codes across source videos raises the present loss from `0.104876` to `0.547940`. Thus:

- D depends strongly on the correct video's code;
- D recognizes that the predicted future endpoint is worse than the true future code;
- improving present reconstruction alone did not improve Fc, because that loss has no gradient
  through Fc (`lambda_recon_pred=0`).

This is useful diagnostic corroboration, not a reason to silently revive prediction-side
reconstruction. That branch was negative in earlier low-capacity regimes and would be a separate
scientific axis in the present high-capacity regime.

## Paired meaning

The registered decision rule says **no winner** when neither arm passes both forecast gates. That
rule applies: residual is not a winning predictor. It is nevertheless the better mechanistic
substrate for a follow-up because it preserves the phenomenon the forecaster is meant to model.
The full-latent arm reduces the copy target itself; this arm leaves a large, healthy temporal
signal and therefore localizes the remaining failure more directly to forecaster learning or to
the moving joint representation/forecast objective.

## Anomaly → mechanism → falsifiable probe

**Anomaly.** Fc loses to zero residual even though the target remains dynamic, high-rank,
source-specific, and reconstructively honest.

**Most plausible mechanism.** Joint `L_flow -> B,Fc` training makes Fc chase a representation whose
temporal coordinates are still being reshaped, while covariance, variance, and present
reconstruction protect geometry but do not make those coordinates easy to forecast. A second
possibility is stricter: the current one-step conditional rectified-flow Fc cannot learn even a
fixed version of these DINO-bottleneck dynamics. This run cannot distinguish the two.

**Single-variable probe.** Repeat the residual recipe from the same Investigation-019 checkpoint
with B frozen for the entire 15,000-step temporal phase (and therefore fixed B_EMA), training only
Fc while retaining the same data, seed, target/noise definition, schedule, and diagnostics. The
prediction is falsifiable:

- if the late copy ratio falls materially toward or below `0.70`, moving joint coordinates were
  the obstruction;
- if it remains above one while copy loss stays fixed, the failure is in Fc/objective learnability
  on this representation, and more geometry work is not the next lever.

Do not combine this probe with SIGReg, a V-JEPA encoder swap, a new horizon, prediction-side
reconstruction, or extra Fc capacity. Any of those would destroy the causal read.

## Evidence limits

- This is one seed and one fixed source-diverse validation batch; it supports a controlled pair,
  not a population confidence interval.
- W&B logs the last training row at 14,950 and last diagnostic at 14,500; completion to 15,000 is
  bound by the finished state and hashed final checkpoint.
- Absolute `L_flow`, `coarse_model_loss`, and `coarse_copy_loss` cannot be ranked across target
  modes. Only the baseline ratios and qualitative copy-loss trajectory are cross-mode evidence.
- The run tests the implemented coarse Phase-1 path only. It says nothing empirical about the
  unimplemented FineFlow, pixel generator, or multi-horizon stages.
