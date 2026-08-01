# Analysis — full-latent prediction (`8r6akjsx`)

## Verdict

**Static-`c` trap.** The run is operationally valid and never collapses to a dead or low-rank code
in the ordinary sense. The failure is temporal: as joint training proceeds, present and future
latents become much easier to copy, pretrained rank and source separation deteriorate, and Fc
becomes more than three times worse than copying the present code forward.

The factual Q1–Q8 record and all final-window values are in
[`METRIC_READOUT.md`](METRIC_READOUT.md).

## What happened, in temporal order

1. **The run inherited the same strong state as the residual arm.** At step 0, online/EMA ranks
   were `364.276/353.867` out of 512, latent cross-video cosine was `0.109707` against encoder
   `0.402768`, and copy loss was `0.962386`.
2. **During and immediately after warmup, all three temporal-health indicators moved in the wrong
   direction.** At step 2,000, copy loss had fallen to `0.883674`, rank to `344.661`, and latent
   cosine had risen to `0.155461`. The copy ratio was already failing at `1.706466`.
3. **The same drift continued through the full schedule.** By step 7,500, copy loss was `0.498067`,
   rank `325.356`, and cosine `0.298463`. By step 14,500 the corresponding values were `0.347529`,
   `293.212`, and `0.327723`.
4. **Fc did not exploit the easier target.** As copy loss shrank, the copy ratio worsened from
   `1.706466` at step 2,000 to `3.241400` at step 14,500. The final-six median is `3.132558`.
5. **The model approached, but did not meaningfully beat, a generic batch answer.** Its final-six
   batch-mean ratio is `0.946963`, with a best point of `0.936746`. That is only about a 5.3%
   improvement over batch mean and remains far from the required 50% improvement.

## Hypothesis update

The full-latent hypothesis was that retaining static and dynamic components in one target might
help a pretrained bottleneck because Fc could transport the whole future code. The preregistered
falsifier was falling copy loss with a ratio near or above one.

The falsifier occurred more strongly than expected. Copy loss fell by about 62% from step 0 to the
late median, while the late copy ratio rose above three. This is not a case where Fc almost learned
the full latent but narrowly lost a strict gate. Joint training made “future equals present” much
cheaper and the learned predictor still performed far worse than that cheaper baseline.

The representation regularizers did exactly what they are designed to do at the coarse level:
spread stayed near one and no dimension died. They did not protect temporal information. A code can
retain hundreds of cross-video feature directions while present and future from the same video
become artificially close. Variance and covariance are therefore necessary collapse guards but
not temporal anti-shortcut objectives.

## Why this is a static-code trap rather than ordinary collapse

The late online rank `296.622` and target rank `293.805` are still well above the historical
shape-aware floor. Latent cosine `0.325100` remains below the encoder's `0.402768`, and dead fraction
is zero. Labeling the run simply “collapsed” would lose the decisive signal.

The trajectory relative to the common warm start is the key:

- online rank falls 18.6%, from fraction `0.7115` to `0.5793`;
- target rank falls 17.0%, from fraction `0.6911` to `0.5738`;
- latent cross-video cosine nearly triples, from `0.109707` to `0.325100`;
- copy loss falls from `0.962386` to `0.362214`.

Online and EMA ranks remain close, so this is not an illusion caused by a lagging target branch.
Both branches move toward a less diverse, more temporally invariant geometry.

## Objective-level mechanism

The full-latent flow target is detached B_EMA future code, but online B is trained through Fc's
undetached present condition. B is simultaneously protected by present reconstruction, variance,
and covariance. None of those common terms requires the directions preserved in `c_t` to encode
change across the 12-frame horizon.

That creates an easy joint-system route: retain enough static visual information to reconstruct
the present and satisfy cross-video geometry, while reducing within-video present/future distance.
EMA then tracks the reshaped online B. The observed falling copy loss, rank contraction, and rising
cross-video cosine are the exact empirical signature of that route. Fc's approximately batch-mean
prediction does not have to learn the remaining sample-specific transition, so the model/copy
ratio gets worse as copy gets cheaper.

This mechanism is an inference from the controlled trajectory, not a proof about individual
parameters. The paired residual arm strengthens it because the only scientific config difference
is target parameterization and that arm shows the opposite copy-loss/rank trajectory.

## Reconstruction is informative but not protective

The decoder remains code-dependent: rolling latents across source videos raises the late present
loss from `0.129828` to `0.535873`, a gap of `0.406016`. It also identifies the predicted future as
worse than the true future: `L_recon_chat=0.187286` versus `L_recon_cplus=0.122533`.

Yet present and true-future reconstruction both deteriorate slightly from step 0. The active
present anchor is not strong enough to preserve the exact pretrained geometry against the
full-latent flow gradient, and it has no direct gradient through Fc because
`lambda_recon_pred=0`. This confirms that “decoder remains honest” and “forecaster learns” are
separate claims.

## Paired meaning

The full-latent arm is not selected. Neither arm passes the two registered prediction gates, so the
investigation has no winner. Relative to residual, full-latent loses substantially more rank,
retains less source separation, makes temporal copy much cheaper, and has a much worse copy ratio.
Its slightly sub-one batch-mean ratio does not outweigh those failures because Q6 only counts when
Q5 also passes.

The result does not prove that a full-latent target is universally wrong. It shows that **this**
jointly trainable bottleneck, EMA target, one-step conditional rectified-flow objective, and
15,000-step schedule exploit a static-code shortcut when asked to predict the full latent.

## Anomaly → mechanism → falsifiable probe

**Anomaly.** A high-rank, pretrained representation becomes progressively easier to copy, while
the supposedly easier full-latent prediction becomes progressively worse relative to copy.

**Most plausible mechanism.** Because B remains trainable, the joint objective can reduce
within-video temporal distance without violating present reconstruction or global covariance/
variance constraints. Fc then learns mostly generic target structure, reflected in a batch ratio
near one, while EMA carries the static drift into the target.

**Single-variable probe.** Do not spend another run trying to tune this full-latent arm in place.
Use the paired residual target and freeze B/B_EMA from the common Investigation-019 source while
training Fc alone. If a fixed target makes the residual copy ratio approach or pass `0.70`, moving
representation coordinates were the shared obstacle. If it remains above one, the current Fc or
flow objective cannot learn these fixed dynamics. This one-variable test is more informative than
adding a regularizer to a target that already demonstrated a temporal-erasure shortcut.

## Evidence limits

- This is one seed and one fixed source-diverse validation batch; it supports a controlled pair,
  not a population confidence interval.
- W&B logs the last training row at 14,950 and last diagnostic at 14,500; completion to 15,000 is
  bound by the finished state and hashed final checkpoint.
- Absolute `L_flow`, `coarse_model_loss`, and `coarse_copy_loss` cannot be ranked across target
  modes. Only the baseline ratios and qualitative copy-loss trajectory are cross-mode evidence.
- The run tests the implemented coarse Phase-1 path only. It says nothing empirical about the
  unimplemented FineFlow, pixel generator, or multi-horizon stages.
