# Investigation 011 Run C - final analysis: `inv011_fixed_position_decoder`

**Status:** FINISHED
**W&B:** `inv011_fixed_position_decoder` (`io74f32b`)
**Group:** `inv011_fixed_position_decoder`
**Created:** 2026-06-30 18:48 UTC
**Pull / read date:** 2026-07-01
**Primary source:** W&B run history and downloaded `output.log`
**Last logged training step:** 14950
**Last diagnostic step:** 14500

Diagnostic metrics (`c_*`, `coarse_*`, `L_recon_*`) log every 500 steps. Training metrics
(`L_flow`, `grad_*`, optimizer health) log every 50 steps. This analysis reads the run by the
project protocol in `KANBAN/README_for_reading_experiments.md`.

## 0. What this run tested

This is investigation 011 Run C. It keeps the same full residual/cosine recipe as Run A
(`new_recon_loss`, `1u69hpfm`) and changes the decoder implementation only:

```text
Run A: learned output queries in D
Run C: fixed tubelet position queries in D
```

The intended decoder rule was:

```text
position tells D where to write;
c tells D what to write.
```

The hypothesis was that removing learned per-output-token query parameters would make
reconstruction depend more honestly on `c`, widen the `L_recon_chat - L_recon_cplus` gap when
prediction is poor, and possibly improve `coarse_vs_copy_ratio`.

## 1. Config highlights

| Field | Value |
|---|---:|
| dataset | `ssv2` |
| max steps | 15000 |
| `horizon_k` | 12 |
| `predict_residual` | true |
| `present_recon_only` | false |
| `lambda_var` | 0.5 |
| `lambda_sigreg` | 5.0 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0.05 |
| `recon_loss_mode` | `cosine` |
| `sigreg_warmup_steps` | 2000 |
| `recon_warmup_steps` | 2000 |
| `lr_bottleneck` | 1e-4 |
| `lr_coarse_flow` | 1e-4 |
| `n_c` | 32 |
| `d_c` | 256 |
| decoder | 512 dim, 4 blocks |
| AGC / global clip | enabled, `grad_clip=0.5` |

This run is directly comparable to Run A `new_recon_loss` because the measured config is the same
on all exposed knobs. The intended difference is the code-level decoder architecture.

## 2. Protocol verdict table

| Q | Question | Result | Evidence |
|---|---|---:|---|
| Q1 | Training alive? | PASS | Finished. `grad_skipped=0` for all logged rows, `grad_has_nan=0`, `instability_warn=0`; `grad_norm` stayed finite, max `3.10`, last diag `2.69`. |
| Q2 | `c_t` alive / video-specific? | PASS | Last diag `c_std_mean=1.005`, `c_dead_dim_frac=0`, `c_cross_video_cosine=0.165`. |
| Q3 | Rich latent, not rank-limited? | FAIL | Last diag `c_effective_rank=50.76`, `c_plus_effective_rank=50.45`. EMA is aligned, but rank is below the current `>60` gate. |
| Q4 | Temporal dynamics, not static `c`? | PARTIAL | `coarse_copy_loss` rose from `0.053` to `1.506`, so present and future latents separate. But the ratio plateau stayed near copy: last-3 `0.988`, `1.028`, `0.971`, mean `0.996`. |
| Q5 | `F_c` beats copy? | FAIL | Final diag ratio `0.971`, best post-10k `0.917`, plateau `0.996`. Gate is `<=0.70`; final model loss `1.462` vs copy loss `1.506` is only a small edge. |
| Q6 | `F_c` uses this video's `c_t` enough to beat batch mean? | FAIL | Final `coarse_vs_batch_mean_ratio=1.067`, post-10k min `1.010`, plateau `1.095`. Gate is `<=0.50`. |
| Q7 | Reconstruction honest about prediction? | FAIL | `L_recon_present` learned to `0.346`, but `L_recon_chat=0.353` vs `L_recon_cplus=0.344`; gap is only `0.0088` final, `0.0099` plateau. |
| Q8 | Verdict | FAIL | Low-rank representation with partial prediction movement; not a passing Phase 1 run. |

## 3. Final metrics

Last diagnostic step is 14500. Plateau means the mean of diagnostic steps 13500, 14000, and 14500.

| Metric | Final diag | Last-3 plateau | Interpretation |
|---|---:|---:|---|
| `coarse_vs_copy_ratio` | 0.9707 | 0.9956 | Better than Run A, but effectively copy-tie and far above `<=0.70`. |
| `coarse_vs_batch_mean_ratio` | 1.0670 | 1.0947 | Fails the batch-mean baseline badly. |
| `coarse_model_loss` | 1.4615 | 1.4980 | Only slightly below copy at the final diag point. |
| `coarse_copy_loss` | 1.5057 | 1.5046 | Copy is hard because `c_t` and `c_{t+k}` are separated. |
| `c_effective_rank` | 50.76 | 50.74 | Regressed below the `>60` gate. |
| `c_plus_effective_rank` | 50.45 | 50.40 | EMA target tracks online `c`; no large target lag. |
| `c_std_mean` | 1.0053 | 1.0051 | Healthy variance. |
| `c_cross_video_cosine` | 0.1652 | 0.1655 | Healthy video-specificity. |
| `c_dead_dim_frac` | 0.0 | 0.0 | No dead-dimension spike. |
| `L_recon_present` | 0.3464 | 0.3465 | Present reconstruction learns under cosine loss. |
| `L_recon_cplus` | 0.3441 | 0.3442 | True-future reconstruction benchmark. |
| `L_recon_chat` | 0.3530 | 0.3541 | Predicted-future reconstruction only slightly worse than true future. |
| `L_recon_chat - L_recon_cplus` | 0.0088 | 0.0099 | Still too small to be an honest prediction-quality readout. |
| `grad_skipped` | 0 | 0 | Stable. |

## 4. Trajectory

Selected diagnostic points:

| step | ratio | batch ratio | copy loss | model loss | rank | plus rank | cosine | recon present | recon cplus | recon chat |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22.9366 | 24.3867 | 0.0533 | 1.2232 | 9.47 | 9.23 | 0.724 | 0.9858 | 0.9878 | 0.9878 |
| 2500 | 1.2131 | 1.3028 | 0.3613 | 0.4383 | 13.15 | 13.14 | 0.132 | 0.3691 | 0.3738 | 0.3720 |
| 5000 | 1.1482 | 1.2668 | 1.0583 | 1.2151 | 37.26 | 36.04 | 0.157 | 0.3553 | 0.3540 | 0.3618 |
| 8000 | 0.9169 | 1.0202 | 1.4008 | 1.2844 | 46.72 | 46.03 | 0.160 | 0.3500 | 0.3475 | 0.3511 |
| 10000 | 0.9870 | 1.0969 | 1.4694 | 1.4504 | 49.11 | 48.53 | 0.164 | 0.3480 | 0.3455 | 0.3529 |
| 12500 | 0.9169 | 1.0098 | 1.5047 | 1.3797 | 50.56 | 50.14 | 0.166 | 0.3468 | 0.3443 | 0.3523 |
| 13500 | 0.9876 | 1.0863 | 1.5034 | 1.4848 | 50.70 | 50.34 | 0.166 | 0.3466 | 0.3442 | 0.3544 |
| 14000 | 1.0285 | 1.1310 | 1.5047 | 1.5476 | 50.76 | 50.39 | 0.165 | 0.3464 | 0.3441 | 0.3549 |
| 14500 | 0.9707 | 1.0670 | 1.5057 | 1.4615 | 50.76 | 50.45 | 0.165 | 0.3464 | 0.3441 | 0.3530 |

The trajectory has four phases:

1. **Warmup / ratio collapse, 0-2500.** Ratio falls from the meaningless init spike to about `1.21`.
   `c_std_mean` moves toward the variance target, and recon loss drops quickly.
2. **Rank expansion, 3000-7000.** Rank climbs from about `18` to `45`; `coarse_copy_loss` rises
   from `0.39` to `1.33`. This is real temporal separation, not static `c`.
3. **Near-copy predictor window, 8000-12500.** Ratio dips below `1.0` at 8000 and 12500, with best
   post-warmup value `0.9169`. This is the useful signal in the run.
4. **End plateau, 13000-14500.** Copy loss, rank, cosine, and recon readouts are basically flat.
   Ratio oscillates around `1.0`; it does not continue improving toward the Phase 1 `0.70` gate.

## 5. Q1 - training health

This run is valid to read. It finished the budget with no skip spiral and no NaN signal:

- `grad_skipped` summed to `0.0` over the logged rows.
- `grad_has_nan` max was `0.0`.
- `instability_warn` max was `0.0`.
- `grad_norm` was finite throughout; max in the downloaded log was `3.10`, far below the
  instability scale that killed early Phase 1 runs.

The fixed-position decoder did not destabilize optimization. The failure is not a training crash or
frozen-step artifact.

## 6. Q2 - representation alive and video-specific

`c_t` is alive by the protocol:

- `c_std_mean` finishes at `1.005`, exactly near the variance target.
- `c_dead_dim_frac` remains `0`.
- `c_cross_video_cosine` falls from `0.724` at initialization to `0.165` at the last diagnostic
  step.

The cross-video cosine result is important: the model is not producing one generic latent for every
video. The representation is video-specific. That means the Q6 failure is not because the
bottleneck has collapsed into identical `c_t` values. It is because `F_c` does not convert the
video-specific `c_t` into a future prediction that beats the batch-mean baseline.

## 7. Q3 - rank and EMA target

This is the main representation regression versus the immediately preceding full-recipe runs.

| Run | Decoder / objective | final rank | plateau rank | final plus rank | ratio plateau |
|---|---|---:|---:|---:|---:|
| inv010 `soft-universe-37` | old relative-MSE recon | 61.08 | 61.07 | 60.41 | 1.0768 |
| Run A `new_recon_loss` | learned-query D, cosine recon | 60.35 | 60.31 | 60.65 | 1.0652 |
| Run C `inv011_fixed_position_decoder` | fixed-position D, cosine recon | 50.76 | 50.74 | 50.45 | 0.9956 |

Run C improves the copy ratio relative to both comparators, but loses about `9-10` effective-rank
points. Since the current Q3 gate is `c_effective_rank > 60`, this is a protocol fail even though
the rank is far above the historical rank-13 ceiling.

The plus-rank follows online rank closely (`50.76` vs `50.45` final), so this is not an EMA lag
problem. The target branch is aligned with a lower-rank online bottleneck.

## 8. Q4 - temporal dynamics versus static `c`

This run is **not** the classic static-`c` trap.

Evidence:

- `coarse_copy_loss` rises from `0.053` at step 0 to `1.506` at step 14500.
- From step 8000 onward, copy loss stays around `1.40-1.51`.
- `c_cross_video_cosine` stays low at the same time.

Rising `coarse_copy_loss` means that the present and future abstract latents are moving apart. The
copy baseline is not becoming trivially easy. This is the good part of the run.

The bad part is that `coarse_model_loss` tracks the copy loss too closely:

```text
last diag:    model 1.4615 / copy 1.5057 = ratio 0.9707
last-3 mean:  model 1.4980 / copy 1.5046 = ratio 0.9956
```

So the pattern is: **temporal separation exists, but `F_c` only barely exploits it.**

## 9. Q5 - copy baseline gate

Run C gets the best late copy-ratio behavior in this family, but it is still not a passing predictor.

| Quantity | Value |
|---|---:|
| final `coarse_vs_copy_ratio` | 0.9707 |
| last-3 plateau | 0.9956 |
| post-10k min | 0.9169 at step 12500 |
| post-10k median | 0.9873 |
| Phase 1 gate | <= 0.70 |

The final point is slightly below copy. The best post-10k point is more meaningfully below copy.
But the run never approaches the required `0.70`, and the late window oscillates around `1.0`
rather than trending down.

This should not be read as "passing prediction." The precise read is:

```text
fixed-position D moved the residual predictor from worse-than-copy / copy-tie
to a noisy near-copy predictor with occasional shallow wins.
```

That is useful signal, not success.

## 10. Q6 - batch-mean baseline gate

Run C fails the batch-mean baseline:

| Quantity | Value |
|---|---:|
| final `coarse_vs_batch_mean_ratio` | 1.0670 |
| last-3 plateau | 1.0947 |
| post-10k min | 1.0098 at step 12500 |
| post-10k median | 1.0916 |
| Phase 1 gate | <= 0.50 |

This is a stricter diagnosis than Q5 in one specific way. A ratio near `1.0` here means that the
model is not beating the generic "use the batch-average future for every clip" guess. Because Q2
passed, this is not caused by every video having the same `c_t`. The bottleneck has video-specific
information; the predictor is not using it in a way that makes the future estimate better than this
generic baseline.

The user's note that "cross video cosine is the problem" is not supported by the metrics. Cross-video
cosine is healthy. The problem is downstream conditioning/prediction, not collapsed video identity.

## 11. Q7 - reconstruction honesty

The fixed-position decoder does learn reconstruction under cosine loss:

- `L_recon_present`: `0.9858 -> 0.3464`
- `L_recon_cplus`: `0.9878 -> 0.3441`
- `L_recon_chat`: `0.9878 -> 0.3530`

But the key diagnostic is the gap:

```text
final chat - cplus = 0.352953 - 0.344121 = 0.008832
plateau gap        = 0.354087 - 0.344152 = 0.009935
```

That is still tiny. The decoder can reconstruct from the predicted future latent almost as well as
from the true future latent, even though Q5 and Q6 say the prediction is not good enough. Therefore
reconstruction is still largely blind as a prediction-quality supervisor.

The fixed-position decoder did not deliver the hoped-for diagnostic widening of
`L_recon_chat - L_recon_cplus`.

## 12. Comparison to Run A and inv010

Run C is better on copy ratio and worse on rank.

| Metric | inv010 `soft-universe-37` | Run A `new_recon_loss` | Run C fixed-position |
|---|---:|---:|---:|
| final ratio | 1.0631 | 1.0575 | **0.9707** |
| plateau ratio | 1.0768 | 1.0652 | **0.9956** |
| best post-10k ratio | 1.0360 | 1.0179 | **0.9169** |
| final batch ratio | 1.1397 | 1.1463 | **1.0670** |
| plateau batch ratio | 1.1549 | 1.1546 | **1.0947** |
| final rank | **61.08** | 60.35 | 50.76 |
| plateau rank | **61.07** | 60.31 | 50.74 |
| final cosine | 0.1624 | **0.1469** | 0.1652 |
| final `chat-cplus` recon gap | 0.0121 | 0.0095 | 0.0088 |

Interpretation:

- The fixed-position decoder is the first recent full-recipe run to end below the copy baseline.
- The magnitude is small: plateau ratio is still essentially `1.0`.
- The batch-mean ratio remains above `1.0`.
- The run loses the rank gate that inv010 and Run A had cleared.
- Reconstruction remains blind and may be slightly less diagnostic than Run A by the final
  `chat-cplus` gap.

So Run C is not a clean win. It is an informative tradeoff: the decoder architecture shift nudged
prediction in the right direction, but it did not solve `F_c`, and it coincided with a lower-rank
abstract space.

## 13. Corrections to the handwritten notes

Most of the manual read was directionally right. The important corrections are:

1. **The copy ratio formula is `coarse_model_loss / coarse_copy_loss`.**
   The notes invert it once. In this run, final `1.4615 / 1.5057 = 0.9707`.

2. **Q3 should be marked as a protocol fail.**
   Rank around `50` is much better than the old rank-13 ceiling and the EMA is healthy, but the
   current gate is `>60`.

3. **Cross-video cosine is not the Q6 problem.**
   Cross-video cosine is healthy at `0.165`. The batch-mean failure means `F_c` does not use the
   already video-specific `c_t` well enough to produce a better future.

4. **`coarse_model_loss` is noisy, but the core issue is not just noisiness.**
   The late model loss remains too close to copy loss. The ratio never approaches the acceptance
   gate even at its best post-10k point.

5. **The Q7 read is correct.**
   `L_recon_present`, `L_recon_cplus`, and `L_recon_chat` are close enough that reconstruction is
   not an honest prediction-quality signal.

## 14. Final diagnosis

`inv011_fixed_position_decoder` is a stable full-length run with alive, video-specific latents.
It is not a static-`c` failure: `coarse_copy_loss` rises substantially, so present and future
abstract latents separate.

The fixed-position decoder produces the best recent copy-ratio movement, including a final ratio
below `1.0` and a best post-10k ratio of `0.9169`. However, the run does not pass Phase 1:
`coarse_vs_copy_ratio` stays near `1.0`, `coarse_vs_batch_mean_ratio` remains above `1.0`, and
`c_effective_rank` regresses to about `50.8`, below the `>60` gate.

The key bottleneck remains predictor skill. `F_c` does not turn video-specific `c_t` into a future
estimate that beats either copy by a large margin or the batch-mean baseline at all. Reconstruction
also remains blind: the decoder still gives nearly the same loss for true future `c_plus` and
predicted future `c_hat`.

**Verdict:** low-rank representation with partial prediction movement; not a passing prediction run.

## 15. Recommended next read / next experiment

Do not iterate on reconstruction objective alone. The fixed-position decoder is worth keeping as a
candidate because it moved the copy ratio in the right direction, but the next experiment should
target the prediction failure directly.

Most direct follow-ups:

1. Add and log direct residual diagnostics: `delta_norm`, `delta_hat_norm`, `delta_hat_std`,
   `delta_std`, `model_minus_copy`, and `implied_rho`.
2. Test an anti-zero-residual objective on `Delta_hat`, preferably with and without reconstruction,
   because this run still looks like a near-zero-residual tie.
3. Re-run a no- or reduced-reconstruction residual arm with the fixed-position decoder only if the
   goal is to see whether the rank loss came from the decoder change or the full recon pressure.
4. Keep Q5 and Q6 as the acceptance gates. Lower `L_recon` or lower `L_flow` alone is not success.
