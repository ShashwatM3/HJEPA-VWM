# Metric readout — fixed residual coordinates

## Evidence boundary

This file reports facts from the unsampled W&B Public API history for
[`r0s6ouwd`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/r0s6ouwd). The history contains
250 training rows from step 0 through step 12,450 and 25 diagnostic rows from step 0 through step
12,000. Training metrics were logged every 50 steps. Fixed-batch diagnostics were logged every 500
steps.

The human stopped the job after approximately four hours. W&B finalized the run as `crashed`. Its
last heartbeat was `2026-08-02T09:56:43Z`, its runtime was 14,647 seconds, and its last row was
step 12,450. The registered 15,000-step run therefore did not finish cleanly. All scientific
statements below are explicitly limited to the observed trajectory.

The W&B project contained 95 runs at the terminal reconciliation: 49 finished, 34 crashed, 11
killed, 1 failed, and 0 running.

## Run identity and effective training contract

| Field | Observed value |
|---|---|
| W&B ID | `r0s6ouwd` |
| Launch commit | `54a207cf9193404401c2c36c3eaf8be09039167c` |
| GPU | one NVIDIA A100-SXM4-80GB |
| Source W&B run | `60yaqw6d` |
| Source checkpoint SHA-256 | `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1` |
| Dataset | full EGO4D, fingerprint `df36af5d...6b1c` |
| Encoder | pinned DINOv3 ViT-B/16, revision `5931719e...96bc` |
| Abstract shape | `N_c=64`, `D_c=512`, internal width `M=512` |
| Temporal task | residual prediction, horizon 12, frame stride 2 |
| Trainable module | `F_c` only |
| Frozen modules | `B`, `B_EMA`, and `D` |
| Optimized objective | `L_flow` only |
| Batch and schedule | batch 64, 15,000 registered steps, 1,500-step warmup, cosine decay |
| Peak `F_c` learning rate | `1e-4` |
| Flow architecture | 6 blocks, 8 heads, condition dropout 0.10 |

The source checkpoint, fresh coarse-flow component hash, data, seed, and step-0 metrics match the
Investigation-020 residual comparator `3y2hxj5t`. The one scientific change is
`optimization_scope: joint -> fc_only`. The combined trainable-state hashes differ because one run
includes `B/F_c/D` and the other includes only `F_c`; the fresh `F_c` component hash itself is the
same: `56a81d8f...96b9`.

The recorded frozen hashes are:

| Module | State hash |
|---|---|
| `B` | `f74f2b4ac8ac181323608130a68242ea80fea39873edc141b5db4497e700b55d` |
| `B_EMA` wrapper | `cd0566f87ed2b2e64b4962e885f481655b7df6d122bcf008055155ea27ac4782` |
| `D` | `81ef4e5c5ae3511b8a12b50eac0c01b3eec16706c13d5346615908a39df1059a` |

## What one optimization step computed

The target was the detached residual

```text
Delta = B_EMA(e_{t+12}) - B_EMA(e_t).
```

The code scaled Gaussian noise by the residual standard deviation, sampled a flow time `tau`, and
formed

```text
z = tau * Delta + (1 - tau) * eps
u_target = Delta - eps
u_hat = F_c(z, tau, c_t)
L_flow = mean((u_hat - u_target)^2).
```

Only `F_c` received gradients. The configured variance, covariance, SIGReg, slot, and present
reconstruction quantities were still computed for logging where applicable, but they were excluded
from the optimized total. Consequently, every logged `loss` equals `L_flow` exactly.

The fixed diagnostic batch used a separate deterministic RNG stream. It disabled condition dropout.
It compared the model against two fixed residual guesses with the same flow-loss ruler:

```text
copy guess:       Delta_hat = 0
batch-mean guess: Delta_hat = mean_batch(Delta)
```

## Complete 500-step diagnostic trajectory

`L_flow` and `grad_norm` in this table come from the random training batch at that same logged step.
The other four loss columns come from the fixed diagnostic batch. The copy loss was exactly
`0.962544` and the batch-mean loss was exactly `0.909427` at every diagnostic row, so they are not
repeated in the table.

| Step | LR mult. | Train `L_flow` | `grad_norm` | Fixed model loss | Model/copy | Model/batch | `L_recon_chat` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0007 | 2.202297 | 0.097068 | 3.021331 | 3.138901 | 3.322235 | 0.231562 |
| 500 | 0.3340 | 1.582466 | 0.285357 | 2.594171 | 2.695118 | 2.852532 | 0.233512 |
| 1,000 | 0.6673 | 1.575262 | 0.201044 | 2.385375 | 2.478198 | 2.622942 | 0.220405 |
| 1,500 | 1.0000 | 0.989078 | 1.290600 | 2.620340 | 2.722306 | 2.881308 | 0.219739 |
| 2,000 | 0.9966 | 1.013804 | 1.125777 | 1.743945 | 1.811807 | 1.917629 | 0.188236 |
| 2,500 | 0.9865 | 0.899286 | 0.642388 | 1.805520 | 1.875778 | 1.985337 | 0.188547 |
| 3,000 | 0.9698 | 0.820377 | 0.589162 | 1.521260 | 1.580457 | 1.672766 | 0.178779 |
| 3,500 | 0.9468 | 1.006928 | 0.657552 | 1.553632 | 1.614089 | 1.708363 | 0.179134 |
| 4,000 | 0.9177 | 0.742400 | 0.465128 | 1.759804 | 1.828284 | 1.935069 | 0.188293 |
| 4,500 | 0.8830 | 0.894857 | 0.476436 | 1.440446 | 1.496498 | 1.583904 | 0.175665 |
| 5,000 | 0.8431 | 0.844702 | 0.443694 | 1.437695 | 1.493641 | 1.580880 | 0.175026 |
| 5,500 | 0.7986 | 0.720549 | 0.369731 | 1.473312 | 1.530643 | 1.620044 | 0.175600 |
| 6,000 | 0.7500 | 0.712469 | 0.504217 | 1.435300 | 1.491152 | 1.578246 | 0.174475 |
| 6,500 | 0.6980 | 0.612212 | 0.398657 | 1.466960 | 1.524044 | 1.613059 | 0.175874 |
| 7,000 | 0.6434 | 0.805759 | 0.349528 | 1.584425 | 1.646080 | 1.742223 | 0.178120 |
| 7,500 | 0.5868 | 0.755386 | 0.390292 | 1.506538 | 1.565162 | 1.656579 | 0.175422 |
| 8,000 | 0.5291 | 0.834014 | 0.415103 | 1.407926 | 1.462713 | 1.548146 | 0.175096 |
| 8,500 | 0.4709 | 0.810201 | 0.347409 | 1.460649 | 1.517488 | 1.606120 | 0.174686 |
| 9,000 | 0.4132 | 0.632208 | 0.350785 | 1.417809 | 1.472980 | 1.559013 | 0.174467 |
| 9,500 | 0.3566 | 0.594680 | 0.411116 | 1.487539 | 1.545424 | 1.635688 | 0.175707 |
| 10,000 | 0.3020 | 0.625220 | 0.319130 | 1.448210 | 1.504565 | 1.592442 | 0.174007 |
| 10,500 | 0.2500 | 0.733749 | 0.269062 | 1.401737 | 1.456283 | 1.541340 | 0.174165 |
| 11,000 | 0.2014 | 0.880246 | 0.338039 | 1.405799 | 1.460503 | 1.545807 | 0.173802 |
| 11,500 | 0.1569 | 0.726398 | 0.276137 | 1.417225 | 1.472374 | 1.558371 | 0.173505 |
| 12,000 | 0.1170 | 0.623544 | 0.249675 | 1.455163 | 1.511788 | 1.600088 | 0.174155 |

The final training-only row was step 12,450. It had `loss=L_flow=0.682081`,
`grad_norm=0.297672`, `grad_skipped=0`, and `lr_mult=0.085481`.

## Q1 — Stability and optimization health

The run was numerically healthy through its human stop.

| Metric | Observed trajectory | Reading |
|---|---|---|
| `loss` | 2.202297 at step 0; minimum 0.475818 at 10,400; 0.682081 at 12,450 | Exactly equal to `L_flow`; the objective optimized as intended. |
| `L_flow` | Same values as `loss` | Fell materially, but this alone does not establish forecasting. |
| `grad_norm` | median 0.390253; maximum 5.187578 at step 850; 0.297672 at cutoff | Finite and far below the skip threshold 150. |
| `grad_skipped` | 0 on all 250 rows | No optimizer update was skipped. |
| `instability_warn` | 0 on all 250 rows | No high-loss/high-gradient warning fired. |
| `grad_has_nan` | 0 on all 25 diagnostic rows | No NaN gradient was observed. |
| `grad_global_norm_postclip` | 0.097068 to 0.500000; median 0.390292 | The global 0.5 clip behaved normally. |
| `grad_param_count` | exactly 70 on all diagnostic rows | Only the intended `F_c` parameter tensors received gradients. |
| `lr_mult` | 0.000667 at step 0; 1.0 at 1,500; 0.085481 at 12,450 | Linear warmup and cosine decay followed the registered schedule. |

By step 12,450, the run had consumed 99.02% of the discrete cumulative learning-rate multiplier
mass of the full 15,000-step schedule. The missing 2,549 updates represented only 0.98% of that
schedule mass.

### AGC metrics

`agc_active` was 1 on every training row. `agc_B_*` and `agc_D_*` were exactly zero on every row,
which is the expected signature of frozen `B` and `D`.

`agc_Fc_any_clipped` was 1 on 10 of 250 logged rows. Those rows were confined to steps 650-2,600.
The maximum number of clipped `F_c` tensors was 4, and the maximum logged ratio was 5.990497 at
step 1,900. No `F_c` clipping event occurred after step 2,600. The late plateau is therefore not an
AGC-clipping artifact.

## Q2 — Representation spread and video specificity

Every value below was bitwise constant across all 25 diagnostic rows.

| Metric | Constant value | Reading |
|---|---:|---|
| `c_std_mean` | 1.103687 | Healthy spread near the variance target. |
| `c_std_median` | 1.099516 | The typical feature dimension is also healthy. |
| `c_dead_dim_frac` | 0.000000 | No dead-dimension population was detected. |
| `e_cross_video_cosine` | 0.402768 | Frozen DINOv3 input reference. |
| `c_cross_video_cosine` | 0.109722 | Strongly video-specific abstract codes. |

The bottleneck reduced the input cross-video cosine by 72.76%. This is a paired comparison on the
same 16-source diagnostic population. The representation is not collapsed.

## Q3 — Feature and slot utilization

Every value below was also bitwise constant across all 25 diagnostic rows.

| Metric | Constant value | Reading |
|---|---:|---|
| `c_effective_rank` | 364.280579 | 71.15% of the 512-dimensional feature space; well above the project gate. |
| `c_plus_effective_rank` | 353.866913 | Healthy fixed target rank; 2.86% below online rank. |
| `c_plus_std_mean` | 1.097573 | Healthy target spread. |
| `c_plus_std_median` | 1.094130 | Healthy typical target dimension. |
| `c_slot_diversity_rank` | 54.454066 | Most of the 64 slot directions are used. |
| `c_slot_diversity_rank_centered` | 58.791698 | Centered slots are not redundant. |
| `c_attn_entropy` | 0.418369 | Attention is selective rather than nearly uniform. |
| `c_attn_entropy_min` | 0.116274 | At least one head is highly selective. |

The representation is high-rank in both feature and slot axes. The predictor failure cannot be
explained by rank-13 collapse or uniform-slot collapse.

## Q4 — Temporal signal and fixed-coordinate proof

`coarse_copy_loss` was exactly `0.9625443220` on every diagnostic row.
`coarse_batch_mean_loss` was exactly `0.9094273448` on every diagnostic row.

The exact stationarity of these losses matters. Copy loss is the mean squared residual magnitude.
It can change under joint bottleneck training even though it is not optimized directly. It cannot
change here because `B` and `B_EMA` are fixed. Its bitwise-constant trajectory therefore shows
that `F_c` saw the same residual coordinate system and the same fixed validation distribution for
the whole observed run.

The residual is not numerically zero. However, its batch mean is informative because the batch-mean
loss is 5.52% lower than the zero-residual copy loss. A useful conditional model must first beat
that unconditional mean before it can claim video-specific forecasting.

## Q5 — Copy gate

The fixed-batch model loss fell from `3.021331` at initialization to a best value of `1.401737` at
step 10,500. This is a 53.61% reduction. The best copy ratio was nevertheless `1.456283`.

The last-six available diagnostic medians, over steps 9,500-12,000, were:

| Metric | Median | Gate |
|---|---:|---:|
| `coarse_model_loss` | 1.432718 | lower is better |
| `coarse_copy_loss` | 0.962544 | fixed baseline |
| `coarse_vs_copy_ratio` | 1.488469 | <= 0.70 |

At the late median, `F_c` was 48.85% worse than zero residual. No diagnostic row beat copy. The
formal copy gate failed at every point.

## Q6 — Batch-mean gate and conditioning

The best batch-mean ratio was `1.541340` at step 10,500. The last-six available median was
`1.575406`, against a gate of `0.50`.

At the late median, `F_c` was 57.54% worse than predicting the same mean residual for every video.
No diagnostic row beat the batch mean. The current metrics therefore do not establish useful
video-specific conditioning, even though the input representation itself is video-specific.

## Q7 — Reconstruction readouts

`D` was fixed, so all correct-code and shuffled-code readouts were exactly stationary:

| Metric | Constant value | Reading |
|---|---:|---|
| `L_recon_present` | 0.124881 | Fixed present-code reconstruction. |
| `L_recon_cplus` | 0.118823 | Fixed true-future-code reconstruction. |
| `L_recon_shuffled_c` | 0.538133 | Another video's code reconstructs very poorly. |
| `L_recon_video_gap` | 0.413252 | The decoder is strongly code- and video-dependent. |

The decoder is not in the historical template-blind regime. Shuffling the code increases loss by
0.413252, so changes in `L_recon_chat` are meaningful readouts of the predicted code.

`L_recon_chat` was the only dynamic reconstruction metric. It fell from `0.231562` to a best value
of `0.173505` at step 11,500. Its last-six median was `0.174081`, a 24.82% improvement from
initialization. The gap to the true-future decode narrowed by 50.99%, but the late predicted-code
loss was still 46.50% above `L_recon_cplus`.

This metric shows that `F_c` learned a nontrivial movement toward decoder-valid future codes. It
does not override the failed copy and batch-mean gates. The run does not log a future-target decode
from the zero-residual or batch-mean code, so reconstruction cannot say whether this movement beats
those baselines in feature space.

## Auxiliary logged losses and mode flags

These values are relevant mainly because they prove which terms were and were not optimized.

| Metric | Observed values | Interpretation |
|---|---|---|
| `L_var` | 0.000329-0.003686; cutoff 0.000787 | Readout on changing train batches; excluded from `loss`. |
| `L_cov` | 0.229413-0.517222; cutoff 0.300427 | Readout on changing train batches; excluded from `loss`. |
| `L_slot` | 0.002545-0.003063; cutoff 0.002748 | Readout only; configured coefficient is zero. |
| `L_sigreg` | 0.006459-0.007931; cutoff 0.007768 | Readout only; configured coefficient is zero. |
| `sigreg_scale` | exactly 0 | SIGReg was inactive. |
| `L_recon` | exactly 0 | Present reconstruction did not train frozen `B/D`. |
| `L_recon_pred` | exactly 0 | Prediction-side reconstruction was inactive. |
| `recon_scale` | exactly 0 | No reconstruction objective entered `loss`. |
| `recon_target_residual` | exactly 0 | Reconstruction targets were absolute features in diagnostics. |
| `recon_mean_norm` | exactly 0 | No feature-mean tracker was active. |
| `prediction_active` | exactly 1 | This was a full-prediction run. |
| `present_recon_only` | exactly 0 | The future branch was active. |
| `whiten_active` | exactly 0 | Raw DINOv3 features were used. |
| `ema_m` | 0.996000-0.996134 | The schedule value was logged, but `ema_updates=false`; it never changed `B_EMA`. |

The small variation in `L_var`, `L_cov`, `L_slot`, and `L_sigreg` comes from different random
training batches. It is not representation drift. Fixed-batch representation diagnostics are the
correct stationarity check, and all of them were exactly constant.

## Q8 — Formal and scientific verdicts

The preregistered run required 15,000 steps. It stopped at 12,450 and W&B did not close cleanly.
The formal registered-run verdict is therefore **Invalid/incomplete**.

The observed scientific trajectory has a separate Reading-Cycle-A classification:
**Healthy rep, no predictor through step 12,450**. The representation and optimizer passed their
checks. `F_c` improved substantially from initialization. It nevertheless remained worse than
both fixed trivial baselines on every diagnostic row.

This distinction prevents two mistakes. The early stop cannot be presented as a completed
15,000-step result. The early stop also does not erase the strong negative trajectory, because the
model had plateaued for thousands of steps and had already consumed 99.02% of the schedule's
cumulative learning-rate mass.
