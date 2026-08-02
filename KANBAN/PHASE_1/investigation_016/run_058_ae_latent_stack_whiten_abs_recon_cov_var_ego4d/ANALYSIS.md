# Lambda-reconstruction audit — historical `lambda_recon=1`, the recent ten runs, and EGO4D run 058

> **Analyzed:** 2026-07-14 (live W&B query at approximately 15:22 GST / 11:22 UTC)
> **W&B project:** `smahalanobis-uc-davis/hjepa-vwm`
> **Primary current run:** `ae_latent_stack_whiten_abs_recon_cov_var_ego4d`
> ([`mvbx96nv`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/mvbx96nv))
> **Current control:** run 057, `Investigation 15 · Whitened latent stack · Covariance plus variance`
> ([`cdvp6hou`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/cdvp6hou))
> **Historical `lambda_recon=1` run:** run 029, `Investigation 07 · Reconstruction weight · 1.00`
> ([`tw685b5g`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/tw685b5g))

This document answers four connected questions:

1. Was `lambda_recon=1` used in the recent ten runs, or anywhere in the project?
2. If it was used, is that result transferable to the current architecture and run 058?
3. Is the current `0.68–0.70` reconstruction plateau evidence that `lambda_recon=0.05`
   sends gradients that are too small to matter?
4. Would changing the current recipe directly to `lambda_recon=1` likely help, do nothing,
   or disrupt representation geometry?

The analysis follows the present-only and full-prediction reading cycles in
[`GUIDES/READING_EXPERIMENTS.md`](../../../../GUIDES/READING_EXPERIMENTS.md). Values below come
from live, unsampled W&B history/config reads, not from remembered dashboard values. The local
Kanban was also read for runs 051–058 and Investigation 07. At analysis time, the local run-058
record still said “planned”; W&B is authoritative for its completed results.

## Executive conclusion

There is **no `lambda_recon=1` run among the ten newest W&B runs**. Eight of the ten use
`lambda_recon=0.05`; the other two are short data/regression smokes with reconstruction off.
Across all 60 live runs, there is exactly one explicit `lambda_recon=1` run: run 029
(`tw685b5g`). It ended at step 200 in a synchronized whole-pod failure, before the first
post-initialization diagnostic checkpoint. It proves only that the 2,000-step reconstruction
ramp made weight 1 numerically survivable through step 200; it does **not** measure the late
reconstruction floor, representation geometry, or prediction quality.

The closest usable empirical evidence is the controlled old weight ladder from 0.05 to 0.50.
At the common step 9,000, raising the weight tenfold lowered diagnostic reconstruction only from
`0.5987` to `0.5855` (about `0.0133` absolute), did not move effective rank off the rank-13
ceiling, and worsened the copy ratio from `1.51` to `1.66`. That is evidence against a simple
“the reconstruction term was too weak” explanation. It is not quantitatively transferable to
run 058 because it used the legacy relative-MSE objective, raw SSv2 features, a learned-query
decoder, the old one-read bottleneck, and full prediction rather than the current cosine,
whitened, fixed-position, latent-stack, present-only system.

The newest EGO4D run gives a stronger and more current diagnosis. Its correct-code validation
loss ends at `0.6782`, but its **wrong-video** loss is already `0.6962`. Only
`0.0180 / (1.0133 - 0.6782) = 5.4%` of the decoder's improvement depends on receiving the
correct video's `c_t`; about 94.6% is available from shared/template structure. At the same time,
validation rank falls from a peak of `80.7` to `52.9`, cross-video cosine rises to `0.863`, and
std falls to `0.419`. This is a template/generalization collapse, not a lack of any learning
signal. A 20× reconstruction weight on the same absolute target is more likely to strengthen
that shortcut and overpower the early covariance/variance shaping than to create useful
video-specific detail.

The direct recommendation is therefore **do not make `lambda_recon=1` the next isolated
change**. First rerun the EGO4D recipe with `recon_residual_target=true` at `lambda_recon=0.05`,
which the run-058 guide pre-registered as the response to a near-zero video gap. If the honest
residual-target reconstruction then plateaus, test `0.05 -> 0.20 -> 1.0` under one otherwise
identical recipe and log per-loss, per-module pre-clipping gradient norms. Raw
`L_recon_present` must remain secondary to the shuffled-code gap and representation gates.

## 1. Live inventory of the ten newest runs

The selection rule was W&B creation time descending, not folder name or stale Kanban numbering.
The live project had 60 runs. The final EGO4D science run is the newest run and was created at
2026-07-13 21:58 UTC (2026-07-14 01:58 GST); it finished at approximately 03:31 UTC / 07:31 GST
after the recent five-and-a-half-hour training window.

| Live order | Run | Created (UTC) | State / last step | Mode and substrate | `lambda_recon` | What it contributes here |
|---:|---|---|---|---|---:|---|
| 60 | [`mvbx96nv`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/mvbx96nv) `ae_latent_stack_whiten_abs_recon_cov_var_ego4d` | Jul 13 21:58 | finished / 14,950 | present-only, EGO4D, whitened cosine | **0.05** | Newest proper experiment; run 058; template/geometry transfer failure. |
| 59 | [`tt4x64hc`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/tt4x64hc) `Investigation 16 · Original-data regression smoke · 100 steps` | Jul 13 20:59 | finished / 50 | full-prediction, SSv2-tiny, raw | **0** | Infrastructure smoke; no reconstruction evidence. |
| 58 | [`29a2ora7`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/29a2ora7) `Investigation 16 · EGO4D data smoke · 500 steps` | Jul 13 20:32 | finished / 450 | full-prediction, EGO4D-tiny, raw | **0** | Data-path smoke; no reconstruction evidence. |
| 57 | [`cdvp6hou`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/cdvp6hou) `Investigation 15 · Whitened latent stack · Covariance plus variance` | Jul 6 05:42 | finished / 14,950 | present-only, SSv2, whitened cosine | **0.05** | Byte-matched science control except dataset/stats; strong present representation. |
| 56 | [`tl5dh73c`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/tl5dh73c) `Investigation 15 · Whitened latent stack · Full geometry` | Jul 5 20:28 | finished / 14,950 | present-only, SSv2, whitened cosine | **0.05** | Strong geometry; SIGReg adds an honesty/decodability tax. |
| 55 | [`nzz64pl6`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/nzz64pl6) `Investigation 15 · Whitened latent stack · Absolute reconstruction` | Jul 5 01:36 | finished / 14,950 | present-only, SSv2, whitened cosine | **0.05** | Good honesty but geometry contracts without active regularizers. |
| 54 | [`lx1b6gw2`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/lx1b6gw2) `Investigation 15 · Whitened latent stack · Residual reconstruction` | Jul 4 11:31 | finished / 14,950 | present-only, SSv2, whitened residual cosine | **0.05** | About 92% video-conditioned, but geometry still contracts. |
| 53 | [`7teohhwc`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/7teohhwc) `Investigation 13 · Sharp slots · Residual reconstruction` | Jul 3 11:17 | external crash / 12,150 | present-only, SSv2, raw residual cosine | **0.05** | Residual target fixes much of the raw-space template shortcut; low-rank. |
| 52 | [`662hfy3c`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/662hfy3c) `Investigation 12 · Sharp slots · Absolute reconstruction` | Jul 2 15:32 | finished / 14,950 | present-only, SSv2, raw cosine | **0.05** | Very low raw recon (`0.293`) but collapsed/template-like code. |
| 51 | [`mtrviiab`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/mtrviiab) `Investigation 11 · Present geometry · Isotropy 12.5, covariance 0.003` | Jul 2 03:48 | external crash / 14,250 | present-only, SSv2, raw cosine | **0.05** | Strong present geometry through the last stable window. |

The user's memory is correct: **8/10 newest runs use 0.05**. The apparent “recent ten” result is
not that weight 1 failed; it is that weight 1 was simply not tried in this modern run family.

Across all 60 live W&B configs, the explicit values were:

| Config value | Run count | Note |
|---:|---:|---|
| `0.05` | 35 | Dominant reconstruction-on operating value. |
| `0` | 3 | Explicit reconstruction-off runs/smokes. |
| `0.10` | 1 | Investigation 07 weight ladder. |
| `0.20` | 2 | One weight arm and one combined capacity arm. |
| `0.50` | 1 | Investigation 07 weight ladder. |
| `1.00` | **1** | Run 029; only 200 steps. |
| key absent / old schema | 17 | Do not infer a value without reading the historical default/config. |

## 2. The only `lambda_recon=1` run

### Run 029 — `tw685b5g`

The only exact match is documented locally in
[`investigation_007/wave_2/run_029_helpful-snow-25`](../../investigation_007/wave_2/run_029_helpful-snow-25/).
Its intended configuration was:

```text
dataset = ssv2                    mode = full prediction
lambda_recon = 1.0               lambda_recon_pred = 0
recon loss = legacy relative MSE lambda_var = 0.5
horizon_k = 12                   n_c = 32
decoder = learned-query 256 x 2  whiten_features = false
recon warmup = 2000              max steps = 15000
```

It was deliberately the saturation extreme of the Investigation 07 weight ladder. All five
Wave-2 jobs stopped together at step 200. The run had no NaNs, no skipped steps, and no divergent
gradient; its last heartbeat aligned with the other four jobs. The Kanban forensics correctly
classify this as an external whole-pod/session/volume event, not a `lambda_recon=1` training
failure.

### Required full-prediction reading cycle

| Q | Question | Result | Evidence |
|---|---|---|---|
| Q1 | Did training stay alive? | **Inconclusive / externally terminated** | Stopped at step 200. `grad_skipped=0`, `grad_has_nan=0`, last `grad_norm=1.153`, max `1.537`. No model-instability signature, but no full run. |
| Q2 | Was `c_t` alive and video-specific? | **Not measurable after training** | Only diagnostic row is initialization: std `0.495`, cosine `0.724`. No step-500 diagnostic exists. |
| Q3 | Was the latent rich? | **Not measurable after training** | Only initialization rank `9.47`. |
| Q4 | Was `c_t` temporally dynamic? | **Not measurable** | Only initialization copy diagnostic. |
| Q5 | Did `F_c` beat copy? | **Not measurable** | Initialization ratio `62.1` is not a trained result. |
| Q6 | Did `F_c` use this video's condition? | **Not measurable** | Initialization batch-mean ratio `12.6` is not a trained result. |
| Q7 | Was reconstruction honest? | **Not measurable** | At initialization: present `1.0246`, true-future `1.0251`, predicted-future `1.0266`; the tiny gap is expected before training. |
| Q8 | Verdict | **Smoke / inconclusive** | It provides infrastructure and early-stability evidence only. |

### What the first 200 steps do establish

At step 200, `recon_scale=0.10`. Therefore the effective weights were only `0.005`, `0.010`,
`0.020`, `0.050`, and `0.100` for configured lambdas 0.05 through 1.00. All five controlled
architectures had the same seed and the same old stack.

| Configured weight | Weighted reconstruction term at step 200 | `grad_norm` after AGC, before global clipping | B tensors AGC-clipped | D tensors AGC-clipped |
|---:|---:|---:|---:|---:|
| 0.05 | 0.00470 | 1.1507 | 8 | 0 |
| 0.10 | 0.00938 | 1.1498 | 8 | 0 |
| 0.20 | 0.01874 | 1.1505 | 8 | 0 |
| 0.50 | 0.04679 | 1.1514 | 8 | 0 |
| **1.00** | **0.09348** | **1.1527** | **8** | **0** |

This is useful but narrow. It says weight 1 did not make the first 200 steps explode. It also
shows why the combined gradient did not grow 20× in that old setup: the bottleneck was already
AGC-clipped on all five arms, while the decoder remained below its AGC threshold.

## 3. How transferable is run 029 to run 058?

The transfer is **low for learning outcome, moderate for early numerical safety**.

| Axis | Run 029 (`tw685b5g`) | Run 058 (`mvbx96nv`) | Transfer consequence |
|---|---|---|---|
| Training purpose | Full future prediction; `L_flow` active | Present reconstruction only; `L_flow=0` | In run 029, recon competed with prediction through shared B. Run 058 has no predictor objective. |
| Dataset | SSv2 | EGO4D | Different content/motion statistics and different shared-template structure. |
| Feature space | Raw V-JEPA features | Fixed offline EGO4D-whitened features | Reconstruction values and gradients are not on the same substrate. |
| Reconstruction formula | Relative MSE | Per-tubelet cosine | The numeric floors and gradient geometry are not comparable. |
| Reconstruction target | Absolute raw features | Absolute whitened features | Both permit shared structure, but the amount and form differ. |
| Bottleneck | Old single cross-attention read plus MLP | Three-block Perceiver-style latent stack with cross-read, self-attention, and MLP | Gradient routing inside B and slot competition changed substantially. |
| Decoder | Learned per-output-token queries that could store a content template | Fixed-position decoder whose zero latent cannot emit position-specific content | The most important old decoder shortcut was explicitly removed. |
| Geometry pressure | Variance floor 0.5 only | Variance floor 0.5 plus covariance 0.01 | Weight 1 changes a different multi-objective balance now. |
| Architecture width | Decoder 256 × 2 | Decoder 512 × 4 | Decoder gradient/capacity regime differs. |
| Code commit | `823eaa5…` | `21d2aa8…` | The relevant files changed by roughly 1,954 insertions / 257 deletions between commits. |
| Shared safety mechanics | 2,000-step recon ramp, AdamW, AGC, global clip 0.5 | Same broad mechanics | The early “probably survivable” result transfers better than the late learning result. |

The old run is therefore not evidence that weight 1 is good or bad for the current reconstruction
floor. It is evidence that weight 1 is not automatically a numerical catastrophe under the ramp.

## 4. The usable historical weight evidence: 0.05 to 0.50

Run 029 did not mature, but the same old architecture had usable controlled arms at 0.05, 0.10,
0.20, and 0.50. The 0.05 control below is run 018 (`yd5958s6`); the other three are the
Investigation 07 arms. They share SSv2, horizon 12, variance weight 0.5, the 256 × 2 decoder,
32 slots, full prediction, the relative-MSE reconstruction objective, seed, optimizer, and
schedules. Step 9,000 is the last common diagnostic age.

| `lambda_recon` | `L_recon_present` @9k | Train `L_recon` @9k | Rank | Std | Cross-video cosine | Copy ratio | Batch-mean ratio | `L_flow` | `grad_norm` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.5987 | 0.6442 | 13.03 | 1.043 | 0.220 | **1.511** | 0.336 | 0.402 | 2.883 |
| 0.10 | 0.5959 | 0.6413 | 12.97 | 1.034 | 0.249 | 1.586 | 0.342 | 0.406 | 2.841 |
| 0.20 | 0.5915 | 0.6375 | 12.98 | 1.043 | 0.251 | 1.564 | 0.338 | 0.403 | 2.928 |
| 0.50 | **0.5855** | **0.6324** | 12.66 | 1.018 | 0.282 | **1.664** | 0.359 | 0.418 | 2.974 |

What this controlled ladder says:

- A 10× weight increase bought only `0.0133` lower diagnostic reconstruction, about a 2.2%
  relative improvement at this floor.
- It did **not** break the rank ceiling. Rank stayed between 12.66 and 13.03.
- It did **not** improve prediction. Every copy ratio failed badly; the 0.50 arm was worse than
  the 0.05 control (`1.664` versus `1.511`).
- Batch-mean ratios passed the weaker conditioning gate, but that never rescues a failed copy gate.
- Combined post-AGC gradient norms were almost invariant across weights. B was AGC-clipped on every
  logged training step in these old runs; D was not.
- The Kanban's extrapolation of a `lambda_recon=1` floor near 0.581 is plausible in this old
  stack, but it remains an extrapolation, not a measured late result.

This is strong evidence against gradient starvation **for the legacy stack**. It is only
qualitative evidence for the new stack, but its direction matters: making the weight much larger
did not unlock a hidden reconstruction regime.

## 5. Run 058: full present-only reading cycle

### Configuration and validity

W&B confirms the intended run-057 twin configuration:

```text
dataset = ego4d
present_recon_only = true
whiten_features = true
whiten_stats_path = logs/whiten/whiten_stats_ego4d_train_seed42.pt
whiten_eps = 1e-4
recon_loss_mode = cosine
recon_residual_target = false
lambda_recon = 0.05
lambda_var = 0.5
lambda_cov = 0.01
lambda_sigreg = 0
n_c = 32
decoder = 512 x 4
steps = 15000
seed = 42
```

The run completed all 15,000 scheduled steps. `present_recon_only=1`,
`prediction_active=0`, `whiten_active=1`, `L_flow=0`, and `L_recon_pred=0` all held. There were
zero skipped steps and zero NaN-gradient diagnostics. Maximum logged `grad_norm` was `7.40` during
the early geometry expansion, well below the skip threshold of 150; the late 20-row mean was
`0.0467`. AGC clipped neither B nor D at any logged step.

### Trajectory

| Step | Present recon | Wrong-video recon | Video gap | Effective rank | Centered slot rank | Cross-video cosine | Std | Train `L_var` | Train `L_cov` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0133 | 1.0133 | 0.0000 | 30.99* | 30.99* | 1.000 | ~0 | 1.000 | 6.726 |
| 500 | 0.9825 | 0.9829 | 0.0004 | 40.56 | 23.96 | 0.596 | 0.573 | 0.202 | 5.295 |
| 1,000 | 0.9082 | 0.9115 | 0.0033 | 65.38 | 30.12 | **0.369** | **0.746** | 0.0438 | 2.685 |
| 2,000 | 0.8193 | 0.8264 | 0.0071 | **80.73** | 29.61 | 0.515 | 0.705 | 0.0084 | 1.368 |
| 5,000 | 0.7095 | 0.7225 | 0.0131 | 63.10 | 30.51 | 0.793 | 0.514 | 0.0020 | 0.415 |
| 7,500 | 0.6872 | 0.7024 | 0.0152 | 51.01 | 30.57 | 0.871 | 0.408 | 0.0007 | 0.248 |
| 10,000 | 0.6820 | 0.6997 | 0.0177 | 57.52 | 30.58 | 0.833 | 0.464 | 0.0010 | 0.272 |
| 12,500 | 0.6789 | 0.6966 | 0.0177 | 53.04 | 30.66 | 0.862 | 0.421 | 0.0006 | 0.257 |
| 14,500 | **0.6782** | **0.6962** | **0.0180** | **52.91** | **30.67** | **0.863** | **0.419** | 0.0005 | 0.213 |

*The near-31 step-zero ranks come from fixed slot identities repeated for every video; with std
approximately zero and cross-video cosine 1, they are mechanical rather than video information.*

### Cycle-B verdict

| Q | Question | Pass? | Evidence |
|---|---|---|---|
| Q1 | Did it train, alive and in the intended mode? | **Pass** | Finished; no skips/NaNs; correct dataset, whitening, absolute target, present-only wiring; no active prediction loss. |
| Q2 | Is `c_t` alive and video-specific? | **Fail** | Final validation std `0.419` versus healthy `0.8–1.2`; cross-video cosine `0.863` versus target below `0.5`; dead dimensions remain zero but that does not rescue weak/shared geometry. |
| Q3 | Is `c_t` rich? | **Fail** | Rank peaks at 80.7 at step 2,000 then contracts to 52.9, below the >60 gate. Centered slot rank stays high, showing slot identity/diversity alone is not enough. |
| Q4 | Does reconstruction learn useful present content? | **Raw loss: yes; video-specific content: mostly no** | Present loss falls to 0.678, but wrong-video loss also falls to 0.696. The gap is only 0.018. |
| Q5 | Do geometry and reconstruction cooperate? | **Fail** | After the early geometry expansion, reconstruction keeps improving while rank/std/video separation collapse. Train regularizer losses look satisfied while fixed-validation geometry fails. |
| Q6 | Verdict | **Collapsed rep / template shortcut** | Stable optimization, but the EGO4D transfer does not preserve run 057's useful representation. |

### The decisive honesty calculation

The decoder's total correct-code improvement is:

```text
1.013300 - 0.678246 = 0.335054
```

The extra improvement attributable to using the correct video rather than a wrong video's code is:

```text
0.696221 - 0.678246 = 0.017975
```

Therefore the video-conditioned share is:

```text
0.017975 / 0.335054 = 0.0536 = 5.4%
```

This is even more template-dominated than run 052's roughly 15% share. The current fixed-position
decoder cannot emit position-specific content from a zero latent, but a highly shared, nonzero
`c_t` can still carry dataset-generic structure that decodes similarly for many videos. The final
cross-video cosine of 0.863 is consistent with exactly that mechanism.

### Train-regularizer versus validation-geometry mismatch

At step 14,500, training-batch `L_var` is only `0.00046`, suggesting the variance hinge is nearly
satisfied on that train batch, while the fixed-validation `c_std_mean` is just `0.419`. Similarly,
training `L_cov` has fallen from 6.73 to 0.21, but validation rank/cosine are unhealthy. The SSv2
control does not show this split: with similarly tiny train `L_var`, validation std is `1.117` and
rank is `208`.

That divergence points to EGO4D train-to-validation generalization, feature-statistics, or shared
content structure—not simply insufficient total gradient magnitude. A larger reconstruction
weight does not directly repair this mismatch and can make memorizing/shared reconstruction more
attractive.

## 6. Run 058 versus the byte-matched SSv2 control, run 057

The code commits differ because the EGO4D data pipeline and frozen-encoder foundation were added,
but the current normalized training hot path and the run-defining architecture/loss configuration
are the same. The intended scientific deltas are dataset plus its matching whitening-statistics
file. Raw reconstruction values across differently whitened datasets are not directly comparable;
trajectory shape, geometry, shuffled-code behavior, and conditioned share are the fairer readouts.

| Metric @ final diagnostic | Run 057 SSv2 | Run 058 EGO4D | Interpretation |
|---|---:|---:|---|
| `L_recon_present` | 0.7129 | 0.6782 | Do not call EGO4D better from this cross-substrate scalar. |
| `L_recon_shuffled_c` | 0.9397 | 0.6962 | EGO4D decoder succeeds almost equally with a wrong code. |
| `L_recon_video_gap` | 0.2268 | 0.0180 | Correct-code dependence almost disappears on EGO4D. |
| Video-conditioned share | ~78.0% | **5.4%** | The central transfer failure. |
| Effective rank | 208.2 | 52.9 | Geometry does not transfer. |
| Centered slot rank | 30.71 | 30.67 | Slots remain distinct in both; this metric alone is insufficient. |
| Cross-video cosine | 0.0585 | 0.8631 | SSv2 codes are video-specific; EGO4D codes are highly shared. |
| `c_std_mean` | 1.117 | 0.419 | SSv2 satisfies spread; EGO4D validation does not. |
| Skips / NaNs | 0 / 0 | 0 / 0 | Difference is learning/generalization, not numerical validity. |
| Cycle-B verdict | **Strong present representation** | **Collapsed rep / template shortcut** | The recipe is not substrate-portable as launched. |

The fact that the collapsed EGO4D run has a numerically lower present reconstruction loss than the
healthy SSv2 run is the clearest warning against optimizing the raw number in isolation.

## 7. What `lambda_recon` actually does in the current code

### Objective and gradient routes

For run 058, the current present-only loss is:

```text
L_total = 0.5 * L_var
        + 0.01 * L_cov
        + lambda_recon * recon_scale * L_recon
```

`L_sigreg`, `L_slot`, and prediction terms are computed/logged where applicable but have zero
weight. The frozen detailed target is detached inside `reconstruction_loss`.

| Term | Reaches bottleneck B? | Reaches decoder D? | Reaches coarse flow `F_c`? |
|---|---:|---:|---:|
| Present `L_recon` | yes | yes | no |
| `L_var` | yes | no | no |
| `L_cov` | yes | no | no |
| `L_flow` | inactive in run 058 | inactive | inactive |

Thus weight 1 cannot directly make prediction better in this present-only run. It changes D's only
data-gradient scale and, more importantly, changes B's reconstruction-versus-geometry direction.

### Two schedules act at once

Reconstruction strength ramps linearly for 2,000 steps. Learning rate warms for 1,500 steps and
then cosine-decays to almost zero at step 15,000.

| Step | LR multiplier | `recon_scale` | Effective recon weight at config 0.05 | Effective recon weight at config 1.0 |
|---:|---:|---:|---:|---:|
| 0 | 0.000667 | 0.00 | 0.0000 | 0.00 |
| 500 | 0.3340 | 0.25 | 0.0125 | 0.25 |
| 1,000 | 0.6673 | 0.50 | 0.0250 | 0.50 |
| 1,500 | 1.0000 | 0.75 | 0.0375 | 0.75 |
| 2,000 | 0.9966 | 1.00 | 0.0500 | 1.00 |
| 7,500 | 0.5868 | 1.00 | 0.0500 | 1.00 |
| 10,000 | 0.3020 | 1.00 | 0.0500 | 1.00 |
| 12,500 | 0.0823 | 1.00 | 0.0500 | 1.00 |
| 14,500 | 0.00338 | 1.00 | 0.0500 | 1.00 |
| 14,950 | 0.000034 | 1.00 | 0.0500 | 1.00 |

The late plateau is partly expected because the optimizer's LR is almost exhausted. But this does
not imply the 0.05 multiplier made all earlier updates ineffective: most reconstruction progress
already happened while LR was substantial.

### Counterfactual scalar balance on run 058

The table below holds the observed run-058 tensors fixed and changes only the scalar multiplier.
It is **not** a simulated alternate trajectory and loss magnitudes are not gradient norms. It does,
however, show how dramatically weight 1 would rebalance the objective from the start.

| Step | Geometry scalar `0.5 L_var + 0.01 L_cov` | Recon scalar at 0.05 | Counterfactual recon scalar at 1.0 |
|---:|---:|---:|---:|
| 500 | 0.1542 | 0.0122 | **0.2430** |
| 1,000 | 0.0487 | 0.0222 | **0.4438** |
| 1,500 | 0.0331 | 0.0314 | **0.6281** |
| 2,000 | 0.0179 | 0.0401 | **0.8019** |
| 7,500 | 0.0028 | 0.0356 | **0.7120** |
| 14,500 | 0.0024 | 0.0344 | **0.6888** |

At 0.05, geometry dominates the first 1,000 steps, is roughly balanced with reconstruction near
step 1,500, and then reconstruction becomes the largest scalar after geometry is mostly satisfied.
At 1.0, reconstruction would already exceed geometry at step 500, exceed it about 9× at step
1,000, and about 45× at step 2,000. That is not merely “give the same solution a stronger final
push”; it changes which solution B is encouraged to form during its critical early organization.

### Why a 20× loss multiplier does not imply a 20× parameter update

The optimizer is AdamW. For a decoder parameter that receives only reconstruction gradient,
scaling the loss by a constant approximately scales Adam's first moment by that constant and its
second moment by the constant squared. The ratio `m / sqrt(v)`—the adaptive update—is therefore
approximately scale-invariant once moments are established. Differences remain through Adam's
epsilon, transient moments, global/AGC clipping, mixed gradients, and decoupled weight decay, but
“20× raw gradient” is not “20× decoder step.”

For B, the situation is different because the gradient is a vector sum:

```text
g_B = 0.5 * grad(L_var)
    + 0.01 * grad(L_cov)
    + lambda_recon * recon_scale * grad(L_recon)
```

Changing the reconstruction weight changes the **direction and relative objective**, not merely
the amplitude. That is exactly why weight 1 risks geometry even if Adam contains its absolute
step size.

The logged `grad_norm` cannot settle the user's hypothesis by itself. It is the norm after
module-wise AGC and before global clip, with all active terms already summed. The current code does
not log unweighted, per-loss gradients for B or D, nor the cosine between reconstruction and
geometry gradients. Run 058's late `grad_norm` near 0.047 and tiny AGC ratios show a calm optimizer;
they do not identify which loss supplied the useful update.

## 8. Why the reconstruction number bottoms out even though learning is active

Several mechanisms are simultaneously visible in the project history.

### 8.1 The newest floor is mostly a shortcut floor

Run 058's wrong-video loss is only 0.018 above its correct-video loss. The model has already
captured most of the easy, shared structure that the absolute target rewards. Increasing the same
loss weight gives more incentive to exploit that same easiest channel. It does not change the
target so that video-specific information becomes necessary.

The residual reconstruction target exists precisely for this case: subtract the per-position
feature mean so the shared template earns zero. The run-058 guide pre-registered this response:
if `L_recon_video_gap` stays near zero, reconsider the residual target on EGO4D.

### 8.2 Lower reconstruction can mean a worse representation

Project examples at the same configured weight make this explicit:

- Run 052 reaches `L_recon_present=0.293` but collapses to rank 13.4, cosine 0.905, and std 0.295.
- Run 055 reaches 0.642 with rank 21.5 and about 86% video-conditioned improvement.
- Run 057 reaches a numerically worse 0.713 but holds rank 208, cosine 0.059, std 1.117, and about
  78% video-conditioned improvement.
- Run 058 reaches 0.678 but only about 5.4% of its improvement is video-conditioned.

These raw losses are not all cross-substrate comparable, but their within-run geometry and honesty
make the general point unambiguous: a low scalar can be bought by a shared, low-rank code.

### 8.3 High-rank geometry carries a decodability cost

Run 055's no-geometry code reconstructs to 0.642 but collapses to rank 21.5. Run 057's covariance
plus variance code holds rank 208 and reconstructs to 0.713. The geometry objective forces B to
use many decorrelated directions rather than the easiest decoder-friendly subspace. Some of the
0.6–0.7 floor is therefore a real content-versus-geometry tradeoff, not lack of optimization.

### 8.4 The old weight sweep already found diminishing returns

On the legacy relative-MSE objective, 10× more weight moved the floor only about 0.013. That does
not prove the same numeric response under cosine whitening, but it strongly warns that the floor
can be structural—latent utilization, decoder access, target shortcut, or compressibility—rather
than weight-bound.

### 8.5 The cosine objective and compressed channel define a different floor

Current reconstruction is mean per-tubelet angular error after normalizing both 1,024-dimensional
feature vectors. It is routed through only 32 × 256 abstract numbers and a fixed-position decoder.
Scaling the loss does not change this information bottleneck, the target's angular structure, or
the decoder architecture.

### 8.6 Cosine LR decay deliberately freezes late improvements

By step 10,000 the LR is 30% of peak; by step 12,500 it is 8.2%; by step 14,500 it is 0.34%.
The final few thousand steps are an asymptotic settling phase. A late plateau does not establish
that the earlier gradient was absent.

## 9. Direct answers to the hypotheses

### “Is `lambda_recon=0.05` so small that reconstruction gradients do not affect training?”

**No, not in that strong form.** Reconstruction falls substantially in every reconstruction-on
run, including from about 1.00 to 0.69 in run 058, so the path is active. In the newest run the
weighted reconstruction scalar is already the dominant late objective term. For D, Adam largely
normalizes constant gradient scaling. The old controlled ladder also shows only tiny gains from
10× more weight.

A weaker statement remains possible: **raising the weight may buy a modest raw-loss improvement
or change B's compromise with geometry.** The current metrics do not contain per-loss gradient
norms, so they cannot quantify exactly how much of B's update comes from reconstruction. That is a
measurement gap, not positive evidence for gradient starvation.

### “Is `lambda_recon=1` too disruptive or extreme?”

There are two different answers:

- **Numerically:** probably survivable. The only weight-1 run was clean through step 200, the
  reconstruction ramp is protective, AGC/global clip exist, and the skip threshold is high. There
  is no evidence that 1.0 automatically explodes training.
- **Scientifically/objectively:** too extreme as the next isolated move. It is a 20× jump that
  makes reconstruction dominate B before geometry forms. It would test a different objective
  balance, not simply whether the same solution needed more gradient.

### “Would weight 1 help the current logic?”

**Likely not in the way intended.** It may lower raw `L_recon` modestly. It is unlikely to fix the
correct-video dependency, validation geometry, or future prediction. On the current absolute
EGO4D target, the most likely failure is an even stronger shared/template solution and weaker
geometry. It also cannot directly improve `F_c`, because run 058 is present-only and the present
anchor never reaches `F_c`.

## 10. Recommended next experiment sequence

### Priority 1 — fix the objective before amplifying it

Run the exact EGO4D twin again with only:

```text
recon_residual_target: false -> true
lambda_recon: keep 0.05
```

This is not an arbitrary new idea. It is the pre-registered run-058 response to an honesty failure,
and prior runs show the residual target removes the per-position template at negligible compute.
The main questions become whether shuffled-code loss returns toward 1.0 in the EGO4D-whitened
space and whether the geometry contraction changes once the shortcut is blocked.

### Priority 2 — instrument the gradient hypothesis directly

Before interpreting a weight sweep, log these values before AGC and global clipping:

- `||grad_B L_recon||` and `||grad_D L_recon||`, both unweighted and weighted;
- `||grad_B (0.5 L_var + 0.01 L_cov)||`;
- cosine similarity between B's reconstruction gradient and geometry gradient;
- per-module parameter-update norm divided by parameter norm after AdamW;
- fraction of steps where the combined pre-global norm exceeds 0.5.

These diagnostics distinguish “recon gradient is tiny,” “recon and geometry cancel,” “Adam
normalizes the scale,” and “clipping saturates the extra weight.” The existing combined
`grad_norm` cannot distinguish them.

### Priority 3 — only then sweep weight under one honest current recipe

Use the same EGO4D whitening file, seed, latent stack, fixed-position decoder, residual target,
covariance/variance weights, schedules, and validation batch. A useful minimal ladder is:

```text
lambda_recon in {0.05, 0.20, 1.00}
```

If only one additional arm is affordable, use `0.20` first. It is a 4× test large enough to move
the relative gradient without jumping immediately to a 20× reconstruction-dominated objective.
The 1.0 arm becomes valuable if 0.20 shows a real honest-loss response without geometry damage.

### Decision metrics

Do not rank arms by raw reconstruction alone. Require, in this order:

1. Q1 validity: no NaNs/skips or sustained pathological clipping.
2. Video dependency: a substantial and growing `L_recon_video_gap`; report the conditioned share.
3. Geometry: rank above 60 and not contracting, std toward 0.8–1.2, cosine below 0.5, centered
   slot rank stable.
4. Raw reconstruction: lower only counts if 2 and 3 hold.
5. Prediction: a later full-prediction transfer must still pass copy and batch-mean gates; no
   present-only result can substitute for that test.

Useful early checkpoints are 2,000, 5,000, 7,500, and 10,000. Run 058 had already peaked in rank
at step 2,000 and was clearly collapsing by step 5,000, so a staged sweep need not spend all
15,000 steps on an arm that repeats the same trajectory.

## Final verdict

`lambda_recon=1` is **not proven dangerous**, because the one historical arm died externally and
was numerically clean while it ran. It is also **not supported as the solution**: the usable old
weight ladder shows strong diminishing returns and no representation or prediction benefit, while
the newest run's apparent floor is dominated by wrong-video/template reconstruction and a
train-to-validation geometry failure.

The most likely outcome of changing only 0.05 to 1.0 today is a modestly better raw scalar with
the same or worse scientific representation. The higher-information move is to make the EGO4D
reconstruction honest first, measure the actual per-loss gradients, and then sweep the weight in a
controlled current-stack experiment.
