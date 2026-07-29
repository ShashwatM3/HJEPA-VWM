# Investigation 020 — Pretrained bottleneck transfer into full prediction

## Status

OPEN — implementation complete; ready for paired launch.

The Investigation-019 source decision is complete. The initialization-only warm start,
parallel temporal-target selector, provenance parity guard, and targeted tests are implemented.
Launch still requires one clean published commit plus the source/resource gates in `GUIDE.md`.

No W&B run exists yet. Do not assign W&B IDs or claim that either arm has started until the live
run URLs are recorded in the corresponding run `DESCRIPTION.md`.

## Question

Starting from the same reconstruction-pretrained bottleneck, does the current Phase-1 coarse system
learn future abstract dynamics better when `F_c` predicts:

1. the temporal residual
   `Delta = B_EMA(e_{t+k}) - B_EMA(e_t)`; or
2. the full future latent `B_EMA(e_{t+k})`?

Both arms keep the same covariance-plus-variance geometry objective. The comparison is about the
temporal target parameterization, not encoder choice, bottleneck shape, regularizer family, decoder
capacity, feature whitening, or prediction-side reconstruction.

## What “full prediction” means here

This remains the implemented Phase-1 system:

```text
context clip -> frozen encoder E -> online B -> c_t
future clip  -> frozen encoder E -> target B_EMA -> c_plus
F_c predicts a rectified-flow velocity toward either c_plus or Delta
D(c_t) reconstructs present frozen features
```

It does not add the unimplemented `FineFlow`, pixel generator, inference sampler, or later training
stages. “Full” means the current future branch and coarse-flow objective are active:
`present_recon_only=false` and `prediction_active=1`.

## Investigation-019 transfer decision: DINOv3

The valid transfer candidates use the same selected high-capacity shape:
`N_c=64`, `D_c=512`, and `M=512`. SigLIP 2 is excluded from this decision. The choice is between
the completed covariance-plus-variance DINOv3 and V-JEPA2 arms.

Late-six diagnostic medians are:

| Metric | [DINOv3 `60yaqw6d`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/60yaqw6d) | [V-JEPA2 `93ildhmk`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/93ildhmk) |
|---|---:|---:|
| `L_recon_present` (lower is better within its feature space) | 0.125322 | 0.222106 |
| `L_recon_video_gap` (higher code dependence is better) | 0.412204 | 0.376610 |
| `c_std_mean` | 1.102973 | 1.130235 |
| `c_cross_video_cosine` (lower is better) | 0.110496 | 0.109080 |
| `c_effective_rank` out of 512 | 362.957 | 389.107 |
| `c_slot_diversity_rank` out of 64 | 54.644 | 56.706 |

Both are stable strong-present-representation runs with zero skipped or NaN updates. V-JEPA2 has a
modest rank and slot-diversity advantage, while DINOv3 retains substantially more decodable,
correct-code-dependent content and has spread closer to the unit target. Raw reconstruction losses
remain feature-space-dependent, so reconstruction alone is not an encoder benchmark; the decision
uses the joint content-and-geometry pattern.

Use DINOv3. Its frozen backend is framewise rather than tubelet-aware, so this is not a claim that
DINOv3 has the stronger temporal prior. The downstream bottleneck receives the full ordered
eight-frame lattice and remains trainable under `L_flow`; the purpose of pretraining was to start
future learning from the strongest information-carrying bottleneck rather than choose an encoder
from architectural analogy. The new runs are the falsifiable test of whether that substrate can be
reorganized for prediction.

This pair does not prove DINOv3 is better than V-JEPA2 for future prediction because encoder is
fixed. After selecting a temporal target, one matched V-JEPA2 control can isolate the temporal-prior
question without turning this two-arm experiment into a two-axis matrix.

The two Investigation-020 arms must use the exact same source checkpoint SHA-256.

## Warm-start contract

This is pretraining transfer, not resume.

Transferred:

- the source checkpoint's live online `B`;
- the matched reconstruction decoder `D`, so the present reconstruction function used to certify
  the bottleneck is not discarded.

Fresh:

- `F_c`;
- AdamW state and learning-rate schedule;
- global step (`0`);
- sampler position and all RNG streams;
- W&B run ID;
- checkpoints and provenance outputs.

Target initialization:

- do not carry the source checkpoint's lagging `B_EMA`;
- after loading the online pretrained `B`, copy it exactly into a fresh `B_EMA`;
- keep `B` trainable. “Pretrained” never means frozen in this investigation.

This makes both arms identical at step 0 in `B`, `B_EMA`, `F_c`, `D`, encoder identity, data order,
and initialization hash. Only `train.predict_residual` may differ.

## Locked common objective

```text
L_total =
    L_flow
  + 0.5 * L_var
  + 0.01 * L_cov
  + 1.0 * recon_scale * L_recon_present
```

Locked off:

```text
lambda_sigreg = 0
lambda_slot = 0
lambda_recon_pred = 0
recon_residual_target = false
whiten_features = false
```

`lambda_recon_pred=0` is deliberate. Earlier work found the prediction-side decoder readout nearly
blind to prediction quality. Turning it on here would add a second temporal gradient path and
confound the target-parameterization comparison.

## Regularizer decision

Use covariance plus variance in both first-wave arms. Do not use SIGReg-only and do not combine
SIGReg with covariance plus variance in this pair.

The live matched V-JEPA2 evidence at the common step 10,500 is:

| Metric | [No geometry `4biwq87o`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4biwq87o) | [Pure SIGReg `utcpfj57`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/utcpfj57) | [Covariance plus variance `guiduvjp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/guiduvjp) |
|---|---:|---:|---:|
| `L_recon_present` (lower is better) | 0.314251 | 0.361520 | 0.332927 |
| `c_std_mean` | 0.246927 | 0.461933 | 0.876897 |
| historical-batch `c_cross_video_cosine` (lower is better) | 0.933076 | 0.761324 | 0.436802 |
| `c_effective_rank` | 12.261 | 54.088 | 117.133 |
| `L_recon_video_gap` (higher is better) | 0.144695 | 0.064914 | 0.110774 |

Covariance plus variance is a strict Pareto improvement over pure SIGReg on these five readouts.
The no-geometry control reconstructs best but has the collapsed spread/rank/separation signature,
which is why reconstruction cannot choose the collapse guard. All three were stable through the
common step. `guiduvjp` was intentionally stopped at step 10,950, so this is a matched common-step
comparison, not a completed-run comparison. All three predate the repaired source-diverse
diagnostic sampler; cosine and rolled-code gap are within-source cross-chunk evidence only.

SIGReg is not dismissed. The [LeJEPA paper](https://arxiv.org/abs/2511.08544) combines SIGReg with
a direct multi-view prediction/alignment loss. The later
[world-model analysis](https://arxiv.org/abs/2605.26379) likewise proves a joint statement about
alignment plus Gaussian regularization under specific latent-distribution and transition
assumptions. This project's conditional rectified-flow objective, frozen encoder, EMA target
bottleneck, slot pooling, and reconstruction anchor are not that theorem's training setup.

After this investigation selects a temporal target, a new investigation may test the incremental
effect of adding SIGReg to the winning full-prediction recipe. It must not rewrite this two-arm
comparison.

## Registered arms

| Folder | Intended W&B role | Only scientific difference |
|---|---|---|
| [`residual_prediction_cov_var/`](residual_prediction_cov_var/) | residual temporal target | `predict_residual=true` |
| [`full_latent_prediction_cov_var/`](full_latent_prediction_cov_var/) | full future latent target | `predict_residual=false` |

The shared design and decision rule are in
[`SWEEP_PLAN_temporal_target.md`](SWEEP_PLAN_temporal_target.md). Parallel launch and recovery are
in [`GUIDE.md`](GUIDE.md).

## Primary outcome

A Phase-1 prediction pass requires both late-window gates:

```text
coarse_vs_copy_ratio <= 0.70
coarse_vs_batch_mean_ratio <= 0.50
```

The run must also pass stability and representation-health checks. `L_flow` alone can never select
the winner, and absolute `L_flow` or absolute coarse losses must not be compared across residual
and full-latent modes because their target/noise scales differ.
