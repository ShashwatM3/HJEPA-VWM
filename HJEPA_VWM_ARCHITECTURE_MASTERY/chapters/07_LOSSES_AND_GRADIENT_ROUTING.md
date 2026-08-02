# 07 — Losses and gradient routing

## Total objective

In full-prediction mode:

```text
L =
    L_flow
  + λ_var L_var
  + 1[λ_cov>0] λ_cov L_cov
  + 1[λ_slot>0] λ_slot L_slot
  + 1[λ_sig>0] λ_sig r_sig(step) L_SIG
  + 1[λ_rec>0] λ_rec r_rec(step) L_recon
  + 1[λ_pred>0] λ_pred r_rec(step) L_recon_pred
```

In present-only mode, `L_flow` is replaced with the graph-connected zero
`abstract.sum()*0`. Collapse and present reconstruction terms can still train `B`.

## Shipped loss defaults

| Term | Weight | Computed every step? | Added by default? |
|---|---:|---:|---:|
| flow | 1 | except present-only | yes |
| variance floor | 0.10 | yes | yes |
| covariance | 0 | yes | no |
| slot diversity | 0 | yes | no |
| SIGReg | 0 | yes | no |
| present reconstruction | 0 | only when active in train; diagnostic otherwise | no |
| predicted reconstruction | 0 | only when active in train; diagnostic otherwise | no |

Computing an inactive regularizer supports calibration, but it must not perturb the random stream.
SIGReg therefore uses its own deterministically seeded generator.

## Flow matching

```text
L_flow = mean_all((û-u)^2)
```

Every batch, slot, and feature element has equal weight. Ordinary target is `c⁺`; residual target is
`Δ`. Gradients reach `F_c` and online conditioning `B`, never the detached target.

## Variance floor

Flatten each example:

```text
c: (B,N_c,D_c) → (B,N_c*D_c)
std_j = population_std_across_batch(c[:,j])
L_var = mean_j max(0, std_target-std_j)
```

`std_target=1.0`; `unbiased=False`. If `B<2`, the function returns zero. It prevents each flattened
coordinate from being constant across videos. It does not decorrelate dimensions or prevent slots
from being redundant.

## Covariance penalty

Pool batch and slots:

```text
z: (B*N_c,D_c)
z ← z - column_mean
C = zᵀz/(B*N_c-1)
L_cov = (sum(C²)-sum(diag(C)²))/D_c
```

This is VICReg's off-diagonal feature decorrelation convention. It attacks correlated feature
dimensions. It does not set diagonal scale; `L_var` handles that.

## Slot-diversity penalty

For each video:

1. subtract its mean slot vector;
2. L2-normalize every residual slot;
3. form the `N_c×N_c` cosine matrix;
4. square and sum off-diagonal values.

The final denominator is:

```text
B*N_c*(N_c-1)
```

Centering is crucial. Without it, the gradient mostly fights a shared mean restored elsewhere by
normalization. This term was historically Goodhart-prone: improving its target metric did not
necessarily produce useful representation/prediction.

## SIGReg

Pool `(B*N_c,D_c)`, optionally subsample at most 512 rows, center features, and sample 128 random
unit projection directions. For every direction, evaluate a BHEP/Epps–Pulley discrepancy from a
standard normal:

```text
T =
 mean_jk exp(-β²(y_j-y_k)²/2)
-2/sqrt(1+β²) mean_j exp(-β² y_j²/(2(1+β²)))
+1/sqrt(1+2β²)
```

with `β=1`. Negative numerical residue is clamped to zero, then projections are averaged.

Properties:

- operates in FP32 with autocast disabled;
- complexity is `O(P*S²)` after row cap;
- tests isotropic standard-normal behavior across random projections;
- uses generator seed `base_seed*1,000,003+step`;
- remains computed but gradient-inactive when its weight is zero.

SIGReg can raise effective rank while moving the online coordinate system faster than the EMA target
can follow, worsening prediction. Hence its optional 2,000-step ramp.

## Reconstruction losses

Cosine mode:

```text
L_recon = mean_(batch,token)(1 - dot(normalize(ê), normalize(stopgrad(e))))
```

Normalization is FP32 with epsilon `1e-6`. Legacy relative MSE divides raw mean squared error by
population target variance floored at `1e-8`.

Present reconstruction routes into `B,D`. Predicted reconstruction routes into `B,F_c,D`.

## Linear loss ramps

For SIGReg and reconstruction:

```text
r(step,warmup) =
  1                         if warmup <= 0
  clamp(step/warmup,0,1)    otherwise
```

With warmup 2,000:

| Step | Ramp |
|---:|---:|
| 0 | 0 |
| 1 | 0.0005 |
| 1,000 | 0.5 |
| 2,000+ | 1 |

This differs from LR warmup, whose first step is nonzero because it uses `(step+1)/warmup`.

## Full gradient-routing matrix

| Objective | Online `B` | `F_c` | `D` | `B_EMA` | frozen `E` | Mean/whitener |
|---|---:|---:|---:|---:|---:|---:|
| `L_flow` | condition path | direct | — | detached | no-grad | buffers only |
| residual `L_flow` | condition path | direct | — | detached present/future | no-grad | buffers only |
| `L_var` | direct | — | — | — | — | — |
| `L_cov` | direct | — | — | — | — | — |
| `L_slot` | direct | — | — | — | — | — |
| `L_SIG` | direct | — | — | — | — | — |
| `L_recon` | through `c_t` | — | direct | — | target detached | mean read only |
| `L_recon_pred` ordinary | condition and endpoint path | endpoint | direct | target detached | target detached | mean read only |
| `L_recon_pred` residual | condition + `c_t` add-back | residual endpoint | direct | target detached | target detached | mean read only |

The mean tracker changes via explicit no-gradient update. The whitener is fixed after offline
configuration.

## Optimizer membership is not gradient routing

The decoder is always placed in AdamW. When both reconstruction weights are zero, it is never called
inside the training loss, so its gradients are `None`; AdamW skips it and its parameters do not
move. Similarly, `F_c` stays in the optimizer in present-only mode but receives no gradients.

## Weight decay partition

Only genuine linear/convolution weight matrices receive the shipped 0.05 AdamW decay. The shared
no-decay/AGC-exclusion predicate removes:

- every parameter with fewer than two dimensions: biases and LayerNorm scale/shift;
- geometry leaves named `queries`, `pos_emb`, `null_condition`, `slot_pos`, `z_type`, `cond_type`;
- zero-initialized residual output/gating submodules marked `is_zero_init`.

This yields up to six stable-order optimizer groups:

```text
B/decay, B/no_decay,
F_c/decay, F_c/no_decay,
D/decay, D/no_decay
```

Older three-group optimizer checkpoints are not structurally compatible. An explicit optimizer reset
is required for intentional migration.

## Adaptive gradient clipping

For each eligible parameter tensor:

```text
bound = λ_module * (||w||₂ + eps)
if ||g||₂ > bound:
    g ← g * bound/(||g||₂+eps)
```

Shipped `eps=0.001`; module factors:

| Module | AGC factor |
|---|---:|
| `B` | 0.20 |
| `F_c` | 0.10 |
| `D` | 0.20 |

Geometry and zero-initialized bridge parameters are excluded so their near-zero weight norm does not
prevent them from opening.

## Global clipping and skip semantics

After AGC, `clip_grad_norm_(all trainable, 0.5)`:

- returns the global norm before its own 0.5 rescaling, but after AGC;
- changes gradients in place;
- is followed by the survivability decision.

If that returned norm is non-finite or greater than 150:

- gradients are zeroed;
- AdamW does not step;
- EMA does not update.

An instability warning is logged when the same norm exceeds 30 and `L_flow>1.0`, but it does not
skip the step.

The diagnostic `grad_global_norm_postclip` is measured later and often pins near 0.5. It must not be
mistaken for the pre-global-clip `grad_norm`.

## What each anti-collapse term can and cannot prove

| Term/metric | Protects against | Does not prove |
|---|---|---|
| variance floor | constant flattened coordinates | decorrelation or semantic richness |
| covariance | feature-dimension correlation | distinct slots or predictability |
| slot loss | centered slot-direction redundancy | cross-video diversity |
| SIGReg | non-isotropic pooled feature law | temporal usefulness |
| reconstruction | some detailed-feature recoverability | predictor uses context |
| effective rank | broad covariance spectrum | causal/semantic factors |

The research history repeatedly demonstrates these separations. Treat no single scalar as a
certificate.
