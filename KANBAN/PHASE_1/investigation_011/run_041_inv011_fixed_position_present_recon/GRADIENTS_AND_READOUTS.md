# Gradient and readout notes

This note exists to prevent the run from being misread as a prediction experiment. It is not. It is
a present-side representation experiment.

## Data Path

Active:

```text
context_clip
  -> FrozenEncoder E
  -> detailed e_t
  -> online Bottleneck B
  -> abstract c_t
  -> fixed-position Decoder D
  -> e_hat_t
```

Inactive:

```text
target_clip
target_detailed e_plus
target_bottleneck B_EMA(e_plus)
c_plus
coarse_flow F_c
z_c / tau_c / u_c / u_c_hat
c_hat
prediction-side reconstruction
copy baseline
batch-mean baseline
```

## Gradient Routing

### Reconstruction

```text
L_recon_present = reconstruction_loss(D(c_t), e_t, mode="cosine")
```

Gradient reaches:

```text
D parameters
B parameters through c_t
```

Gradient does not reach:

```text
FrozenEncoder E
e_t target
F_c
B_EMA
```

`e_t` is detached inside `reconstruction_loss` by `as_target`.

### SIGReg

```text
L_sigreg(c_t)
```

Gradient reaches:

```text
B parameters
```

Gradient does not reach:

```text
D
F_c
FrozenEncoder E
B_EMA
```

SIGReg pools `abstract` as `(B * N_c, D_c)` and pushes random 1-D projections toward standard
normal behavior. In practical terms, it pressures the bottleneck to use the 256 feature dimensions
more evenly and to avoid low-rank geometry.

### Variance Floor

```text
L_var(c_t) = mean_j max(0, 1.0 - Std(c_j))
```

Gradient reaches:

```text
B parameters
```

It is one-sided. Once a coordinate has enough batchwise spread, that coordinate stops receiving
variance-floor pressure. This makes it a safety rail rather than a full geometry objective.

## Combined Objective

The present-only objective for this run is:

```text
L_total =
    0.5 * L_var(c_t)
  + 5.0 * sigreg_scale * L_sigreg(c_t)
  + 0.05 * recon_scale * L_recon_present(D(c_t), e_t)
```

Scales:

```text
sigreg_scale = linear_ramp(step, 2000)
recon_scale  = linear_ramp(step, 2000)
```

Expected scale checkpoints:

| Step | `sigreg_scale` | `recon_scale` |
|---:|---:|---:|
| 0 | 0.00 | 0.00 |
| 500 | 0.25 | 0.25 |
| 1000 | 0.50 | 0.50 |
| 1500 | 0.75 | 0.75 |
| 2000+ | 1.00 | 1.00 |

## Why The Gradients Can Help Each Other

Reconstruction gives `c_t` a content reason to exist. Without it, SIGReg can create a well-shaped
latent distribution that is not necessarily useful for decoding frozen V-JEPA features.

SIGReg gives reconstruction a geometry constraint. Without it, reconstruction might choose a
low-rank or correlated code that decodes present features but is not a healthy representation space
for later prediction.

Variance floor is the simple collapse guard. It is less expressive than SIGReg, but it protects
against the direct constant-code failure mode.

## How The Gradients Can Fight

The run can fail in several interpretable ways:

| Pattern | Likely meaning |
|---|---|
| `L_recon_present` falls, rank stays low | Reconstruction found a decodable but low-rank code. |
| Rank rises, `L_recon_present` stalls | SIGReg shaped geometry but content did not become decodable. |
| `c_std_mean` good, rank low | Variance exists but dimensions are correlated or redundant. |
| Cross-video cosine high | Representation is becoming video-independent. |
| Slot rank very low | The 32 slots may be redundant even if feature rank looks okay. |
| AGC clips `B` or `D` heavily | The joint objective may be too aggressive or badly scaled. |

## Present-Only Reading Cycle

Use Q1, Q2, Q3, Q5-prime, and a reconstruction-specific Q7:

| Question | Metrics |
|---|---|
| Q1: Training alive? | `grad_skipped`, `grad_has_nan`, `grad_norm`, `instability_warn` |
| Q2: `c_t` alive/video-specific? | `c_std_mean`, `c_dead_dim_frac`, `c_cross_video_cosine` |
| Q3: Rich latent? | `c_effective_rank`, plus `c_std_mean` sanity |
| Q5-prime: Present reconstruction works? | `L_recon_present`, `c_effective_rank`, `c_cross_video_cosine`, `c_std_mean` |
| Q7: Reconstruction honestly uses `c_t`? | `L_recon_present`, and ideally a future shuffled-`c` diagnostic if added later |

Skip:

- Q4 temporal dynamics;
- Q5 copy gate;
- Q6 batch-mean gate;
- `L_recon_cplus`;
- `L_recon_chat`.

Those require the future/prediction branch and are intentionally unavailable here.

## Recommended Verdict Labels

Use these labels when filling `OBSERVATIONS.md`:

| Label | Pattern |
|---|---|
| Invalid | Q1 fail, skipped-step spiral, NaN, or early crash. |
| Collapsed rep | Q2 fail: high cross-video cosine, dead dims, or weak std. |
| Low-rank decodable | Reconstruction improves but rank stays below target. |
| Pretty geometry, weak content | Rank/std healthy but `L_recon_present` stalls. |
| Strong present representation | Reconstruction improves and rank/std/cosine all pass. |

## What Would Be Most Convincing

The strongest outcome would be:

```text
L_recon_present lower than or comparable to Run C's ~0.346 cosine value
c_effective_rank > 60
c_cross_video_cosine < 0.5
c_std_mean around 1.0
grad_skipped = 0
```

That would say the bottleneck can carry present content with healthy geometry when prediction is not
part of the objective.

## Caveat

A strong present representation is necessary but not sufficient for Phase 1 prediction. It would not
prove `F_c` can forecast. It would only remove one major ambiguity: whether the bottleneck/decoder
channel itself can work.
