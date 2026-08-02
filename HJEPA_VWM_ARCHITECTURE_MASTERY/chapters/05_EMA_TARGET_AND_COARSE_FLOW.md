# 05 — EMA target and coarse flow

## The target is an EMA bottleneck, not an EMA encoder

At construction, the target bottleneck is a deep copy of online `B`. Every target parameter is
marked `requires_grad=False`, and the module stays in evaluation mode. It maps future detailed
features to:

```text
c⁺ = stop_gradient(B_EMA(E(x_future)))
```

Because `E` is frozen, current and future clips share exactly the same encoder instance and weights.
There is no `E_EMA`.

## EMA transition

After a successful optimizer step:

```text
θ_EMA ← m θ_EMA + (1-m) θ_online
```

The implementation uses in-place linear interpolation with weight `1-m` toward online parameters.
It updates corresponding parameter pairs in strict order. If the gradient is non-finite or exceeds
the skip threshold, neither optimizer nor EMA moves.

The update order is:

```text
backward → AGC → global clip/check → optimizer.step() → B_EMA update
```

Updating the teacher before the optimizer would give it a one-step lag different from the documented
schedule.

## EMA momentum schedule

Shipped values:

```text
m_start = 0.996
m_end   = 0.9999
total   = 105,000 steps
```

The schedule is:

```text
p = clamp(step / 105000, 0, 1)
m(step) = m_end - (m_end-m_start) * 0.5 * (1 + cos(πp))
```

| Step | Momentum | Approximate `1/(1-m)` timescale |
|---:|---:|---:|
| 0 | 0.996000000 | 250 steps |
| 15,000 | 0.996193111 | 262.7 steps |
| 105,000 | 0.999900000 | 10,000 steps |

Phase 1 lasts 15,000 steps by default but shares the 105,000-step latent-stage denominator. Thus its
teacher momentum changes only modestly during Phase 1. Do not claim it reaches 0.9999 at step 15,000.

## Coarse flow contract

```text
F_c(z_τ, τ, c_t) → û
```

All latent arguments have shape `(B,N_c,D_c)` and `τ` has shape `(B,)`. At shipped defaults, the
model processes two streams of 32 tokens at width 256.

## Rectified-flow training path

For ordinary full-latent prediction:

```text
ε ~ Normal(0,I)
τ ~ Uniform[0,1)
z_τ = (1-τ)ε + τc⁺
u   = c⁺ - ε
L_flow = mean((F_c(z_τ,τ,c_t) - u)^2)
```

The target velocity is constant along the straight interpolation. `c⁺` is detached. Gradients enter
`F_c` directly and online `B` through the conditioning `c_t`; they do not enter `B_EMA` or `E`.

## Input identities

Before concatenation:

```text
z stream:    z_τ + slot_pos + z_type
condition:   c_t + slot_pos + cond_type
```

- `slot_pos`: `(N_c,D_c)`, small-normal initialization with std 0.02, shared across streams so slot
  `i` can align with slot `i`;
- `z_type`: `(1,1,D_c)`, zero initialized;
- `cond_type`: `(1,1,D_c)`, zero initialized;
- `null_condition`: `(N_c,D_c)`, zero initialized and learned.

The concatenated sequence is `(B,2N_c,D_c)`. The position/type additions happen to local copies; the
caller retains the raw `z_τ` for endpoint arithmetic.

## Classifier-free condition dropout

During training, each batch example independently drops its whole condition with probability 0.1.
A dropped example replaces all `c_t` slots with the learned null-condition table, after which
slot/type codes are still added.

Diagnostics can pass an explicit all-false drop mask, guaranteeing conditioned evaluation even while
the module is in training mode. In evaluation mode with no override, no random condition dropout
occurs.

The current repository trains the unconditional branch but does not implement classifier-free
guidance sampling.

## Flow-time embedding

For `D=D_c`, half of the channels use sine and half cosine:

```text
freq_i = exp(-log(10000) * i / floor(D/2))
emb(τ) = concat(sin(τ freq_i), cos(τ freq_i))
```

Odd dimensions are padded and truncated. This fixed embedding passes through:

```text
D → 4D → D
```

with SiLU between the linear layers.

## Six AdaLN-Zero blocks

Each block receives the 64-token combined sequence and the processed time embedding.

1. The time embedding passes through SiLU and a zero-initialized `D→6D` projection.
2. Its output splits into attention shift, scale, gate and MLP shift, scale, gate.
3. Non-affine LayerNorm is modulated for self-attention.
4. Eight-head full self-attention mixes both streams.
5. The attention output is multiplied by its per-example gate and added residually.
6. A second modulated non-affine LayerNorm enters `D→4D→D` GELU MLP.
7. The MLP output is gated and added residually.

At initialization all modulation shifts/scales/gates are zero, so each block is the identity. Unlike
a DiT with a separate zeroed output head, this implementation then applies an affine LayerNorm to
the first `N_c` tokens. Its initial velocity is therefore a normalized version of the marked
`z` stream, not an all-zero tensor.

## Output

After six blocks:

```text
u_hat = LayerNorm(x[:, :N_c])
```

Only the noised-future half is returned. The condition half influences it through full self-attention
once gates open.

## Exact parameter formula

Let `D=D_c`, `N=N_c`, and `L=6`.

```text
learned input codes = 2*N*D + 2*D
time MLP            = 8*D^2 + 5*D
one AdaLN block     = 18*D^2 + 15*D
final LayerNorm     = 2*D

P_Fc = 2*N*D + 2*D + (8*D^2+5*D) + L*(18*D^2+15*D) + 2*D
```

At `N=32,D=256,L=6`:

```text
P_Fc = 7,643,904
```

The flow count depends strongly on `D_c²` and only linearly on `N_c`. Bottleneck width `M` does not
change `F_c`.

## Attention-logit scale

The combined sequence length is `2N_c=64`. With eight heads and batch 64, one full attention-logit
tensor contains:

```text
64 * 8 * 64 * 64 = 2,097,152 elements
```

That is 4 MiB in BF16 per block before accounting for kernel-specific materialization and backward
state. There are six blocks.

## Residual-prediction mode

When `predict_residual=True`, the target is a purely temporal difference between two EMA outputs:

```text
c_EMA,present = B_EMA(e_t)
Δ = c_EMA,future - c_EMA,present
σ = max(std(Δ), 1e-6)
ε = σ * Normal(0,I)
z_τ = (1-τ)ε + τΔ
u = Δ - ε
```

The standard deviation is taken over all elements with PyTorch's default sample correction. Both
EMA terms are detached, so the residual does not mix temporal change with an online-versus-teacher
gap.

The one-step endpoint predicts a residual:

```text
Δ_hat = z_τ + (1-τ)û
c_hat = c_online,present + Δ_hat
```

The copy baseline in this mode is zero residual. This keeps the copy comparison meaningful:
predicting no change corresponds exactly to copying the present latent.

## One-step endpoint and the inference boundary

In ordinary mode:

```text
c_hat = z_τ + (1-τ)û
```

This is used by diagnostics and the optional prediction-side reconstruction objective. It is not a
multi-step sampler. There is no solver, step schedule, guidance scale, horizon input, or rollout
loop in current Phase 1.

## Failure interpretations

- `F_c` loss below an unconditional batch-mean baseline but not below copy means it models average
  target structure without useful temporal conditioning.
- Conditioned and shuffled-condition losses being equal means it ignores the present latent.
- A residual predictor near zero is a copy model, even if the latent representation itself is
  healthy.
- High condition dropout is not “more robust” by definition; it can encourage unconditional
  prediction.
- Good `c` rank does not prove good `F_c`; representation and prediction gates are separate.
