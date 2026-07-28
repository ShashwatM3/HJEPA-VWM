# 20 — Five end-to-end traces

These traces are designed for whiteboard rehearsal. Say every shape, state transition, and gradient
boundary aloud.

## Trace 1 — shipped V-JEPA full-prediction step

### Data

One sample at start `a`, default `k=4`:

```text
context indices = a+[0,2,4,6,8,10,12,14]
target indices  = a+[4,6,8,10,12,14,16,18]
```

Six exact frames overlap. Shared crop/color jitter produces two `(8,3,256,256)` tensors. Batch:
`(64,8,3,256,256)`.

### Frozen feature path

V-JEPA normalization and tubelet-2 encoding:

```text
(64,8,3,256,256)
→ (64,4,16,16,1024)
→ e_t (64,1024,1024)
```

Same frozen `E` produces `e_future`. Neither path records gradients.

### Online bottleneck

```text
e_t
→ in_proj (64,1024,256)
→ reshape (256,256,16,16) because B*T_e=64*4
→ two ConvNeXt blocks
→ flatten + learned pos (64,1024,256)
→ 32 learned queries
→ 3× [cross-attn + self-attn + MLP]
→ Identity abstract projection
→ LayerNorm
→ c_t (64,32,256)
```

### Teacher

```text
c⁺ = stopgrad(B_EMA(e_future)) = (64,32,256)
```

### Flow

```text
ε (64,32,256)
τ (64,)
z_τ=(1-τ)ε+τc⁺
```

`F_c` adds slot/type identity, concatenates condition:

```text
(z stream 32 + condition 32) → (64,64,256)
→ six 8-head AdaLN-Zero blocks
→ first 32 + LayerNorm
→ û (64,32,256)
```

Loss is MSE to `c⁺-ε`.

### Representation/loss

`L_var` is active at 0.1. Covariance, slot, and SIG are evaluated but zero-weighted. `D` is not
called in the training objective because both recon weights are zero.

### Update

```text
backward
→ AGC B=.2,Fc=.1
→ global clip=.5
→ skip if returned norm nonfinite or >150
→ AdamW B/Fc (D has None grads)
→ EMA B teacher at m(step)
```

`E` never moves.

## Trace 2 — frame encoder substitution

Keep the exact same batch and trainable abstract contract. Select DINOv3:

```text
(64,8,3,256,256)
→ flatten (512,3,256,256)
→ frame microbatches of 8
→ last_hidden_state (frame,261,768)
→ strip CLS+4 registers
→ (64,8,256,768)
→ e (64,2048,768)
```

Downstream changes:

- bottleneck input projection is `768→M`;
- learned detailed position table has 2,048 rows;
- cross-attention memory length doubles;
- decoder outputs `(64,2048,768)`;
- `B` count becomes 5,033,731 at `M=256`;
- `D` count becomes 2,106,368;
- `c`, flow shapes, flow count, losses, schedule, and EMA semantics stay unchanged.

This is the encoder seam's purpose: change detailed geometry without changing the external abstract
API.

## Trace 3 — present-only residual-feature reconstruction

Recipe switches:

```text
present_recon_only=true
lambda_recon>0
recon_residual_target=true
```

### Absent work

- target clip need not transfer;
- no target encoder call;
- no `B_EMA` forward;
- no noise/time interpolation;
- no `F_c` call;
- no copy/batch-mean diagnostics.

### Active path

```text
context → no_grad E → e_t → online B → c_t
```

Mean tracker receives `e_t`. First batch copies its per-position mean; later batches EMA update.

```text
target = e_t - mean
prediction = D(c_t)
L_recon = tokenwise cosine(prediction,target)
```

Weighted loss also includes active geometry terms. Gradients reach `B,D`; `F_c` parameters remain
unchanged despite optimizer membership. `B_EMA` still exists and may be EMA-updated from `B` on
successful steps, but it is not a prediction target in this mode.

Run verdict uses stability, geometry, reconstruction, and shuffled-video gap—not copy gates.

## Trace 4 — full temporal-residual prediction plus predicted reconstruction

Switches:

```text
predict_residual=true
lambda_recon_pred>0
```

Compute three bottleneck outputs:

```text
c_online = B(e_t)                         gradients
c_ema_present = stopgrad(B_EMA(e_t))      no gradient
c_ema_future  = stopgrad(B_EMA(e_future)) no gradient
Δ = c_ema_future-c_ema_present
```

Scale:

```text
σ=max(std_all(Δ),1e-6)
ε=σN(0,I)
z=(1-τ)ε+τΔ
u=Δ-ε
û=F_c(z,τ,c_online)
Δ_hat=z+(1-τ)û
c_hat=c_online+Δ_hat
```

Losses:

```text
L_flow=MSE(û,u)
L_recon_pred=cosine(D(c_hat),e_future)
```

Gradient routes:

- `D` from reconstruction;
- `F_c` from both flow and reconstruction;
- `B` from condition and direct `c_online` add-back;
- neither EMA term nor `E`.

Copy baseline is `Δ_hat=0`, with loss `mean(Δ²)`.

## Trace 5 — exact remote resume to completion

### Before launch

1. local code/tests pass;
2. commit and push exact SHA;
3. remote fast-forward to that SHA;
4. verify persistent dataset/cache/checkpoint roots;
5. materialize dataset and encoder identities;
6. inspect checkpoint hash/schema/provenance;
7. resolve unique log/provenance output;
8. use checkpoint W&B ID with `resume=must`.

### Load

Before mutation, validate:

```text
encoder fingerprint
dataset fingerprint or explicit narrow transfer
run provenance
sampler state = function(next_step,count,batch)
module state keys/shapes
optimizer groups/state tensor shapes
mean/whitener identities
```

Then load online/EMA/flow/decoder, optimizer, optional buffers, RNG states. Suppose `next_step=7500`
on SSv2 tiny:

```text
epoch = 7500//62 = 120
batch index = 7500%62 = 60
batch_offset = 60*64 = 3840
```

The first resumed loader uses exactly that stored/validated position.

### Continue

At step 7,500, recompute peak LRs from config and multiply by schedule scale 0.586824. Step RNG is
`42*1,000,003+7500`. Diagnostics cannot change future draws.

### Finish

At next-step 15,000:

- atomically save `phase1_step15000.pt`;
- hash it;
- write/update run summary path+hash;
- finish W&B;
- verify zero process exit;
- verify final provenance and acceptance window.

Only this complete chain is a reproducible continuation.

## Trace rehearsal questions

For each trace, answer:

1. What tensor enters/exits each boundary?
2. Which operation owns normalization?
3. Which state receives gradient, EMA mutation, buffer mutation, or no mutation?
4. Which random generator chooses each stochastic value?
5. What identity would reject an incompatible cached/checkpointed object?
6. Which metrics determine the scientifically correct verdict?
