# 08 — Training step and schedules

## Shipped optimizer schedule

| Setting | Value |
|---|---:|
| optimizer | AdamW |
| betas | `(0.9,0.95)` |
| weight decay | 0.05 on eligible matrices |
| batch size | 64 |
| Phase 1 steps | 15,000 |
| warmup | 1,500 |
| peak LR `B` | `1e-4` |
| peak LR `F_c` | `2e-4` |
| peak LR `D` | `1e-4` |
| global clip | 0.5 |
| skip threshold | 150 |

Each module's decay/no-decay groups share its peak LR.

## Learning-rate multiplier

```text
if step < 1500:
    scale = max(1e-8,(step+1)/1500)
else:
    p = min((step-1500)/(15000-1500),1)
    scale = 0.5*(1+cos(πp))
```

`step` is zero-based. The loop's default updates are 0 through 14,999.

| Step | LR scale | `B/D` LR | `F_c` LR |
|---:|---:|---:|---:|
| 0 | 0.000666667 | `6.6667e-8` | `1.3333e-7` |
| 1,499 | 1.000000 | `1.0e-4` | `2.0e-4` |
| 1,500 | 1.000000 | `1.0e-4` | `2.0e-4` |
| 2,000 | 0.996619 | `9.9662e-5` | `1.9932e-4` |
| 2,500 | 0.986522 | `9.8652e-5` | `1.9730e-4` |
| 5,000 | 0.843121 | `8.4312e-5` | `1.6862e-4` |
| 7,500 | 0.586824 | `5.8682e-5` | `1.1736e-4` |
| 10,000 | 0.301960 | `3.0196e-5` | `6.0392e-5` |
| 12,500 | 0.082256 | `8.2256e-6` | `1.6451e-5` |
| 14,500 | 0.003381 | `3.3808e-7` | `6.7616e-7` |
| 14,950 | 0.000033846 | `3.3846e-9` | `6.7692e-9` |
| 14,999 | `1.3539e-8` | `1.3539e-12` | `2.7078e-12` |
| 15,000+ | 0 | 0 | 0 |

The `--steps` CLI override changes loop length, not `stage1_steps`. A 2,000-step run is a prefix of
the 15,000-step schedule; a run beyond 15,000 has zero learning rate unless the schedule config is
also changed.

## Exact training-step order

### 1. Seed this step

```text
step_seed = base_seed*1,000,003 + step
```

Torch CPU and all CUDA RNGs are reset to it. Thus flow noise, `τ`, and condition dropout are pure
functions of seed and step, independent of prior diagnostic calls.

### 2. Transfer inputs

Context moves to the device non-blocking. Target moves only in full-prediction mode. Present-only
mode deliberately avoids the target transfer and encoder pass.

### 3. Clear gradients

```text
optimizer.zero_grad(set_to_none=True)
```

`None` gradients also ensure inactive decoder/flow parameters are skipped by AdamW.

### 4. Enter autocast

On CUDA with shipped BF16 settings, the trainable stack and encoder run under bfloat16 autocast,
except explicit FP32/FP64 numerical islands such as SIGReg and whitening.

### 5. Encode and bottleneck

Full prediction:

```text
e_t      = no_grad(E(context))
c_t      = B(e_t)
e_future = no_grad(E(target))
c⁺       = no_grad(B_EMA(e_future))
```

Present-only:

```text
e_t = no_grad(E(context))
c_t = B(e_t)
```

If enabled, whitening is applied to encoder outputs at this seam.

### 6. Construct the flow problem

In full mode, construct ordinary or residual target, draw `ε` and `τ`, interpolate, call `F_c`, and
compute mean squared flow loss. Present-only assigns a scalar zero.

### 7. Update the reconstruction mean

If residual reconstruction is active, update its running mean from current training `e_t` before
forming this step's target.

### 8. Compute representation terms

Variance, covariance, slot diversity, and SIGReg are all evaluated. SIGReg has a dedicated
per-step generator so its optional calculation cannot shift the main RNG.

### 9. Assemble weighted loss

Apply active loss weights and linear ramps. Invoke `D` only for training reconstruction objectives
that are enabled.

### 10. Backward

One `loss.backward()` constructs all trainable gradients.

### 11. Clip

Module-specific AGC acts first; global norm clip acts second.

### 12. Decide survivability

Use the post-AGC/pre-global-rescale norm returned by global clipping. Non-finite or `>150` skips the
entire state transition.

### 13. Commit

On a valid step:

```text
optimizer.step()
B_EMA ← EMA(B)
```

The function reports metrics either way. The outer loop advances its global iteration; checkpoints
record completed updates/next-step semantics so resume remains explicit.

## Present-only mode

This is a representation-capacity experiment, not a dynamics run.

- target clip is not required;
- target encoder/bottleneck path is skipped;
- `F_c` is not called;
- base loss is graph-connected zero;
- `lambda_recon` must be positive;
- prediction residual and prediction reconstruction are invalid;
- variance/covariance/SIG/slot terms can still act.

Metrics expose `present_recon_only=1` and `prediction_active=0`. Do not use its excellent rank as
evidence of successful future prediction.

## EMA and loss ramps are different schedules

| Schedule | Denominator | Shape | At step 0 |
|---|---:|---|---:|
| LR warmup/decay | 1,500 then 13,500 | linear then cosine down | `1/1500` |
| recon/SIG ramp | 2,000 | linear up | 0 |
| EMA momentum | 105,000 | cosine up | 0.996 |

They share a step counter but not an endpoint.

## Logging and checkpoint cadence

Shipped:

```text
log_every = 50
diag_every = 500
checkpoint_every = 2500
```

Over 15,000 updates:

- 300 ordinary log points;
- 30 diagnostic points;
- every diagnostic step is also a log step;
- scheduled checkpoint next-step values: 2,500, 5,000, 7,500, 10,000, 12,500, 15,000;
- final save targets the 15,000 path again, atomically replacing it with final state.

The code uses next-step semantics for checkpoint filenames/state. “Checkpoint 2500” means updates
0–2499 are complete and update 2500 is next.

## Fixed diagnostic batch and RNG isolation

The validation batch:

- is constructed once;
- uses up to 16 examples and must contain more than one;
- is reused at every diagnostic cadence;
- runs under a deterministic forked RNG seeded from `base_seed*1,000,003+900001`;
- restores the training RNG afterward;
- forces no condition dropout.

This turns metric changes into model changes rather than validation-batch noise. It also means a
biased fixed batch can bias every diagnostic; EGO's historical same-source batch is a known example.

## Default work volume

At batch 64 and 15,000 steps:

```text
960,000 example presentations
7,680,000 context RGB frames
7,680,000 target RGB frames in full mode
```

These counts include repeated dataset items across epochs and overlap inside pairs. Present-only
halves the pair-level encoder window count.

## Training configuration in one block

```text
batch=64, steps=15k, warmup=1.5k
AdamW β=(0.9,0.95), wd=0.05
LR: B=1e-4, Fc=2e-4, D=1e-4
AGC: B=.20, Fc=.10, D=.20, eps=.001
global clip=.5, skip>150
warn if grad>30 and L_flow>1
EMA=.996→.9999 over 105k
λ_var=.1; all other optional weights=0
recon/SIG ramps=2k
```

## Schedule traps

- Warmup step zero is not zero LR.
- Reconstruction/SIG step zero is zero weight.
- `--steps` does not redefine cosine decay.
- EMA does not reach its configured end during default Phase 1.
- A skipped update also skips EMA, but the training loop still needs unambiguous next-step
  accounting.
- Post-clip gradient diagnostics near 0.5 are expected and do not reveal the raw spike magnitude.
