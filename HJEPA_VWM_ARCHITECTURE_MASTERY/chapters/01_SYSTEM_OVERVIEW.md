# 01 — System overview

## The executable graph

```text
video file
   │
   ├── deterministic context window ── augment ── normalize ── frozen E ── e_t
   │                                                                  │
   │                                                                  ▼
   │                                                           online B ── c_t
   │                                                                  │
   │                                           ┌──────────────────────┼─────────────┐
   │                                           │                      │             │
   │                                           ▼                      ▼             ▼
   │                                      collapse losses        feature D      F_c condition
   │                                                               │             │
   │                                                               ▼             │
   │                                                           reconstruct e_t    │
   │                                                                             │
   └── deterministic target window ─── augment ── normalize ── frozen E ── e_{t+k}
                                                                        │
                                                                        ▼
                                                                 EMA B (no grad)
                                                                        │
                                                                        ▼
                                                                       c⁺
                                                                        │
                       ε, τ ── z_τ=(1-τ)ε+τc⁺ ──────────────────────────┤
                                                                        ▼
                                                                velocity target
                                                                   u=c⁺-ε
                                                                        │
                                    F_c(z_τ, τ, c_t) ───────────────────┘
                                               │
                                               ▼
                                      MSE(û_θ, u)
```

The two raw windows pass through the same frozen encoder weights. The target path differs only after
encoding: detailed target features enter `B_EMA`, while context features enter online `B`.

## A forward pass in 14 statements

1. The dataset selects one video and chooses a deterministic start for the current epoch and sample.
2. It asks Decord for only the union of frame indices needed by context and target.
3. It converts `uint8` RGB to float `[0,1]`, resizes, crops, and applies shared color jitter.
4. Context and target clips have shape `(B,8,3,256,256)`.
5. The frozen encoder applies its own normalization and emits `e_t` and, unless present-only,
   `e_{t+k}`.
6. Optional exact whitening maps detailed features into a standardized coordinate system.
7. Online `B` maps `e_t` to `c_t`.
8. In a no-gradient target path, `B_EMA` maps `e_{t+k}` to `c⁺`.
9. Collapse-control diagnostics/losses inspect the online abstract representation.
10. Flow matching samples `ε` and `τ`, constructs `z_τ`, and asks `F_c` for velocity `û_θ`.
11. `F_c` is conditioned on `c_t`, with classifier-free condition dropout during training.
12. `D` may reconstruct present detailed features from `c_t`; an optional prediction-reconstruction
    branch decodes the one-step endpoint estimate.
13. Weighted losses backpropagate through trainable modules, then AGC and global clipping constrain
    gradients.
14. A successful optimizer step updates online weights and then EMA-updates `B_EMA`; a skipped step
    does neither.

## What each learned module is responsible for

### Frozen encoder `E`

`E` supplies a rich, pinned feature coordinate system. It is not adapted to this task in Phase 1.
The encoder seam makes three families look identical downstream: all return a 3-D tensor
`(B,N_e,D_e)` and a machine-checkable `EncoderSpec`.

### Online bottleneck `B`

`B` is the representation learner. It must turn thousands of frozen detailed tokens into a small
ordered set of abstract slots without collapsing across examples, dimensions, or slots. It combines
local ConvNeXt mixing on the detailed lattice with learned-query cross-attention and latent
self-attention.

### Target bottleneck `B_EMA`

`B_EMA` is a slowly moving teacher for the bottleneck only. It prevents the flow target from moving
as abruptly as the online representation. It never receives gradients, and it is updated after—not
before—a successful optimizer step.

### Coarse flow `F_c`

`F_c` learns a conditional vector field in abstract space. Given a noisy/interpolated future latent,
a continuous flow time, and current abstract context, it predicts the velocity toward the future
abstract target. It is not an autoregressive Transformer, a categorical predictor, or a diffusion
noise-prediction model.

### Feature decoder `D`

`D` asks whether abstract slots retain enough information to reconstruct the frozen encoder's
detailed token grid. Its output is feature space. It has fixed 3-D sinusoidal output positions and
uses abstract slots as cross-attention memory.

## Which modules receive gradients?

| Signal | `B` | `F_c` | `D` | `B_EMA` | `E` |
|---|---:|---:|---:|---:|---:|
| flow loss, ordinary target | yes through condition | yes | no | no | no |
| flow loss, residual mode | yes through condition/add-back | yes | no | no | no |
| present reconstruction | yes | no | yes | no | no |
| predicted reconstruction | yes | yes | yes | no | no |
| variance/covariance/slot/SIG regularizers | yes | no | no | no | no |
| EMA update | copied update | no | no-gradient mutation | target | no |

“No gradient” does not always mean “never changes”: `B_EMA` changes by EMA assignment, and feature
statistics change by buffer updates.

## Operational boundaries

The architecture is only one layer of the system:

```text
dataset builders
    → manifests and identity
        → deterministic loader/sampler
            → model/loss step
                → diagnostics and W&B
                    → checkpoint + exact resume
                        → remote launch/monitor/recovery
```

The project treats provenance as part of correctness. A low loss from a run whose encoder revision,
dataset inventory, or resume lineage is unknown is not considered equivalent evidence.

## What is deliberately absent

The following do not exist in the current executable Phase 1 graph:

- a trainable or EMA target visual encoder;
- a fine-scale flow `F_e`;
- RGB, VAE-latent, or pixel generation;
- teacher-forced detailed-future prediction;
- an inference-time ODE/SDE integrator or rollout API;
- a future-offset/horizon embedding;
- multi-horizon batching;
- checkpoint upload to W&B.

The original brief discusses several of these. They belong to the future design and must never be
used to answer “what executes now?”

## One-step endpoint interpretation

Flow matching trains at arbitrary `τ`. If the predicted velocity were constant and exact along the
remaining path, an estimate of the endpoint is:

```text
c_hat = z_τ + (1 - τ) * û_θ(z_τ, τ, c_t)
```

The implementation uses this algebraic estimate for diagnostics and optional predicted-feature
reconstruction. It does not claim that this is a production sampler. Iterative numerical integration
is a future inference concern.

## The central research problem

The model must satisfy two different demands:

1. `B` must preserve a broad, non-collapsed abstract representation of the video.
2. `F_c` must genuinely use current context to predict change rather than copy the present latent,
   emit a batch prototype, or exploit overlap between windows.

History shows these are separable. Runs have achieved good latent rank and variance while the
predictor still tied a zero-residual/copy baseline. This is why the monitoring system evaluates
representation health and conditional prediction health independently.

## Scale at shipped defaults

For V-JEPA2 with `M=256`:

| Component | Parameters | Trainable? |
|---|---:|---:|
| frozen `E` | 325,971,328 | no |
| online `B` | 4,837,123 | yes |
| `B_EMA` | 4,837,123 | no-gradient EMA |
| `F_c` | 7,643,904 | yes |
| `D` | 2,172,160 | yes |
| trainable total | 14,653,187 | yes |
| checkpoint model bundle excluding `E` | 19,490,310 | mixed |

The frozen encoder dominates parameter count but does not create optimizer state. The flow model is
the largest trainable component at the shipped width; scaling bottleneck `M` changes that balance.

## Oral-exam answer: “Describe the architecture end to end”

> Each sample produces an eight-frame context clip and an eight-frame future-shifted clip. A pinned,
> frozen pretrained encoder maps both to detailed spatiotemporal tokens. The online bottleneck
> locally mixes the context token lattice, then uses learned abstract queries and three latent
> attention blocks to compress it into 32 slots of width 256. A no-gradient EMA copy of only this
> bottleneck maps target features to future slots. Coarse flow matching interpolates Gaussian noise
> to those future slots and trains a six-block AdaLN-Zero Transformer to predict the velocity while
> conditioning on present slots. Optional collapse penalties and a cross-attention feature decoder
> regularize the representation. AdamW updates the online bottleneck, flow, and decoder; successful
> steps then update the EMA target. Deterministic sampling, strict encoder/dataset fingerprints,
> complete RNG and sampler checkpointing, fixed-batch diagnostics, W&B provenance, and remote launch
> rules make the training result reproducible and interpretable.
