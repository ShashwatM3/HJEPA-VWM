# 16 — Planned future hierarchy

This chapter isolates design intent from executable code. Every item below is **PLANNED** unless
explicitly contrasted with current Phase 1.

## Why a hierarchy

The intended world model separates:

- coarse abstract dynamics: what changes at a semantic/structural level;
- fine detailed dynamics: how encoder-level detail evolves given the coarse future;
- frame generation: how predicted detailed state maps to visual output.

This creates a causal dependency:

```text
current video
→ current detailed e_t
→ current abstract c_t
→ predicted future abstract c_hat
→ predicted future detailed e_hat
→ predicted visual frame/latent
```

Each stage should prove itself before downstream generation can hide its failure.

## Current versus original brief

| Concern | Original brief target | Current Phase 1 |
|---|---|---|
| input | four 128×128 frames | eight 256×256 frames |
| encoder | smaller trainable online encoder | pinned frozen pretrained encoder |
| target encoder | EMA encoder | same frozen encoder, no copy |
| detailed latent | about `256×384` tokens/features | V-JEPA `1024×1024`; frame encoders `2048×768` |
| abstract latent | `32×256` | `32×256` shipped default |
| bottleneck target | teacher path | EMA copy of bottleneck only |
| coarse flow | six blocks | six blocks implemented |
| fine flow | eight blocks at detailed width ~384 | absent |
| frame generator | 12 blocks in VAE-latent space | absent |
| horizon input | later multi-horizon intent | absent |

The original numbers explain direction, not current code.

## Planned fine flow `F_e`

Conceptual contract:

```text
F_e(z_e, τ_e, e_t, future_abstract_condition) → detailed velocity
```

It would predict future detailed features conditioned on both current detail and the future abstract
state. The coarse future supplies global intent; current detailed tokens supply texture/local state.

## Teacher-forcing sequence

The safe curriculum is:

1. train/validate coarse representation and `F_c`;
2. train `F_e` first with stopped true future abstract target `stopgrad(c⁺)`;
3. once fine prediction works under teacher forcing, feed stopped predicted `stopgrad(c_hat)`;
4. compare degradation caused specifically by coarse prediction;
5. only later consider end-to-end gradient coupling.

Starting `F_e` on bad `c_hat` would confound fine-model capacity with coarse-predictor failure.

## Planned gradient boundary

During the isolated fine stage:

```text
L_fine must not update F_c
```

The future abstract condition is stopped. This preserves the meaning of the already-measured coarse
stage. If fine loss were allowed to reshape `F_c` immediately, an apparent fine improvement could
destroy copy-gate performance or change the abstract coordinate system.

The project can deliberately relax this later, but it should be an explicit joint-training phase.

## Planned frame/VAE-latent generator

The original plan places a generator downstream of predicted detailed features, likely operating in
a pretrained VAE latent space rather than raw pixels:

```text
G(noised frame latent, time, stopgrad(e_hat), optional context) → frame-latent velocity
```

Key safety boundary:

```text
frame-generation loss initially must not rewrite e_hat/F_e
```

Otherwise a powerful generator can compensate for a weak world model, making generated frames look
plausible without validating predicted representation dynamics.

## What current `D` is not

The feature reconstruction decoder cannot be promoted by renaming:

- it maps abstract slots directly to encoder features;
- it has no image/VAE latent input;
- it has no flow time;
- it is not generative over visual uncertainty;
- it exists as a representation-training anchor.

A future generator needs a new explicit module/interface and tests.

## Planned multi-horizon support

Current data `k` changes the pair while the model sees no horizon. Multi-horizon training requires:

1. sample or enumerate horizon per example;
2. include horizon in dataset/batch identity;
3. provide a horizon embedding to `F_c` and likely `F_e`;
4. bind supported horizon set to config/provenance;
5. evaluate copy/batch-mean gates separately by horizon;
6. define inference rollout units consistently with decoded FPS;
7. prevent mixed-horizon metrics from hiding failures.

`CoarseFlow` already has a natural place alongside slot/type embeddings, but no such parameter exists
today.

## Planned inference integration

Training predicts a vector field at random `τ`. Production inference requires a numerical path from
source noise to endpoint:

```text
dz/dτ = F(z,τ,condition)
z(0) ~ Normal
integrate τ: 0→1
```

Design decisions not yet implemented:

- Euler/Heun/RK/ODE solver;
- number and spacing of steps;
- deterministic seed API;
- conditional/unconditional guidance;
- latent normalization/whitening at boundaries;
- how to chain horizons;
- online versus EMA bottleneck for conditioning/targets;
- uncertainty sampling and evaluation.

The one-step diagnostic endpoint is not a substitute.

## Planned rollout semantics

An autoregressive rollout could:

1. encode observed context;
2. integrate `F_c` to a future abstract;
3. integrate `F_e` to future detail;
4. generate/decode a visual frame/clip;
5. decide whether the next step conditions on predicted latents or re-encoded generated frames;
6. repeat with a specified horizon.

Each choice changes error accumulation. Re-encoding generated output adds encoder/generator
distribution shift; pure latent rollout may drift away from decodable visual states.

## Required gates before adding a stage

### Coarse stage gate

- stable;
- `c` non-collapsed and effective rank above project target;
- temporal separation is not static;
- copy ratio ≤0.70;
- batch-mean ratio ≤0.50.

### Fine stage teacher-forced gate

- detailed-flow loss beats fixed baselines;
- predictions preserve token geometry/rank;
- reconstruction/probe readouts are video-specific;
- no gradient enters `F_c`.

### Predicted-condition fine gate

- quantify teacher-forced versus predicted-condition gap;
- attribute degradation to coarse error;
- maintain provenance for both conditions.

### Generator gate

- visual metrics/human evaluation;
- conditional dependence on predicted detail;
- no shortcut that ignores `e_hat`;
- stopped upstream gradients in isolated phase.

## Future checkpoint/provenance implications

New stages require:

- new schema or backward-compatible explicit keys;
- `F_e`/generator optimizer and EMA policies;
- solver/inference config;
- VAE/checkpoint immutable identity;
- fine/frame noise RNG streams;
- stage-specific sampler state;
- gradient-boundary tests;
- separate W&B acceptance panels;
- storage planning for much larger checkpoints.

Adding a module without extending provenance would make the scientific comparison incomplete.

## Future shape questions that must be answered

- Does `F_e` operate on flattened `N_e×D_e` or factorized `T_e×H_e×W_e`?
- How are V-JEPA tubelets aligned to desired future RGB frames?
- Are frame-encoder and V-JEPA fine flows architecture-shared or adapter-specific?
- Does detailed prediction use raw or whitened feature coordinates?
- Is `c_hat` one latent per clip horizon or a trajectory?
- Does the generator create one frame, eight-frame clip, or VAE video tensor?
- How is horizon represented across 12-FPS EGO and variable-timing SSv2?

The current repository does not answer these in executable code. An airtight engineer says “open
design decision” rather than inventing a generic ML convention.

## Architecture-preserving implementation order

```text
source-diverse diagnostic repair
→ finish external latent-shape evidence
→ obtain passing coarse-prediction recipe
→ specify/test F_e interface
→ teacher-forced fine flow
→ predicted coarse condition
→ specify/test generator
→ stopped-detail frame generation
→ explicit joint training
→ multi-horizon
→ numerical inference/rollout
```

This sequence keeps failure attribution local.
