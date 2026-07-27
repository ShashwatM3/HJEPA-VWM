# Architecture from RGB to evidence

**Visual reference · audited 2026-07-25**

[Course home](../README.md) · [Tensor atlas](tensor-atlas.md) ·
[Gradient map](gradient-map.md) · [Training schedule](training-schedule.md) ·
[Full system chapter](../chapters/01_SYSTEM_OVERVIEW.md)

This page compresses the executable architecture, state transitions, and planned hierarchy into one
reference. Solid Mermaid nodes are implemented. Nodes explicitly labeled **PLANNED** are not
executable.

| Label | Meaning |
|---|---|
| **IMPLEMENTED / TRAINABLE** | executes in Phase 1 and receives optimizer gradients when its route is active |
| **FROZEN** | executes but has no trainable state transition |
| **EMA** | changes by assignment after a successful optimizer update, never by backpropagation |
| **PLANNED** | architecture intent that is absent from current code |
| **STOP** | detached target or mutation guard |

## Current Phase 1: paired inputs

```mermaid
flowchart LR
    A["Video file<br/>Decord · one thread"]
    B["Deterministic paired windows<br/>context: a+[0:14:2]<br/>target: a+k+[0:14:2]"]
    C["Shared raw transform<br/>resize · crop · brightness/contrast/saturation"]
    D["FROZEN E inputs<br/>(B,8,3,256,256) each"]
    A --> B --> C --> D
```

> **Temporal warning:** shipped `k=4` shares six of eight decoded frames. The first normal even
> non-overlapping horizon is `k=16`.

## Current Phase 1: online and teacher paths

```mermaid
flowchart LR
    XC["context clip"]
    XT["target clip"]
    E1["same frozen E"]
    E2["same frozen E"]
    EC["e_t<br/>V-JEPA: B×1024×1024<br/>frame encoder: B×2048×768"]
    ET["e_future<br/>same detailed geometry"]
    W1["optional fixed whitening"]
    W2["same optional fixed whitening"]
    B["trainable B<br/>2× ConvNeXt<br/>learned-query cross-attention<br/>3× latent refinement"]
    BE["EMA B copy<br/>no gradients"]
    CT["online c_t<br/>B×32×256"]
    CP["detached c_plus<br/>B×32×256"]

    XC --> E1 --> EC --> W1 --> B --> CT
    XT --> E2 --> ET --> W2 --> BE --> CP
    B -. "successful-step EMA assignment" .-> BE
```

Key facts:

- Production detailed geometry comes from the resolved `EncoderSpec`.
- Whitening, when enabled, sits at the common detailed-feature seam and is seen by every consumer.
- `B_EMA` is a deep copy of `B`, updated only after successful AdamW.
- There is no target encoder. The shared encoder `E` is already frozen; the EMA exists only at `B`.

## Three learning surfaces

| Surface | Core computation | Direct gradient destinations |
|---|---|---|
| geometry | variance floor, optional covariance, slot diversity, SIGReg on online `c_t` | `B` |
| dynamics | `z=(1-τ)ε+τc_plus`; `u=c_plus-ε`; `F_c(z,τ,c_t)→u_hat` | `F_c`; `B` through live condition |
| content | `D(c_t)→e_hat_t` or `D(c_hat)→e_hat_future` | present: `D,B`; predicted: `D,F_c,B` |

Targets from `E` and `B_EMA` are detached. A metric or target may participate in a forward
calculation without becoming a gradient destination.

## Inside coarse flow

```mermaid
flowchart LR
    Z["z stream<br/>z_tau + slot_pos + z_type<br/>32×D_c"]
    C["condition stream<br/>c_t or null + slot_pos + condition_type<br/>32×D_c"]
    CAT["concatenate<br/>64×256"]
    ADA["6× AdaLN-Zero blocks<br/>8-head self-attention + MLP<br/>time supplies six shift/scale/gates"]
    OUT["velocity<br/>LayerNorm(first 32) → u_hat"]
    Z --> CAT
    C --> CAT
    CAT --> ADA --> OUT
```

Condition dropout replaces an entire example's condition stream with the learned null condition.
The `z` and condition type identities prevent concatenation from erasing stream roles.

## Successful state transition

```mermaid
flowchart LR
    L["backward<br/>all active objectives"]
    A["AGC<br/>B .20 · F_c .10 · D .20"]
    G["global clip<br/>max norm .5"]
    S{"nonfinite or<br/>preclip norm >150?"}
    X["zero gradients<br/>no AdamW · no EMA"]
    O["AdamW<br/>B · F_c · D with gradients"]
    M["EMA<br/>B_EMA ← mB_EMA+(1-m)B"]
    L --> A --> G --> S
    S -- yes --> X
    S -- no --> O --> M
```

The optional residual-reconstruction mean is a mutable buffer updated before backward. A later
gradient skip does not roll that buffer update back.

## Full experiment architecture

```mermaid
flowchart LR
    BM["builder manifests<br/>SSv2/EGO selection and chunks"]
    DI["dataset identity<br/>paths · sizes · frame counts · hashes"]
    TS["training state<br/>model · EMA · optimizer · buffers · RNG · sampler"]
    DG["diagnostics<br/>stability · collapse · rank · copy · content honesty"]
    EV["evidence<br/>W&B · provenance · checkpoint path/SHA"]
    BM --> DI --> TS --> DG --> EV
```

The model graph alone is not the experiment. Dataset/encoder/whitening identities, exact sampler
position, RNG state, tracking identity, and durable checkpoint evidence are architectural state.

## Planned hierarchy — not executable

```mermaid
flowchart LR
    C["PLANNED<br/>coarse integration<br/>numerically integrate F_c"]
    F["PLANNED<br/>fine flow F_e<br/>true stopped c_plus, then stopped c_hat"]
    P["PLANNED<br/>frame/VAE generator<br/>condition on stopped e_hat"]
    R["PLANNED<br/>rollout<br/>multi-horizon · solver · guidance · chaining · uncertainty"]
    C -.-> F -.-> P -.-> R
```

Current `D` reconstructs frozen encoder features. It is not the planned frame/VAE generator.
Current `c_hat` is a one-step algebraic endpoint estimate, not a numerical inference rollout.

## Memory sentence

> Frozen detail → trainable abstract slots → EMA future target → conditional abstract flow →
> optional feature reconstruction, enclosed by deterministic data and strict experiment identity.

Sources: current [data](../../data.py), [encoders](../../encoders.py),
[models](../../models.py), [training](../../train.py), and the
[audited source ledger](../SOURCE_LEDGER.md).
