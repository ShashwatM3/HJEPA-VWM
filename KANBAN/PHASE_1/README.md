# Phase 1 — Research KANBAN

Phase 1 trains **coarse dynamics only**: frozen V-JEPA 2 encoder, trainable
bottleneck `B`, EMA bottleneck `B_EMA`, and coarse flow `F_c`, with a variance
floor on abstract latent `c_t`. The question is whether this stack learns
non-collapsed representations and beats copy/batch-mean baselines at a fixed
horizon.

**Authoritative spec:** [`AGENT_FILES/PHASES/PHASE_1.md`](../AGENT_FILES/PHASES/PHASE_1.md)

**History source:** [`AGENT_FILES/COMPLETE_FULL_CHAT`](../AGENT_FILES/COMPLETE_FULL_CHAT) (full user–agent transcript, ~11k lines).

**Project context (later phases):**

| Phase | Adds | Spec |
|---|---|---|
| 1 | Coarse flow `F_c` | `AGENT_FILES/PHASES/PHASE_1.md` |
| 2 | Fine flow `F_e` + hierarchy tests | `AGENT_FILES/PHASES/PHASE_2.md` |
| 3 | VAE + frame generator `D` | `AGENT_FILES/PHASES/PHASE_3.md` |
| 4 | Multi-horizon coarse (`h_k`) | `AGENT_FILES/PHASES/PHASE_4.md` |

---

## Current status (June 2026)

| Investigation | Question | Status | Runs |
|---|---|---|---|
| [001](investigation_001/) | Can Phase 1 train without numerical blow-up? | **CLOSED** | 1 |
| [002](investigation_002/) | Is dataloader throughput sufficient for full SSv2? | **CLOSED** | 3 |
| [003](investigation_003/) | Why does `c_t` collapse (low rank / high cross-video cosine)? | **CLOSED** | 6 |
| [004](investigation_004/) | Is VICReg-C needed on top of a strong variance floor? | **PAUSED** | 1 (Run A only) |
| [005](investigation_005/) | Can we finish the 15k acceptance run with the winning config? | **ACTIVE** | 1 |

**Winning config (from investigation 003):** full SSv2, `horizon_k=12`, `lambda_var=0.5`, no slot loss, init fixes baked in.

**Active work:** Resume from pre-spike checkpoint (~7500) after `elated-snowflake-15` grad-skip death spiral at step 8500. See [investigation_005](investigation_005/).

---

## W&B run index (named in chat)

| Run name | W&B id | Investigation |
|---|---|---|
| `youthful-pond-1` | x4pwz33d | 002 |
| `efficient-aardvark-2` | fz7ztfc8 | 002 |
| `charmed-haze-4` | gj8ypv0d | 002 |
| `peachy-terrain-5` | 1chv2608 | 001 |
| `exalted-lion-6` | wv69n7n5 | 003 |
| `cerulean-snow-13` | (name only) | 003 |
| `elated-snowflake-15` | jhodg49x | 005 |

Runs 2–5 (collapse sequence) have **no W&B names in chat** — slug folders under investigation_003.

---

## How to use this folder

Read [`../PROTOCOL.md`](../PROTOCOL.md) before editing. One investigation + one run
folder at a time unless cross-linking.
