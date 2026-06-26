# Phase 1 — Research KANBAN

Phase 1 trains **coarse dynamics only**: frozen V-JEPA 2 encoder, trainable
bottleneck `B`, EMA bottleneck `B_EMA`, and coarse flow `F_c`, with a variance
floor on abstract latent `c_t`. The question is whether this stack learns
non-collapsed representations and beats copy/batch-mean baselines at a fixed
horizon.

**Authoritative spec:** [`AGENT_FILES/PHASES/PHASE_1.md`](../AGENT_FILES/PHASES/PHASE_1.md)

**History source:** [`AGENT_FILES/COMPLETE_FULL_CHAT`](../AGENT_FILES/COMPLETE_FULL_CHAT) + W&B run list (June 2026).

---

## Current status (June 2026)

| Investigation | Question | Status | Runs |
|---|---|---|---|
| [001](investigation_001/) | Can Phase 1 train without numerical blow-up? | **CLOSED** | 1 |
| [002](investigation_002/) | Is dataloader throughput sufficient for full SSv2? | **CLOSED** | 4 |
| [003](investigation_003/) | Why does `c_t` collapse? | **CLOSED** | 9 |
| [004](investigation_004/) | Is VICReg-C needed beyond `lambda_var=0.5`? | **PAUSED** | 1 (Run A) |
| [005](investigation_005/) | Can we finish the 15k acceptance run? | **ACTIVE** | 3 |
| [006](investigation_006/) | Does a reconstruction anchor break the rank ceiling? | **ACTIVE** | 2 (`fanciful-lake-18`, `easy-blaze-19`) |
| [007](investigation_007/) | What binds the ~0.60 reconstruction capacity floor? | **OPEN** | 0 (8-run sweep, pending launch) |

**Winning config:** full SSv2, `horizon_k=12`, `lambda_var=0.5`, no slot loss.

**Active work:** [`investigation_006`](investigation_006/) — reconstruction anchor. Run 1
(`fanciful-lake-18`, option 1) **removed the Mode-B cliff** but did not break the rank ceiling
(~13.3) or fix the copy gate. Run 2 (`easy-blaze-19`, option 3, predicted-latent anchor through
`F_c`) is a **negative result**: no prediction gain + mild representation harm, because the
reconstruction channel is **capacity-saturated at ~0.60** (`c` rebuilds only ~40% of `e` at
128:1) and so is blind to prediction quality. This spawned **[`investigation_007`](investigation_007/)** —
an 8-run, 8-GPU-parallel OFAT sweep over `lambda_recon` × decoder size × `n_c` to find what
**binds** that floor (weight-, decoder-, or capacity-bound). Design:
[`investigation_007/SWEEP_PLAN`](investigation_007/SWEEP_PLAN_decoder_capacity.md); execution:
[`investigation_007/GUIDE`](investigation_007/GUIDE.md). CLI flags implemented; pending launch.

---

## Complete W&B run index (hjepa-vwm)

All runs from project dashboard, in W&B creation order:

| # | Run name | ID | Runtime | Investigation | BRIEF |
|---|---|---|---|---|---|
| 1 | `youthful-pond-1` | x4pwz33d | 2m25s | 002 | smoke |
| 2 | `efficient-aardvark-2` | fz7ztfc8 | 5m16s | 002 | smoke |
| 3 | `comfy-glade-3` | 0mgmqxxi | 5m29s | 002 | smoke |
| 4 | `charmed-haze-4` | gj8ypv0d | 4m27s | 002 | smoke |
| 5 | `peachy-terrain-5` | 1chv2608 | 3h53m | 001 | **Run 1** |
| 6 | `exalted-lion-6` | wv69n7n5 | 14m18s | 003 | P1 diag |
| 7 | `sleek-leaf-7` | rpxyg9qt | 1h39m | 003 | **Run 2** |
| 8 | `serene-cloud-8` | dhp1i3fk | 1h32m | 003 | **Run 3** |
| 9 | `confused-butterfly-9` | m30jxiye | 1s | 003 | fail |
| 10 | `skilled-waterfall-10` | 27i1r9qi | 1h8m | 003 | **Run 4** (ran k=4, raw slot — inert) |
| 11 | `olive-terrain-11` | q40nq0l3 | 1h58m | 003 | Run 5a (first centered-slot k=12) |
| 12 | `copper-sky-12` | ejror834 | 2h24m | 003 | **Run 5b** (slot Goodhart confirmed) |
| 13 | `cerulean-snow-13` | 4lo4j7qb | 3h7m | 003 | **Run 6** ✓ |
| 14 | `jolly-forest-14` | 8bkeeuio | 1h32m | 003 | Run 6b (winning-config repeat, crashed @3900) |
| 15 | `elated-snowflake-15` | jhodg49x | 5h25m | 005 | 15k fail |
| 16 | `drawn-elevator-16` | 0n5mx3qf | 3h29m | 005 | resume fail |
| 17 | `royal-cherry-17` | 0xv4upvb | 4h52m | 005 | AGC resume — skip-free, rank collapse |
| 18 | `fanciful-lake-18` | yd5958s6 | 6h29m | 006 | recon anchor (λ=0.05) — cliff removed, rank ceiling held, copy gate failed |
| 19 | `easy-blaze-19` | 3syv6wp2 | 6h10m | 006 | option 3 (λ_pred=0.05) — negative: capacity floor blocks prediction gain |

Project URL: https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm

**Mapping notes (resolved via W&B MCP, 2026-06-25):** all run configs are now confirmed
against the W&B config panel. `olive-terrain-11` = first centered-slot k=12 run (slot=0.05,
cov=0.0027); `jolly-forest-14` = winning-config repeat (var=0.5, k=12, no slot), crashed
@3900. `skilled-waterfall-10` actually ran at **k=4** (launch drift) on the inert raw slot
loss. See each run's DESCRIPTION for the verified command/config.

---

## How to use this folder

Read [`../PROTOCOL.md`](../PROTOCOL.md) before editing. One investigation + one run
folder at a time unless cross-linking.
