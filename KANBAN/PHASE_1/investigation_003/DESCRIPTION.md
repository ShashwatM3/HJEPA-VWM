# Investigation 003 — Why does `c_t` collapse (low rank / video-agnostic codes)?

**Status:** CLOSED  
**Opened:** 2026-06-10 (after `peachy-terrain-5` showed `c_effective_rank ~5`)  
**Closed:** 2026-06 (after `cerulean-snow-13` validated `lambda_var=0.5` + `horizon_k=12`)

## Question

Why does abstract latent `c_t` use only ~2% of its 256 dimensions (`c_effective_rank ~5`),
and why does `c_cross_video_cosine` climb toward video-independent codes — despite
healthy variance-floor metrics on some runs?

## Why it matters

Phase 1 acceptance requires non-collapsed `c_t` and `F_c` beating copy baseline.
Low rank means the bottleneck is not carrying future-relevant structure; Phase 2
(fine flow hierarchy) is pointless on a collapsed coarse state.

## Parent context

- Branched from: [investigation_001](investigation_001/) (`peachy-terrain-5` rank ~5)
- Plan docs: [`AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/`](../../AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/)
- Spec: [`AGENT_FILES/PHASES/PHASE_1.md`](../../AGENT_FILES/PHASES/PHASE_1.md) §12 collapse gates

## Runs in this investigation

| Run | BRIEF | Role |
|---|---|---|
| [`exalted-lion-6`](exalted-lion-6/) | P1 | **First** — tiny diagnostic; slot redundancy + feature correlation dominant |
| [`sleek-leaf-7`](sleek-leaf-7/) | **Run 2** | Full SSv2 baseline / VICReg Run A @3500 |
| [`confused-butterfly-9`](confused-butterfly-9/) | — | Failed launch (1s) |
| [`serene-cloud-8`](serene-cloud-8/) | **Run 3** | Aggressive slot loss + k=4 — Goodhart |
| [`skilled-waterfall-10`](skilled-waterfall-10/) | **Run 4** | k=12 slot loss inert (loss/metric mismatch) |
| [`olive-terrain-11`](olive-terrain-11/) | — | Intermediate k=12 attempt (config TBD on W&B) |
| [`copper-sky-12`](copper-sky-12/) | **Run 5** | Centered slot loss — Goodhart + grad spikes |
| [`jolly-forest-14`](jolly-forest-14/) | — | Intermediate pre-elated run (config TBD) |
| [`cerulean-snow-13`](cerulean-snow-13/) | **Run 6** | **Win:** `lambda_var=0.5`, no slot loss |

## Spawned

- [investigation_004](investigation_004/) — VICReg-C path (Run A executed; Run B not run)
- [investigation_005](investigation_005/) — complete 15k with winning config
