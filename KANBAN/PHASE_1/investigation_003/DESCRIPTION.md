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

Listed in W&B chronological order (the BRIEF "Run N" labels are a coarser narrative
overlay; verified configs are in each run's DESCRIPTION).

| Run | BRIEF | Role (verified vs W&B) |
|---|---|---|
| [`exalted-lion-6`](exalted-lion-6/) | P1 | **First** — tiny diagnostic; flag-based init ablation → init is a non-lever; pointed at VICReg-C |
| [`sleek-leaf-7`](sleek-leaf-7/) | **Run 2** | Full SSv2 baseline / VICReg Run A @3500; per-head metric exposes slot collapse (1.62) |
| [`serene-cloud-8`](serene-cloud-8/) | **Run 3** | k=4, **raw** slot loss 0.25 — early Goodhart |
| [`confused-butterfly-9`](confused-butterfly-9/) | — | Failed launch (1s, k=4 slot=0.25) |
| [`skilled-waterfall-10`](skilled-waterfall-10/) | **Run 4** | **k=4** (not 12 — launch drift), raw slot 0.05 **inert** → exposed loss/metric bug |
| [`olive-terrain-11`](olive-terrain-11/) | Run 5a | **First centered-slot k=12** run (slot=0.05) — Goodhart begins; killed @3900 |
| [`copper-sky-12`](copper-sky-12/) | **Run 5b** | Re-run of olive — Goodhart confirmed (rank→4.8, cosine 0.84) + grad spikes |
| [`cerulean-snow-13`](cerulean-snow-13/) | **Run 6** | **Win:** `lambda_var=0.5`, k=12, no slot |
| [`jolly-forest-14`](jolly-forest-14/) | Run 6b | Winning-config repeat (var=0.5, k=12); reproduced cerulean, crashed @3900 |

## Spawned

- [investigation_004](../investigation_004/) — VICReg-C path (Run A executed; Run B not run)
- [investigation_005](../investigation_005/) — complete 15k with winning config
