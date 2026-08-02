# LEARN/ — The HJEPA-VWM curriculum

> **Who this is for.** You. You have an ML background (you know what a tensor,
> a loss, a gradient, and a transformer are) but no assumed knowledge of the
> specific concepts this project uses. Every concept is taught where it
> appears, with the *why* before the *what*.
>
> **What this is.** A topic-by-topic walk through the entire project — the
> architecture, the reasoning behind every component, and the ML operations
> (training recipes, schedules, logging, metrics, failure modes, cost math).
> The goal is intuition: after reading, you should be able to reason about
> questions like "should we add SIGReg back?" or "how would multi-horizon
> change the pipeline?" without asking anyone.
>
> **How it relates to other docs.** `AGENT_FILES/KNOWLEDGE/` is the *spec* —
> written to constrain a coding agent. These files are the *textbook* —
> written to teach a human. Where they overlap, **`BRIEF_V0_3.md` wins on
> Phase 1 operating numbers** (validated on real runs); `UNDERSTANDING.md`
> §2.6 wins on architecture shapes; `config.py` defaults are not always the
> same as the operating CLI flags — see file 07. These files win on
> explanation.
>
> **Prerequisite concepts** (JEPA, flow matching, ViTs, EMA targets, etc.)
> are taught in [`../LEARN_THEORY/`](../LEARN_THEORY/00-INDEX.md) — general
> ML knowledge, not codebase-specific. Read that folder first if any concept
> in Part I feels unfamiliar.

---

## Reading order

The files build on each other. Read in order the first time; jump around
afterwards.

### Part I — What we're building (architecture)

| File | Topic | The question it answers |
|---|---|---|
| [`01-BIG-PICTURE.md`](01-BIG-PICTURE.md) | World models, JEPA, latent prediction | Why predict latents instead of pixels? What is the two-level hierarchy for? |
| [`02-FROZEN-ENCODER.md`](02-FROZEN-ENCODER.md) | ViTs, V-JEPA 2, tubelets, freezing | Where do the 1024 tokens come from? Why frozen? Why this specific model? |
| [`03-BOTTLENECK.md`](03-BOTTLENECK.md) | ConvNeXt, cross-attention queries, c_t | How do 1024 tokens become 32 slots? Why each sub-module exists. |
| [`04-FLOW-MATCHING.md`](04-FLOW-MATCHING.md) | Rectified flow, velocity fields, conditioning | How do we predict the future? Why not a plain MSE predictor? |
| [`05-EMA-AND-STOP-GRAD.md`](05-EMA-AND-STOP-GRAD.md) | Self-supervised targets, EMA, stop-gradient | Why does the target come from a slow copy of the bottleneck? |

### Part II — Keeping it honest (collapse & measurement)

| File | Topic | The question it answers |
|---|---|---|
| [`06-COLLAPSE-AND-METRICS.md`](06-COLLAPSE-AND-METRICS.md) | Collapse modes, variance floor, the 7 diagnostic families | How can self-supervised learning silently fail, and how do we catch it? |

### Part III — How we train it (MLOps)

| File | Topic | The question it answers |
|---|---|---|
| [`07-TRAINING-RECIPE.md`](07-TRAINING-RECIPE.md) | LR, warmup, cosine decay, AdamW, clipping, bf16, batch/steps/epochs | What is every number in `TrainConfig` and why is it that value? |
| [`08-DATA-PIPELINE.md`](08-DATA-PIPELINE.md) | Video decoding, dataloaders, CPU/GPU bottlenecks, throughput | Why was training slow, what did we fix, what's the cost math? |
| [`09-EXPERIMENT-OPS.md`](09-EXPERIMENT-OPS.md) | W&B, logging cadence, checkpoints, tmux, RunPod, resume | How do we run, watch, and recover a multi-hour training job? |

### Part IV — When it breaks & where it goes (judgment)

| File | Topic | The question it answers |
|---|---|---|
| [`10-FAILURE-MODES.md`](10-FAILURE-MODES.md) | Gradient explosions, NaN, precision, overtraining — Run 1 as a case study | What killed Run 1, mechanically, and what are the general lessons? |
| [`11-PHASES-AND-MULTI-HORIZON.md`](11-PHASES-AND-MULTI-HORIZON.md) | Phases 1–4, acceptance gates, h_k | What's the roadmap? How does multi-horizon actually work? |
| [`12-GLOSSARY.md`](12-GLOSSARY.md) | Every term in one place | Quick lookup when you forget what something means. |
| [`13-FRAMES-TUBELETS-AND-HORIZON.md`](13-FRAMES-TUBELETS-AND-HORIZON.md) | Frames, tubelets, and horizon offsets | What exactly is the context clip, target clip, and `horizon_k` in frame/tubelet terms? |

---

## The 60-second version of the whole project

We are building a **video world model**: a system that watches a few frames
of video and predicts what the *state of the world* will be later — first as
compressed latents, then (Phase 3) as watchable pixels.

**Intended architecture (full v0 — Phases 1–3, plus Phase 4 multi-horizon):**

```
                         ┌──────── FROZEN encoder E (V-JEPA 2 ViT-L/16) ────────┐
context clip x_{≤t}   ──►  E  ──► e_t  (1024 tokens × 1024 dims — detailed)     │
                         └───────────────────────────────────────────────────────┘
                                            │
                                            ▼
                              bottleneck B  ──► c_t  (32 × 256 — abstract)
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         │                                  │                                  │
         ▼                                  │                                  │
  F_c — coarse flow                         │     TARGET (stop-grad + EMA):    │
  rectified flow: noise → c⁺                │     future clip x_{≤t+k}         │
  conditioned on c_t                        │       ──► E ──► e⁺ ──► B_EMA     │
         │                                  │              ──► c⁺_{t+k}        │
         ▼                                  │                                  │
  predicted abstract future ĉ⁺              │                                  │
         │                                  │                                  │
         ▼                                  │                                  │
  F_e — fine flow [Phase 2]                 │                                  │
  rectified flow: noise → e⁺                │                                  │
  conditioned on (e_t, stopgrad(ĉ⁺))       │                                  │
         │                                  │                                  │
         ▼                                  │                                  │
  predicted detailed future ê⁺              │                                  │
         │                                  │                                  │
         ▼                                  │                                  │
  D — frame generator [Phase 3]             │                                  │
  flow in frozen VAE latent space           │                                  │
         │                                  │                                  │
         ▼                                  │                                  │
  predicted future frames (pixels)          │                                  │
         └──────────────────────────────────┴──────────────────────────────────┘

Phase 4: same stack, but F_c takes a horizon embedding h_k — one shared
predictor, k ∈ {4, 8, 16, 32} (file 11).
```

Three flow-matching networks (`F_c`, `F_e`, `D`) share one recipe (file 04).
The hierarchy is the bet: abstract `c` steers detailed `e`, which steers
pixels — verified by bypass tests (shuffled-c, shuffled-e), not assumed.

**Cross-cutting machinery (all latent-training phases):** targets from an EMA
copy of the bottleneck on the *future* clip, with gradients blocked on the
target branch; variance floor on `c_t`; seven diagnostic metric families.

**What's implemented in code today:** Phase 1 only — `E`, `B`, `B_EMA`, `F_c`.
`F_e`, `D`, and `h_k` are spec'd but not built. See § below.

Everything else in this curriculum is the detail behind this diagram.

---

## Honest state of the project as of writing (2026-06-23)

- **Phase / stage:** Phase 1, Stage 1 only — coarse dynamics (`B`, `B_EMA`,
  `F_c`). `F_e`, `D`, and multi-horizon `h_k` are not built yet.
- **Code:** Implemented at repo root (`train.py`, `models.py`, `losses.py`,
  `diagnostics.py`, `data.py`, `config.py`); not a planning-only repo.
- **Investigations** (see [`KANBAN/PHASE_1/README.md`](../../KANBAN/PHASE_1/README.md)):
  - **001** CLOSED — Run 1 (`peachy-terrain-5`) gradient explosion at step
    ~10750 (file 10).
  - **002** CLOSED — dataloader throughput sufficient for full SSv2.
  - **003** CLOSED — collapse / low rank; **winning config:** full SSv2,
    `--horizon-k 12`, `--lambda-var 0.5`, no slot loss; best run
    **`cerulean-snow-13`** (rank ~13.7+, cosine ~0.20, beats copy baseline).
  - **004** PAUSED — VICReg-C (`--lambda-cov`) not needed yet given strong
    variance floor.
  - **005** ACTIVE — complete the **15k acceptance run**; `elated-snowflake-15`
    hit a grad-skip death spiral at step 8500; **`drawn-elevator-16`** resume
    from pre-spike checkpoint — verify outcome on W&B.
- **15k acceptance:** Not cleanly complete. Short-window success on
  `cerulean-snow-13`; full 15k failed once; resume in progress.
- **Operating vs default hyperparameters:** `config.py` defaults
  `horizon_k=4`, `lambda_var=0.10`. Validated operating values are
  **`--horizon-k 12 --lambda-var 0.5`** (file 07). Recommended launch:
  `python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5`
- **Open tensions:** `c_effective_rank` ~13–14 vs spec soft-target >60;
  whether gentle `lambda_cov` helps; optimizer stability late in long runs
  (file 10 §7 — a second failure mode distinct from Run 1).
