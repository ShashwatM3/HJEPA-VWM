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
> written to teach a human. Where they overlap, the spec wins on numbers;
> these files win on explanation.

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
| [`06-COLLAPSE-AND-METRICS.md`](06-COLLAPSE-AND-METRICS.md) | Collapse modes, variance floor, the 5 diagnostics | How can self-supervised learning silently fail, and how do we catch it? |

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

---

## The 60-second version of the whole project

We are building a **video world model**: a system that watches a few frames
of video and predicts what the *state of the world* will be a moment later —
not the pixels, but a compact internal description.

The pipeline:

```
video frames
   → frozen pretrained encoder (V-JEPA 2)   → e_t  (1024 tokens — detailed)
   → trainable bottleneck                    → c_t  (32 tokens — abstract)
   → flow-matching predictor F_c             → predicted future c_{t+k}
```

Only the bottleneck and the predictor train. The target for prediction comes
from an EMA (slow-moving) copy of the bottleneck applied to the *future*
clip, with gradients blocked. A small variance floor keeps `c_t` from
collapsing to a constant. Five diagnostic metrics watch for subtler
failures. Phase 1 (now) proves the coarse level works; Phases 2–4 add a fine
predictor, a frame generator, and multi-horizon prediction.

Everything else in this curriculum is the detail behind that paragraph.

---

## Honest state of the project as of writing (2026-06-10)

- Phase 1 implementation: **done** and smoke-tested.
- Run 1 of Phase 1 training: **crashed at step ~10750** (gradient explosion
  at peak LR — see `10-FAILURE-MODES.md`, it's the best teaching material in
  the repo).
- Fixes landed: lower peak LR, tighter clipping, NaN skip-guard, 15k-step
  budget. Run 2 pending.
- Known open question: `c_effective_rank` stuck at ~5 of 256 — the bottleneck
  is using ~2% of its capacity. Whether this is a dataset-size artifact, an
  initialization artifact, or a missing-regularizer problem is the live
  research question (see `06-COLLAPSE-AND-METRICS.md` §6 and
  `03-BOTTLENECK.md` §7).
