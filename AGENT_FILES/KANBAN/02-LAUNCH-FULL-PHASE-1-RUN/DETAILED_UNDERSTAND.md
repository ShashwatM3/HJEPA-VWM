# 02 — Launch the full Phase 1 training run

> **Run 1 failed** (2026-06-10, `peachy-terrain-5`, step ~10750, gradient
> explosion at peak LR). Hyperparameters retuned in commit `<TBD>`; this
> folder now describes the **Run 2 / 15k-step** launch. The original
> 30k-step plan is preserved in `PHASE_1.md` §10 as the intended *full*
> Phase 1 budget, but is no longer the default for ssv2_tiny. See
> [`POSTMORTEM_RUN1.md`](POSTMORTEM_RUN1.md) for the full diagnosis.

## What this task is

Execute the **15,000-step** Phase 1 training run (revised from the original
30k spec). The aim is the same as PHASE_1.md §10–§12: produce a checkpoint
plus acceptance-gate measurements that mark the project's **Phase 1 complete**.
The 15k budget is a deliberate compromise — see "Why 15k not 30k" below.

> **Note on naming.** "Phase 1" in this folder name refers to the project's
> Phase 1 — Stages 0 + 1 of the architecture (frozen V-JEPA 2 encoder,
> trainable bottleneck `B`, EMA bottleneck `B_EMA`, coarse flow `F_c`,
> variance floor on `c_t`). This is *not* the "Plan Phase" numbering used by
> the KANBAN folder names. Both `01-OPTIMIZE-DATALOADER` and this folder are
> KANBAN phases of the operational plan; the actual training they launch is
> the project's Phase 1.

## Why this is being done

### What 200-step smoke runs proved (and didn't prove)

The previous smoke runs (`--steps 100` then `--steps 200`) validated:

- Encoder loads and stays frozen.
- Dataloader produces real video batches (no decord crashes, after the VP9
  `num_threads=1` fix).
- Forward + backward + optimizer step + EMA update all run without NaN.
- `L_var` drops over 200 steps — the variance floor is becoming inactive as the
  bottleneck spreads `c_t` across dimensions.

What they did **not** prove:

- That `L_flow` actually trains. At step 199, `lr_mult = 0.02` — we are at 2%
  of full LR. The flow predictor `F_c` has not meaningfully been updated.
- That `coarse_vs_copy_ratio` drops below 1.0. The Phase 1 acceptance gate
  (PHASE_1.md §12) requires this ratio to be **≤ 0.70** at step ≥ 10k.
- That `c_t` is genuinely future-relevant and not memorising frame statistics.

The 30k run is what produces this evidence.

### What this run delivers

| Artefact | Where |
|---|---|
| 5 intermediate checkpoints | `/workspace/checkpoints/phase1_step{2500,5000,7500,10000,12500}.pt` |
| 1 final checkpoint | `/workspace/checkpoints/phase1_step15000.pt` |
| ~300 logged training steps on W&B | hjepa-vwm project, run URL printed at launch |
| 30 diagnostic snapshots | one every 500 steps, on the fixed `val_batch` |
| Acceptance-gate-grade metrics | PHASE_1.md §12 values, measurable from W&B, **interpreted on the 15k-equivalent scale** (see below) |

## Why 15k not 30k

The 30k+10k-warmup spec in PHASE_1.md §10 was calibrated for the full SSv2
corpus (~170k clips). On `ssv2_tiny` (~4k clips), 30k steps × 64 batch =
**480 epochs of the subset** — extreme overtraining, and the cause of
Run 1's instability (Adam variance estimate drifted; pre-clip gradients
spiked into the 24–27 range around step 8.7k; peak LR at step ~10550
triggered explosion in bf16). The shorter 15k run:

- Stays within the same architecture and design (no code or loss changes).
- Reaches near-peak LR by step 1.5k (vs step 10k), so any instability shows
  up early — within 30–45 min — rather than 3 hours in.
- Halves wall-clock (~5 h instead of ~10 h) and pod cost (~$12 instead of
  ~$24).
- Tests the same hypothesis: *does this architecture learn coarse dynamics
  on real video without diverging or collapsing?*

A successful 15k run on `ssv2_tiny` justifies a 30k run on full SSv2 as a
follow-up (a separate KANBAN plan phase).

## Success criteria / goals (interpreted on the 15k scale)

The PHASE_1.md §12 thresholds were sized for the 30k trajectory. On 15k,
treat them as "PASS-equivalent" targets — the *direction* and the
*magnitude relative to where we'd be on 30k* matter more than meeting the
exact value:

| Gate (PHASE_1.md §12) | 30k threshold | 15k PASS-equivalent (Run 2 target) |
|---|---|---|
| Stage 1 stability | 30k steps complete without OOM/NaN | 15k steps complete without OOM/NaN; `grad_skipped` total = 0 |
| `coarse_vs_copy_ratio` (val) | ≤ 0.70 after step ≥ 10k | **trending under 1.5 by step 12500; < 1.0 = strong** |
| `coarse_vs_batch_mean_ratio` (val) | ≤ 0.50 after step ≥ 10k | < 0.80 by step 12500 |
| `c_effective_rank` | > 60 on val | **> 30 at step 12500** |
| `c_t` variance health | < 15% dims with std < 0.1× median | unchanged |
| `c_cross_video_cosine` | well below ~0.5; no drift toward 1.0 | unchanged |
| Gradient health | no repeated NaN; grad norm logged | **`grad_skipped` total = 0; no NaN** |
| Final checkpoint loadable | `torch.load(/workspace/checkpoints/phase1_step15000.pt)` works | unchanged (path adjusted) |

## Pre-flight checklist (before launch)

The launch command is short, but four things must be true first:

1. **Plan Phase 01 complete** — `s/step` is acceptable, you have the timing
   number recorded, and the pod is on the latest commit of
   `phase1-v0.2-frozen-encoder`.
2. **`ssv2_tiny` exists** at `/workspace/data/ssv2_tiny/{train,validation}` with
   ~4k / ~350 symlinks (see [`../../SETUPS/SETUP.md`](../../SETUPS/SETUP.md) A12).
3. **W&B is logged in** on the pod — the smoke runs already showed it is
   (`wandb: Currently logged in as: smahalanobis`).
4. **`HF_HOME` is set to `/workspace/hf_cache`** so the V-JEPA 2 checkpoint
   persists on the network volume across pod restarts.

## Risks and stop conditions

| Symptom (during run) | Likely cause | Action |
|---|---|---|
| `grad_has_nan = 1` at any diagnostic | numerical blowup | Abort immediately; debug; do not auto-resume |
| **`grad_skipped > 0` (new in Run 2)** | a single batch produced pre-clip grad above the skip threshold (50) | Once is tolerable (the skip guard ate it); >2 across the run = stop and investigate, likely instability |
| Pre-clip `grad_norm` repeatedly > 10 | precursor to explosion | Stop early; tighten `grad_clip` further; halve peak LR; restart |
| `coarse_vs_copy_ratio` > 5 at step ≥ 7500 | F_c not learning future structure | Stop, screenshot W&B, escalate to tech lead before spending more pod time |
| `L_var` rising sharply mid-run | variance floor failing; possible directional collapse | Stop, inspect `c_cross_video_cosine` trajectory |
| Pod disconnects with "Zero GPU" | RunPod issue ([troubleshooting](https://docs.runpod.io/pods/troubleshooting/zero-gpus)) | Per [`../../SETUPS/SETUP.md`](../../SETUPS/SETUP.md) FAQ — terminate pod, deploy new pod with the same network volume, resume from latest checkpoint via `--resume /workspace/checkpoints/phase1_stepXXXXX.pt` |
| Out-of-memory at runtime | Unlikely with frozen encoder + batch 64; possible if pod RAM is tiny | Halve `global_batch` via a temporary config edit; restart from latest checkpoint; flag to tech lead |
| W&B run shows healthy curves through step 12.5k+ | normal | let it finish |

## Cost estimate

At RunPod A100 80GB pricing (approx. $2–3 / hr at time of writing) on the
**15k-step run** with the dataloader-decode optimization landed in Plan
Phase 01:

| s/step (post-Plan-Phase-01) | 15k wall-clock | Pod cost |
|---|---|---|
| 1.25 (observed in Run 1 — the realistic number) | ~5.2 h | **$11–16** |
| 1.0 (warm-cache best case) | ~4.2 h | $9–13 |
| 0.6 (after Plan Phase 03 — not yet executed) | ~2.5 h | $5–8 |

## How this fits the bigger picture

Phase 1 is the *first* of four project phases ([`../../AGENTS.md`](../../AGENTS.md)
§5). It must pass acceptance before Phase 2 begins
([`../../AGENT-BEHAVIOUR/PROTOCOL.md`](../../AGENT-BEHAVIOUR/PROTOCOL.md) §2). The
output of this run determines:

- Whether to start Plan Phase 02 of Phase 2 (build `F_e` on top of this `B_EMA`).
- Whether to re-run on the **full SSv2** for a stronger result (deferred —
  ssv2_tiny first per project plan).
- Whether the architecture choice was correct at all (a clean `coarse_vs_copy`
  pass at step 10k is the first real evidence beyond synthetic sanity).

## What this task is NOT

- **NOT** Phase 2 work. Do not launch fine-flow training, do not edit `F_e`,
  do not modify shuffled-c tests.
- **NOT** further hyperparameter tuning beyond the Run 1 → Run 2 retune
  documented in [`POSTMORTEM_RUN1.md`](POSTMORTEM_RUN1.md). The set of
  changes there (peak LR halved, clip tightened, warmup shortened, step
  budget cut, NaN/Inf skip guard added) is the *only* tuning permitted
  inside this folder. Anything beyond that — `lambda_var`, EMA schedule,
  `global_batch`, encoder choice — requires re-opening
  [`../../AGENT-BEHAVIOUR/PROTOCOL.md`](../../AGENT-BEHAVIOUR/PROTOCOL.md)
  §5 and tech-lead approval.
- **NOT** a "full SSv2" run. That is a follow-up after `ssv2_tiny` passes
  acceptance.
