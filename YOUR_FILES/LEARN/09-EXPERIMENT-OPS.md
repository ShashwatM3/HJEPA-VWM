# 09 — Experiment Operations: Running, Watching, and Recovering Training Jobs

> **What you'll understand after this file:** the full operational story —
> staged validation before spending money, W&B logging, tmux and remote-run
> hygiene, checkpoint/resume mechanics, the RunPod persistence model, and
> the human/agent Kanban workflow this project uses.

---

## 1. The staged-validation philosophy: spend compute in increasing amounts

The most expensive failure is the one you discover 5 hours in. The
codebase is built as a ladder of ever-more-expensive validations, each
gating the next:

| Stage | Command / mechanism | Costs | Catches |
|---|---|---|---|
| Contract tests | `pytest tests/` (no torch/GPU needed) | seconds, free | shape math, config consistency, token-order assumptions |
| Module smoke | `smoke_test_models()` (synthetic e_t, no encoder download) | seconds | wiring, gradient boundaries (the `grad is None` asserts) |
| **Stage 0** | `python train.py --stage0-only` | ~1 min on pod | encoder loads & is frozen, one real train step runs, loss finite, EMA moved, encoder params bit-identical |
| Short run | `--steps 200 --log-every 50` | ~5 min | dataloader on real data, s/step measurement, W&B wiring |
| Full run | `--steps 15000` | ~6 h, ~$10 | the actual science |

Stage 0 deserves emphasis because it's this project's signature move: a
*synthetic* end-to-end step (random pixels, real modules) that asserts the
invariants which, if broken, would silently corrupt a full run — frozen
encoder unchanged, EMA actually moving, finite loss. One minute of pod
time buys insurance on six hours. The generalizable rule: **before any
long job, run the cheapest experiment that exercises every invariant the
long job depends on.**

## 2. Logging: print + W&B, and what each is for

From `run_training`: metrics print to stdout *and* log to **Weights &
Biases** (project `hjepa-vwm`) every `log_every=50` steps, with the
diagnostic panel merged in every 500.

Why both channels:

- **stdout** (captured in tmux) is the survivable record — it lives on the
  pod, works offline, and is what you paste for debugging. It's also the
  crash record: a traceback appears here, not in W&B.
- **W&B** gives time-series plots, run comparison, and a `config` snapshot
  (the entire `Config` dataclass is serialized at init — so every run
  permanently records the *exact* hyperparameters that produced it; when
  you ask "what LR did peachy-terrain-5 use?", the answer is in its
  Overview tab, not in your memory or git archaeology).

Reference runs to compare against:

| Run | Role | Outcome |
|---|---|---|
| `peachy-terrain-5` | Run 1 postmortem | Gradient explosion at step ~10750 (file 10) |
| `cerulean-snow-13` | Investigation 003 win | `k=12`, `λ_var=0.5`; rank ~13.7+, beats copy |
| `elated-snowflake-15` | Investigation 005 fail | Grad-skip death spiral at step 8500 (file 10 §7) |
| `drawn-elevator-16` | Investigation 005 resume | Resume from step ~7500 with lower flow LR — verify on W&B |

Full run index: [`KANBAN/PHASE_1/README.md`](../../KANBAN/PHASE_1/README.md).

Operational notes learned in Run 1:

- W&B is wrapped in try/except and **degrades to a warning** if login or
  network fails — a logging outage must never kill a training run.
- The W&B *charts* tab sometimes fails to render while data is flowing
  fine (the Overview summary numbers update). Hard-reload, switch the
  x-axis, or read `wandb-summary.json` — but check raw data before
  concluding the *run* has a problem. Distinguish "dashboard broken" from
  "training broken."
- `wandb.log(metrics, step=step)` keys everything to the global step, so
  resumed runs continue the same x-axis.

## 3. tmux: keeping jobs alive without you

SSH sessions die — laptop sleeps, Wi-Fi drops — and a process started in a
plain SSH session dies with it (SIGHUP). **tmux** is a terminal
multiplexer: it owns the terminal session on the *server*, your SSH client
merely views it.

The working set of commands (all you need):

```bash
tmux new -s phase1        # create session named "phase1"
# ... launch training inside it ...
# detach: Ctrl-b then d   (training keeps running)
tmux attach -t phase1     # reconnect later, from any SSH session
tmux ls                   # what sessions exist?
```

Project convention: every long-running pod job launches inside
`tmux new -s phase1`. (And a Run-1 lesson now baked into HUMAN_TASKS:
fresh pods don't even *have* tmux — `apt-get update -qq && apt-get install
-y tmux` is part of the pre-flight.)

## 4. The RunPod persistence model (the trap that bit us)

A RunPod pod = an ephemeral **container** + a persistent **volume**
mounted at `/workspace`. The asymmetry is everything:

| Survives pod stop/restart | Does NOT survive |
|---|---|
| `/workspace/data` (dataset) | pip-installed packages (live in the container's site-packages) |
| `/workspace/checkpoints` | apt-installed tools (tmux!) |
| `/workspace/hf_cache` (HF_HOME → encoder weights) | environment variables, shell history |
| the repo clone (if under /workspace) | anything in `/root`, `/tmp` |

This is why a fresh pod greeted us with `ModuleNotFoundError:
transformers` despite everything having "worked yesterday." The pre-flight
sequence (now codified in `KANBAN/02-.../HUMAN_TASKS.md` H2.1–H2.2):

```bash
cd /workspace/hierarchal-jepa-flow-world-model
pip install -r requirements.txt
apt-get update -qq && apt-get install -y tmux
wandb login                              # paste API key
export HF_HOME=/workspace/hf_cache       # so the 1.2GB encoder isn't re-downloaded
python train.py --stage0-only            # Stage 0 gate before anything long
```

`HF_HOME` matters doubly: it avoids a multi-GB re-download *and* keeps the
encoder cache on the volume so even a container rebuild doesn't repay it.

Related pinning lesson: `requirements.txt` pins `transformers>=4.53,<5`
because a v5 release broke the V-JEPA 2 loading path (the
`torch.float8_e8m0fnu` incident). On ephemeral machines you reinstall
deps constantly — **unpinned dependencies make every pod restart a dice
roll.** Pin anything whose breakage you've already witnessed.

## 5. Checkpointing and resume

Every 2,500 steps (and at the end), `save_checkpoint` writes to
`/workspace/checkpoints/phase1_step{N}.pt`:

- `bottleneck`, `target_bottleneck`, `coarse_flow` state dicts —
  note the EMA module is saved (resume must preserve the smoothing
  history, file 05 §5), and the encoder is *not* (always re-loaded from
  HF cache — saves ~1.2GB per checkpoint).
- `optimizer` state — AdamW's m/v. Resuming without it restarts the
  optimizer's statistics cold and causes a transient instability bump;
  resuming with it continues seamlessly.
- `global_step` and the full serialized `config`.

Resume: `python train.py --resume /workspace/checkpoints/phase1_step7500.pt`
— reloads modules + optimizer and continues the step counter (so LR/EMA
schedules pick up exactly where they left off; both are pure functions of
`step`, which is what makes resume *correct* and not just *possible*).

After a grad-skip death spiral (file 10 §7), resume from a **pre-spike**
checkpoint — not the final one — and consider lowering coarse-flow LR:
`--lr-coarse-flow 1e-4` (run `drawn-elevator-16` strategy).

Two honest limitations worth knowing (fine for Phase 1 scale, would need
fixing for serious runs): the dataloader's RNG state isn't saved (resumed
runs see a re-shuffled stream), and a crash mid-write could leave a
truncated checkpoint (atomic-rename writing is the standard fix).

## 6. Watching a run: the milestone discipline

A multi-hour run isn't "launch and pray" — it has scheduled checkups
(codified in the Kanban as Milestones A/B/C at steps ~1k / ~7.5k /
~12.5k). At each one, a human pastes current metrics; the interpretation
runbook is file 06 §4 (*stability → aliveness → distinctness → usefulness
→ richness*). The critical early checkpoint is just-after-warmup (~step
1.5–2k): that's when LR reaches peak — the maximum-risk moment Run 1
taught us about — so seeing stable `grad_norm` there materially de-risks
the rest of the run.

Quick reference for "is it healthy right now":

```
grad_skipped == 0 always          # any skip → stop and investigate
grad_norm: O(1–10), trending flat/down, no upward drift
L_var ≈ 0 after first ~500 steps
c_std_mean ≈ 1.0, c_dead_dim_frac ≈ 0
c_cross_video_cosine < 0.5
ratios falling over time; gates at end: copy ≤ 0.70, batch-mean ≤ 0.50
c_effective_rank: watch it grow (current best ~13–14; spec soft-target >60)
```

## 7. The human/agent Kanban workflow

Training research is tracked under **`KANBAN/PHASE_1/`** — one folder per
investigation, one subfolder per W&B run:

```
KANBAN/PHASE_1/
├── README.md                    ← status table, full W&B run index
├── investigation_001/           ← numerical stability (Run 1)
├── investigation_003/           ← collapse / rank (cerulean-snow-13 win)
├── investigation_005/           ← 15k acceptance (active)
└── ...
```

Each run folder holds `DESCRIPTION.md`, `OBSERVATIONS.md`, `NEXT_STEPS.md`.
Read [`KANBAN/PROTOCOL.md`](../../KANBAN/PROTOCOL.md) before editing.

Legacy launch docs also live under `AGENT_FILES/KANBAN/02-LAUNCH-FULL-PHASE-1-RUN/`
(including `POSTMORTEM_RUN1.md` — the source for file 10).

The handoff protocol is the point: every place where progress depends on
information only one actor can obtain is an explicit, named pause — not an
implicit assumption. It's the same philosophy as `as_target()` for
gradient boundaries (file 05): **make the invisible coordination structure
explicit and reviewable.**

## 8. Questions to test yourself

1. Why does Stage 0 exist when contract tests and smoke tests already
   pass? *(It's the only stage that exercises the real pretrained encoder
   + a real optimizer step + EMA on the real device — the invariants the
   6-hour run depends on.)*
2. A fresh pod: which three things must you reinstall/re-set before
   training, and why are they gone? *(pip deps, tmux, env vars like
   HF_HOME — the container is ephemeral; only /workspace persists.)*
3. Why must the EMA module be in the checkpoint? *(Resume would otherwise
   reset the target to the online weights, destroying the smoothing
   history and destabilizing the predictor's objective.)*
4. Why is W&B wrapped in try/except? *(A logging/network failure must
   degrade to a warning, never kill a 6-hour training job.)*
5. Your W&B charts are blank but Overview shows fresh numbers. What do you
   conclude? *(Rendering issue, not a training issue — verify via raw
   summary data before touching the run.)*
