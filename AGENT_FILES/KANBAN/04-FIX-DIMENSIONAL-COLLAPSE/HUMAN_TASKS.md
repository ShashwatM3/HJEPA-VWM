# 04 — Human tasks: Fix dimensional collapse of `c_t`

> **Read first:** [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) and
> [`../README.md`](../README.md) (handoff convention).
>
> **Your role:** you have the only SSH access to the pod, the only RunPod
> console, and the only eyes on W&B. The agent writes/gates the code; **you
> run it and report metrics back**. Items are `H4.x` and cross-reference the
> agent's `A4.x` in [`TASKS.md`](TASKS.md).
>
> Pod pre-flight recovery (fresh-pod dep/tmux/HF_HOME) is the same as
> [`../02-LAUNCH-FULL-PHASE-1-RUN/HUMAN_TASKS.md`](../02-LAUNCH-FULL-PHASE-1-RUN/HUMAN_TASKS.md)
> H2.1–H2.2 — reuse it; it is not repeated in full here.

---

## Task H4.1 — Pull the branch and pre-flight

**Status:** [NOT DONE]

**Description:**
Get the pod onto the commit that has the instrumentation + flags (after the
agent finishes A4.0–A4.2), and confirm the environment.

**Instructions:**

1. On the pod: `cd /workspace/hierarchal-jepa-flow-world-model && git fetch origin && git pull`.
2. Run the H2.1-style pre-flight (transformers version, wandb login, HF_HOME,
   ssv2 counts). Recover per H2.1 step 0 if anything is wiped.
3. `python train.py --stage0-only` → confirm it passes and now prints
   attention-entropy (proves A4.0 landed).

**Report back to the agent:**
- `git log -1 --oneline`, the Stage 0 pass line (incl. attention-entropy), and
  the four pre-flight values.

---

## Task H4.2 — Fix-1 short instrumented job

**Status:** [NOT DONE]

**Description:**
A short run with the init flags on to see whether sharper init changes the
at-init/early picture (A4.1). Cheap; this is *not* the main experiment.

**Instructions:**

1. In a tmux session, launch a short run (~300–500 steps) on `ssv2_tiny`
   with the init flags the agent specifies (e.g. `zero_init_out_mlp=True`,
   `query_init=orthogonal`), `--log-every 50 --diag-every 100`.
2. Let it reach the first couple of diagnostic ticks.

**Report back to the agent:**
- Step-0 and early `attention_entropy`, `slot_diversity_rank`,
  `c_effective_rank`, plus `c_std_mean`, `grad_norm`, `grad_skipped`. A W&B
  screenshot is fine.

---

## Task H4.3 — Main event: the larger-data A/B run(s)

**Status:** [NOT DONE]

**Description:**
The real experiment: VICReg-C on a larger/full dataset (A4.2). **You decide
the run matrix** per [`ANALYSIS_AND_DECISIONS.md`](ANALYSIS_AND_DECISIONS.md)
§3 — discuss with the tech lead:

- **2-run (full attribution):** Run A = no reg, full SSv2; Run B = VICReg-C,
  full SSv2.
- **1-run (cheaper):** Run B only; ship if rank **and** copy-ratio both improve.

**Instructions:**

1. Confirm the dataset is available at `/workspace/data/ssv2` (full) or the
   chosen larger subset.
2. Launch in tmux with the agent-specified flags (for Run B: `lambda_cov=<value>`
   the agent gives, init flags as chosen in H4.2). Detach.
3. Share the W&B run URL(s) immediately, then report at milestones
   (~10%, ~50%, end), emphasizing `c_effective_rank` **and**
   `coarse_vs_copy_ratio` together, plus the standard guards.

**Report back to the agent:**
- W&B URL(s); milestone values for `c_effective_rank`, `coarse_vs_copy_ratio`,
  `coarse_vs_batch_mean_ratio`, `c_dead_dim_frac`, `c_cross_video_cosine`,
  `slot_diversity_rank`, `grad_has_nan`, `grad_skipped`; final checkpoint path.

---

## Task H4.4 — Relay the verdict / next decision

**Status:** [NOT DONE]

**Description:**
Take the agent's interpretation (A4.4/A4.5) to the tech lead, including the
supervisor sign-off question (a kept covariance term reverses the v0.2
"no covariance loss initially" directive).

**Instructions:**

1. Share the agent's per-run verdict and recommendation with the tech lead.
2. Relay the decision back to the agent so it can close the folder.

**Report back to the agent:**
- The tech lead's decision: lock the fix / escalate to SIGReg / revisit task
  difficulty / other.
