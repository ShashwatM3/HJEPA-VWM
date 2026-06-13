# 02 — Tasks: Launch the full Phase 1 training run (coding agent)

> **Read first:**
> - [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) — what this run delivers.
> - [`HUMAN_TASKS.md`](HUMAN_TASKS.md) — the bulk of this folder's work is the
>   human's (SSH, tmux, watching W&B). Read it so you know what the human is
>   doing and what they will ask of you.
> - [`../README.md`](../README.md) — the agent/human handoff convention.
>
> **This folder is mostly human work.** The agent's role in Plan Phase 02 is:
>
> 1. Prepare any last-minute code fixes if the human asks before launch.
> 2. **Standby** during the 5–14 hour training run.
> 3. Interpret W&B values/screenshots the human shares at milestones.
> 4. Help write the final acceptance-gate report after the run finishes.
>
> Do not try to launch training directly — the agent has no SSH access to the
> pod. Wait for the human at every step.

---

> **⏸ PAUSE — BLOCKED ON HUMAN.**
>
> Do not start Task A2.1 until:
>
> 1. Plan Phase 01 is fully closed (see
>    [`../01-OPTIMIZE-DATALOADER/`](../01-OPTIMIZE-DATALOADER/)).
> 2. The human has completed [`HUMAN_TASKS.md`](HUMAN_TASKS.md) **H2.1**
>    (pre-flight check) and reports back the four pre-flight values.

---

## Task A2.1 — Confirm pre-flight values from the human

**Status:** [DONE] — all five pre-flight values verified.

**Description:**
When the human reports the pre-flight values from H2.1, verify them and
either green-light the launch (A2.2) or send the human back to fix a
prerequisite. Step 0 (the `transformers` dep check) catches the most common
failure mode — a fresh pod with wiped pip site-packages.

**Instructions:**

The human will send back five values. Cross-check each:

| # | Value | Expected | If wrong, instruct the human to… |
|---|---|---|---|
| 0 | `transformers` version | `4.5x.y` (anywhere in `>=4.53,<5`) | `pip install -r requirements.txt` (fresh pod); then re-login wandb and re-export HF_HOME (the inline recovery block in H2.1 step 0). |
| 1 | `git log -1 --oneline` | Includes the dataloader-optimization commit hash from A1.3 (`5e78caa`) | `git fetch origin && git pull origin phase1-v0.2-frozen-encoder` |
| 2 | `ssv2_tiny` train count | ~4,000 | Re-run `make_subset.py` (see [`../../SETUPS/SETUP.md`](../../SETUPS/SETUP.md) A12) |
| 3 | W&B login state | `Currently logged in as: <username>` | `wandb login` and re-confirm |
| 4 | `HF_HOME` | `/workspace/hf_cache`, directory non-empty | `export HF_HOME=/workspace/hf_cache` + append to `~/.bashrc` |

**How to verify:**

- All five values match expected.
- If any didn't, the human has reported the corrective action and the
  re-verified value.

---

> **⏸ PAUSE — BLOCKED ON HUMAN.**
>
> The launch itself is **H2.2, H2.3, H2.4** — these run sequentially on the
> pod (tmux, launch, detach). Wait for the human to report back the W&B run
> URL from H2.3.

---

## Task A2.2 — Acknowledge the W&B run URL and watch for early-stop signals

**Status:** [NOT DONE]

**Description:**
When the human pastes back the W&B run URL, record it and start watching
for the early-stop signals listed in
[`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md). The agent does not have
W&B access — it relies on the human checking the dashboard and sharing
values/screenshots at the milestones in **H2.5**. Run 2 schedule is:

**Instructions:**

1. Record the W&B run URL in a chat message back to the human.
2. Remind the human of the three milestones in H2.5 (Run 2 / 15k schedule):
   - Milestone A: step ~1,000 → health check (no NaN, no `grad_skipped`,
     `L_var` falling, pre-clip `grad_norm` mostly < 10).
   - Milestone B: step ~7,500 → **first acceptance-gate read** (~50% wall-clock).
   - Milestone C: step ~12,500 → final acceptance values, last abort window.
3. Confirm to the human that the agent is in standby and they should ping
   when they reach each milestone, sharing the relevant metric values.

**How to verify:**

- The W&B URL is recorded.
- The human has acknowledged the milestone plan.

---

> **⏸ EXTENDED PAUSE — BLOCKED ON HUMAN (long).**
>
> The 30k run takes 5–14 hours. The human will detach tmux and let it run.
> The agent should expect long quiet periods between handoffs. Do **not**
> hallucinate W&B values; do **not** assume metrics are healthy without
> reports from the human.

---

## Task A2.3 — Interpret Milestone A (step ~1,000)

**Status:** [NOT DONE]

**Description:**
At Milestone A, the human shares the values (or a screenshot) of the
metrics logged so far. Interpret and respond. Run 2 milestone A is at
step ~1000 (vs Run 1's step ~2000) because warmup is now 1500 steps, so
we're already near peak LR.

**Instructions:**

Look for these signals in the human's report:

| Signal | Healthy reading | Action if unhealthy |
|---|---|---|
| `loss`, `L_flow`, `L_var` | all finite, no NaN | Abort if any NaN |
| `grad_has_nan` | always 0 | Abort if ever 1 |
| `grad_skipped` (Run 2 new field) | always 0 at this point | If > 0: warmup is already hitting the 50× clip threshold — abort and re-tune (peak LR is still too high) |
| Pre-clip `grad_norm` recent values | mostly < 10 | If > 10 repeatedly: precursor to instability — abort and tighten clip / lower LR |
| `L_var` direction since step 0 | falling toward 0 | Sharp drop to near-zero is fine (variance grew above threshold); rising is a collapse signal — investigate |
| `L_flow` direction | falling | If flat or rising past step 800: model isn't learning |

Respond to the human with: "Milestone A healthy, continue" OR "Milestone A
shows <specific issue>, recommend <action>".

**How to verify:**

- Response sent.
- If aborting was recommended, this folder's status is updated and Plan
  Phase 02 is paused for debug.

---

## Task A2.4 — Interpret Milestone B (step ~7,500) — the acceptance gate

**Status:** [NOT DONE]

**Description:**
The Run 2 mid-run acceptance read. The 15k run's PASS-equivalent target is
`coarse_vs_copy_ratio < 1.5` by step 12500; at step 7500 we want to see it
already trending below ~3 to be on track.

**Instructions:**

Look for these in the human's report:

| Signal | Healthy at step 7500 | Action |
|---|---|---|
| `coarse_vs_copy_ratio` (latest) | < 2.0 and falling | Healthy — let it run |
| `coarse_vs_copy_ratio` (latest) | 2.0 – 5.0 | Borderline — let it run to step ~10000 and re-check; warn human to keep eyes on W&B |
| `coarse_vs_copy_ratio` (latest) | > 5.0 | **Tell the human to stop** the run via `tmux attach -t phase1` + `Ctrl+C`. Screenshot W&B. Escalate to tech lead. Do not relaunch without a fix. |
| `coarse_vs_batch_mean_ratio` | < 1.2 and falling | Same logic as above |
| `c_effective_rank` | rising; > 15 desirable at step 7500 | Stagnation at < 10 is a concern (matches Run 1's failure mode); note it |
| `c_cross_video_cosine` | falling below 0.5 | Drift toward 1.0 = directional collapse — abort |
| `grad_has_nan` | always 0 | Any 1 = abort |
| `grad_skipped` total | 0 | > 0 = the skip guard fired; once is tolerable, > 2 = abort and re-tune |

Respond to the human with: "Milestone B passed, continue" OR "Milestone B
warning: <details>, recommend <action>".

**How to verify:**

- Response sent.
- If the run was stopped, the agent has logged the W&B values that triggered
  the stop in a chat message to the human (so the tech lead can review).

---

## Task A2.5 — Interpret Milestone C (step ~12,500) and final exit

**Status:** [NOT DONE]

**Description:**
Late-run check on the 15k schedule. The run will exit at step 15000 about
~45 min after this point. Compare against the **15k PASS-equivalent**
targets (see [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) — these
are not the strict PHASE_1.md §12 thresholds, which were sized for 30k).

**Instructions:**

| Gate | 15k PASS-equivalent | Action if missed |
|---|---|---|
| `coarse_vs_copy_ratio` | < 1.5 (trending; < 1.0 = strong) | Soft warning — note it; the final 2.5k steps may close the gap |
| `coarse_vs_batch_mean_ratio` | < 0.80 | Soft warning |
| `c_effective_rank` | > 30 | Soft warning (Run 1 stuck at ~5 — anything > 15 here is real progress) |
| `c_dead_dim_frac` | < 0.15 (hard stop at 0.30) | If > 0.30, recommend stop |
| `c_cross_video_cosine` | well below ~0.5 | Note any drift toward 1.0 |
| `grad_has_nan` (any logged step) | always 0 | If ever 1, fail |
| `grad_skipped` total | 0 (≤ 2 acceptable with warning) | > 2 = stop and investigate |

Respond with: "On track for clean acceptance" OR "Borderline — expect to
need a soft-warning note in the final report" OR "Run should be stopped".

**How to verify:**

- Response sent.

---

> **⏸ PAUSE — BLOCKED ON HUMAN.**
>
> Wait for the run to actually exit at step 15k. The human's
> [`HUMAN_TASKS.md`](HUMAN_TASKS.md) **H2.6** ends with a final-state report:
> checkpoint path + the final metric values for every PHASE_1.md §12 gate
> (interpreted on the 15k PASS-equivalent scale).

---

## Task A2.6 — Help write the Phase 1 acceptance-gate report

**Status:** [NOT DONE]

**Description:**
When the human reports the final values from H2.6, draft the acceptance-gate
report for the tech lead. The report follows
[`../../PHASES/PHASE_1.md`](../../PHASES/PHASE_1.md) §12 row by row.

**Instructions:**

Draft a report with this structure:

```
# Phase 1 acceptance-gate report (15k Run 2)

W&B run: <URL from H2.3>
Final checkpoint: /workspace/checkpoints/phase1_step15000.pt
Wall-clock: <real time, from human's report>

## Gates (PHASES/PHASE_1.md §12, interpreted on 15k PASS-equivalent scale)

| Gate                              | 15k target | Final value | Status |
|-----------------------------------|------------|-------------|--------|
| Stage 0 sanity                    | pass       | <pass/fail> | <PASS/FAIL>
| 15k steps complete, no NaN        | yes        | <yes/no>    | <PASS/FAIL>
| `grad_skipped` total              | 0 (≤2 WARN)| <value>     | <PASS/WARN/FAIL>
| Frozen encoder unchanged          | yes        | <yes/no>    | <PASS/FAIL>
| `coarse_vs_copy_ratio` (val)      | < 1.5      | <value>     | <PASS/FAIL/WARN>
| `coarse_vs_batch_mean_ratio`      | < 0.80     | <value>     | <PASS/FAIL/WARN>
| `c_effective_rank`                | > 30       | <value>     | <PASS/FAIL/WARN>
| `c_dead_dim_frac`                 | < 0.15     | <value>     | <PASS/FAIL/WARN>
| `c_cross_video_cosine`            | well < 0.5 | <value>     | <PASS/FAIL/WARN>
| Final checkpoint loadable         | yes        | <yes/no>    | <PASS/FAIL>

## Soft warnings

<any borderline values; trajectory notes>

## Recommendation

<one of: "Proceed to Phase 2", "Re-run on full SSv2 first", "Debug X before Phase 2">
```

Fill in every cell with the human's reported values. Mark PASS/FAIL/WARN per
the thresholds. The recommendation should follow this logic:

- **All gates PASS** → "Proceed to Phase 2."
- **All gates PASS except c_effective_rank borderline (50–60)** → "Proceed to
  Phase 2 with a note. Consider full-SSv2 re-run later if Phase 2 reveals
  weak fine-flow performance."
- **Any acceptance ratio FAIL** → "Do not proceed to Phase 2. Re-evaluate
  with tech lead: more steps, full SSv2, or design review."
- **NaN anywhere** → "Do not proceed. Debug before any further training."

**How to verify:**

- Report drafted and sent to the human for review/sharing with tech lead.

---

## Task A2.7 — Close out Plan Phase 02

**Status:** [NOT DONE]

**Description:**
Once the report is delivered and the human + tech lead have decided next
steps, close this folder.

**Instructions:**

1. Mark all `[NOT DONE]` items in this file as `[DONE]`.
2. Verify all `[NOT DONE]` items in [`HUMAN_TASKS.md`](HUMAN_TASKS.md) are
   `[DONE]`.
3. Tell the human: "Plan Phase 02 closed. Phase 1 acceptance report
   delivered. Awaiting decision on Phase 2 / full SSv2 re-run / debug."

**How to verify:**

- All tasks in both files closed.
- Handoff message delivered.
