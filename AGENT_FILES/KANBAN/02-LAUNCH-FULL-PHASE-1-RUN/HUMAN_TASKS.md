# 02 — Tasks: Launch the full Phase 1 training run (human operator)

> **Read first:**
> - [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) — what this run delivers.
> - [`TASKS.md`](TASKS.md) — the agent's role (standby + interpretation).
> - [`../README.md`](../README.md) — the agent/human handoff convention.
>
> **You own this folder.** The agent can't SSH to the pod, can't open tmux, and
> can't watch W&B. Your job is to launch the training, watch the dashboard at
> three milestones, and report back the final values for the acceptance gates.
>
> Order: do tasks in numerical order. After each task, send the agent the
> values listed under **Report back to the agent**.

---

> **⏸ WAIT FOR THE AGENT FIRST.**
>
> Do not start H2.1 until Plan Phase 01 is fully closed (the agent has
> declared it complete in A1.5).

---

## Task H2.1 — Pre-flight check on the pod

**Status:** [DONE] — transformers 4.57.6, commit `5e78caa` clean, tiny=4002/348, W&B authenticated (proven by H1.2 run `gj8ypv0d` minutes prior; `wandb status` cosmetic doc gap noted), HF_HOME=`/workspace/hf_cache` with `hub`+`xet`.

**Corresponds to:** the first pause point in [`TASKS.md`](TASKS.md), before A2.1.

**Description:**
Confirm all five prerequisites for launching the 30k run are in place. Note
that **pod redeploys wipe `/usr/local/lib/python3.11/dist-packages/` and
`~/.netrc`** — anything not on the `/workspace` network volume must be
re-provisioned. The dep check (step 0 below) catches this.

**Instructions:**

```bash
# From your laptop, SSH into the pod:
ssh runpod-jepa
cd /workspace/hierarchal-jepa-flow-world-model

# 0. Fresh-pod check: is `transformers` installed? If not, this is a fresh
#    pod and pip site-packages + W&B login + HF_HOME were wiped.
python -c "import transformers; print('transformers:', transformers.__version__)" \
    2>&1 | head -1
#    If you see "ModuleNotFoundError: No module named 'transformers'", run:
#       pip install -r requirements.txt
#       wandb login            # paste API key
#       export HF_HOME=/workspace/hf_cache
#       grep -q 'HF_HOME=/workspace/hf_cache' ~/.bashrc \
#         || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
#    Then re-run the dep check and continue to step 1.

# 1. Latest code
git status                                          # must be clean
git log -1 --oneline                                # latest dataloader commit

# 2. ssv2_tiny counts
ls /workspace/data/ssv2_tiny/manifest.json
echo "tiny train: $(find /workspace/data/ssv2_tiny/train -maxdepth 1 -type l | wc -l)"
echo "tiny val:   $(find /workspace/data/ssv2_tiny/validation -maxdepth 1 -type l | wc -l)"

# 3. W&B authentication
wandb status

# 4. HF cache location
echo "HF_HOME=$HF_HOME"
ls /workspace/hf_cache 2>/dev/null | head -3
```

**How to verify:**

- Step 0 prints a non-error `transformers: 4.5x.y` line.
- `git status` is clean.
- The commit hash matches what the agent pushed in A1.3
  (`5e78caa Decode only needed frames in dataloader`).
- Train count is approximately 4,000; val count approximately 350.
- `wandb status` confirms login.
- `HF_HOME` is `/workspace/hf_cache` and the directory has cached
  encoder files in it (e.g. `hub`, `xet`).

If any check fails, fix it before continuing (see
[`../../SETUPS/SETUP.md`](../../SETUPS/SETUP.md) A10, A11, A12 and the inline
recovery sequence under step 0).

**Report back to the agent:**

- `transformers` version from step 0.
- The commit hash from `git log -1 --oneline`.
- Train and val symlink counts.
- W&B login confirmation (your wandb username will be in the output).
- `HF_HOME` value and whether `/workspace/hf_cache` has files in it.

---

## Task H2.2 — Start a tmux session

**Status:** [NOT DONE]

**Description:**
Long-running training survives SSH disconnect only inside tmux (or
similar). Create a fresh tmux session named `phase1`.

**Instructions:**

```bash
# 0. Fresh-pod check: is tmux installed? RunPod base images often skip it.
which tmux 2>/dev/null || { echo "tmux missing — installing..."; \
    apt-get update -qq && apt-get install -y tmux; }

# Check whether a session already exists:
tmux ls 2>/dev/null

# If a stale `phase1` session exists from a previous attempt, kill it:
# tmux kill-session -t phase1

# Create new session:
tmux new -s phase1
```

You should now see a green status bar at the bottom of your terminal showing
`phase1`.

**How to verify:**

- The tmux status bar is visible.
- `echo $TMUX` (inside the session) prints a non-empty path.

**Report back to the agent:**

- Nothing required for this task; it's a stepping-stone to H2.3.

---

## Task H2.3 — Launch the 15k Phase 1 training command (Run 2)

**Status:** [NOT DONE]

**Corresponds to:** the second pause point in [`TASKS.md`](TASKS.md), before A2.2.

**Description:**
Run the **15k-step** Phase 1 training command (revised after Run 1's
gradient-explosion crash — see
[`POSTMORTEM_RUN1.md`](POSTMORTEM_RUN1.md)).

**Instructions:**

Inside the tmux session:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python train.py --data ssv2_tiny --steps 15000
```

Notes:

- The `--steps 15000` flag is explicit even though it now matches the
  config default; this makes the intent obvious in shell history.
- Do **not** add `--log-every 1` — the default `log_every=50` is what the
  project expects. Per-step logging would spam tmux and slow throughput.

Within ~30 seconds you should see:
- `step=0 {...}` printed with finite numbers and **`grad_has_nan: 0.0,
  grad_skipped: 0.0`** (Run 2 logs the new `grad_skipped` field).
- A W&B run URL of the form
  `https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/<id>`.

**How to verify:**

- `step=0 {...}` appears with finite values.
- The W&B URL prints and opens to a live run in the browser.
- The next few step lines (50, 100, ...) also print finite values.

**Report back to the agent:**

- The W&B run URL (full link).
- Whether `step=0` printed `grad_has_nan: 0.0` (or whatever it printed
  for that field).

---

## Task H2.4 — Detach tmux and let it run

**Status:** [NOT DONE]

**Description:**
Detach the tmux session so you can disconnect SSH (or close your laptop)
without killing training.

**Instructions:**

Inside the tmux session, with training visibly running:

1. Press **`Ctrl+B`**, release.
2. Press **`D`**.

You return to a regular shell prompt. Training continues on the pod.

**How to verify:**

- `tmux ls` (outside the session) shows `phase1: 1 windows`, no
  `(attached)` suffix.
- After 5 minutes, the W&B dashboard shows additional logged steps (the
  run is still active).
- You can `exit` the SSH session entirely — the run keeps logging on W&B.

**Report back to the agent:**

- Nothing required; the next handoff is the Milestone A check in H2.5.

---

## Task H2.5 — Check W&B at three milestones (Run 2, 15k schedule)

**Status:** [NOT DONE]

**Corresponds to:** the three pause points in [`TASKS.md`](TASKS.md) (A2.3, A2.4, A2.5).

**Description:**
Don't babysit the pod, but do check W&B at three points so problems are
caught early. After each milestone, share values with the agent for
interpretation. The Run 2 schedule pulls milestones earlier than Run 1
because the warmup is shorter (1500 vs 10000) and the run itself is half
as long (15000 vs 30000).

> **Always also check `grad_skipped` and pre-clip `grad_norm`** at every
> milestone — these are the Run 1-explosion warning lights. See
> [`POSTMORTEM_RUN1.md`](POSTMORTEM_RUN1.md).

### Milestone A — step ~1,000 (~15 min wall-clock after launch)

Warmup completes at step 1500, so by step 1000 we're at lr_mult ≈ 0.67 and
the model is already learning at near-peak LR. This is the earliest point
the Run 1 failure mode would surface.

Open the W&B run URL. Read off (latest values):

- `loss`, `L_flow`, `L_var`
- `grad_has_nan` at the latest logged step
- **`grad_skipped`** total (sum or count of >0 events since step 0) — should be 0
- Pre-clip `grad_norm` recent values — should mostly be < 10
- `L_var` trajectory (is it falling toward 0?)
- `L_flow` trajectory (falling?)

**Report back to the agent** for Milestone A:

- The above values.
- A screenshot of the W&B Summary tab is fine if quicker.

### Milestone B — step ~7,500 (~50% wall-clock, ~2.5 hr in)

The acceptance-gate first read.

Read off:

- `coarse_vs_copy_ratio` at the latest 500-step boundary (7000, 7500, ...)
- `coarse_vs_batch_mean_ratio` (same)
- `c_effective_rank`
- `c_cross_video_cosine`
- `grad_has_nan`, `grad_skipped` totals — both should be 0
- Pre-clip `grad_norm` recent values

**Report back to the agent** for Milestone B.

If the agent recommends stopping the run:

```bash
ssh runpod-jepa
tmux attach -t phase1
# Press Ctrl+C to stop training
```

### Milestone C — step ~12,500 (~85% wall-clock, ~4.5 hr in)

Late-run check, last opportunity to abort before the final acceptance.

Read off:

- `coarse_vs_copy_ratio` (latest snapshot)
- `coarse_vs_batch_mean_ratio` (latest)
- `c_effective_rank`
- `c_dead_dim_frac`
- `c_std_mean`, `c_std_median`
- `c_cross_video_cosine`
- Whether `grad_has_nan` or `grad_skipped` was ever > 0 at any logged step
- Pre-clip `grad_norm` recent values

Also confirm intermediate checkpoints exist on the pod:

```bash
ssh runpod-jepa
ls /workspace/checkpoints/phase1_step*.pt
```

(With `checkpoint_every=2500`, by step 12500 you should see
`phase1_step2500.pt`, `phase1_step5000.pt`, `phase1_step7500.pt`,
`phase1_step10000.pt`, and `phase1_step12500.pt`.)

**Report back to the agent** for Milestone C.

**How to verify:**

- All three milestones reported.
- Run is still running after Milestone C (unless the agent recommended stop).

---

## Task H2.6 — Final inspection after step 15,000

**Status:** [NOT DONE]

**Corresponds to:** the final pause point in [`TASKS.md`](TASKS.md), before A2.6.

**Description:**
The run exits at step 30,000 (or earlier if you aborted on the agent's
recommendation). Verify the final checkpoint, then read off the final
acceptance-gate metric values from W&B for the agent's report.

**Instructions:**

```bash
ssh runpod-jepa
tmux attach -t phase1
# Confirm the last lines show the final checkpoint save + clean wandb finish.
# Detach with Ctrl+B then D, or close the tmux session entirely if done:
#   tmux kill-session -t phase1

cd /workspace/hierarchal-jepa-flow-world-model

# Final checkpoint check:
ls -lh /workspace/checkpoints/phase1_step15000.pt

# Confirm it loads:
python - <<'PY'
import torch
ckpt = torch.load("/workspace/checkpoints/phase1_step15000.pt", map_location="cpu")
print("global_step:", ckpt["global_step"])
print("keys:", list(ckpt.keys()))
PY
```

Then on W&B, record the **final** values of:

| W&B field | Note where you found it |
|---|---|
| `coarse_vs_copy_ratio` | last 500-step boundary |
| `coarse_vs_batch_mean_ratio` | last 500-step boundary |
| `c_effective_rank` | last 500-step boundary |
| `c_std_mean`, `c_std_median` | last 500-step boundary |
| `c_dead_dim_frac` | last 500-step boundary |
| `c_cross_video_cosine` | last 500-step boundary |
| Any `grad_has_nan = 1` anywhere? | full run |
| Final `loss`, `L_flow`, `L_var` | last logged step |
| Total run duration | top of the W&B run page |

**How to verify:**

- Final checkpoint exists at `/workspace/checkpoints/phase1_step15000.pt`.
- `global_step` from the loaded checkpoint equals `15000`.
- All metric values recorded.

**Report back to the agent:**

- Final checkpoint path and confirmed `global_step`.
- Every metric in the table above.
- Total wall-clock duration.
- Whether `grad_has_nan` was ever non-zero.

The agent will use these to draft the acceptance-gate report (A2.6).

---

## Task H2.7 — Receive the agent's acceptance-gate report

**Status:** [NOT DONE]

**Description:**
The agent drafts a structured report from your H2.6 numbers. Read it, share
it with the tech lead, and respond with the decision on next steps (Phase 2
/ full SSv2 / debug).

**Instructions:**

1. Receive the report from the agent.
2. Cross-check that all values match what you reported in H2.6 (the agent
   shouldn't make any up, but it's cheap to verify).
3. Share with tech lead.
4. Once a decision is made, tell the agent:
   - "Proceed to Phase 2" — agent opens
     [`../../PHASES/PHASE_2.md`](../../PHASES/PHASE_2.md).
   - "Re-run on full SSv2" — different plan; agent helps you author the new
     command.
   - "Debug before any more training" — agent goes into debug mode.

**How to verify:**

- The report has been delivered.
- The tech lead has reviewed it.
- A decision is recorded and shared back with the agent.

**Report back to the agent:**

- The final decision string from above.
