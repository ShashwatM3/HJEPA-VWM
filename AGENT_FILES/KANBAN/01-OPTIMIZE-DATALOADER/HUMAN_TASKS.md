# 01 — Tasks: Optimize the dataloader (human operator)

> **Read first:**
> - [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) — why we are doing this.
> - [`TASKS.md`](TASKS.md) — what the agent is doing in parallel and where it
>   pauses on your input.
> - [`../README.md`](../README.md) — the agent/human handoff convention.
>
> **Order:** do tasks in numerical order. Mark `[NOT DONE]` → `[DONE]` once
> the **How to verify** check passes. After each task, send the agent the
> values listed under **Report back to the agent**.

---

> **⏸ WAIT FOR THE AGENT FIRST.**
>
> Do not start H1.1 until the agent has reported that **Task A1.3 is `[DONE]`**
> (the code change is committed and pushed to `phase1-v0.2-frozen-encoder`).

---

## Task H1.1 — Pull on pod and re-run Stage 0 sanity

**Status:** [DONE] — pod `9bf9a825ee7b` (fresh redeploy) required `pip install -r requirements.txt` first; then Stage 0 passed with bit-identical metrics to prior pod (`loss=4.67935, L_flow=4.581477, L_var=0.978734`).

**Corresponds to:** the first pause point in [`TASKS.md`](TASKS.md), after A1.3.

**Description:**
Pull the agent's latest commit on the pod and confirm Stage 0 sanity still
passes. Stage 0 uses synthetic tensors, so this only checks that the new
code compiles and the model path is unbroken — it does not test the
dataloader change yet.

**Instructions:**

```bash
# From your laptop, SSH into the pod:
ssh runpod-jepa    # or paste the SSH command from the RunPod Connect tab

# On the pod:
cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline      # should show the new "Decode only needed frames" commit

python train.py --stage0-only
```

**How to verify:**

- `git log -1 --oneline` shows the commit hash the agent pushed in A1.3.
- `python train.py --stage0-only` prints
  `Stage 0 sanity passed: {...}` with `'grad_has_nan': 0.0`.
- No traceback.

**Report back to the agent:**

- The commit hash from `git log -1 --oneline`.
- Pass/fail for Stage 0 (and the last output line in full if it failed).

---

## Task H1.2 — Timed 200-step benchmark on `ssv2_tiny`

**Status:** [DONE] — `real=4m42.195s` → 1.41 s/step. Step-0 `L_flow=2.8726, L_var=0.4267, grad_has_nan=0` (bit-identical to baseline). Step-50 `L_flow=3.38, L_var=0.42`. Run charmed-haze-4 (gj8ypv0d) clean exit, no NaN, no decord errors.

**Corresponds to:** the first pause point in [`TASKS.md`](TASKS.md), after A1.3.

**Description:**
Measure new `s/step` and confirm model inputs are bit-identical (or close
enough that step-0 metrics match the pre-optimization baseline).

**Instructions:**

```bash
# On the pod, still in the repo:
export HF_HOME=/workspace/hf_cache     # if not already set in ~/.bashrc

time python train.py --data ssv2_tiny --steps 200 --log-every 50
```

You can also set `HF_HOME` permanently:

```bash
grep -q 'HF_HOME=/workspace/hf_cache' ~/.bashrc \
  || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
```

**How to verify:**

- The run completes 200 steps with no NaN and no decord crashes.
- The `time` block at the end prints `real`, `user`, `sys` values.
- Step 0 and step 50 metrics are printed.

**Report back to the agent:**

- The exact `real` time printed by `time` (e.g. `2m14s`).
- The step-0 values of `L_flow` and `L_var`.
- The step-50 values of `L_flow` and `L_var`.
- Whether `grad_has_nan` was 0 at step 0 (look in the step-0 log line).

(Pre-optimization baseline for reference — the new step-0 numbers should be
within ~5% of these:
`L_flow ≈ 2.87`, `L_var ≈ 0.43` at step 0;
`L_flow ≈ 3.38`, `L_var ≈ 0.42` at step 50.)

---

## Task H1.3 — Check vCPU count and RAM on the pod

**Status:** [DONE] — nproc=128, 2× AMD EPYC 7742 64-Core (Threads/core=1), 2 TiB RAM. No upgrade is meaningful.

**Corresponds to:** the first pause point in [`TASKS.md`](TASKS.md), after A1.3.

**Description:**
Confirm the pod has enough CPU resources to keep 8 dataloader workers
genuinely parallel.

**Instructions:**

```bash
# On the pod:
nproc
lscpu | grep -E "Model name|^CPU\(s\):|Thread"
free -h
```

**How to verify:**

- All three commands ran without error.

**Report back to the agent:**

- The `nproc` integer (e.g. `32`).
- The CPU model name line from `lscpu` (e.g.
  `Model name: Intel(R) Xeon(R) Platinum 8480C`).
- The Threads-per-core line if present.
- The first row of `free -h` (total RAM, e.g. `Mem: 256Gi`).

---

## Task H1.4 — Pod-tier upgrade decision

**Status:** [DONE] — Keeping current pod (nproc=128 already top-tier; bottleneck is per-worker decord-seek on VP9, not cores). Next folder: Plan Phase 02.

**Corresponds to:** the second pause point in [`TASKS.md`](TASKS.md), after
the agent writes its decision note in A1.4.

**Description:**
Based on the agent's decision note (`s/step` table) and your `nproc`
reading, decide whether to keep the current pod or redeploy at a higher
CPU tier. The agent cannot make this call — it depends on your billing
budget and how many runs you expect to do.

**Instructions:**

Reference decision table (from the agent's A1.4 note):

| 1A.5 s/step | 1B.1 vCPUs | Recommended decision |
|---|---|---|
| ≤ 0.7 | any | Keep the pod. |
| 0.7 – 1.0 | ≥ 16 | Keep the pod. |
| 0.7 – 1.0 | 12–15 | Keep unless tech lead disagrees. |
| 0.7 – 1.0 | < 12 | Upgrade. |
| > 1.0 | any | Try Plan Phase 03 first (code-side); upgrade hardware only if 03 isn't enough. |

If upgrading: follow [`../../SETUPS/SETUP.md`](../../SETUPS/SETUP.md) Path A
step **A4 Path 4b** — *Deploy new pod with the same network volume*. Then:

```bash
# On the NEW pod, after SSH:
cd /workspace/hierarchal-jepa-flow-world-model
git log -1 --oneline                     # confirm latest commit is present

# Re-run H1.1 and H1.2 to confirm throughput on the new tier:
python train.py --stage0-only
time python train.py --data ssv2_tiny --steps 200 --log-every 50
```

**How to verify:**

- A decision is recorded in chat or commit message:
  - "Keeping current pod" — done, no action.
  - "Upgrading to <tier>" — new pod is up, `H1.1` and `H1.2` rerun
    successfully on it.

**Report back to the agent:**

- The final decision string.
- If upgraded: the new pod's `nproc`, RAM, and the new `s/step` from the
  re-run.
