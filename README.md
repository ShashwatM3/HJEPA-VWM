# HJEPA-VWM

Hierarchical JEPA-Flow video world model — research codebase for Phase 1 (coarse abstract
latent prediction). This README is an entry point; deeper detail lives in linked docs.

Index of all guides: [`GUIDES/README.md`](GUIDES/README.md)

---

## New to this project? Read in this order

### 1. Operator playbooks (start here if you run experiments)

| # | Guide | Why |
|---|---|---|
| 1 | [**`GUIDES/EXPERIMENT_LIFECYCLE.md`**](GUIDES/EXPERIMENT_LIFECYCLE.md) | Full loop: brainstorm → plan → implement → pod → train → analyze → KANBAN |
| 2 | [**`GUIDES/MLOPS.md`**](GUIDES/MLOPS.md) | RunPod, W&B, volumes, `run_history.py` |
| 3 | [**`GUIDES/READING_EXPERIMENTS.md`**](GUIDES/READING_EXPERIMENTS.md) | How to read a W&B run (Q1–Q8) |

### 2. Project knowledge (architecture, code map, metrics)

| Topic | Document |
|---|---|
| **Current architecture narrative (v0.3)** | [`GUIDES/latest_brief.md`](GUIDES/latest_brief.md) |
| **Original design (v0.1)** | [`GUIDES/original_brief.pdf`](GUIDES/original_brief.pdf) |
| **File map** (training code, tests, MLOps scripts) | [`GUIDES/CODEBASE_STRUCTURE.md`](GUIDES/CODEBASE_STRUCTURE.md) |
| **Metric glossary + experiment history** | [`GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`](GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md) |

Briefs are **historical narrative**, not immutable spec — shipped behavior is in `config.py` and
[`AGENT_FILES/AGENTS.md`](AGENT_FILES/AGENTS.md).

### 3. Experiment record

| Topic | Document |
|---|---|
| **What we tried (Phase 1 KANBAN)** | [`KANBAN/PHASE_1/README.md`](KANBAN/PHASE_1/README.md) |
| **KANBAN update rules** | [`KANBAN/PROTOCOL.md`](KANBAN/PROTOCOL.md) |

### 4. Infrastructure

| Topic | Document |
|---|---|
| **First RunPod deploy** | [`AGENT_FILES/SETUPS/SETUP.md`](AGENT_FILES/SETUPS/SETUP.md) |
| **Fresh pod bootstrap** | [`AGENT_FILES/SETUPS/NEW_POD.md`](AGENT_FILES/SETUPS/NEW_POD.md) |
| **Volume / data paths** | [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](AGENT_FILES/SETUPS/VOLUME_LAYOUT.md) |

W&B project: **`smahalanobis-uc-davis/hjepa-vwm`**

---

## Within-video drift probe (offline diagnostic)

`drift_probe.py` measures **within-video temporal drift**: how far the frozen V-JEPA embedding
of a window travels over time (`1 − cos(e_t, e_{t+k})` for a ladder of offsets `k`), and how far
the bottleneck latent travels over the **same** windows (`1 − cos(c_t, c_{t+k})`, with the
bottleneck loaded from any training checkpoint). It never compares two different videos.

It answers two questions the training dashboards cannot: **how much change does the encoder even
see at each horizon** (the signal budget behind the copy baseline), and **what fraction of that
change survives the bottleneck** for a given run/checkpoint. Runs entirely offline — no training
run is launched or modified, and past runs are evaluated from their saved checkpoints.

### How to run

**Option 1 — on RunPod (the normal path).** The pod has torch, the SSv2 data at
`/workspace/data/`, and the run checkpoints under `/workspace/ckpt/` — so this is where the probe
is meant to run. Launch inside tmux so an SSH drop cannot kill the first (encoder-heavy) pass:

```bash
tmux new -s drift_probe
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache

# Sanity: the probe's unit tests actually execute here (they skip on torch-less machines).
python -m pytest tests/test_drift_probe.py -q

# Step 1 (one-time-ish): V-JEPA-only reference curve.
# Builds the probe manifest + encoder feature cache under logs/drift_probe/.
python drift_probe.py --data ssv2 --probe-videos 64 --latent-curve off

# Step 2: evaluate checkpoints (repeat --ckpt to overlay runs/steps;
# the encoder side is read from the cache, so this part is cheap).
python drift_probe.py --data ssv2 --probe-videos 64 \
  --ckpt /workspace/ckpt/<run-dir>/phase1_step7500.pt \
  --ckpt /workspace/ckpt/<run-dir>/phase1_step15000.pt

# Detach: Ctrl-B then D.  Reattach: tmux attach -t drift_probe
```

Outputs land in `logs/drift_probe/` on the pod. The probe logs **nothing to W&B** by design —
the graphs are local PNG files; view them by copying them to your laptop.

**Viewing the graphs (copying outputs off the pod).** Run these on your LAPTOP (a local
terminal, not the SSH session). RunPod's proxy SSH (`<pod-id>@ssh.runpod.io`) does **not**
support `scp`/`sftp`, so pipe the bytes through plain `ssh` instead:

```bash
# One PNG (substitute your own <pod-id>@ssh.runpod.io connect string):
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519 \
  "cat /workspace/hierarchal-jepa-flow-world-model/logs/drift_probe/graph1_ssv2_validation_n64_seed42.png" \
  > ~/Desktop/graph1_drift.png
open ~/Desktop/graph1_drift.png

# All PNGs + JSON (excludes the ~1 GB feature cache):
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519 \
  "tar -C /workspace/hierarchal-jepa-flow-world-model/logs -czf - --exclude='*.pt' drift_probe" \
  > ~/Desktop/drift_probe.tgz
tar -xzf ~/Desktop/drift_probe.tgz -C ~/Desktop && open ~/Desktop/drift_probe/
```

If your Connect tab also offers a **direct TCP** SSH command (`ssh root@<ip> -p <port> ...`),
normal `scp -P <port> -i <key> root@<ip>:<remote-path> <local-path>` works over that one.
Alternatives: open the PNG via Cursor/VS Code Remote-SSH's file explorer, or via the pod's
JupyterLab if the RunPod Connect tab exposes one.

**Option 2 — on a personal device (laptop/workstation).** Works only if the machine has the
dependencies and a local SSv2 copy; there is no pod magic in the script itself:

```bash
pip install -r requirements.txt          # torch, decord, transformers, matplotlib, ...
export JEPA_DATA_ROOT=/path/to/data      # parent dir containing ssv2/ or ssv2_tiny/
export HF_HOME=/path/to/hf_cache         # optional; first run downloads V-JEPA 2 ViT-L (~1.2 GB)

# Reference curve on the tiny subset (CPU works — the 64-video encode is slow but one-time).
python drift_probe.py --data ssv2_tiny --probe-videos 64 --latent-curve off

# Latent curves need checkpoint .pt files copied from the pod (e.g. via scp) first.
python drift_probe.py --data ssv2_tiny --probe-videos 64 \
  --ckpt ~/checkpoints/<run-dir>/phase1_step15000.pt
```

Without torch installed, only `python drift_probe.py --help` works (by design); without a local
dataset, use Option 1. Keep ONE manifest/cache location per probe definition — the point of the
manifest is that every checkpoint is measured on identical videos and windows.

Useful flags: `--offsets 2,4,8,12,16,24,32` (measured ladder), `--graph2-offsets 16,24,32`
(offsets averaged into the per-video graph; defaults to the non-overlapping ones, k > 14),
`--encoder-curve on|off` / `--latent-curve on|off` (per-curve toggles), `--use-ema` (evaluate
`B_EMA` instead of the online bottleneck), `--out-dir logs/drift_probe`.

Outputs under `--out-dir`: `results_<tag>.json` (all raw drift matrices + Spearman
faithfulness scores), `graph1_<tag>.png` (X = offset k, Y = mean drift; V-JEPA reference vs
latent curves), and `graph2_<ckpt>_<tag>.png` per checkpoint (X = probe videos sorted by
V-JEPA drift, Y = per-video mean drift; staircase vs dots). The probe set is pinned by
`manifest_<tag>.json`, so every invocation measures identical videos and windows — checkpoints
from different runs are directly comparable. The first invocation pays one frozen-encoder pass
over the probe windows; afterwards the feature cache makes each checkpoint evaluation cheap.

Notes: checkpoints are self-describing (the bottleneck is rebuilt from the config stored inside
the `.pt`), so no architecture flags are needed. Checkpoints predating the sharp-slot attention
change (commit `cc0e318`) need the matching code checkout. If `matplotlib` is missing the probe
still writes the JSON and skips PNGs.

---

## For coding agents

Start at **[`AGENT_FILES/AGENTS.md`](AGENT_FILES/AGENTS.md)** — architecture, shapes, training step,
invariants, doc precedence (including why briefs are **not** ground truth), and mandatory read order.

Then read **[`AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`](AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md)** before
changing code.

Human–agent workflow detail: [`GUIDES/EXPERIMENT_LIFECYCLE.md`](GUIDES/EXPERIMENT_LIFECYCLE.md).

---
