# MLOps — RunPod, W&B, and experiment tooling

> **Audience:** The ML engineer running training jobs.  
> **Companion:** [`EXPERIMENT_LIFECYCLE.md`](EXPERIMENT_LIFECYCLE.md) for *when* to do each step; this file for *how* the infrastructure works.

---

## Stack overview

```text
Laptop (Cursor + git)
    │  git push
    ▼
RunPod GPU pod (SSH)
    │  train.py → wandb.log
    ▼
W&B project hjepa-vwm
    │  MCP / run_history.py
    ▼
KANBAN run folder (OBSERVATIONS, ANALYSIS, NEXT_STEPS)
```

| Piece | Role |
|---|---|
| **Git repo** | Code only — lives at `/workspace/hierarchal-jepa-flow-world-model/` on the pod |
| **Network volume** | Persistent `/workspace` — data, checkpoints, HF cache survive pod termination |
| **RunPod pod** | Ephemeral GPU machine — SSH in, pull code, launch `train.py` |
| **W&B** | Experiment tracking — metrics, config, run grouping |
| **`run_history.py`** | Pull full metric history via W&B Public API (offline analysis) |
| **`parse_logs.py`** | Parse console `step=N {dict}` logs → JSON |

---

## RunPod

### Volume layout

```text
/workspace/
├── hierarchal-jepa-flow-world-model/   # git clone (code)
├── data/ssv2/                          # full dataset (symlinks)
├── data/ssv2_tiny/                     # smoke subset (make_subset.py)
├── ssv2_raw/                           # raw .webm backing files
├── checkpoints/                        # default checkpoint dir
├── stats/                              # encoder/dataset-bound whitening envelopes
├── preflight/                          # resource + exact run-provenance JSON
└── hf_cache/                           # pinned Hugging Face encoder snapshots
```

Full reference: [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../AGENT_FILES/SETUPS/VOLUME_LAYOUT.md).

Local override: `export JEPA_DATA_ROOT=/path/to/data` (dataset parent only).

### Fresh pod checklist

Every **new pod** (new image, terminated and redeployed):

1. Complete the canonical five-step [`AGENT_FILES/SETUPS/NEW_POD.md`](../AGENT_FILES/SETUPS/NEW_POD.md) bootstrap — system packages, cache paths, clone/checkout, requirements, then W&B login.
2. Read the **investigation** `GUIDE.md` (e.g. `KANBAN/PHASE_1/investigation_007/GUIDE.md`) for sweep-specific launch.
3. Read the **run** `GUIDE.md` inside the run folder for that experiment's exact command.

Repeat deploys on an existing pod: skip NEW_POD if packages and W&B login persist; still `git pull`.

### SSH and long jobs

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519   # from RunPod Connect tab
nvidia-smi
tmux new -s train                                   # survives SSH disconnect
```

Official docs: [RunPod SSH](https://docs.runpod.io/pods/configuration/use-ssh),
[Manage Pods](https://docs.runpod.io/pods/manage-pods).

### Multi-GPU sweeps

Each config gets **one process per GPU** via `CUDA_VISIBLE_DEVICES` — not DDP. Each run needs its
**own checkpoint directory**. Group runs in W&B with `WANDB_RUN_GROUP`.

Example pattern: [`KANBAN/PHASE_1/investigation_007/GUIDE.md`](../KANBAN/PHASE_1/investigation_007/GUIDE.md).

### Code sync on pod

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git pull --ff-only
python -m pip install -r requirements.txt
python -m pytest -q
python -c "from models import smoke_test_models; smoke_test_models()"
```

---

## Weights & Biases

| Setting | Value |
|---|---|
| Entity | `smahalanobis-uc-davis` |
| Project | `hjepa-vwm` |
| Auth | `wandb login` on pod, or `WANDB_API_KEY` in environment |

### What gets logged

| Cadence | Default interval | Examples |
|---|---|---|
| Training step | `log_every` = 50 | `L_flow`, `grad_norm`, `grad_skipped`, `L_recon` |
| Diagnostics | `diag_every` = 500 | `c_effective_rank`, `coarse_vs_copy_ratio`, `L_recon_present` |

Encoder experiments additionally update W&B config with the resolved `EncoderSpec`, feature and
dataset fingerprints, dependency/git identity, initialization/data-order hashes, and stats
fingerprint. For queries and dashboards, `resolved_encoder_spec` and
`resolved_feature_fingerprint` are the authoritative encoder geometry and identity. In particular,
do not read alternate-encoder geometry from legacy `model.d_e` or `model.encoder_repo` fields,
which remain only for historical checkpoint/config compatibility. The complete copy also lives at
`resolved_provenance.encoder_spec`. Runtime provenance includes CUDA, cuDNN, GPU model, and the
explicit data/model/training/diagnostic seed streams. Whitening additionally binds the transform
seed, exact clip budget (12,800 by default), token-row count, and eigensolver. `--require-wandb` makes
initialization, logging, and final checksum recording fatal
instead of silently continuing. The small provenance/stats artifacts may be uploaded; giant
checkpoints are not uploaded automatically, and the final checkpoint is referenced by SHA-256.

Dataset identity schema v2 opens each split container once to bind decoded frame count in
addition to sorted path and byte size. On a full corpus, the first provenance/preflight pass
can therefore take noticeable time; a corrupt or zero-frame container is a hard pre-run
failure. Resource-preflight JSON uses CUDA events and keeps diagnostics outside the measured
training interval. It reports `encoder_throughput` and `training_step_throughput` with
examples/s, frames/s, and detailed tokens/s, alongside encoder-only and total peak memory.

Metric meanings: [`GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`](PROBLEMS_METRICS_AND_EXPERIMENTS.md).

### Reading a run

Use [`READING_EXPERIMENTS.md`](READING_EXPERIMENTS.md) — fixed Q1–Q8 cycles. Do not improvise a
generic ML metric tour.

### W&B MCP (in Cursor)

Enable the **user-wandb** MCP server so agents can pull real trajectories instead of guessing.

- Entity: `smahalanobis-uc-davis`
- Project: `hjepa-vwm`
- Setup: [W&B MCP documentation](https://docs.wandb.ai/guides/hosting/mcp-server)

**Agent skills** (install if missing):

- [`.claude/skills/read-wandb-run/SKILL.md`](../.claude/skills/read-wandb-run/SKILL.md)
- [`.agents/skills/wandb-primary/SKILL.md`](../.agents/skills/wandb-primary/SKILL.md)

### CLI export (no MCP)

```bash
python run_history.py --run <run_id> --report
python run_history.py --run <run_id> -o logs/<name>.json
python run_history.py --run <run_id> --format parse_logs -o logs/<name>/output.log
```

Requires `wandb login` or `WANDB_API_KEY`. Uses unsampled `scan_history()` per W&B Public API.

### Run grouping

```bash
python train.py \
  --wandb-entity smahalanobis-uc-davis \
  --wandb-project hjepa-vwm \
  --wandb-group manual_phase1_recipe \
  --wandb-name "Manual Phase 1 · configs/train.yaml" \
  --require-wandb
```

---

## Checkpoints and logs

| Path | Contents |
|---|---|
| YAML `checkpoint_dir` or operator override `--checkpoint-dir` (default `/workspace/checkpoints`) | Atomic `phase1_step*.pt` — B/B_EMA/F_c/D, optimizer, fixed buffers, RNG/sampler, resolved encoder/dataset/run identity |
| `logs/` (convention, under repo) | Console captures from `train.py` redirects |

The frozen encoder weights are **not** checkpointed; its exact repository, immutable revision,
preprocessing/precision contract, parameter count, and feature fingerprint are. Resume rejects a
different encoder, dataset, run recipe, or optimizer unless the corresponding explicit migration
flag is supplied. Whitening buffers are embedded so a strict resume survives deletion of the
external stats file.

Resume:

```bash
python train.py \
  --resume /workspace/checkpoints/phase1_step15000.pt
```

Losses, schedules, modes, and optimizer settings always come from `configs/train.yaml`. Only dataset,
encoder, `N_c`, `D_c`, and bottleneck mixer width remain scientific CLI overrides.

---

## Operator paths (first time vs repeat)

| Situation | Doc |
|---|---|
| First time on RunPod (volume migration, subset, first train) | [`AGENT_FILES/SETUPS/SETUP.md`](../AGENT_FILES/SETUPS/SETUP.md) Path A |
| Code changed locally → train again | [`AGENT_FILES/SETUPS/SETUP.md`](../AGENT_FILES/SETUPS/SETUP.md) Path B |
| Brand-new pod image | [`AGENT_FILES/SETUPS/NEW_POD.md`](../AGENT_FILES/SETUPS/NEW_POD.md) |
| SSH / billing / zero-GPU issues | [RunPod docs](https://docs.runpod.io/pods/manage-pods) and [`AGENT_FILES/SETUPS/SETUP.md`](../AGENT_FILES/SETUPS/SETUP.md) FAQ |
