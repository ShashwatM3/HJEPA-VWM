# Codebase structure

This note maps **what lives where** in the HJEPA-VWM repository: training code, MLOps helpers,
agent docs, and experiment history. It is written for humans onboarding to the project and for
agents that need a fast orientation before tracing a specific path in code.

**Agents:** read [`AGENT_FILES/AGENTS.md`](../AGENT_FILES/AGENTS.md) first for architecture, shapes, and
invariants. Use this file when you need to know *which file to open*, not *how the tensor flows*.

---

## Repository layout (top level)

```text
HJEPA-VWM/
├── config.py              # All defaults, paths, dimensions
├── data.py                # Video dataset + dataloader (SSv2 .webm / EGO4D .mp4 chunks)
├── make_subset.py         # Build ssv2_tiny symlink subset
├── select_ego4d_uids.py   # Pick EGO4D source UIDs + download batches from ego4d.json
├── chunk_ego4d.py         # Chunk EGO4D 540ss videos into 4s/12fps/256px .mp4 clips
├── make_ego4d_subset.py   # Build ego4d_tiny symlink subset
├── models.py              # nn.Module classes (encoder, B, F_c, D, …)
├── losses.py              # Pure tensor losses (no parameters)
├── diagnostics.py         # Collapse probes, baselines, AGC helpers
├── train.py               # Training loop, CLI, checkpoints, W&B logging
├── parse_logs.py          # Console log → JSON
├── run_history.py         # W&B Public API export + reports
├── drift_probe.py         # Offline within-video drift probe (V-JEPA vs latent)
├── rank_probe.py          # Offline effective-rank probe for frozen V-JEPA embeddings
├── whiten_stats.py        # Offline whitening stats for frozen V-JEPA features
├── tests/                 # Contract and gradient-routing tests
├── requirements.txt
├── pyproject.toml         # Black + Ruff
├── README.md              # Human entry point (short)
├── GUIDES/                # Operator playbooks + project knowledge
├── AGENT_FILES/           # Agent behaviour, setups
└── KANBAN/                # Phase 1 experiment history (living record)
```

Runtime data and checkpoints live **outside** the git repo on RunPod under `/workspace/`. See
[`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../SETUPS/VOLUME_LAYOUT.md).

---

## Training pipeline (the six core files)

These flat files are the entire Phase 1 implementation. There is no `src/` package.

| File | Owns | Does not own |
|---|---|---|
| `config.py` | `ModelConfig`, `TrainConfig`, `Config`; path defaults; locked dimensions | CLI parsing (that is `train.py`) |
| `data.py` | `SSV2Dataset`, clip windows, encoder normalization | Model forward passes |
| `models.py` | `FrozenEncoder`, `Bottleneck` (+ `BottleneckLatentBlock`), `TargetBottleneck`, `CoarseFlow`, `Decoder`, `FeatureMeanTracker`, `FeatureWhitener` | Loss math, optimizer |
| `losses.py` | `flow_matching_loss`, `variance_floor`, `reconstruction_loss`, `as_target`, … | Any `nn.Parameter` |
| `diagnostics.py` | `variance_stats`, `coarse_baselines`, `reconstruction_readouts`, AGC/decay grouping | Training loop |
| `train.py` | `train_step`, `run_diagnostics`, EMA, checkpoints, `argparse`, `wandb.init` | New module architectures |

**Typical read order for a change:**

1. `config.py` — is there already a knob?
2. `train.py` — where is it wired into the step or diagnostics?
3. `models.py` / `losses.py` / `diagnostics.py` — the actual math
4. `tests/test_*.py` — existing contracts

**Module construction order** (must stay consistent):

```text
encoder, bottleneck, target_bottleneck, coarse_flow, decoder
```

---

## Data path

| File | Role |
|---|---|
| `make_subset.py` | One-time: symlinks ~4k train / ~348 val clips into `ssv2_tiny/` + `manifest.json` |
| `select_ego4d_uids.py` | One-time: reads `ego4d.json`, picks scenario-diverse source UIDs (~210 h), splits train/val by SOURCE video, emits hour-balanced download batch files |
| `chunk_ego4d.py` | One-time (per batch): ffmpeg-chunks downloaded EGO4D 540ss videos into 4 s / 12 fps / 256 px `.mp4` clips under `data/ego4d/`; idempotent, batch-friendly |
| `make_ego4d_subset.py` | One-time: symlinks ~4k train / ~350 val chunks into `ego4d_tiny/` + `manifest.json` (≤10 chunks per source video) |
| `data.py` | Loads `.webm`/`.mp4` via decord; yields `(context_clip, target_clip)` tensors `(8,3,256,256)` |

Key semantics:

- Context window ends at time `t`; target window ends `horizon_k` **original** frames later.
- Only the 16 frame indices needed are decoded (not full videos).
- Encoder path uses ImageNet/V-JEPA mean/std — not `[-1,1]` VAE normalization.
- Datasets: `--data ssv2 | ssv2_tiny | ego4d | ego4d_tiny`. EGO4D chunks are pre-encoded to
  12 fps, so `frame_stride`/`horizon_k` keep the same real-time meaning as on SSv2
  (build procedure: [`AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md`](../AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md)).

Override dataset parent locally: `export JEPA_DATA_ROOT=/path/to/data`.

---

## Training and evaluation entry points

| Command | Purpose |
|---|---|
| `python train.py --stage0-only` | Synthetic one-step sanity (encoder load + shapes) |
| `python train.py --data ssv2_tiny --steps 500` | Short smoke run |
| `python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5` | Typical full Phase 1 experiment (CLI overrides defaults) |

Important CLI groups (full list in `train.py` `parse_args()`):

- **Data:** `--data`, `--horizon-k`, `--seed`
- **Schedule:** `--steps`, `--resume`, `--log-every`, `--diag-every`, `--checkpoint-dir`
- **Losses:** `--lambda-var`, `--lambda-recon`, `--lambda-recon-pred`, `--lambda-sigreg`, …
- **Modes:** `--present-recon-only`, `--predict-residual`, `--recon-residual-target`, `--recon-loss-mode`, `--whiten-features` (+ `--whiten-stats-path`, `--whiten-eps`)
- **Optimizer:** `--lr-bottleneck`, `--lr-coarse-flow`, `--no-agc`, `--grad-skip-threshold`

Checkpoints write to `checkpoint_dir` (default `/workspace/checkpoints`). The frozen encoder is
**never** checkpointed — it reloads from Hugging Face.

---

## MLOps and analysis helpers

These are **not** imported by the training loop. Use them for log export, run comparison, and
KANBAN evidence.

| File | Role |
|---|---|
| `parse_logs.py` | Parses `step=N {dict}` console lines → structured JSON |
| `run_history.py` | Pulls full metric history from W&B Public API; `--report` for Phase 1 summaries |
| `drift_probe.py` | Within-video temporal drift: frozen V-JEPA embedding drift vs bottleneck latent drift (loaded from checkpoints) on a pinned probe set; writes JSON + PNGs, no W&B logging |
| `rank_probe.py` | Frozen-encoder effective rank: applies the `c_effective_rank` covariance-rank formula to cached V-JEPA `e` tokens over the drift-probe manifest; writes JSON + optional PNG, no W&B logging |
| `whiten_stats.py` | Offline whitening statistics (mean + covariance eigendecomposition) of frozen V-JEPA training-set features; `train.py --whiten-features` consumes its `.pt` output (never imports the script), no W&B logging |

W&B project: **`smahalanobis-uc-davis/hjepa-vwm`**.

```bash
# Export one run's history + text report
python run_history.py --run <run_id> --report

# Parse a saved console log
python parse_logs.py logs/my_run.txt -o logs/my_run.json

# Within-video drift probe: V-JEPA reference curve + latent curves from checkpoints
# (usage details in README "Within-video drift probe")
python drift_probe.py --data ssv2 --probe-videos 64 \
  --ckpt /workspace/ckpt/<run-dir>/phase1_step15000.pt

# V-JEPA effective-rank budget over the same fixed probe-set namespace
python rank_probe.py --data ssv2 --probe-videos 64
```

Prefer the **W&B MCP server** in Cursor for interactive metric pulls (see
[`GUIDES/MLOPS.md`](MLOPS.md)).

---

## Tests

| Path | What it guards |
|---|---|
| `tests/test_phase1_contract.py` | Config, subset manifest, flat-file deliverables |
| `tests/test_reconstruction_loss.py` | Reconstruction loss modes and detach behaviour |
| `tests/test_optimizer_and_flow.py` | Optimizer groups and flow loss contracts |
| `tests/test_agc.py` | Adaptive gradient clipping contracts |
| `tests/test_decoder.py` | Fixed-position decoder invariants |
| `tests/test_residual_recon_target.py` | `FeatureMeanTracker` + shuffled-c readouts |
| `tests/test_present_recon_only.py` | Present-only mode gradient routing |
| `tests/test_sigreg.py` | SIGReg loss and logging RNG isolation |
| `tests/test_bottleneck_attention.py` | Bottleneck latent-stack identity-at-init + sharp-attention diagnostics |
| `tests/test_drift_probe.py` | Drift-probe pure helpers: offsets, windows, drift matrices, Spearman, checkpoint-config rebuild |
| `tests/test_rank_probe.py` | Rank-probe pure helpers: covariance spectrum, entropy-rank formula, energy ranks, report validation |
| `tests/test_whitening.py` | `FeatureWhitener` round-trip/stats contracts and train-step/checkpoint whitening wiring |

Quick local checks (no encoder download):

```bash
pytest -q
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

---

## Documentation map

| Path | Audience | Contents |
|---|---|---|
| [`AGENT_FILES/AGENTS.md`](../AGENT_FILES/AGENTS.md) | **Agents (start here)** | Architecture, shapes, training step, invariants, doc precedence |
| [`AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`](../AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md) | Agents | Phase discipline, escalation |
| [`AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md`](../AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md) | Agents | Naming map, docstrings, detach rules |
| [`GUIDES/README.md`](README.md) | Humans + agents | Index of all guides (this folder) |
| [`GUIDES/latest_brief.md`](latest_brief.md) | Both | Architecture narrative v0.3 — **not ground truth** |
| [`GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`](PROBLEMS_METRICS_AND_EXPERIMENTS.md) | Both | Metric glossary + experiment problem history |
| [`GUIDES/CODEBASE_STRUCTURE.md`](CODEBASE_STRUCTURE.md) | Both | This file |
| [`AGENT_FILES/SETUPS/SETUP.md`](../AGENT_FILES/SETUPS/SETUP.md) | Humans | Laptop ↔ RunPod operator guide |
| [`AGENT_FILES/SETUPS/NEW_POD.md`](../AGENT_FILES/SETUPS/NEW_POD.md) | Humans | Fresh pod bootstrap |
| [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../AGENT_FILES/SETUPS/VOLUME_LAYOUT.md) | Both | `/workspace` path contract |

---

## Experiment history (`KANBAN/`)

KANBAN is the **living record** of what was tried in Phase 1. It is not a build spec.

```text
KANBAN/
├── PROTOCOL.md
└── PHASE_1/
    ├── README.md
    └── investigation_NNN/ ...
```

W&B reading cycles: [`GUIDES/READING_EXPERIMENTS.md`](READING_EXPERIMENTS.md).  
Experiment lifecycle: [`GUIDES/EXPERIMENT_LIFECYCLE.md`](EXPERIMENT_LIFECYCLE.md).

---

## What is not implemented yet

Phase 1 only. The following appear in phase docs but **do not exist** in code yet:

- `FineFlow` (`F_e`), Stage 2/3 training
- VAE wrapper, pixel frame generator, `eval.py`
- Multi-horizon embedding (`h_k`)
- Shuffled-c / zero-c fine-flow bypass tests

Do not assume these modules exist — grep before editing.
