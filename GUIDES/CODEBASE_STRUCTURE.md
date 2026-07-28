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
├── config.py              # Typed defaults + strict YAML experiment loader
├── configs/
│   └── train.yaml         # The only editable Phase-1 experiment recipe
├── data.py                # Video dataset + dataloader (SSv2 .webm / EGO4D .mp4 chunks)
├── encoders.py            # Generic raw-clip frozen-encoder seam + private adapters
├── provenance.py          # Dataset/run identity + atomic checkpoint/artifact envelopes
├── make_subset.py         # Build ssv2_tiny symlink subset
├── select_ego4d_uids.py   # Pick EGO4D source UIDs + download batches from ego4d.json
├── chunk_ego4d.py         # Chunk EGO4D 540ss videos into 4s/12fps/256px .mp4 clips
├── make_ego4d_subset.py   # Build ego4d_tiny symlink subset
├── models.py              # EncoderSpec-driven Phase-1 trainable modules
├── losses.py              # Pure tensor losses (no parameters)
├── diagnostics.py         # Collapse probes, baselines, AGC helpers
├── train.py               # Training loop, CLI, checkpoints, W&B logging
├── parse_logs.py          # Console log → JSON
├── run_history.py         # W&B Public API export + reports
├── evaluate_checkpoint_diagnostics.py # Paired encoder/latent checkpoint cosine evaluator
├── drift_probe.py         # Encoder-generic within-video feature-vs-latent drift probe
├── rank_probe.py          # Encoder-generic effective-rank probe over the shared cache
├── whiten_stats.py        # Encoder/dataset/seed-bound offline whitening statistics
├── tests/                 # Contract and gradient-routing tests
├── requirements.txt
├── pyproject.toml         # Black + Ruff
├── README.md              # Human entry point (short)
├── GUIDES/                # Operator playbooks + project knowledge
├── AGENT_FILES/           # Agent behaviour, SSH access, setups, knowledge dossiers
├── .agents/skills/        # Cross-agent repository skills (canonical copies)
├── .claude/skills/        # Claude Code skill links/copies
└── KANBAN/                # Phase 1 experiment history (living record)
```

Runtime data and checkpoints live **outside** the git repo on RunPod under `/workspace/`. See
[`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../SETUPS/VOLUME_LAYOUT.md).

---

## Training pipeline (the seven core files)

These flat files are the entire Phase 1 implementation. There is no `src/` package.

| File | Owns | Does not own |
|---|---|---|
| `config.py` | `EncoderConfig`, `ModelConfig`, `TrainConfig`, `Config`; strict YAML loading; path defaults; locked dimensions | CLI parsing (that is `train.py`) |
| `data.py` | Deterministic raw `[0,1]` `ClipBatch` values, context/future windows, shared transforms, resumable sample order | Encoder normalization or backend selection |
| `encoders.py` | Raw-clip contract, normalization/precision/frame-microbatching, immutable `EncoderSpec`, private registry/adapters, real smoke CLI | Latent architecture, losses, or training orchestration |
| `models.py` | `EncoderSpec`-driven `Bottleneck`, `TargetBottleneck`, `CoarseFlow`, `Decoder`, `FeatureMeanTracker`, and `FeatureWhitener`; narrow historical V-JEPA geometry reader | Encoder loading/preprocessing, loss math, optimizer |
| `losses.py` | `flow_matching_loss`, `variance_floor`, `reconstruction_loss`, `as_target`, … | Any `nn.Parameter` |
| `diagnostics.py` | `variance_stats`, `coarse_baselines`, `reconstruction_readouts`, AGC/decay grouping | Training loop |
| `provenance.py` | Dataset/run fingerprints, EncoderSpec serialization, atomic JSON/Torch writes, strict whitening/cache envelopes | Model math or network authentication |
| `train.py` | `train_step`, diagnostics, EMA, strict resume/checkpoints, parity/resource preflight, five-axis scientific CLI, W&B policy | Encoder backend classes or new module architectures |

**Typical read order for a change:**

1. `configs/train.yaml` and `config.py` — is there already a recipe field?
2. `encoders.py` — if the change touches frozen features or preprocessing
3. `train.py` — where is it wired into the step or diagnostics?
4. `models.py` / `losses.py` / `diagnostics.py` — the actual math
5. `tests/test_*.py` — existing contracts

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
| `data.py` | Loads `.webm`/`.mp4` via decord; yields typed raw `ClipBatch` values with context and optional target tensors `(B,8,3,256,256)` |

Key semantics:

- Context window ends at time `t`; target window ends `horizon_k` **original** frames later.
- Only the 16 frame indices needed are decoded (not full videos).
- `data.py` emits canonical raw `[0,1]` pixels. `encoders.FrozenEncoder` owns the
  selected adapter's normalization exactly once; no common caller imports processor rules.
- Present-only mode decodes context frames only. Full mode decodes context and target
  together so one deterministic crop/jitter is shared by both windows.
- Datasets: YAML `data.dataset`, with the hot override
  `--data ssv2 | ssv2_tiny | ego4d | ego4d_tiny`. EGO4D chunks are pre-encoded to
  12 fps, so `frame_stride`/`horizon_k` keep the same real-time meaning as on SSv2
  (build procedure: [`AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md`](../AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md)).

Override dataset parent locally: `export JEPA_DATA_ROOT=/path/to/data`.

---

## Training and evaluation entry points

| Command | Purpose |
|---|---|
| `python encoders.py --smoke --encoder vjepa2_vitl16 --batch-size 1` | Real pinned-adapter shape/revision/freeze/memory report (downloads weights if absent) |
| `python encoders.py --smoke --encoder siglip2_vitb16 --batch-size 1` | Real pinned SigLIP 2 vision-only patch-tower smoke |
| `python encoders.py --smoke --encoder dinov3_vitb16 --batch-size 1` | Real pinned DINOv3 patch-token smoke; requires accepted checkpoint access |
| `python train.py --stage0-only` | Synthetic one-step sanity (encoder load + shapes) |
| `python train.py --resource-preflight ...` | Exact one-step forward/backward/diagnostic memory and throughput report |
| `python train.py --preflight-only --provenance-out run.json ...` | Materialize a no-step immutable run identity for paired comparison |
| `python train.py` | Run the repository's single `configs/train.yaml` recipe |
| `python train.py --encoder dinov3_vitb16 --n-c 16 --d-c 512` | Full experiment with only active sweep axes overridden |

Configuration interface:

- **Single YAML recipe:** `configs/train.yaml`, always loaded; encoder details, batch/schedule,
  losses, reconstruction/whitening modes, decoder/background architecture, optimizer/AGC,
  cadence, runtime, and W&B defaults. Every leaf has allowed-value/range guidance inline.
- **Five scientific CLI overrides:** `--data`, `--encoder`, `--n-c`, `--d-c`,
  `--bottleneck-mixer-dim`.
- **Operations:** strict W&B, atomic RNG/sampler resume, explicit optimizer reset/dataset
  transfer/legacy flags, frame-count-bound provenance comparison, CUDA-event resource preflight
  with examples/frames/tokens throughput

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
| `evaluate_checkpoint_diagnostics.py` | Restores current Phase-1 checkpoints through the strict loader and reports paired encoder/live-bottleneck cross-video cosine on one corrected fixed source-diverse batch |
| `drift_probe.py` | Encoder-generic within-video detailed/latent drift; strict versioned feature cache; checkpoint EncoderSpec/whitener reconstruction; JSON/PNG plus optional W&B artifact |
| `rank_probe.py` | Encoder-generic raw/effective rank over the shared strict probe manifest/cache; frame layouts add pre-concatenation per-frame norms/ranks; identity-bearing JSON/plot plus optional W&B artifact |
| `whiten_stats.py` | Deterministic context-only stats through the same factory/preprocessing; atomic encoder/dataset-bound envelope consumed strictly by training; inspect and optional W&B artifact modes |

W&B project: **`smahalanobis-uc-davis/hjepa-vwm`**.

```bash
# Export one run's history + text report
python run_history.py --run <run_id> --report

# Parse a saved console log
python parse_logs.py logs/my_run.txt -o logs/my_run.json

# Within-video drift probe: selected encoder reference + checkpoint latent curves
# (usage details in README "Within-video drift probe")
python drift_probe.py --data ssv2 --probe-videos 64 \
  --ckpt /workspace/ckpt/<run-dir>/phase1_step15000.pt

# SigLIP effective-rank budget over the same fixed probe-set namespace
python rank_probe.py --data ssv2 --encoder siglip2_vitb16 --probe-videos 64
```

Prefer the **W&B MCP server** in Cursor for interactive metric pulls (see
[`GUIDES/MLOPS.md`](MLOPS.md)).

---

## Tests

| Path | What it guards |
|---|---|
| `tests/test_phase1_contract.py` | Config, subset manifest, flat-file deliverables |
| `tests/test_encoders.py` | Raw-input validation, normalization once, both dense layouts, frame ordering/microbatching, fingerprints, sticky freeze, registry/revision failures, and V-JEPA legacy parity |
| `tests/test_encoder_run_contract.py` | Paired initialization, deterministic resume/transfer, checkpoint-before-mutation, B>1 honesty, CLI and diagnostic RNG contracts |
| `tests/test_experiment_config.py` | Strict duplicate/key/type/legacy-field YAML validation, exact five-axis CLI, precedence, cache synchronization, and shipped recipes |
| `tests/test_provenance.py` | Dataset/runtime/seed identity, narrow transfer, and strict atomic whitening/feature-cache envelope validation |
| `tests/test_reconstruction_loss.py` | Reconstruction loss modes and detach behaviour |
| `tests/test_optimizer_and_flow.py` | Optimizer groups and flow loss contracts |
| `tests/test_agc.py` | Adaptive gradient clipping contracts |
| `tests/test_decoder.py` | Fixed-position decoder invariants |
| `tests/test_residual_recon_target.py` | `FeatureMeanTracker` + shuffled-c readouts |
| `tests/test_present_recon_only.py` | Present-only mode gradient routing |
| `tests/test_sigreg.py` | SIGReg loss and logging RNG isolation |
| `tests/test_bottleneck_attention.py` | Bottleneck latent-stack identity-at-init + sharp-attention diagnostics |
| `tests/test_evaluate_checkpoint_diagnostics.py` | Offline evaluator restoration, fixed-batch, one-forward, inference/eval, output, and failure contracts with fake modules |
| `tests/test_drift_probe.py` | Drift-probe pure helpers: offsets, windows, drift matrices, Spearman, checkpoint-config rebuild |
| `tests/test_rank_probe.py` | Rank-probe pure helpers: covariance spectrum, entropy-rank formula, energy ranks, report validation |
| `tests/test_whitening.py` | `FeatureWhitener` round-trip/stats contracts and train-step/checkpoint whitening wiring |
| `tests/test_chunk_ego4d.py` | ffmpeg availability, conservative worker/thread limits, atomic chunk writes, and encode-failure handling |
| `tests/test_select_ego4d_uids.py` | grouped/metadata-only UID rejection and authoritative-tier provenance |

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
| [`AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md`](../AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md) | Humans + agents | SSH key/alias setup, remote authorization, Git sync, tmux launch, monitoring, and revocation |
| [`AGENT_FILES/GUIDE_AUTONOMOUS_REMOTE_RUN.md`](../AGENT_FILES/GUIDE_AUTONOMOUS_REMOTE_RUN.md) | Agents | Unattended execution state machine from a provider SSH command plus a KANBAN run guide |
| [`.agents/skills/run-remote-experiment/SKILL.md`](../.agents/skills/run-remote-experiment/SKILL.md) | Agents | Auto-discovered SSH/RunPod/tmux experiment-execution workflow |
| [`AGENT_FILES/KNOWLEDGE/encoders/README.md`](../AGENT_FILES/KNOWLEDGE/encoders/README.md) | Both | Encoder research, shipped SigLIP/V-JEPA status, DINO gate, and paired-run execution plan |
| [`GUIDES/README.md`](README.md) | Humans + agents | Index of all guides (this folder) |
| [`GUIDES/latest_brief.md`](latest_brief.md) | Both | Architecture narrative v0.3 — **not ground truth** |
| [`GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`](PROBLEMS_METRICS_AND_EXPERIMENTS.md) | Both | Metric glossary + experiment problem history |
| [`GUIDES/CODEBASE_STRUCTURE.md`](CODEBASE_STRUCTURE.md) | Both | This file |
| [`AGENT_FILES/SETUPS/SETUP.md`](../AGENT_FILES/SETUPS/SETUP.md) | Humans | Laptop ↔ RunPod operator guide |
| [`AGENT_FILES/SETUPS/NEW_POD.md`](../AGENT_FILES/SETUPS/NEW_POD.md) | Humans + agents | Canonical five-step fresh-pod bootstrap and W&B handoff |
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
