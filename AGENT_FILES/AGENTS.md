# AGENTS.md - HJEPA-VWM living reference for coding agents

This is the entry point for agents working in this repository. It is deliberately
implementation-grounded: use it to build a correct mental model before changing
code, then verify the details by tracing the referenced files yourself.

Do not treat this document as permission to stop reading code. It is a map of the
current system and the intended system, not a substitute for tracing the exact
function, class, and gradient path you are about to edit.

## 0. First rules

1. Read this file first, then read `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`.
2. Before touching code, read `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md`.
3. Before touching `config.py`, `data.py`, paths, or subsets, read
   `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`.
4. Before changing architecture, read this file and trace `config.py` + the hot path in code.
   Optional context (not prescriptions): [`GUIDES/latest_brief.md`](../GUIDES/latest_brief.md),
   [`GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`](../GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md).
   See §2 below — **briefs are not ground truth.**
5. Do not infer this project from generic ML habits. The important bugs here are
   gradient-routing, target-branch, shape, and diagnostic-contract bugs.
6. Do not put experiment-run history, W&B run narratives, or KANBAN contents in
   this file. This document is about the project architecture and current code.
7. When your change alters implementation, metrics, paths, ops, or experiment
   record, **update the agent-maintained living docs in the same PR/session** —
   see §2.2. Do not leave code and docs diverged.

## 1. What this project builds

HJEPA-VWM is a hierarchical JEPA-flow video world model. It does not train a
pixel diffusion model. Its research claim is that a compact abstract latent can
carry predictive video state, and that a larger detailed latent can later recover
local visual detail under the control of that abstract state.

The intended full v0 hierarchy is:

```text
context clip x_{<=t}
  -> frozen video encoder E
  -> detailed latent e_t
  -> trainable bottleneck B
  -> abstract latent c_t
  -> coarse flow F_c predicts future abstract c_{t+k}
  -> fine flow F_e predicts future detailed e_{t+k}
  -> frame generator renders pixels in frozen VAE latent space
```

The current implemented code is not the full hierarchy. It implements the Phase
1 coarse system plus several gated Phase-1 training mechanisms:

- implemented: data loading, `FrozenEncoder`, `Bottleneck`, `TargetBottleneck`,
  `CoarseFlow`, feature-space reconstruction `Decoder`, losses, diagnostics,
  optimizer/EMA/checkpoint training loop, and tests.
- not implemented: `FineFlow`, Stage 2/3 training, shuffled-c fine-flow tests,
  Phase-3 pixel frame generator, VAE wrapper, inference rollout sampler,
  `eval.py`, and Phase-4 multi-horizon embedding.

When a doc describes `F_e`, frame generation, or multi-horizon prediction, treat
that as intended design until code exists. When code and planning docs differ for
implemented behavior, trace code first and then reconcile the docs explicitly.

## 2. Documentation, precedence, and research posture

### Source precedence (when sources conflict)

Use this order — **code always wins over prose**:

1. **Root implementation files + tests** for anything implemented today:
   `config.py`, `data.py`, `models.py`, `losses.py`, `diagnostics.py`,
   `train.py`, `make_subset.py`, `tests/`.
2. **This file (`AGENTS.md`)** — implementation-grounded map of shapes, modules,
   training step, shipped defaults, invariants.
3. **Living factual guides** (must stay aligned with code when it changes):
   - [`GUIDES/CODEBASE_STRUCTURE.md`](../GUIDES/CODEBASE_STRUCTURE.md) — file map
   - [`GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`](../GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md) — metric glossary grounded in `diagnostics.py` / `train.py`
4. **`KANBAN/PHASE_1/`** — what was tried, measured, and learned (experiment history,
   not a build spec).
5. **Architecture briefs** (historical intent and past reasoning only):
   [`GUIDES/latest_brief.md`](../GUIDES/latest_brief.md),
   [`GUIDES/original_brief.pdf`](../GUIDES/original_brief.pdf).
6. **`README.md`**, other [`GUIDES/`](../GUIDES/README.md) playbooks — operator orientation.

### Briefs are not ground truth

The briefs capture **a point-in-time research narrative**. Successful research architectures
**inevitably evolve** as experiments falsify assumptions. Treat briefs as:

- **Useful context** — why the hierarchy exists, what we thought at v0.1/v0.3, what empirically
  seemed to help in past runs.
- **Not a constraint** — do not refuse a sound idea because a brief did not anticipate it.
- **Not an override** — if a brief says X and `config.py` / `train.py` do Y, **Y is current
  behavior** until the human approves a change.
- **Not a creativity ban** — propose new mechanisms, loss terms, or diagnostics when evidence
  (KANBAN, W&B, code gaps) supports them. Escalate only for locked invariants in §15.

When a brief recommendation differs from shipped defaults (e.g. brief suggests `--horizon-k 12`
while `config.py` has `horizon_k=4`), **state both** and default to code unless the human is
running a deliberate experiment.

### 2.1 Human-owned docs (read; do not edit unless the human asks)

These are **human-curated**. Use them for orientation. Do not rewrite them to match your code
change — update the agent-maintained docs instead (§2.3).

| Path | Why human-owned |
|---|---|
| [`README.md`](../README.md) | Project entry point and read order for newcomers — humans set navigation. |
| [`GUIDES/README.md`](../GUIDES/README.md) | Guide catalog and grouping — humans decide what belongs in the index. |
| [`GUIDES/EXPERIMENT_LIFECYCLE.md`](../GUIDES/EXPERIMENT_LIFECYCLE.md) | How **humans** run the research loop with AI — workflow policy, not implementation truth. |
| [`GUIDES/latest_brief.md`](../GUIDES/latest_brief.md) | Research **narrative** at milestones — humans update when intent story changes, not on every refactor. |
| [`GUIDES/original_brief.pdf`](../GUIDES/original_brief.pdf) | Frozen v0.1 baseline. |
| [`AGENT_FILES/SETUPS/SETUP.md`](SETUPS/SETUP.md) | First-time human deploy (laptop → RunPod) — operator journey, not agent contract. |
| [`AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md`](AGENT-BEHAVIOUR/CODE_DESIGN.md) | Coding conventions — humans set style; agents follow, not redefine. |
| [`AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`](AGENT-BEHAVIOUR/PROTOCOL.md) | Agent governance — humans set rules; agents follow §0–§2 here first. |
| [`KANBAN/PROTOCOL.md`](../KANBAN/PROTOCOL.md) | KANBAN triad rules — humans set record-keeping policy. |

### 2.2 Agent-maintained living docs (you keep these current)

**Rule:** If you change the thing a doc describes, update that doc in the **same session**
before finishing. Prefer minimal, factual diffs — no drive-by rewrites.

#### A. Implementation truth (code changes)

| Doc | Update when | How to update |
|---|---|---|
| **`AGENT_FILES/AGENTS.md`** (this file) | Any change to modules, shapes, training step, defaults, invariants, diagnostics, CLI flags, or gradient routing. | Edit the affected section (§4–§15). Sync tables with `config.py`. Do not paste KANBAN run stories here. |
| [`GUIDES/CODEBASE_STRUCTURE.md`](../GUIDES/CODEBASE_STRUCTURE.md) | Root files added/removed/renamed; new tests; MLOps scripts; doc paths change. | Update repo tree, file tables, and typical read order. Keep one-line roles accurate. |
| [`GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`](../GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md) | New/changed W&B keys; new diagnostics in `diagnostics.py`; renamed metrics; new failure mode worth glossary entry. | Add or edit glossary entry tied to the logging function in code. Note cadence (`log_every` / `diag_every`). Append problem history only when a run conclusion is confirmed (often with KANBAN). |
| [`GUIDES/READING_EXPERIMENTS.md`](../GUIDES/READING_EXPERIMENTS.md) | Acceptance gates change; a Q1–Q8 panel set changes; new mode flags alter which cycle applies. | Edit the affected question block (metrics table + W&B panel search). Keep full-prediction vs present-only cycles separate. |

#### B. Infrastructure and ops (paths, pod, tooling)

| Doc | Update when | How to update |
|---|---|---|
| [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](SETUPS/VOLUME_LAYOUT.md) | `config.py` path defaults change; volume directory contract changes; dataset layout changes. | Update path table, target tree, verification commands. Do not rewrite operator SSH steps (those stay in `SETUP.md`). |
| [`AGENT_FILES/SETUPS/NEW_POD.md`](SETUPS/NEW_POD.md) | `requirements.txt` changes; bootstrap commands change; default git branch changes; smoke-test commands change. | Update pip install block, verification snippets, and example `train.py` one-liner if defaults shift. |
| [`GUIDES/MLOPS.md`](../GUIDES/MLOPS.md) | W&B project/name changes; checkpoint naming changes; `run_history.py` / `parse_logs.py` CLI changes; volume layout changes. | Update stack diagram paths, CLI examples, and “what gets logged” tables to match `train.py`. |

#### C. Experiment record (KANBAN)

| Doc | Update when | How to update |
|---|---|---|
| [`KANBAN/PHASE_1/README.md`](../KANBAN/PHASE_1/README.md) | A run finishes or status/verdict changes; new investigation opens/closes. | Add row to run index with W&B id, mode, verdict label from `READING_EXPERIMENTS.md`. Update investigation status table. Pull metrics from W&B — never invent values. |
| **`KANBAN/PHASE_1/investigation_*/…`** | Human asks you to plan, launch, or analyze an experiment (see human workflow in `EXPERIMENT_LIFECYCLE.md`). | Follow triad: `DESCRIPTION.md` before launch; after run: `OBSERVATIONS.md`, `NEXT_STEPS.md`, optional `METRIC_READOUT.md` / `ANALYSIS.md`. Obey [`KANBAN/PROTOCOL.md`](../KANBAN/PROTOCOL.md). Do not delete history. |

**KANBAN checklist after a run analysis:**

1. Run folder triad updated with verdict + W&B link.
2. Parent investigation `OBSERVATIONS.md` synthesis if the thread moved.
3. `KANBAN/PHASE_1/README.md` run row and verdict column updated.
4. If a new metric or failure mode mattered, add glossary entry in `PROBLEMS_METRICS_AND_EXPERIMENTS.md`.

#### What agents must not do to living docs

- Do not edit human-owned docs (§2.1) to “sync” implementation — use §2.2 instead.
- Do not copy brief prose into `AGENTS.md` or replace code truth with narrative.
- Do not rewrite KANBAN history; append corrections with dates.
- Do not update `latest_brief.md` unless the human explicitly asks for a narrative refresh.

### Drift to know

- Old docs often mention 30k Phase-1 steps. Current defaults: `stage1_steps=15000`,
  `warmup_steps=1500`, `grad_clip=0.5`, `grad_skip_threshold=150.0` (`config.py`).
- `latest_brief.md` may list experiment CLI overrides (`--horizon-k 12`, `--lambda-var 0.5`)
  that differ from shipped defaults (`horizon_k=4`, `lambda_var=0.1`).
- `FineFlow` / Phase 2 is not implemented unless the human explicitly requests it.

## 3. Repository map

Root implementation files:

| Path | Role |
|---|---|
| `config.py` | Dataclass configuration, path contract, dimensions, optimizer/loss knobs. |
| `data.py` | Clip-per-file video dataset and dataloader. Produces context/target clip pairs from SSv2 `.webm` symlinks or EGO4D `.mp4` chunks. |
| `make_subset.py` | Builds `ssv2_tiny` as symlinks plus `manifest.json`. |
| `select_ego4d_uids.py` | Selects scenario-diverse EGO4D source-video UIDs, train/validation split, and download batches from `ego4d.json`. |
| `chunk_ego4d.py` | Chunks downloaded EGO4D 540ss videos into 4-second, 12 FPS, 256px-shorter-side H.264 `.mp4` clips under `data/ego4d`. |
| `make_ego4d_subset.py` | Builds `ego4d_tiny` as symlinks into `data/ego4d` plus `manifest.json`. |
| `models.py` | All `nn.Module` classes currently implemented. |
| `losses.py` | Pure tensor losses and detach helper. No parameters. |
| `diagnostics.py` | Collapse metrics, baseline comparisons, AGC, weight-decay grouping. |
| `train.py` | Stage-0 sanity, Stage-1 training, CLI, optimizer, EMA, checkpoints. |
| `parse_logs.py` | Parses `step=N {dict}` console logs into JSON. |
| `run_history.py` | W&B Public API export/report helper for logged metrics. |
| `drift_probe.py` | Offline within-video temporal drift probe: frozen-encoder drift vs bottleneck-latent drift over a pinned probe set, evaluated from checkpoints. |
| `rank_probe.py` | Offline frozen-encoder effective-rank probe: applies the `c_effective_rank` covariance-rank formula to cached V-JEPA `e` tokens over the drift-probe manifest. |
| `whiten_stats.py` | Offline whitening statistics for frozen V-JEPA features: single-pass fp64 mean/covariance over training-set encoder tokens, saved as the eigendecomposition consumed by `train.py --whiten-features`. |
| `requirements.txt` | Runtime and dev dependencies. |
| `pyproject.toml` | Black and Ruff configuration. |
| `tests/` | Unit tests for contracts, losses, optimizer grouping, AGC, decoder, modes. |

Agent and architecture docs:

| Path | Role |
|---|---|
| `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md` | How agents operate, when to ask, phase discipline. |
| `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` | Flat-file layout, naming, docstrings, detach rules. |
| `AGENT_FILES/AGENT-BEHAVIOUR/WORKFLOW.md` | Redirect → [`GUIDES/EXPERIMENT_LIFECYCLE.md`](../GUIDES/EXPERIMENT_LIFECYCLE.md). |
| `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md` | RunPod `/workspace` data/checkpoint/cache layout. |
| `AGENT_FILES/KNOWLEDGE/encoders/README.md` | Frozen-encoder research index: DINOv3-B and SigLIP 2-B dossiers plus the encoder-pluggability/parallel-experiment plan. These describe proposed work, not shipped implementation. |
| `GUIDES/latest_brief.md` | Architecture narrative (v0.3) — **historical intent, not ground truth**. |
| `GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md` | Metric glossary + experiment problem history. |
| `GUIDES/CODEBASE_STRUCTURE.md` | File map: training code, MLOps, docs, KANBAN. |
| `GUIDES/` | Operator playbooks + project knowledge (see [`GUIDES/README.md`](../GUIDES/README.md)). |
| `KANBAN/PHASE_1/README.md` | Phase 1 experiment index and W&B run table. |
| `KANBAN/PROTOCOL.md` | KANBAN update rules (triad, analysis files). |

`KANBAN/` and `AGENT_FILES/KANBAN/` exist, but their contents are not part of
this living reference. Use them only if the human specifically asks for
task-history or current task-management context.

## 4. Canonical terminology and tensor shapes

Names in code follow the symbol map from `CODE_DESIGN.md`.

| Symbol | Code name | Meaning | Shape |
|---|---|---|---|
| `x` | `context_clip` | Context video window ending at time `t` | `(B, 8, 3, 256, 256)` |
| `x_{<=t+k}` | `target_clip` | Future clip window ending `horizon_k` frames later | `(B, 8, 3, 256, 256)` |
| `E` | `encoder` | Frozen V-JEPA 2 ViT-L/16 video encoder | module |
| `e_t` | `detailed` | Detailed context latent from frozen encoder | `(B, 1024, 1024)` |
| `B` | `bottleneck` | Trainable compressor from detailed to abstract | module |
| `c_t` | `abstract` | Current abstract latent | `(B, 32, 256)` by default |
| `B_EMA` | `target_bottleneck` | EMA copy of `B`, never backpropagated | module |
| `e_plus` | `target_detailed` | Future detailed latent from frozen encoder | `(B, 1024, 1024)` |
| `c_plus` | `target_abstract` | Future abstract EMA target | `(B, 32, 256)` |
| `F_c` | `coarse_flow` | Rectified-flow velocity predictor for `c_plus` | module |
| `D` in current code | `decoder` | Feature reconstruction decoder `c -> e_hat`; not the Phase-3 pixel generator | module |
| `c_hat` | `c_hat` / endpoint | One-step estimate of future abstract latent | `(B, 32, 256)` |

Shape derivation:

- Encoder patch size is 16, tubelet size is 2.
- Each 8-frame clip becomes 4 temporal tubelets.
- Each 256x256 frame has a 16x16 spatial patch grid.
- `N_ctx = (8 / 2) * (256 / 16)^2 = 4 * 256 = 1024`.
- `D_e = 1024` comes from `facebook/vjepa2-vitl-fpc64-256`.
- `N_c = 32` and `D_c = 256` are the abstract bottleneck defaults.

Do not introduce functions that touch tensors without shape-contract docstrings.
When modifying tensor paths, trace shape assertions in `models.py`,
`losses.py`, and tests.

## 5. Data path, from disk to batch

Runtime assets live outside the repo under `/workspace` on RunPod:

```text
/workspace/hierarchal-jepa-flow-world-model/  # git repo, code only
/workspace/data/ssv2/                         # full SSv2 symlink layout
/workspace/data/ssv2_tiny/                    # symlink smoke subset
/workspace/data/ego4d/                        # EGO4D chunk corpus, real .mp4 files
/workspace/data/ego4d_tiny/                   # symlink smoke subset
/workspace/ssv2_raw/                          # raw .webm backing files
/workspace/ego4d_raw/                         # transient EGO4D CLI raw downloads + kept manifests
/workspace/checkpoints/                       # training checkpoints
/workspace/hf_cache/                          # Hugging Face / diffusers cache
```

Only `JEPA_DATA_ROOT` overrides the dataset parent path. Do not add path
auto-detection.

`make_subset.py`:

1. Loads `<data_root>/ssv2/labels.json` as `video_id -> class label`.
2. Groups existing `.webm` symlinks by class for `train` and `validation`.
3. Selects deterministic per-class subsets.
4. Creates idempotent symlinks in `<data_root>/ssv2_tiny`.
5. Writes `manifest.json`.

EGO4D build helpers:

1. `select_ego4d_uids.py` reads `/workspace/ego4d_raw/ego4d.json`, filters
   unsuitable source videos, selects ~210 source hours, splits train/validation
   by source UID, and writes four balanced batch UID files plus
   `selection_manifest.json`.
2. `chunk_ego4d.py` processes the raw `.mp4` files currently present in
   `/workspace/ego4d_raw/v2/video_540ss`, writes real 4-second H.264 `.mp4`
   chunks to `/workspace/data/ego4d/{train,validation}`, and rebuilds a
   cumulative `chunk_manifest.json` by rescanning the output tree.
3. `make_ego4d_subset.py` creates symlink-only `/workspace/data/ego4d_tiny`
   from the chunk corpus, capped per source video for diversity.

`data.py`:

1. `SSV2Dataset` indexes the sorted union of `.webm` and `.mp4` files in the
   selected split. The class name is historical; the contract is a dataset root
   with `train/` and `validation/` clip files.
2. `_open_video_reader` opens decord `VideoReader(..., num_threads=1)`. The
   single-thread setting avoids known VP9/decord packet errors while dataloader
   workers still parallelize across videos.
3. `_window_indices` selects two 8-frame stride-2 windows:
   - context indices: `start + i * frame_stride`
   - target window ends at `start + (T - 1) * stride + horizon_k`
   - too-short videos are padded by clamping to the last frame.
4. `_decode_frames` decodes only the 16 needed frame indices, not the whole
   video.
5. Frames become float tensors in `[0, 1]`, channel-first.
6. `_resize_shorter_side` resizes so the shorter side is 256.
7. `_crop` uses random crop for train and center crop for validation.
8. `_color_jitter` applies one brightness/contrast/saturation sample to the
   combined context+target stack for train only.
9. `_normalize_encoder` applies V-JEPA/ImageNet mean/std:
   `(0.485, 0.456, 0.406)` and `(0.229, 0.224, 0.225)`.
10. The item returns `(context_clip, target_clip)`, both
    `(8, 3, 256, 256)`.

No horizontal flip, temporal flip, rotation, tubelet dropout, or `[-1, 1]`
normalization belongs on the encoder path. VAE `[-1, 1]` normalization is a
future Stage-4 concern, not current code.

## 6. Current model components

All current trainable modules live in `models.py`.

### FrozenEncoder

`FrozenEncoder` wraps `transformers.AutoModel.from_pretrained(
"facebook/vjepa2-vitl-fpc64-256", attn_implementation="sdpa")`.

Contracts:

- Calls `model.get_vision_features(clip)`.
- Input is `(B, 8, 3, 256, 256)`, already encoder-normalized.
- Output is `(B, 1024, 1024)`.
- All parameters have `requires_grad=False`.
- `.train()` is overridden to keep the wrapped model in eval mode.
- `forward` is decorated with `torch.no_grad()`.
- The same instance encodes context and target clips.

Never put encoder parameters in an optimizer. Never add a target encoder.

### Bottleneck

`Bottleneck` maps `e_t` or `e_plus` from `(B, 1024, 1024)` to
`(B, N_c, D_c)`.

Pipeline:

1. `in_proj`: linear `D_e=1024 -> bottleneck_mixer_dim=256`.
2. Reshape temporal-major tokens into `(B * 4, 256, 16, 16)`.
3. Apply two shared `ConvNeXtBlock`s per temporal slot.
4. Flatten back to `(B, 1024, 256)` and add the learned `pos_emb` memory tags.
5. `to_kv`: linear `256 -> D_c=256` produces the memory tokens.
6. A Perceiver-style latent processor of `bottleneck_latent_blocks=3`
   `BottleneckLatentBlock`s refines the `N_c=32` learned query slots. Each
   block runs three zero-init residual updates on the slot stream:
   sharpened-cosine cross-attention read from the 1024 memory tokens
   (`SharpCrossAttention`, zero-init `o_proj`), slot self-attention for slot
   competition (zero-init `out_proj`), and a per-slot MLP (zero-init last
   layer).
7. Final LayerNorm.

Important current initialization:

- Query slots are initialized with `nn.init.orthogonal_`.
- Every latent-block residual output (`cross_attn.o_proj`,
  `self_attn.out_proj`, `mlp[-1]`) is zero-initialized and tagged
  `is_zero_init = True`, so at step 0 the module returns
  `LayerNorm(queries)` for EVERY input (identity-at-init), and all
  input-dependence grows in through training. Tests enforce this at several
  depths.
- Zero-init and learned geometry parameters are excluded from AGC and weight
  decay by predicates in `diagnostics.py`.

`Bottleneck(..., return_attn=True)` returns per-head attention weights of the
FINAL latent block's cross-attention read for diagnostics:
`(B, heads, N_c, N_ctx)`.

### TargetBottleneck

`TargetBottleneck` is a deepcopy of `Bottleneck`:

- It is initialized from the online bottleneck.
- All parameters have `requires_grad=False`.
- It stays in eval mode.
- It updates only through `_update_ema` in `train.py`.
- Its `forward` runs under `torch.no_grad()` and returns `as_target(abstract)`.

It produces `c_plus = B_EMA(e_plus)`, the stop-gradient target for the coarse
flow. It must never receive gradients or optimizer steps.

### CoarseFlow

`CoarseFlow` predicts rectified-flow velocity in abstract space:

```text
F_c(z_c, tau_c, c_t) -> u_c_hat
```

Inputs:

- `z_c`: noised target latent `(B, N_c, D_c)`.
- `tau_c`: flow time `(B,)`.
- `abstract`: current `c_t` condition `(B, N_c, D_c)`.

Structure:

- learned `null_condition` for condition dropout.
- learned `slot_pos`, `z_type`, and `cond_type` embeddings that stamp slot and
  stream identity before concatenation.
- sinusoidal timestep embedding plus MLP.
- six `AdaLNBlock`s by default.
- each `AdaLNBlock` uses LayerNorm without affine, multi-head self-attention,
  MLP, and zero-initialized modulation producing six shift/scale/gate tensors.
- concatenates `[z_c stream, condition stream]` into `2 * N_c` tokens.
- returns only the first `N_c` output tokens, normalized.

Condition dropout:

- During training, each example drops the condition with probability
  `condition_dropout = 0.10`.
- Diagnostics can pass `condition_drop=_no_drop(...)` to force real conditioning.

There is no horizon embedding in current code. Phase 4 will need a deliberate
extension point here.

### Decoder

`Decoder` is the current feature reconstruction decoder. It is not the planned
Phase-3 pixel/frame generator.

Purpose:

- Decode an abstract latent `(B, N_c, D_c)` back to frozen detailed features
  `(B, N_ctx, D_e)`.
- Provide optional reconstruction pressure so `c_t` remains information-rich.
- Support both present reconstruction `D(c_t) -> e_t` and prediction-side
  reconstruction `D(c_hat) -> e_plus`.

Structure:

- `kv_proj`: projects abstract slots to decoder width.
- deterministic fixed 3D tubelet position codes are registered as the
  non-trainable buffer `fixed_pos`.
- initial cross-attention queries are normalized fixed positions, values come
  from `c`.
- `DecoderBlock`s use fixed positions as query information but do not add
  position as output content.
- output projects decoder width back to `D_e`.

Invariant: the decoder may know where output tubelets are, but it must not use
a learned per-output-token content template. Tests enforce that `fixed_pos` is
a buffer and that zero latent cannot emit position-specific content.

### FeatureMeanTracker

`FeatureMeanTracker` supports the residual reconstruction target
(`cfg.train.recon_residual_target`):

- Buffers only (`mean` fp32 `(N_ctx, D_e)`, `initialized` flag), zero
  parameters — it can never enter the optimizer, AGC, weight decay, or EMA.
- `update(detailed)` folds a training batch into an EMA per-tubelet-position
  mean; the first batch initializes it directly. Train-step only; diagnostics
  read it but never update it.
- `subtract(features)` returns `features - mean` for the reconstruction target.
- It travels outside the five-module bundle as an optional keyword argument and
  is checkpointed under the optional `recon_feature_mean` key; old checkpoints
  load fine and the mean re-warms.

### FeatureWhitener

`FeatureWhitener` supports fixed offline whitening of the frozen V-JEPA
features (`cfg.train.whiten_features`, motivated by investigation_014's finding
that `e` has pooled entropy rank ~193/1024 with a long low-energy tail):

- Buffers only (`mean`, `whiten_mat`, `unwhiten_mat` fp32, `initialized` flag),
  zero parameters — it can never enter the optimizer, AGC, weight decay, or EMA.
- Statistics come from ONE offline pass over the training set
  (`whiten_stats.py`); `configure` builds the ZCA pair
  `W = U (Lambda + eps I)^{-1/2} U^T` and its inverse. Never per-batch
  whitening.
- `whiten`/`unwhiten` run the 1024x1024 matmul in fp32 with autocast disabled
  (the SIGReg WALK_FIXES F2 precision rule) and return the input dtype.
- It is applied at ONE seam — inside `train._coarse_forward` /
  `train._present_forward`, right after each frozen-encoder call — so `B`,
  `B_EMA`, the flow targets, the reconstruction targets, the mean tracker, and
  every diagnostic all live in the same whitened space.
- It travels outside the five-module bundle as an optional keyword argument and
  is checkpointed under the optional `feature_whitener` key so checkpoints are
  self-describing; `drift_probe.py` rebuilds it from the checkpoint and refuses
  to evaluate a whitened-space checkpoint on raw features.

## 7. Flow math and implemented loss functions

All current flow losses use rectified flow:

```text
eps ~ N(0, I)
tau ~ Uniform(0, 1)
z_tau = (1 - tau) * eps + tau * target
u_target = target - eps
L_flow = mean((u_hat - u_target)^2)
```

`losses.py` owns the pure tensor primitives:

- `as_target(x)`: centralized stop-gradient helper, returns `x.detach()`.
- `interpolate(target, eps, tau)`: computes `(1 - tau) * eps + tau * target`.
- `velocity_target(target, eps)`: computes `target - eps`.
- `flow_matching_loss(u_hat, u_target)`: mean squared velocity error.
- `residual_target(target_future, target_present)`: returns detached
  `Delta = target_future - target_present` and scalar `sigma = std(Delta)`.
- `reconstruction_loss(pred_detailed, target_detailed, mode)`.
- `variance_floor(abstract)`.
- `sigreg_loss(abstract)`.
- `covariance_floor(abstract)`.
- `slot_diversity_loss(abstract)`.

### Coarse full-latent objective

Default mode predicts the full future abstract target:

```text
flow_target = c_plus
eps_c = randn_like(c_plus)
z_c = interpolate(c_plus, eps_c, tau_c)
u_c = c_plus - eps_c
u_c_hat = F_c(z_c, tau_c, c_t)
L_flow = mean((u_c_hat - u_c)^2)
```

Gradient from `L_flow` reaches `B` through `c_t` and reaches `F_c`. It does not
reach the frozen encoder, `B_EMA`, `e_plus`, or `c_plus`.

### Residual prediction mode

When `cfg.train.predict_residual` is true:

```text
c_present_ema = B_EMA(e_t)
Delta = c_plus - c_present_ema
sigma = std(Delta)
eps_c = sigma * randn_like(Delta)
flow_target = Delta
```

`F_c` predicts residual velocity. The prediction-side reconstruction endpoint is
converted back to a future latent by `c_hat = c_t + Delta_hat`.

The copy/no-change baseline remains comparable because "copy" becomes
"predict zero residual".

### Reconstruction loss

`reconstruction_loss` always detaches the target detailed features.

Modes:

- `cosine` (default): L2-normalize each tubelet vector along `D_e`, then
  return `mean(1 - cosine(pred, target))`.
- `relative_mse`: return raw MSE divided by `Var(target)`, retained as an
  explicit legacy comparison mode.

Current reconstruction paths:

- Present anchor: `decoder(c_t)` vs `e_t`. Trains `D` and `B`, not `F_c`.
- Prediction-side anchor: `decoder(c_hat)` vs `e_plus`. Trains `D`, `F_c`,
  and `B` through the coarse conditioning path.
- Diagnostic readouts also score `decoder(c_plus)` and `decoder(c_hat)` under
  `no_grad`.

Residual reconstruction target (`recon_residual_target`, default off): both
anchors score against the per-position residual `e - mean` instead of the
absolute features, where `mean` is the `FeatureMeanTracker` EMA per-tubelet
mean of `e_t`. This removes the video-independent template component from the
objective so reconstruction pressure must route video-specific content through
`c_t` (the run-052 collapse fix). Gradient routing is unchanged; requires an
active recon anchor (validated in `finalize_training_config`). Do not confuse
it with `predict_residual`, which is the temporal residual for `F_c`.

### Variance floor

`variance_floor(abstract)`:

1. Reshapes `c_t` to `(B, N_c * D_c)`.
2. Computes per-coordinate std across batch.
3. Applies hinge `max(0, std_target - std)`.
4. Averages over coordinates.

Formula:

```text
L_var = mean_j max(0, 1.0 - Std(c_j))
```

It prevents constant-code collapse. It does not decorrelate dimensions or force
distinct slots.

### SIGReg, covariance, and slot losses

These exist in code but are off by default:

- `sigreg_loss`: stochastic projection/BHEP-style normality statistic pushing
  pooled `c_t` rows toward isotropic `N(0, I)`. It uses a per-step generator in
  `train_step` so logging it does not perturb the global RNG when its weight is
  zero.
- `covariance_floor`: VICReg-C off-diagonal covariance penalty on pooled
  `B * N_c` rows over `D_c` dimensions.
- `slot_diversity_loss`: centers slots within each video, normalizes residual
  slot vectors, and penalizes squared off-diagonal slot cosine similarities.

Do not assume these should be turned on. Their weights default to zero. If you
change them, explain the gradient effect and watch the core prediction metrics.

## 8. Exact current training step

`train.train_step` is the authoritative implementation.

Inputs:

- `batch = (context_clip, target_clip)`.
- `modules = (encoder, bottleneck, target_bottleneck, coarse_flow, decoder)`.
- optimizer over `B`, `F_c`, and `Decoder`.

Normal prediction mode:

1. Move clips to device. In `present_recon_only`, skip moving/using
   `target_clip`.
2. `optimizer.zero_grad(set_to_none=True)`.
3. Enter CUDA bf16 autocast when available and configured.
4. `_coarse_forward`:
   - with no grad: `detailed = encoder(context_clip)`; when whitening is
     active, `detailed = whitener.whiten(detailed)`.
   - trainable: `abstract = bottleneck(detailed)`.
   - with no grad: `target_detailed = encoder(target_clip)`, whitened the same
     way.
   - target branch: `target_abstract = target_bottleneck(target_detailed)`.
5. Build flow target:
   - full-latent: `flow_target = target_abstract`, `eps_c = randn_like`.
   - residual: target-present from `B_EMA(detailed)`, `Delta`, scaled noise.
6. Sample `tau_c = rand(B)`.
7. Build `z_c`, `u_c`, and `u_c_hat`.
8. Compute `flow_loss`.
9. Always compute for logging: `var_loss`, `cov_loss`, `slot_loss`,
   `sigreg_l`.
10. Start total loss as `flow_loss`.
11. Add `lambda_var * var_loss`.
12. Add optional active terms:
    - `lambda_cov * L_cov` if `lambda_cov > 0`.
    - `lambda_slot * L_slot` if `lambda_slot > 0`.
    - `lambda_sigreg * sigreg_scale * L_sigreg` if `lambda_sigreg > 0`.
13. If reconstruction anchors are active, compute a linear `recon_scale`.
14. If `lambda_recon > 0`, add present reconstruction.
15. If `lambda_recon_pred > 0`, build one-step endpoint:
    `endpoint = z_c + (1 - tau_c) * u_c_hat`; in residual mode,
    `c_hat = abstract + endpoint`, otherwise `c_hat = endpoint`; add
    prediction-side reconstruction.
16. Exit autocast and call `loss.backward()`.
17. Apply module-specific AGC to `B`, `F_c`, and `Decoder` if enabled.
18. Apply global `clip_grad_norm_` with `grad_clip`.
19. If the returned norm is non-finite or greater than
    `grad_skip_threshold`, zero grads and skip `optimizer.step()`.
20. Otherwise step optimizer.
21. Compute EMA momentum with `ema_cosine`.
22. If the optimizer step was not skipped, update `B_EMA` from `B`.
23. Return a flat metrics dict.

Present-only reconstruction mode:

- `_present_forward` encodes only the context.
- `flow_loss` is zero.
- Future branch, residual mode, `F_c`, and `lambda_recon_pred` are disabled.
- `lambda_recon` must be positive or `finalize_training_config` raises.
- Non-prediction regularizers still follow their configured weights.

## 9. Total objective in code

In normal mode:

```text
L_total =
    L_flow
  + lambda_var * L_var
  + [lambda_cov * L_cov if lambda_cov > 0]
  + [lambda_slot * L_slot if lambda_slot > 0]
  + [lambda_sigreg * sigreg_scale * L_sigreg if lambda_sigreg > 0]
  + [lambda_recon * recon_scale * L_recon if lambda_recon > 0]
  + [lambda_recon_pred * recon_scale * L_recon_pred if lambda_recon_pred > 0]
```

`sigreg_scale` and `recon_scale` are linear ramps from 0 to 1 over their warmup
step counts.

Terms may be computed for logging even when their weights are zero. Do not
mistake a logged scalar for an active gradient source.

## 10. Optimizer, clipping, EMA, and checkpoints

Optimizer:

- `make_optimizer` creates AdamW over `B`, `F_c`, and `Decoder`.
- The decoder is always included so checkpoints are consistent; when no
  reconstruction loss is active, it receives no gradients and does not move.
- Each module is split into decay and no-decay groups by
  `diagnostics.partition_decay_params`.
- Decayed: genuine Linear/Conv weight matrices.
- No-decay and no-AGC: 1-D params, biases, LayerNorm weights, learned geometry
  params named `queries`, `null_condition`, `slot_pos`, `z_type`, `cond_type`,
  and submodules tagged `is_zero_init`.

LR schedule:

- `lr_scale`: linear warmup for `warmup_steps`, then cosine decay to zero over
  `stage1_steps`.
- `apply_lr_schedule` updates every optimizer group from peak base LRs.
- `peak_base_lrs` rebuilds base LRs from the current config so resuming does not
  double-apply saved scheduled LRs.

AGC and global clipping:

- `adaptive_gradient_clip` enforces per-tensor `||g|| <= lambda * (||w|| + eps)`
  on eligible tensors.
- Current default AGC lambdas: `B=0.20`, `F_c=0.10`, `Decoder=0.20`.
- Global `clip_grad_norm_` then clips all trainable params to `grad_clip=0.5`.
- `grad_norm` in `train_step` is the norm returned by `clip_grad_norm_` before
  the global rescale, after AGC.
- `gradient_health` reports a post-clip norm and should not be read as the raw
  gradient magnitude.

EMA:

```text
m(step) = end - (end - start) * 0.5 * (1 + cos(pi * clamp(step / total, 0, 1)))
B_EMA <- m * B_EMA + (1 - m) * B
```

Defaults: start `0.996`, end `0.9999`, denominator `105000`.

EMA updates only after successful optimizer steps. If a step is skipped, the EMA
target does not move.

Checkpoints:

- `save_checkpoint` writes `global_step`, `bottleneck`, `target_bottleneck`,
  `coarse_flow`, `decoder`, `optimizer`, and serialized config, plus the
  optional `recon_feature_mean` (mean tracker) and `feature_whitener`
  (whitening buffers) keys when those modules are active.
- The frozen encoder is never checkpointed; reload it from Hugging Face.
- `load_checkpoint` supports older checkpoints without decoder state.
- It rejects checkpoints from the old learned-query decoder architecture.
- It skips incompatible optimizer state group counts with a warning and keeps
  the model weights loaded.

## 11. Diagnostics and what each one proves

Diagnostics are pure functions in `diagnostics.py` and are run on a fixed
validation batch in `train.run_diagnostics`.

Representation health:

- `variance_stats(c_t)` logs `c_std_mean`, `c_std_median`,
  `c_dead_dim_frac`.
- `cross_video_cosine(c_t)` logs mean pairwise cosine between batch examples.
  High values indicate video-independent collapse.
- `effective_rank(c_t)` pools batch and slots, computes covariance over
  `D_c`, then logs `exp(entropy(normalized_eigenvalues))`.
- `slot_diversity_rank(c_t)` computes within-video effective rank over the
  `N_c` slots.
- `attention_entropy(bottleneck, detailed)` computes normalized per-head
  cross-attention entropy. It is useful but weaker than actual slot/rank
  metrics.

Prediction baselines:

- `coarse_baselines` disables condition dropout and compares `F_c` against:
  - copy/no-change baseline.
  - batch-mean future target baseline.
- It logs model/copy/batch-mean losses and ratios.
- In residual mode, copy means predict zero residual.

Reconstruction readouts:

- `L_recon_present`: `decoder(c_t)` vs `e_t`.
- `L_recon_cplus`: `decoder(c_plus)` vs `e_plus`.
- `L_recon_chat`: `decoder(c_hat)` vs `e_plus`.
- `L_recon_shuffled_c`: `decoder(roll(c_t, 1))` vs this video's target — decodes
  ANOTHER video's latent against each target (deterministic roll, RNG-free).
- `L_recon_video_gap`: `L_recon_shuffled_c - L_recon_present`. Near zero means
  the decode barely depends on which video's latent it received — the run-052
  template-collapse signature. In residual-target mode every readout scores the
  residual `e - mean`, matching the training objective.

Gradient and stability:

- `grad_norm`: per-step post-AGC/pre-global-rescale norm from training.
- `grad_skipped`: whether optimizer/EMA update was skipped.
- `instability_warn`: `grad_norm` and `L_flow` are both above warning
  thresholds.
- `agc_*`: module-level AGC ratios and clipped tensor counts.
- `grad_global_norm_postclip`, `grad_has_nan`, `grad_param_count`: diagnostic
  pass gradient health.

Offline probes (not part of the training loop):

- `drift_probe.py` measures WITHIN-video temporal drift on a pinned probe set:
  `1 - cos(e_t, e_{t+k})` from the frozen encoder (a constant of the dataset,
  cached after one encoding pass) and `1 - cos(c_t, c_{t+k})` from any
  checkpoint's bottleneck, plus Spearman faithfulness between the two. It
  compares windows of the SAME video only — never two different videos. It is
  an analysis helper like `run_history.py`; training code never imports it and
  it logs nothing to W&B. Checkpoints trained with `whiten_features` carry
  their whitener; the probe applies it to the cached raw features before the
  bottleneck (the cache itself always stores raw V-JEPA features). See README
  "Within-video drift probe" for usage.
- `rank_probe.py` measures the entropy effective rank of frozen V-JEPA `e`
  features over the same pinned manifest/cache namespace as `drift_probe.py`.
  Its pooled-token metric is the direct `e`-side analog of
  `c_effective_rank`: stack `(B * N_ctx, D_e)` anchor-window tokens, center the
  covariance, rank eigenvalues via entropy. It also reports per-video token
  ranks and cross-video mean-vector rank. It is an offline analysis helper;
  training code never imports it and it logs nothing to W&B.

Future Phase-2/3 diagnostics, not implemented yet:

- fine-flow shuffled-`c` conditioning test (distinct from reconstruction
  `L_recon_shuffled_c`, which is implemented).
- zero-c condition test for `F_e`.
- teacher-vs-predicted fine-flow gap.
- decoder dependency test with shuffled `e_hat`.
- full `eval.py`.

## 12. Configuration defaults and CLI overrides

`config.py` is the source of current implemented defaults.

Model defaults:

| Field | Default |
|---|---|
| `encoder_repo` | `facebook/vjepa2-vitl-fpc64-256` |
| `encoder_frozen` | `True` |
| `encoder_patch` / `encoder_tubelet` | `16` / `2` |
| `t_ctx`, `h`, `w` | `8`, `256`, `256` |
| `n_ctx`, `n_tgt` | properties, both `1024` |
| `n_c`, `d_c`, `d_e` | `32`, `256`, `1024` |
| bottleneck blocks / heads | `2` ConvNeXt blocks, `8` cross-attn heads |
| `bottleneck_latent_blocks` | `3` Perceiver-style latent blocks |
| `f_c_blocks`, `f_c_dim`, `f_c_heads` | `6`, `256`, `8` |
| `condition_dropout` | `0.10` |
| reconstruction decoder dim / blocks / heads | `256`, `2`, `8` |

Training defaults:

| Field | Default |
|---|---|
| `global_batch` | `64` |
| `stage1_steps`, `max_steps` | `15000`, `15000` |
| `lr_bottleneck`, `lr_coarse_flow`, `lr_decoder` | `1e-4`, `2e-4`, `1e-4` |
| `warmup_steps`, `total_latent_steps` | `1500`, `105000` |
| `adam_betas`, `weight_decay` | `(0.9, 0.95)`, `0.05` |
| `grad_clip`, `grad_skip_threshold` | `0.5`, `150.0` |
| `agc_enabled` | `True` |
| `agc_lambda_bottleneck`, `agc_lambda_coarse_flow`, `agc_lambda_decoder` | `0.20`, `0.10`, `0.20` |
| `ema_m_start`, `ema_m_end`, `ema_schedule_steps` | `0.996`, `0.9999`, `105000` |
| `lambda_var`, `var_floor_std_target` | `0.10`, `1.0` |
| `lambda_sigreg`, `lambda_cov`, `lambda_slot` | `0.0`, `0.0`, `0.0` |
| `lambda_recon`, `lambda_recon_pred` | `0.0`, `0.0` |
| `recon_loss_mode`, `recon_warmup_steps` | `cosine`, `2000` |
| `recon_residual_target`, `recon_mean_momentum` | `False`, `0.99` |
| `present_recon_only`, `predict_residual` | `False`, `False` |
| `whiten_features`, `whiten_stats_path`, `whiten_eps` | `False`, `""`, `1e-4` |
| `horizon_k`, `frame_stride` | `4`, `2` |
| `precision` | `bf16` |
| `log_every`, `diag_every`, `checkpoint_every` | `50`, `500`, `2500` |

Data/path defaults:

| Field | Default |
|---|---|
| `data_root` | `os.environ["JEPA_DATA_ROOT"]` or `/workspace/data` |
| `dataset` | `ssv2_tiny` (choices: `ssv2`, `ssv2_tiny`, `ego4d`, `ego4d_tiny`) |
| `num_workers`, `pin_memory` | `8`, `True` |
| `checkpoint_dir` | `/workspace/checkpoints` |
| `hf_cache_dir` | `/workspace/hf_cache` |
| `seed` | `42` |

Current `train.py` CLI flags include data selection, steps, resume, seed,
logging cadence, `horizon_k`, all optional loss weights, reconstruction modes,
residual/present-only switches, LRs, AGC toggles, gradient skip threshold,
decoder capacity, `n_c`, and checkpoint directory. Read `parse_args()` before
adding or changing any experiment knob.

## 13. Stage and phase status

Be precise about "stage" vs "phase":

- Training stages are the model curriculum.
- Agent phases are implementation work packages.

Current code:

| Training stage | Implemented? | Notes |
|---|---|---|
| Stage 0 sanity | yes | `python train.py --stage0-only` loads the encoder, runs one synthetic step, verifies the frozen encoder, and checks the exact dtype-rounded `B_EMA` transition. |
| Stage 1 coarse dynamics | yes | `B`, `B_EMA`, `F_c`, optional reconstruction/regularizer knobs. |
| Stage 2 fine teacher forcing | no | Requires `FineFlow`; Phase 2 spec only. |
| Stage 3 predicted-coarse fine | no | Requires detached `c_hat` into `F_e`; Phase 2 spec only. |
| Stage 4 pixel/frame generation | no | Requires VAE, frame generator, inference/eval; Phase 3 spec only. |
| Phase 4 multi-horizon | no | Deferred; no horizon embedding in code. |

Do not start a later phase unless the human explicitly asks. Do not quietly
implement planned modules while fixing current Phase-1 code.

## 14. Planned full pipeline, not current code

This is the intended end state from the phase docs.

Phase 2 adds `FineFlow`:

- predicts velocity toward `e_plus`.
- conditions on `e_t` plus `c_cond`.
- Stage 2 uses `c_cond = c_plus`.
- Stage 3 uses `c_cond = as_target(c_hat)` after a ramp.
- `L_e` must not backpropagate into `F_c` through `c_hat`.
- shuffled-c and zero-c tests prove that detailed prediction actually depends
  on abstract state.

Phase 3 adds frame generation:

- frozen `diffusers.AutoencoderKL` VAE.
- future frame target in VAE latent space.
- frame generator trains with rectified-flow loss in patched VAE latent space.
- latent stack and VAE are frozen; only frame generator trains.
- decoder dependency tests prove frames depend on predicted detailed state.

Phase 4 adds multi-horizon coarse prediction:

- horizon set `(4, 8, 16, 32)`.
- per-sample horizon sampling.
- learned horizon embedding `h_k`.
- one shared `F_c` handles all horizons.
- data pipeline must sample target windows ending at each selected horizon.

Again: none of these are currently implemented. Check for classes/functions
before assuming they exist.

## 15. Non-negotiable invariants

Do not violate these without explicit human approval:

1. The encoder is frozen, shared, and never optimized.
2. There is no target encoder. Only `B` has an EMA copy.
3. `B_EMA` never receives gradients and never enters an optimizer group.
4. All target branch outputs are stop-gradient.
5. Use `as_target()` for detach boundaries; do not scatter unexplained
   `.detach()` calls.
6. `c_t` on the conditioning path is not detached during latent training.
7. In future Stage 3, `c_hat` must be detached before feeding `F_e`.
8. In future Stage 4, `e_hat` and the latent stack must be detached/frozen
   before training the pixel generator.
9. Encoder input normalization is ImageNet/V-JEPA stats, not `[-1, 1]`.
10. No tubelet dropout on frozen encoder inputs.
11. SSv2 direction-sensitive transforms must not include horizontal/temporal
    flips unless a human approves a changed data contract.
12. `c_t` must remain a bottleneck. Do not casually widen `N_c`, `D_c`, or add
    bypasses that let future prediction ignore the abstract path.
13. Diagnostics are part of the architecture. A component is not done if the
    bypass/collapse test that proves it is real is absent.
14. Changing locked dimensions, gradient routing, or phase sequencing requires
    escalation.
15. Do not import KANBAN or run-history content into this architectural entry
    file.

## 16. How to verify your understanding before edits

For any code change, trace the relevant path in this order:

1. Find the public function/class with `rg -n "class|def"`.
2. Read its docstring and call sites.
3. Trace tensor shapes from `config.py`.
4. Identify which loss terms can reach it.
5. Identify detach/EMA/frozen boundaries.
6. Check diagnostics or tests that assert the intended contract.
7. Add or update tests when you change a contract.

Useful local checks:

```bash
python -m py_compile config.py data.py models.py losses.py diagnostics.py train.py make_subset.py select_ego4d_uids.py chunk_ego4d.py make_ego4d_subset.py
pytest -q
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

RunPod checks that may download or require data:

```bash
python -c "from data import smoke_test_dataloader; smoke_test_dataloader()"
python -c "from models import smoke_test_encoder; smoke_test_encoder()"
python train.py --stage0-only
```

Use `pytest tests/test_phase1_contract.py -q` when touching config, subset, or
flat-file deliverables. Use targeted tests in `tests/test_*` when touching AGC,
optimizer grouping, reconstruction, decoder, SIGReg, or present-only mode.

## 17. File-level implementation guide

`config.py`:

- Holds dataclasses and path defaults.
- Includes properties for token geometry.
- Avoid inline magic constants in hot paths. Add config fields first.
- CLI overrides are applied in `train.main`, not here.

`data.py`:

- Owns only clip-per-file video loading and preprocessing for SSv2 and EGO4D
  chunk roots.
- Keep context/target transforms shared where intended.
- Do not change video sampling semantics without updating docs and tests.

`make_subset.py`:

- Symlink-only. Never copy or re-encode video bytes.
- Idempotent reruns should be safe.
- Keep manifest output deterministic.

`select_ego4d_uids.py`, `chunk_ego4d.py`, `make_ego4d_subset.py`:

- Offline dataset-preparation helpers, not training dependencies.
- Keep UID selection deterministic and source-video-level train/validation
  splitting intact.
- Keep EGO4D chunking idempotent and batch-friendly; raw downloads are
  transient, but manifests and `ego4d.json` are provenance.

`models.py`:

- Owns only modules with parameters/buffers.
- Keep construction order `(encoder, bottleneck, target_bottleneck,
  coarse_flow, decoder)` consistent with `train.py`.
- Do not put losses or optimizer logic here.

`losses.py`:

- Pure tensor math only.
- No `nn.Module`, no trainable parameters.
- Detach targets through `as_target()`.

`diagnostics.py`:

- Pure probes and optimizer-support utilities.
- Any new bypass/collapse test should return a flat `dict[str, float]`.
- Keep AGC and weight-decay exclusion predicates shared.

`train.py`:

- Orchestrates modules, losses, optimizer, EMA, logging, checkpoints.
- Keep train-step gradient paths readable.
- If a new loss is added, document exactly which modules it trains and add a
  test for the gradient contract.

`parse_logs.py`, `run_history.py`, `drift_probe.py`, `rank_probe.py`, and
`whiten_stats.py`:

- Analysis helpers, not training dependencies.
- Do not couple core training to these scripts.
- `drift_probe.py` specifics: the probe set is pinned by a manifest JSON and the
  frozen-encoder features are cached under `--out-dir` (default
  `logs/drift_probe/`). Never regenerate an existing manifest in place —
  checkpoint results are only comparable when the manifest is identical. The
  bottleneck is rebuilt from the config serialized inside the checkpoint;
  window sampling and transforms reuse `data.py` validation helpers. Pure
  helpers are unit-tested in `tests/test_drift_probe.py`.
- `whiten_stats.py` specifics: single-pass fp64 mean/covariance over training
  split encoder tokens; saves raw eigenvalues so `cfg.train.whiten_eps` can be
  swept at load time without recomputing. Training never imports the script —
  `train._build_whitener` only reads its `.pt` artifact.
- `rank_probe.py` specifics: it shares the drift-probe manifest/cache naming by
  default, encodes only anchor windows, and builds manifests compatible with the
  default drift offset ladder so a rank-only first run does not poison later
  drift comparisons. Pure helpers are unit-tested in `tests/test_rank_probe.py`.

## 18. Current limitations to keep visible

- The project is currently coarse-latent only.
- The reconstruction `Decoder` reconstructs frozen encoder features, not pixels.
- There is no inference sampler for real future generation yet.
- There is no fine-flow shuffled-c proof yet.
- There is no frame-generator dependency proof yet.
- Some docs preserve older constants and planned-stage placeholders. Verify
  against code before using them in implementation.

The right agent behavior is to preserve the current working contracts while
making progress on the specific requested change, not to opportunistically
complete the whole research roadmap.
