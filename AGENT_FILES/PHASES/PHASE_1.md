# PHASE_1.md — Coarse Hierarchy (Stages 0 + 1) — v0.2 (frozen encoder)

> **⚠️ v0.2 update.** This phase now builds: a **frozen pretrained V-JEPA 2 ViT-L/16 encoder** (`E`,
> dim 1024) + a **trainable bottleneck** `B` → `c_t` + an **EMA bottleneck** `B_EMA` + coarse flow
> `F_c`, with a **variance floor on `c_t`** (no SIGReg). It also logs the three required monitors
> (variance, cross-video cosine, effective rank of `c_t`). Read
> [`AGENT_FILES/KNOWLEDGE/BRIEF_V0_2.md`](../KNOWLEDGE/BRIEF_V0_2.md),
> [`SUPERVISOR_FEEDBACK_EXPLAINED.md`](../KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md), and
> [`FROZEN_ENCODER_RESEARCH.md`](../KNOWLEDGE/FROZEN_ENCODER_RESEARCH.md) before this doc.
>
> **Agent instruction:** Execute this document top-to-bottom. When finished, the repo trains a
> coarse-only world model on RunPod: **frozen encoder + trainable bottleneck + EMA bottleneck +
> coarse flow `F_c`**, with the variance floor. Stage 1 must pass non-collapse and baseline-beating
> gates before Phase 2 begins.
>
> **Prerequisites:** Complete **`AGENT_FILES/SETUPS/SETUP.md` Path A** (first time) or **Path B** (subsequent runs). SSv2 at `/workspace/data/ssv2/`; `ssv2_tiny` created by `make_subset.py`. Read [`AGENT_FILES/AGENTS.md`](../AGENTS.md), then [`AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`](../AGENT-BEHAVIOUR/PROTOCOL.md), [`AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md`](../AGENT-BEHAVIOUR/CODE_DESIGN.md), [`AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md`](../KNOWLEDGE/UNDERSTANDING.md) §0–§3 + §2.6, and [`AGENT_FILES/KNOWLEDGE/BRIEF_V0_2.md`](../KNOWLEDGE/BRIEF_V0_2.md) §1–§7, §9–§11.

---

## Workflow — execute in this exact order

| Step | Section | Deliverable | Verify before continuing |
|---|---|---|---|
| 1 | §2 | Confirm RunPod data layout (see [`AGENT_FILES/SETUPS/SETUP.md`](../SETUPS/SETUP.md) A8 if first time) | `ls /workspace/data/ssv2/train \| head` shows symlinks |
| 2 | §3 | `requirements.txt`, `pyproject.toml`, `.gitignore`, `.pre-commit-config.yaml` | `pip install -r requirements.txt` succeeds |
| 3 | §4 | `config.py` — all §2.6 constants | Import test: `python -c "from config import Config; print(Config())"` |
| 4 | §5 | `make_subset.py` + run it → `ssv2_tiny/` | Manifest exists; symlink count ≈ 4000 train / 256 val |
| 5 | §6 | `data.py` — dataset + dataloader | `python -c "from data import smoke_test_dataloader; smoke_test_dataloader()"` |
| 6 | §7 | `models.py` — FrozenEncoder (E), B, B_EMA, F_c | `python -c "from models import smoke_test_models; smoke_test_models()"` |
| 7 | §8 | `losses.py` — flow matching + variance floor | Unit smoke on synthetic tensors |
| 8 | §9 | `diagnostics.py` — Phase 1 probes | Each function returns a metrics dict on fake batch |
| 9 | §10 | `train.py` — Stage 0 sanity + Stage 1 loop | Stage 0 passes in < 2 min |
| 10 | §11 | Full smoke train on `ssv2_tiny` | 500 steps, no NaN, W&B logging live |
| 11 | §12 | Acceptance gates | All thresholds in §12 met or reported |
| 12 | §13 | `README.md` Phase 1 section | Human-readable run instructions |

**Do not start Phase 2 until §12 passes.**

---

## §2. RunPod data paths (code defaults)

**Volume reference:** [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../SETUPS/VOLUME_LAYOUT.md) — current volume contents, target tree, migration map.

**Human operator:** Volume setup is in **[`AGENT_FILES/SETUPS/SETUP.md`](../SETUPS/SETUP.md) Path A** (steps A7–A8). This section is what the **agent hardcodes** in `config.py`.

**Expected layout on the volume:**

```
/workspace/data/ssv2/train/          ← symlinks to .webm in /workspace/ssv2_raw/...
/workspace/data/ssv2/validation/
/workspace/data/ssv2/labels.json     ← maps video_id → class label string
/workspace/data/ssv2_tiny/           ← created by make_subset.py in §5
/workspace/checkpoints/              ← training checkpoints (sibling to repo)
```

Code defaults (override via `JEPA_DATA_ROOT` for local dev only):

```python
DATA_ROOT = os.environ.get("JEPA_DATA_ROOT", "/workspace/data")
SSV2_FULL = f"{DATA_ROOT}/ssv2"
SSV2_TINY = f"{DATA_ROOT}/ssv2_tiny"
CHECKPOINT_DIR = "/workspace/checkpoints"
```

---

## §3. Project scaffolding

Create at repo root:

**`requirements.txt`** (minimum):
```
torch>=2.1
torchvision
decord
numpy
wandb
diffusers          # listed now; used in Phase 3 only
transformers>=4.49 # must include V-JEPA 2 / 2.1 support (vjepa2 model family)
accelerate
einops
```

> **Encoder availability note (resolved).** The originally-scoped ViT-B/16 (`D_e=768`) has **no
> `transformers`/HF repo** — only a `torch.hub` entrypoint (`vjepa2_1_vit_base_384`). We therefore use
> **V-JEPA 2 ViT-L/16** via the clean HF path `facebook/vjepa2-vitl-fpc64-256` (`D_e=1024`), whose
> native resolution **256 matches ours**. Loading is `AutoModel.from_pretrained(...)` +
> `get_vision_features`. If this repo id fails to load, **escalate to the human** — do not silently
> substitute a different encoder.

**`.gitignore`:** `__pycache__/`, `wandb/`, `*.pt`, `.env`, `archive/`

**`.pre-commit-config.yaml`:** black (line-length 100) + ruff, per `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` §8.

---

## §4. `config.py`

Single flat `@dataclass` hierarchy. Include the naming map from `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` §3 as a module-level comment block.

**Must expose (all values from `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §2.6):**

```python
@dataclass
class ModelConfig:
    # Frozen encoder (V-JEPA 2 ViT-L/16) — verify exact HF repo id at load
    encoder_repo: str = "facebook/vjepa2-vitl-fpc64-256"
    encoder_frozen: bool = True
    t_ctx: int = 8            # context frames (tubelet-2 -> 4 temporal tokens)
    h: int = 256
    w: int = 256
    n_ctx: int = 1024         # (8/2) * (256/16)^2 = 4 * 256
    n_tgt: int = 1024         # clip-level target, same geometry
    n_c: int = 32
    d_e: int = 1024            # frozen encoder embedding dim
    d_c: int = 256
    bottleneck_convnext_blocks: int = 2
    bottleneck_cross_attn_heads: int = 8
    f_c_blocks: int = 6
    f_c_dim: int = 256
    f_c_heads: int = 8
    condition_dropout: float = 0.10
    # NOTE: no tubelet_dropout (removed in v0.2 — frozen encoder)
    # NOTE: no encoder_depth/heads — set by the pretrained checkpoint

@dataclass
class TrainConfig:
    global_batch: int = 64
    stage1_steps: int = 30_000
    # Phase 1 trains ONLY stage 1; later stages added in Phase 2/3
    max_steps: int = 30_000
    # NOTE: no lr_encoder — encoder is frozen
    lr_bottleneck: float = 2e-4
    lr_coarse_flow: float = 4e-4
    warmup_steps: int = 10_000
    total_latent_steps: int = 105_000  # for cosine LR denominator (Phase 2+)
    adam_betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    ema_m_start: float = 0.996
    ema_m_end: float = 0.9999
    ema_schedule_steps: int = 105_000
    lambda_var: float = 0.10           # variance-floor weight (replaces SIGReg)
    var_floor_std_target: float = 1.0  # hinge target in L_var
    horizon_k: int = 4                 # single fixed horizon for Phases 1-3 (Phase 4 multi-horizon)
    frame_stride: int = 2
    precision: str = "bf16"
    log_every: int = 50
    diag_every: int = 500
    checkpoint_every: int = 5000

@dataclass
class DataConfig:
    data_root: str = field(default_factory=lambda: os.environ.get("JEPA_DATA_ROOT", "/workspace/data"))
    dataset: str = "ssv2_tiny"  # CLI override: ssv2 | ssv2_tiny
    num_workers: int = 8
    pin_memory: bool = True

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    debug_shapes: bool = True
    checkpoint_dir: str = "/workspace/checkpoints"
    seed: int = 42
```

Add CLI parsing in `train.py` (or a tiny `cli.py`) for `--data {ssv2,ssv2_tiny}`, `--steps`, `--resume`, `--seed`.

---

## §5. `make_subset.py`

Build SSv2-tiny for 4–5 hour smoke runs. **Spec:**

**Purpose:** Stratified random subset reusing existing symlinks.

**Inputs:**
- Source train: `{DATA_ROOT}/ssv2/train/`
- Source val: `{DATA_ROOT}/ssv2/validation/`
- Labels: `{DATA_ROOT}/ssv2/labels.json`
- Args: `--train-per-class` (default 23), `--val-per-class` (default 2), `--seed` (default 42)

**Outputs:**
- `{DATA_ROOT}/ssv2_tiny/train/` — symlinks to resolved `.webm` targets
- `{DATA_ROOT}/ssv2_tiny/validation/`
- `{DATA_ROOT}/ssv2_tiny/manifest.json` — selected video_ids, per-class counts, seed

**Algorithm:**
1. Load labels.json → `{video_id: class_label}`.
2. For each split in `[train, validation]`:
   - Enumerate symlinks; extract video_id from filename (strip `.webm`).
   - Group by class_label.
   - Per class: deterministic shuffle with seed; take first N (`train_per_class` or `val_per_class`).
   - Resolve symlink → absolute `.webm` path; create new symlink in tiny dir.
3. Write manifest.json.

**Edge cases:** idempotent reruns (skip existing symlinks); log and skip broken symlinks; log undersized classes.

**Self-check at end:** print total counts, per-class min/max/mean, assert all symlinks readable.

**Run once on RunPod:**
```bash
cd /workspace/hierarchal-jepa-flow-world-model
python make_subset.py --train-per-class 23 --val-per-class 2 --seed 42
```

Expected: ~4,002 train symlinks (174 classes × 23, modulo undersized), ~348 val (174 × 2).

---

## §6. `data.py`

### 6.1 Dataset contract (v0.2)

Each `__getitem__` returns:
- `context_clip`: `(T=8, C=3, H=256, W=256)` float32, **encoder-normalized** (V-JEPA processor stats)
- `target_clip`: `(T=8, C=3, H=256, W=256)` float32, encoder-normalized — the future window ending
  at `t+k` (`k = cfg.train.horizon_k`, single horizon in Phases 1–3)

**Clip sampling:**
- Load video via `decord` (CPU, per-worker).
- SSv2 ~12 fps; sample at **stride 2**. Take an 8-frame **context** window ending at `t` and an
  8-frame **target** window ending at `t + k`.
- Require the video to be long enough for both windows; deterministically skip/clip-pad too-short
  videos and log counts.
- Random temporal start index per epoch (different crops across epochs).

**Augmentations (train):**
- Resize shorter side to 256, random crop 256×256.
- Color jitter: brightness=0.4, contrast=0.4, saturation=0.4, hue=0. (Apply the **same** crop/jitter
  to context and target windows for consistency.)
- **No** horizontal flip, temporal flip, or rotation (SSv2 labels are direction-sensitive).
- Apply the frozen encoder's normalization **after** geometric/color augmentation.

**Eval transform:** shorter side 256, center crop 256×256, no color jitter.

### 6.2 Dataloader

- `batch_size = global_batch` (64) unless OOM → document grad accumulation in README if needed.
- `num_workers=8`, `pin_memory=True`.
- `drop_last=True` for training.

### 6.3 No tubelet dropout (v0.2)

Tubelet dropout is **removed** — the frozen encoder never saw dropped tokens at pretraining. The
dataloader returns full clips; the encoder consumes all tokens. (If a shortcut knob is wanted later,
mask **inside the bottleneck**, per `UNDERSTANDING.md` §3.2.1.)

### 6.4 `smoke_test_dataloader()`

Load 2 batches from `ssv2_tiny`, print shapes `(B,8,3,256,256)` for both context and target, assert
no NaN, and confirm values match the encoder processor's expected range (not [-1,1]).

---

## §7. `models.py` — Phase 1 modules only

Implement per `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §3. Use `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` naming map.

### 7.1 Frozen encoder `E` (`FrozenEncoder`)

A thin wrapper around the pretrained **V-JEPA 2 ViT-L/16** (do **not** implement a patchifier or
position embeddings — they are internal to the encoder).

- Load: `AutoModel.from_pretrained(cfg.model.encoder_repo)` (+ matching `AutoVideoProcessor`).
- `requires_grad=False` on all params; `eval()`; forward under `torch.no_grad()`.
- Input: clip `(B, 8, 3, 256, 256)`. Output: `e` `(B, 1024, 1024)` via `get_vision_features` /
  `last_hidden_state`.
- Used for **both** context (`e_t`) and target (`e_plus`) — the same frozen instance.
- Assert at construction: total trainable params of `E` == 0.

### 7.2 (removed) — there is no from-scratch online encoder in v0.2

### 7.3 Bottleneck `B` (`Bottleneck`)

Per `UNDERSTANDING.md` §3.3:
1. Input projection `1024 → mixer width` (no kept-mask / zero-fill; all 1024 tokens are real).
2. Reshape to `(B·4, 16, 16, ·)` (4 temporal slots × 16×16 spatial); per-slot 2D ConvNeXt mixing —
   **2 blocks**, weights shared across the 4 temporal-slot grids.
3. Cross-attention: 32 **learned query embeddings** (dim 256) → keys/values from projected tokens
   (→256), 8 heads.
4. Output MLP block + LayerNorm → `c_t` `(B, 32, 256)`.

The bottleneck is applied identically to `e_t` (→ `c_t`) and, via the EMA copy, to `e_plus`
(→ `c_plus`).

### 7.4 EMA bottleneck branch (`TargetBottleneck` / `B_EMA`)

- A single EMA copy of `B` (the encoder is frozen and shared — **no** target encoder):
  - `requires_grad=False` on all params; always `eval()`.
- Forward: `target_clip` → frozen `E` → `e_plus` → `B_EMA` → `c_plus`; wrap with `as_target()`.
- Init: Stage 0 copies `B → B_EMA` (no encoder copy).

### 7.5 Coarse flow `F_c` (`CoarseFlow`)

Per `UNDERSTANDING.md` §3.6:
- 6 DiT blocks, dim 256, 8 heads, adaLN-Zero from `τ`.
- Conditioning: concat `[z_c || c_t]` → 64 tokens; read out first 32 as velocity prediction.
- Condition dropout 10%: replace `c_t` with learned null embedding `(32, 256)`.
- Output: `v_hat` (= `u_c_hat`) `(B, 32, 256)`.
- (Phase 4 adds a learned horizon embedding `h_k` input; not wired in Phase 1.)

### 7.6 `smoke_test_models()`

Synthetic batch B=2:
- Forward context clip through frozen `E` then `B` → `c_t` `(2,32,256)`; confirm `E` has no grad.
- Forward target clip through `E` then `B_EMA` → `c_plus` detached.
- Sample `τ`, `z_0`, build `z_τ`, run `F_c` → velocity shape `(2,32,256)`.
- Backward on `L_flow` stub → gradients reach **B, F_c only**; **not** `E`, **not** `B_EMA`.

---

## §8. `losses.py`

Pure tensor math — no `nn.Parameter`.

### 8.1 `flow_matching_loss(u_hat, u_target)`

`mean((u_hat - u_target) ** 2)` over all non-batch dims. Shared by coarse/fine/frame losses.

### 8.2 Building blocks

```python
def interpolate(z_target, eps, tau):
    """z = (1 - tau) * eps + tau * z_target. tau shape (B,1,1) broadcastable."""
def velocity_target(z_target, eps):
    """u = z_target - eps."""
```

### 8.3 `variance_floor(c_t, std_target=1.0)` (replaces SIGReg)

Variance floor on `c_t` only:
- Input: `(B, N_c, D_c)`.
- Flatten across slots/features per batch; compute **per-dimension std** across the batch.
- `L_var = mean_j( max(0, std_target - std_j) )`.
- Return scalar loss.

Do **not** implement SIGReg, VICReg, or covariance losses (v0.2 supervisor directive).

### 8.4 Phase 1 total loss

```python
L = L_flow + cfg.train.lambda_var * variance_floor(c_t)   # lambda_var = 0.10
```

`L_flow` is the coarse flow-matching loss (`flow_matching_loss(v_hat, v_target)` with
`v_target = c_plus - z_0`). No `L_e` in Phase 1. Variance floor on `c_t` active from step 0.

---

## §9. `diagnostics.py`

Each function: pure, returns `dict[str, float]`.

| Function | Measures | Threshold source |
|---|---|---|
| `variance_stats(c)` | per-dim std of `c_t` (mean, frac near-zero) | §2.6 / §9.1 collapse flags |
| `cross_video_cosine(c)` | mean pairwise cosine of `c_t` across **different** videos | §9.1b (healthy ≪ ~0.5) |
| `effective_rank(c, D)` | covariance effective rank of `c_t` | §2.6 rank floors |
| `coarse_baselines(F_c, batch, ...)` | L_flow for model vs copy vs batch-mean | ratio ≤ 0.70 / ≤ 0.50 after 10k steps |
| `gradient_health(model)` | global grad norm, NaN check | hard stop on NaN |

> The first three (variance, cross-video cosine, effective rank of `c_t`) are the supervisor's
> **required minimum** monitors — log all three to W&B.

**Copy baseline:** predict `c_plus` by outputting current `c_t` unchanged (identity in abstract space).

**Batch-mean baseline:** predict `c_plus` as mean of all `c_plus` in the current batch of 64 clips.

Run diagnostics every `diag_every` steps (500) on a **fixed small val batch** (cache 64 clips at init).

---

## §10. `train.py`

### 10.1 Stage 0 — sanity (no real data)

Script section or `--stage0-only` flag:

1. Build all Phase 1 modules on GPU; **load + freeze the encoder** (assert 0 trainable encoder params).
2. `B_EMA.load_state_dict(B.state_dict())` (no encoder copy).
3. Synthetic batch: random context `(B,8,3,256,256)` and target `(B,8,3,256,256)`.
4. One forward + backward + optimizer step + **bottleneck** EMA update.
5. Assert: no NaN in loss or grads; `B_EMA` params changed slightly; **encoder params unchanged**.

Exit 0 in < 2 minutes.

### 10.2 Stage 1 training loop

**Optimizer:** AdamW with param groups (encoder is **frozen — not in any group**):
- `B`: lr 2e-4
- `F_c`: lr 4e-4

**LR schedule:** linear warmup 10k steps, then cosine decay to 0 over remaining `(stage1_steps - warmup)` = 20k steps.

**EMA:** after each optimizer step, update **`B_EMA` only**:
```python
m = ema_cosine(step, start=0.996, end=0.9999, total=105_000)
for p_online, p_ema in zip(B.parameters(), B_EMA.parameters()):
    p_ema.data.lerp_(p_online.data, 1 - m)
# No encoder EMA — E is frozen and shared.
```

**Training step (Stage 1):**

```
1. Load batch: context_clip, target_clip
2. with no_grad: e_t = E(context_clip)          # frozen encoder
3. c_t = B(e_t)                                  # trainable
4. with no_grad: e_plus = E(target_clip); c_plus = as_target(B_EMA(e_plus))
5. Sample z_0 ~ N(0,I), tau ~ U(0,1) per example
6. z_tau = (1-tau)*z_0 + tau*c_plus
7. v_target = c_plus - z_0
8. v_hat = F_c(z_tau, tau, c_t)                  # condition dropout inside F_c
9. L_flow = flow_matching_loss(v_hat, v_target)
10. L = L_flow + lambda_var * variance_floor(c_t)
11. backward, clip grad 1.0, step, EMA update (B only)
```

**AMP:** bf16 autocast (encoder forward under `no_grad`).

**Checkpointing:** save `{step, B, F_c, B_EMA, optimizer, config}` (encoder not saved — reload from
HF) to `/workspace/checkpoints/phase1_step{step}.pt` every 5000 steps and at end.

**W&B:** log `L_flow`, `L_var`, **variance(c_t)**, **cross_video_cosine(c_t)**, **effective_rank(c_t)**,
LR, EMA m, grad norm every `log_every` / `diag_every`.

### 10.3 CLI

```bash
# Stage 0
python train.py --stage0-only

# Smoke (500 steps)
python train.py --data ssv2_tiny --steps 500

# Full Phase 1 (30k steps, ~4-5 hr on A100 80GB with tiny data)
python train.py --data ssv2_tiny --steps 30000

# Full dataset (production — long run)
python train.py --data ssv2 --steps 30000
```

Default `--data ssv2_tiny`.

---

## §11. Intermediate verification (before acceptance)

After 500-step smoke:
- [ ] No NaN in loss
- [ ] W&B run shows decreasing L_flow (no requirement to beat baselines yet)
- [ ] Checkpoint saves and loads

After 10k+ steps on tiny (or full val subset):
- [ ] Run `coarse_baselines` on val — record ratios

---

## §12. Acceptance gates — Phase 1 complete when ALL pass

| Gate | Criterion | Source |
|---|---|---|
| Stage 0 | Synthetic forward/backward/EMA, no NaN | §10.1 |
| Shapes | All modules match §2 shape table | smoke tests |
| Training stability | 30k steps on `ssv2_tiny` completes without OOM/NaN | RunPod |
| Frozen encoder | 0 trainable encoder params; encoder weights unchanged after a step | §7.1 |
| F_c vs copy | val L_flow ratio ≤ **0.70** after step ≥ 10k | §2.6 |
| F_c vs batch-mean | val L_flow ratio ≤ **0.50** after step ≥ 10k | §2.6 |
| `c_t` health | eff. rank `c_t` > 60 on val | §2.6 |
| `c_t` no collapse (variance) | < 15% dims with std < 0.1× median (warning); hard stop if > 30% | §9.1 |
| `c_t` no collapse (cosine) | cross-video cosine well below ~0.5 (warn if drifting to 1.0) | §9.1b |
| Gradient health | no repeated NaN; grad norm logged | §9 |
| Checkpoint | final checkpoint at step 30000 loadable | manual |
| README | documents `--data`, paths, expected runtime | §13 |

Report all metric values to the human with checkpoint path.

---

## §13. README.md (Phase 1 section)

Must include:
- What Phase 1 builds (plain language: **frozen pretrained encoder**, trainable bottleneck, coarse predictor, **EMA bottleneck** targets, variance floor).
- What "beats copy/batch-mean baselines" means.
- One-time data migration — see `AGENT_FILES/SETUPS/SETUP.md` step A8 (human operator).
- `make_subset.py` then `train.py` commands.
- Default paths and `JEPA_DATA_ROOT` override.
- Expected runtime: ~4–5 hours for 30k steps on A100 80GB with `ssv2_tiny`.
- What is **not** in Phase 1: `F_e`, frame generator, Stages 2–4.

---

## §14. Files delivered at end of Phase 1

```
config.py
make_subset.py
data.py
models.py          # FrozenEncoder, Bottleneck, B_EMA, CoarseFlow only
losses.py
diagnostics.py
train.py
requirements.txt
.gitignore
.pre-commit-config.yaml
README.md
```

**Not in scope for Phase 1:** `FineFlow`, `FrameGenerator`, `eval.py`, VAE loading, Stages 2–4 logic, shuffled-c test.

---

## §15. Handoff to Phase 2

Leave:
- Final checkpoint at `/workspace/checkpoints/phase1_step30000.pt` (or latest).
- W&B run URL in README or commit message.
- Note any soft warnings (rank borderline, etc.).

Phase 2 will **add** `FineFlow`, extend `train.py` for Stages 2–3, extend `diagnostics.py` — without rewriting Phase 1 module internals. See [`AGENT_FILES/PHASES/PHASE_2.md`](PHASE_2.md).
