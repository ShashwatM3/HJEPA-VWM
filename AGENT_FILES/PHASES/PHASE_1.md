# PHASE_1.md — Coarse Hierarchy (Stages 0 + 1)

> **Agent instruction:** Execute this document top-to-bottom. When finished, the repo trains a coarse-only world model on RunPod: encoder + bottleneck + EMA target branch + coarse flow `F_c`, with SIGReg regularization. Stage 1 must pass non-collapse and baseline-beating gates before Phase 2 begins.
>
> **Prerequisites:** Complete **`AGENT_FILES/SETUPS/SETUP.md` Path A** (first time) or **Path B** (subsequent runs). SSv2 at `/workspace/data/ssv2/`; `ssv2_tiny` created by `make_subset.py`. Read [`AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`](../AGENT-BEHAVIOUR/PROTOCOL.md), [`AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md`](../AGENT-BEHAVIOUR/CODE_DESIGN.md), [`AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md`](../KNOWLEDGE/UNDERSTANDING.md) §0–§3 + §2.6, and [`AGENT_FILES/KNOWLEDGE/hierarchical_jepa_flow_architecture_brief.pdf`](../KNOWLEDGE/hierarchical_jepa_flow_architecture_brief.pdf) §1–§7, §9–§11.

---

## Workflow — execute in this exact order

| Step | Section | Deliverable | Verify before continuing |
|---|---|---|---|
| 1 | §2 | Confirm RunPod data layout (see [`AGENT_FILES/SETUPS/SETUP.md`](../SETUPS/SETUP.md) A8 if first time) | `ls /workspace/data/ssv2/train \| head` shows symlinks |
| 2 | §3 | `requirements.txt`, `pyproject.toml`, `.gitignore`, `.pre-commit-config.yaml` | `pip install -r requirements.txt` succeeds |
| 3 | §4 | `config.py` — all §2.6 constants | Import test: `python -c "from config import Config; print(Config())"` |
| 4 | §5 | `make_subset.py` + run it → `ssv2_tiny/` | Manifest exists; symlink count ≈ 4000 train / 256 val |
| 5 | §6 | `data.py` — dataset + dataloader | `python -c "from data import smoke_test_dataloader; smoke_test_dataloader()"` |
| 6 | §7 | `models.py` — patchifier, E, B, E_bar, B_bar, F_c | `python -c "from models import smoke_test_models; smoke_test_models()"` |
| 7 | §8 | `losses.py` — flow matching + SIGReg | Unit smoke on synthetic tensors |
| 8 | §9 | `diagnostics.py` — Phase 1 probes | Each function returns a metrics dict on fake batch |
| 9 | §10 | `train.py` — Stage 0 sanity + Stage 1 loop | Stage 0 passes in < 2 min |
| 10 | §11 | Full smoke train on `ssv2_tiny` | 500 steps, no NaN, W&B logging live |
| 11 | §12 | Acceptance gates | All thresholds in §12 met or reported |
| 12 | §13 | `README.md` Phase 1 section | Human-readable run instructions |

**Do not start Phase 2 until §12 passes.**

---

## §2. RunPod data paths (code defaults)

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
diffusers   # listed now; used in Phase 3 only
transformers
accelerate
```

**`.gitignore`:** `__pycache__/`, `wandb/`, `*.pt`, `.env`, `archive/`

**`.pre-commit-config.yaml`:** black (line-length 100) + ruff, per `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` §8.

---

## §4. `config.py`

Single flat `@dataclass` hierarchy. Include the naming map from `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` §3 as a module-level comment block.

**Must expose (all values from `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §2.6):**

```python
@dataclass
class ModelConfig:
    t_ctx: int = 4
    h: int = 128
    w: int = 128
    patch_t: int = 1
    patch_h: int = 16
    patch_w: int = 16
    n_ctx: int = 256
    n_tgt: int = 64
    n_c: int = 32
    d_e: int = 384
    d_c: int = 256
    encoder_depth: int = 12
    encoder_heads: int = 6
    encoder_mlp_ratio: int = 4
    bottleneck_convnext_blocks: int = 2
    bottleneck_cross_attn_heads: int = 8
    f_c_blocks: int = 6
    f_c_dim: int = 256
    f_c_heads: int = 8
    tubelet_dropout: float = 0.40
    condition_dropout: float = 0.10

@dataclass
class TrainConfig:
    global_batch: int = 64
    stage1_steps: int = 30_000
    # Phase 1 trains ONLY stage 1; later stages added in Phase 2/3
    max_steps: int = 30_000
    lr_encoder: float = 2e-4
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
    lambda_e_reg: float = 0.02
    lambda_c_reg: float = 0.10
    sigreg_m: int = 1024
    sigreg_knots: int = 17
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

### 6.1 Dataset contract

Each `__getitem__` returns:
- `context_clip`: `(T=4, C=3, H=128, W=128)` float32, normalized to **[-1, 1]**
- `future_frame`: `(C=3, H=128, W=128)` float32, **[-1, 1]**

**Clip sampling:**
- Load video via `decord` (CPU, per-worker).
- SSv2 ~12 fps; sample 5 frames with **stride 2** → covers ~10 source frames (~0.83 s).
- First 4 frames = context; 5th = target `y`.
- Random temporal start index per epoch (different crops across epochs).

**Augmentations (train):**
- Resize shorter side to 128, random crop 128×128.
- Color jitter: brightness=0.4, contrast=0.4, saturation=0.4, hue=0.
- **No** horizontal flip, temporal flip, or rotation (SSv2 labels are direction-sensitive).

**Eval transform:** shorter side 128, center crop 128×128, no color jitter.

### 6.2 Dataloader

- `batch_size = global_batch` (64) unless OOM → document grad accumulation in README if needed.
- `num_workers=8`, `pin_memory=True`.
- `drop_last=True` for training.

### 6.3 Tubelet dropout (applied in model forward, not dataloader)

Document in docstring: dataloader returns full clips; tubelet dropout happens in patchifier path per `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §3.2.1.

### 6.4 `smoke_test_dataloader()`

Load 2 batches from `ssv2_tiny`, print shapes, assert value range in [-1.2, 1.2], no NaN.

---

## §7. `models.py` — Phase 1 modules only

Implement per `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §3. Use `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` naming map.

### 7.1 Shared patchifier (`PatchEmbed`)

- `Conv3d` kernel `(1,16,16)`, stride `(1,16,16)`, out_channels `D_e=384`.
- Context: `(B,3,4,128,128)` → `(B,256,384)`.
- Target: `(B,3,1,128,128)` → `(B,64,384)`.
- **3D sin-cos position embeddings** (factorized T+H+W, summed). Target temporal index = 4 (frame after context). Precompute tables; not learned.
- Context path: apply tubelet dropout **after** pos embed (40%, keep ~60%).
- Target path: no dropout.

Return `(tokens, kept_mask)` for context; `(tokens, None)` for target.

### 7.2 Online encoder `E` (`OnlineEncoder`)

- ViT-S: depth 12, dim 384, 6 heads, MLP ratio 4, pre-norm, no CLS.
- Input: variable ~154 tokens (train) or 256 (eval).
- Output: `e_t` same shape as input.

### 7.3 Bottleneck `B` (`Bottleneck`)

Per §3.3:
1. Reconstruct spatial grid from `e_t` + kept_mask; zero-fill dropped slots.
2. Per-frame 2D ConvNeXt mixing — **2 blocks**, shared weights across 4 context frames.
3. Cross-attention: 32 **learned query embeddings** (dim 256, no input conditioning) → keys/values from projected tokens (384→256), 8 heads.
4. Output MLP block + LayerNorm → `c_t` `(B, 32, 256)`.

Target side (for EMA branch): reshape 64 tokens to 8×8 grid, same ConvNeXt blocks, cross-attn → `c_plus`.

### 7.4 EMA target branch (`TargetBranch`)

- Separate `TargetEncoder` + `TargetBottleneck` — structurally identical to E/E_bar but:
  - `requires_grad=False` on all params.
  - Always `eval()`.
  - No tubelet dropout on target path.
- Forward: `future_frame` → patchify target → `E_bar` → `e_plus` → `B_bar` → `c_plus`.
- Wrap outputs with `as_target()` (detach).

Init: `copy_weights_from_online()` — Stage 0 copies `E→E_bar`, `B→B_bar`.

### 7.5 Coarse flow `F_c` (`CoarseFlow`)

Per §3.6:
- 6 DiT blocks, dim 256, 8 heads, adaLN-Zero from `τ_c`.
- Conditioning: concat `[z_c || c_t]` → 64 tokens; read out first 32 as velocity prediction.
- Condition dropout 10%: replace `c_t` with learned null embedding `(32, 256)`.
- Output: `u_c_hat` `(B, 32, 256)`.

### 7.6 `smoke_test_models()`

Synthetic batch B=2:
- Forward context through E, B → shapes OK.
- Forward target through TargetBranch → detached.
- Sample `τ_c`, `ε_c`, build `z_c`, run F_c → velocity shape OK.
- Backward on L_c stub → gradients reach E, B, F_c; not E_bar.

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

### 8.3 `sigreg(latents, m=1024, knots=17)`

Sketched Isotropic Gaussian Regularization (LeJEPA / le-wm reference):
- Input: `(B, N, D)` — use `e_t` or `c_t` (online branch only).
- Random 1D projections + Epps-Pulley characteristic function test.
- Return scalar loss.

Reference impl for structure: `lucas-maes/le-wm` (GitHub). Do not copy blindly — match API in docstring.

### 8.4 Phase 1 total loss

```python
L = L_c + lambda_c_reg * SIGReg(c_t) + lambda_e_reg * SIGReg(e_t)
```

No `L_e` in Phase 1. SIGReg on both latents active from step 0 (locked decision).

---

## §9. `diagnostics.py`

Each function: pure, returns `dict[str, float]`.

| Function | Measures | Threshold source |
|---|---|---|
| `latent_std_stats(e, c)` | per-dim mean/std | §2.6 collapse flags |
| `effective_rank(x, D)` | covariance effective rank | §2.6 rank floors |
| `coarse_baselines(F_c, batch, ...)` | L_c for model vs copy vs batch-mean | ratio ≤ 0.70 / ≤ 0.50 after 10k steps |
| `gradient_health(model)` | global grad norm, NaN check | hard stop on NaN |

**Copy baseline:** predict `c_plus` by outputting current `c_t` unchanged (identity in abstract space).

**Batch-mean baseline:** predict `c_plus` as mean of all `c_plus` in the current batch of 64 clips.

Run diagnostics every `diag_every` steps (500) on a **fixed small val batch** (cache 64 clips at init).

---

## §10. `train.py`

### 10.1 Stage 0 — sanity (no real data)

Script section or `--stage0-only` flag:

1. Build all Phase 1 modules on GPU.
2. `TargetBranch.copy_weights_from_online()`.
3. Synthetic batch: random `(B,4,3,128,128)` and `(B,3,128,128)`.
4. One forward + backward + optimizer step + EMA update.
5. Assert: no NaN in loss or grads; EMA params changed slightly from online.

Exit 0 in < 2 minutes.

### 10.2 Stage 1 training loop

**Optimizer:** AdamW with param groups:
- `E`: lr 2e-4
- `B`: lr 2e-4
- `F_c`: lr 4e-4

**LR schedule:** linear warmup 10k steps, then cosine decay to 0 over remaining `(stage1_steps - warmup)` = 20k steps.

**EMA:** after each optimizer step, update `E_bar`, `B_bar`:
```python
m = ema_cosine(step, start=0.996, end=0.9999, total=105_000)
for p_online, p_ema in zip(E.parameters(), E_bar.parameters()):
    p_ema.data.lerp_(p_online.data, 1 - m)
# same for B / B_bar
```

**Training step (Stage 1):**

```
1. Load batch: context_clip, future_frame
2. Patchify context → e_t = E(...); c_t = B(e_t, mask)
3. target_detailed, target_abstract = TargetBranch(future_frame)  # detached
4. Sample eps_c, tau_c ~ U(0,1) per example
5. z_c = interpolate(target_abstract, eps_c, tau_c)
6. u_c = velocity_target(target_abstract, eps_c)
7. u_c_hat = F_c(z_c, tau_c, c_t)   # condition dropout inside F_c
8. L_c = flow_matching_loss(u_c_hat, u_c)
9. L = L_c + lambda_c_reg*SIGReg(c_t) + lambda_e_reg*SIGReg(e_t)
10. backward, clip grad 1.0, step, EMA update
```

**AMP:** bf16 autocast + GradScaler if needed.

**Checkpointing:** save `{step, model state, optimizer, ema, config}` to `/workspace/checkpoints/phase1_step{step}.pt` every 5000 steps and at end.

**W&B:** log `L_c`, `SIGReg_e`, `SIGReg_c`, LR, EMA m, grad norm, diagnostic dict every `log_every`.

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
- [ ] W&B run shows decreasing L_c (no requirement to beat baselines yet)
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
| F_c vs copy | val L_c ratio ≤ **0.70** after step ≥ 10k | §2.6 |
| F_c vs batch-mean | val L_c ratio ≤ **0.50** after step ≥ 10k | §2.6 |
| Latent health | eff. rank `c_t` > 60, `e_t` > 90 on val | §2.6 |
| No collapse | < 15% dims with std < 0.1× median (warning); hard stop if > 30% | §2.6 |
| Gradient health | no repeated NaN; grad norm logged | §9 |
| Checkpoint | final checkpoint at step 30000 loadable | manual |
| README | documents `--data`, paths, expected runtime | §13 |

Report all metric values to the human with checkpoint path.

---

## §13. README.md (Phase 1 section)

Must include:
- What Phase 1 builds (plain language: encoder, bottleneck, coarse predictor, EMA targets).
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
models.py          # PatchEmbed, OnlineEncoder, Bottleneck, TargetBranch, CoarseFlow only
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
