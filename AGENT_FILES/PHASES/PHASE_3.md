# PHASE_3.md — Frame Generation (Stage 4) + Inference — v0.2 (frozen encoder)

> **⚠️ v0.2 update.** Inherits the **frozen encoder** (`D_e=1024`), **EMA bottleneck** (`B_EMA`), and
> **variance floor** (no SIGReg). `e_hat` (the fine-flow output `D` conditions on) now has the
> encoder-derived dim — `1024` for the clip-level default, shape per the §14 #35 decision. Frames are
> processed at **256×256** (VAE latent `(B,4,32,32)`), and the **render target is frame `t+k`** (the
> last frame of the future clip) under the recommended default. Numbers below tagged `128`/`384`/
> `TargetBranch` are updated inline; exact `e_hat`/`D` token counts lock with the §14 #35 decision.
>
> **Agent instruction:** Execute this document top-to-bottom. Completes **v0**. Adds frozen VAE + frame generator `D`, Stage 4 training, multi-step inference rollout, and standalone evaluation with all seven bypass tests from brief §9.
>
> **Prerequisites:** Phase 2 acceptance gates passed ([`AGENT_FILES/PHASES/PHASE_2.md`](PHASE_2.md)). Checkpoint at step 105k. Latent hierarchy verified (shuffled-c ≥ 2.0).

---

## Workflow — execute in this exact order

| Step | Section | Deliverable | Verify before continuing |
|---|---|---|---|
| 1 | §2 | Extend `config.py` for Stage 4 + VAE | import OK |
| 2 | §3 | VAE wrapper + `FrameGenerator` in `models.py` | encode/decode smoke |
| 3 | §4 | `L_frame` in `losses.py` | synthetic test |
| 4 | §5 | Inference rollout (Heun ODE) | produces `e_hat`, `c_hat` from context |
| 5 | §6 | Extend `train.py` — Stage 4, freeze latent stack | 500-step smoke |
| 6 | §7 | `eval.py` — seven diagnostics on checkpoint | all tests run |
| 7 | §8 | Full Stage 4 train to 150k steps | stable, frames decode |
| 8 | §9 | Acceptance gates | v0 complete |
| 9 | §10 | Final README — full pipeline | |

**After §9, v0 is done. Stage 5 (optional polish) is out of scope.**

---

## §2. Config extensions

```python
stage4_steps: int = 45_000
max_steps: int = 150_000
stage4_start: int = 105_000

lr_frame_generator: float = 2e-4
stage4_warmup_steps: int = 3_000
stage4_total_steps: int = 45_000   # cosine over remaining 42k after warmup

vae_model_id: str = "stabilityai/sd-vae-ft-mse"
vae_scale_factor: float = 0.18215   # diffusers convention for SD VAE latents
hf_cache_dir: str = "/workspace/hf_cache"

# Frame generator (§2.6)
d_blocks: int = 12
d_dim: int = 512
d_heads: int = 8
d_mlp_ratio: int = 4
n_vae_tokens: int = 64      # 8x8 patched from 4x16x16
d_vae_token: int = 16       # 4 channels * 2x2 patch

# Inference ODE
inference_heun_steps: int = 4   # for c_hat and e_hat rollout at eval/inference
```

---

## §3. Models — VAE + FrameGenerator

### 3.1 Frozen VAE (`VAEWrapper`)

Load via `diffusers.AutoencoderKL.from_pretrained(vae_model_id, cache_dir=hf_cache_dir)`.

- `requires_grad=False`, always `eval()`.
- **Encode:** render-target frame (frame `t+k`) `(B,3,256,256)` in **[-1,1]** → latent `(B,4,32,32)`.
  - Use `.latent_dist.mode()` (deterministic) for training targets per brief §4.4.
  - Multiply by `vae_scale_factor` when storing/using latents if your diffusers version expects it — **pick one convention and use it consistently in D and decode**. Document in code comment.
  - **Note:** the VAE uses its own [-1,1] pixel range; the *encoder* path uses the V-JEPA processor
    normalization. Keep the two normalizations separate.
- **Decode:** latent → RGB `[-1,1]` for visualization.

### 3.2 VAE latent patchify for D

Per `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §2.4 and §3.8 (resolution 256):

1. VAE latent `a_y`: `(B, 4, 32, 32)`.
2. Patchify with **2×2 spatial patches** → `(B, 256, 16)` where 16 = 4×2×2.
3. Linear proj 16 → 512 for transformer.
4. Add 2D sin-cos pos embed on 16×16 grid.

### 3.3 `FrameGenerator` (`D`)

- 12 DiT blocks, dim 512, 8 heads, MLP ratio 4 (~38M params).
- Self-attention on 256 VAE-latent tokens.
- Cross-attention: queries from VAE tokens; keys/values from `as_target(e_hat)` projected **1024→512**
  (`e_hat` carries the encoder-derived dim; token count per §14 #35).
- adaLN-Zero on flow time `τ_x`.
- Output: proj 512→16, unpatchify → `(B, 4, 32, 32)` velocity in patched space.

### 3.4 Smoke test

- Encode random frame → patchify → D forward with random `e_hat` → velocity shape `(B,256,16)`.
- Backprop L_frame updates **D only** — no grad to E, B, flows, VAE.

---

## §4. `L_frame` in `losses.py`

Stage 4 only:

```python
a_y = vae.encode(target_frame_tk)        # frame t+k; patched to (B, 256, 16)
eps_x ~ N(0, I)
tau_x ~ U(0,1)
z_x = interpolate(a_y, eps_x, tau_x)
u_x = velocity_target(a_y, eps_x)

# e_hat from latent stack — see §5 for training-time construction
u_x_hat = frame_generator(z_x, tau_x, as_target(e_hat))

L_frame = flow_matching_loss(u_x_hat, u_x)
```

**Total Stage 4 loss:** `L = L_frame` only (no L_c, L_e, SIGReg — latent stack frozen).

---

## §5. Inference rollout (`inference.py` or `models.py` helpers)

Training-time in Stage 4 uses **one-step** `e_hat` for speed (same as Phase 2 formula extended to fine flow):

```python
e_hat_train = z_e + (1 - tau_e) * u_e_hat   # with sampled tau_e, z_e from training batch
```

**Inference / eval** uses **multi-step Heun ODE** (4 steps, locked for v0):

### 5.1 Coarse rollout (`c_hat` at inference)

```
Start: z = eps_c ~ N(0,I) at tau=0
For each step in Heun integrator from 0→1:
    u = F_c(z, tau, c_t)
    z = heun_step(z, u, dt)
Output: c_hat_final = z at tau=1
```

### 5.2 Fine rollout (`e_hat` at inference)

```
Condition on c_hat_final (detached)
Start: z = eps_e ~ N(0,I)
Integrate F_e with Heun 0→1
Output: e_hat_final
```

### 5.3 Frame generation

```
z_x = integrate D with Heun 0→1, conditioned on e_hat_final
a_pred = final z_x unpatchified
image = vae.decode(a_pred)
```

Expose:

```python
def predict_next_frame(context_clip, models, cfg) -> Tensor:
    """Returns (B, 3, 256, 256) in [-1, 1] (rendered frame t+k)."""
```

---

## §6. `train.py` — Stage 4

### 6.1 Freeze latent world model

At `step >= 105_000`:

```python
# E is already frozen; freeze the trainable latent modules too:
for p in chain(B.parameters(), F_c.parameters(), F_e.parameters()):
    p.requires_grad = False
# B_EMA already no grad; VAE already frozen; E already frozen
# Only D.train(); D parameters require grad
```

**No EMA updates** during Stage 4 (bottleneck frozen).

### 6.2 Optimizer

New AdamW param group for D only, lr `2e-4`, same betas/weight decay.

**LR schedule:** 3k warmup, cosine decay over remaining 42k (Stage 4 local schedule — separate from latent cosine).

### 6.3 Stage 4 training step

```
1. Load context_clip, target frame t+k (latent stack frozen — need frame t+k for VAE target)
2. With torch.no_grad():
     e_t = E(context); c_t = B(e_t)            # frozen encoder + frozen bottleneck
     # Build e_hat for conditioning via one-step fine flow at random tau:
     sample eps_e, tau_e, z_e, u_e, u_e_hat = F_e(..., c_cond=c_plus or c_hat per Stage 4 policy)
     
     # Locked: use predicted coarse at inference-like path for D conditioning
     # Compute c_hat one-step from F_c (no grad)
     # Compute e_hat one-step from F_e with c_cond = as_target(c_hat)
     e_hat = one_step_e(...)

3. a_y = vae.encode_patchified(target_frame_tk)
4. Sample eps_x, tau_x; flow matching through D
5. L_frame only; backward; step
```

**Stage 4 `c_cond` for building `e_hat`:** use `as_target(c_hat)` one-step from F_c (predicted-coarse regime) — matches deployment. Do not use teacher `c_plus` in Stage 4.

### 6.4 Resume

```bash
python train.py --data ssv2_tiny --steps 150000 \
  --resume /workspace/checkpoints/checkpoint_step105000.pt
```

### 6.5 Checkpoint

Final v0 checkpoint: step **150000** at `/workspace/checkpoints/checkpoint_step150000.pt`.

---

## §7. `eval.py`

Standalone script — loads checkpoint, runs **all seven tests** from brief §9 on a fixed val batch:

| Test | Function | Pass criterion |
|---|---|---|
| Latent std | `latent_std_stats` | no hard collapse (>30% dead dims) |
| Effective rank | `effective_rank` | c_t > 60, e_t > 90 |
| Coarse baseline | `coarse_baselines` | ratios ≤ 0.70 / ≤ 0.50 |
| Shuffled-c | `shuffled_c_test` | ratio ≥ 2.0 at end of latent training |
| Teacher vs predicted | `teacher_vs_predicted_gap` | logged, gap not catastrophic |
| Gradient health | N/A at eval | skipped (train-time) |
| Decoder dependency | `decoder_dependency_test` | shuffled `e_hat` worsens L_frame or visual quality score |

### 7.1 `decoder_dependency_test`

1. Generate frames with real `e_hat`.
2. Shuffle `e_hat` across batch; regenerate.
3. Measure L_frame increase ratio or LPIPS/ MSE vs ground truth — **shuffled must be worse**.

If shuffled `e_hat` produces equally good frames, D ignores the world model — **fail**.

### 7.2 CLI

```bash
python eval.py --checkpoint /workspace/checkpoints/checkpoint_step150000.pt \
  --data ssv2_tiny --output-dir /workspace/checkpoints/eval_results/
```

Save: metrics JSON, optional PNG grid of predicted vs ground-truth frames.

---

## §8. Smoke and full run

```bash
# Stage 4 smoke (500 steps from 105k ckpt)
python train.py --data ssv2_tiny --steps 105500 --resume checkpoint_step105000.pt

# Full v0 (150k total)
python train.py --data ssv2_tiny --steps 150000 --resume checkpoint_step105000.pt

# Eval
python eval.py --checkpoint /workspace/checkpoints/checkpoint_step150000.pt --data ssv2_tiny
```

Stage 4 alone: 45k steps × ~2 sec/step ≈ 25 hours on tiny (F_e frozen but D + VAE encode add cost) — document actual measured throughput in README.

---

## §9. Acceptance gates — v0 complete

| Gate | Criterion |
|---|---|
| VAE | encode/decode round-trip on val frame visually recognizable |
| Stage 4 stable | 45k steps, no NaN, L_frame decreases |
| Freeze verified | no latent param grad in Stage 4 |
| Decoder dependency | shuffled e_hat test fails (ratio > 1.2 on L_frame or clear visual degradation) |
| Inference | `predict_next_frame` returns sane RGB tensor |
| eval.py | all seven tests execute without crash; hard thresholds met or flagged |
| End-to-end | context → predicted next frame PNG saved |
| Checkpoint 150k | loadable; resume produces same stage |
| README | full pipeline documented: clone → pull → train → eval on RunPod |

Report metrics, sample images path, eval JSON to human.

---

## §10. Final README structure

1. **Architecture one-paragraph** (from `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §1).
2. **Network volume layout** (`AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md` §3).
3. **One-time setup** (data migration, make_subset, HF cache).
4. **Phase summary** — what each phase added.
5. **Full training command** — 0→150k with resume points:
   - After Phase 1: resume at 30k
   - After Phase 2: resume at 105k
   - Phase 3: to 150k
6. **Eval command**.
7. **Runtime estimates** (A100 80GB, ssv2_tiny vs ssv2).
8. **What v0 does not include:** Stage 5 polish, multi-horizon t+2/t+4, unit test suite.

---

## §11. Files delivered / modified in Phase 3

```
config.py          # Stage 4 + VAE + inference
models.py          # + VAEWrapper, FrameGenerator, rollout helpers
losses.py          # L_frame (if separated)
train.py           # Stage 4 freeze + train
eval.py            # NEW — seven tests
inference.py       # optional; may live in models.py if small
README.md          # complete v0 documentation
requirements.txt   # ensure diffusers/transformers pinned
```

---

## §12. Stage 5 — explicitly out of scope

Brief §6 Stage 5 ("optional polish" — small LR tuning of D or brief F_e unfreeze) is **not part of v0**. Do not implement unless the human opens a new phase doc later.

If all Stages 1–4 pass acceptance gates, **stop**. v0 is shipped.
