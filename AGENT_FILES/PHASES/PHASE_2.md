# PHASE_2.md — Detailed Hierarchy (Stages 2 + 3) — v0.2 (frozen encoder)

> **⚠️ v0.2 update.** Inherits Phase 1's **frozen V-JEPA 2 ViT-L/16 encoder** (`D_e=1024`), trainable
> bottleneck, **EMA bottleneck** (`B_EMA`), and **variance floor** (no SIGReg). All `e_t`-dependent
> dims below change from `384` to **`1024`**, and the context token count is **`N_ctx=1024`**.
>
> **⚠️ OPEN DESIGN DECISION (confirm before coding Phase 2):** the supervisor's update specified the
> **coarse** path only. With a frozen *video* encoder, the recommended default for the fine flow is a
> **clip-level detailed target** `e_plus = E(x_{≤t+k})` of shape `(1024, 1024)` (same geometry as
> `e_t`), with the frame generator (Phase 3) rendering the **last** future frame `t+k`. The
> alternative is a frame-level detailed target. **Exact `F_e` token counts lock once the human
> confirms** (see [`UNDERSTANDING.md`](../KNOWLEDGE/UNDERSTANDING.md) §14 #35). Until then, treat the
> `64`-token target shapes below as placeholders for the chosen geometry.
>
> **Agent instruction:** Execute this document top-to-bottom. Extends the Phase 1 codebase. When finished, the full **latent world model** trains through Stages 2 and 3: fine flow `F_e` with teacher-forced then predicted-coarse conditioning, shuffled-c bypass test passes, hierarchy verified.
>
> **Prerequisites:** Phase 1 acceptance gates passed. Checkpoint at step 30k loadable. Read [`AGENT_FILES/PHASES/PHASE_1.md`](PHASE_1.md) for existing modules — **do not rewrite** `FrozenEncoder`, `Bottleneck`, `B_EMA`, `CoarseFlow`, or data pipeline except where this doc explicitly says extend.

---

## Workflow — execute in this exact order

| Step | Section | Deliverable | Verify before continuing |
|---|---|---|---|
| 1 | §2 | Extend `config.py` for Stages 2–3 | New step boundaries import cleanly |
| 2 | §3 | Add `FineFlow` to `models.py` | smoke test forward + backward |
| 3 | §4 | Extend `losses.py` for L_e + Stage 3 mix | synthetic tensor test |
| 4 | §5 | `c_hat` one-step construction helper | detached correctly |
| 5 | §6 | Extend `diagnostics.py` | shuffled-c, zero-c, teacher gap |
| 6 | §7 | Extend `train.py` — Stages 2–3, ramp, transitions | resume from Phase 1 ckpt |
| 7 | §8 | 2k-step smoke on `ssv2_tiny` Stage 2 | L_e finite, no NaN |
| 8 | §9 | Train to step 105k (or staged smoke sections) | metrics logged |
| 9 | §10 | Acceptance gates | all thresholds met |
| 10 | §11 | Update README Phase 2 section | |

**Do not start Phase 3 until §10 passes.**

---

## §2. Config extensions

Add to `TrainConfig`:

```python
stage2_steps: int = 25_000   # steps 30k → 55k
stage3_steps: int = 50_000   # steps 55k → 105k
max_steps: int = 105_000     # Phase 2 trains through end of Stage 3

stage2_start: int = 30_000
stage3_start: int = 55_000
stage4_start: int = 105_000  # Phase 3 only; not trained here

c_cond_ramp_steps: int = 5_000   # linear α: 0→1 over first 5k of Stage 3

lr_fine_flow: float = 4e-4

# Shuffled-c thresholds (§2.6)
shuffled_c_ratio_stage2_min: float = 1.5
shuffled_c_ratio_stage3_min: float = 2.0
```

**Stage detection from global `step`:**

| Stage | Step range | Active modules in loss |
|---|---|---|
| 1 | `[0, 30_000)` | B, F_c + variance floor (encoder frozen; unchanged from Phase 1) |
| 2 | `[30_000, 55_000)` | B, F_c, F_e (E frozen) — teacher-forced `c_cond = c_plus` |
| 3 | `[55_000, 105_000)` | B, F_c, F_e (E frozen) — ramp then `c_cond = stopgrad(c_hat)` |

Optimizer param group: add `F_e` at lr `4e-4` from step 30k. Optionally reset LR warmup is **not** applied — continue cosine from Phase 1 schedule (warmup already completed during Stage 1).

---

## §3. `FineFlow` in `models.py`

Per `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` §3.7.

### Structure

- 8 transformer blocks, dim 384, 8 heads, MLP ratio 4 (~14M params). `F_e`'s working dim stays 384;
  the **memory** width follows the encoder (`D_e=1024`).
- Per block: **self-attention** on `z_e` (target tokens) → **cross-attention** to memory → **MLP**,
  all adaLN-Zero modulated by `τ_e`. (Target token count = `1024` for the clip-level default, or `64`
  for a frame-level target — pending the §14 #35 decision.)

### Cross-attention memory (v0.2 dims)

1. `proj_e = Linear(1024 → 384)(e_t)` — project frozen-encoder memory to `F_e`'s width `(B, 1024, 384)`.
2. `proj_c_cond = Linear(256 → 384)(c_cond)` — shape `(B, 32, 384)`.
3. `memory = concat([proj_e, proj_c_cond], dim=1)` — shape `(B, 1024 + 32, 384)`.
4. Cross-attention: queries from `z_e`, keys/values from `memory`.

### Condition dropout (10%)

Single Bernoulli per example: replace **entire memory** with learned null memory parameter (truncated to actual sequence length). Both `e_t` and `proj_c_cond` dropped together.

### Forward signature

```python
def forward(self, z_e, tau_e, e_t, c_cond, *, condition_drop=None) -> Tensor:
    """
    Args:
        z_e: (B, 64, 384) noised target latent
        tau_e: (B,) flow time
        e_t: (B, N_ctx_post, 384) context detailed latent
        c_cond: (B, 32, 256) coarse condition (c_plus or c_hat)
    Returns:
        u_e_hat: (B, 64, 384) predicted velocity
    """
```

### Smoke test

- Stage 2 mode: `c_cond = target_abstract`.
- Backward from L_e reaches E, B, F_e, F_c — **not** through detached targets.
- Verify **no grad** from L_e into F_c via `c_hat` path in Stage 3 (see §5).

---

## §4. Loss extensions in `losses.py`

### 4.1 Fine flow loss

Same primitives as coarse:

```python
z_e = interpolate(e_plus, eps_e, tau_e)   # e_plus detached
u_e = velocity_target(e_plus, eps_e)
u_e_hat = fine_flow(z_e, tau_e, e_t, c_cond)
L_e = flow_matching_loss(u_e_hat, u_e)
```

**Independent `τ`:** sample `tau_c ~ U(0,1)` and `tau_e ~ U(0,1)` separately per example (locked decision).

### 4.2 Stage 2 total loss

```python
L = L_c + lambda_fine * L_e + lambda_var * variance_floor(c_t)   # no SIGReg (v0.2)
```

`lambda_fine = 1.0`. `c_cond = target_abstract` (already detached from EMA branch).

### 4.3 Stage 3 total loss

Same formula. `c_cond` from ramp (§7.3). `L_c` still active — both flows train jointly.

**Critical:** `c_hat` used in `c_cond` must be `.detach()` / `as_target()` so **L_e does not backprop into F_c through c_hat** (brief §5, non-negotiable).

---

## §5. `c_hat` construction

One-step Euler (locked for v0 training):

```python
def predict_abstract_one_step(z_c, tau_c, u_c_hat):
    """One-step rectified-flow prediction of c_plus from current noised state.

    c_hat = z_c + (1 - tau_c) * u_c_hat

    Used as F_e conditioning in Stage 3. Must be detached before feeding F_e.
    """
    # broadcast tau_c to (B, 1, 1)
    c_hat = z_c + (1 - tau_c) * u_c_hat
    return c_hat
```

In Stage 3 training step:
1. Compute `u_c_hat = F_c(z_c, tau_c, c_t)` (same forward as L_c).
2. `c_hat = predict_abstract_one_step(z_c, tau_c, u_c_hat)`.
3. `c_cond = as_target(c_hat)` when ramp α = 1; blend during ramp (§7.3).

**Do not** backprop Stage 3 L_e into `u_c_hat` / F_c via `c_hat`.

---

## §6. Diagnostics extensions

Add to `diagnostics.py`:

### 6.1 `shuffled_c_test(fine_flow, batch, c_hat_real, ...)`

For each batch of 64:
1. Compute L_e with real `c_hat` (or `c_plus` in Stage 2).
2. Permute `c_hat` across batch dimension (each clip gets another clip's coarse prediction).
3. Compute L_e with shuffled condition.
4. Return `ratio = L_e_shuffled / L_e_real` (mean over batch).

**Pass:** ratio ≥ 1.5 by end of Stage 2 (step 55k); ≥ 2.0 by end of Stage 3 (step 105k).

If ratio ≈ 1.0, F_e ignores abstract conditioning — **architecture failure**.

### 6.2 `zero_c_ablation`

Same as shuffled but replace `c_cond` with learned null embedding. Loss should increase similarly to shuffled.

### 6.3 `teacher_vs_predicted_gap`

Compare L_e with `c_cond = c_plus` vs `c_cond = stopgrad(c_hat)` on same batch. Gap should shrink during Stage 3 but not collapse to zero instantly.

Log all three every `diag_every` on fixed val batch.

---

## §7. `train.py` extensions

### 7.1 Resume from Phase 1

```bash
python train.py --data ssv2_tiny --steps 105000 --resume /workspace/checkpoints/phase1_step30000.pt
```

Load model, optimizer, EMA, **global step**. Continue LR cosine and EMA m schedule from loaded step (do not restart warmup).

### 7.2 Stage 2 step (teacher-forced)

```
# Steps 30k ≤ step < 55k
c_cond = c_plus   # = as_target(B_EMA(E(target_clip))), already detached

# Same coarse path as Phase 1 for L_c (L_flow)
# Fine path (e_plus = frozen E on target clip):
z_e, u_e, u_e_hat from eps_e, tau_e, e_plus, F_e(..., c_cond)
L = L_flow + L_e + lambda_var * variance_floor(c_t)   # no SIGReg
```

### 7.3 Stage 3 step (predicted-coarse with ramp)

```
# Steps 55k ≤ step < 105k
# Coarse forward (for both L_c and c_hat):
u_c_hat = F_c(z_c, tau_c, c_t)
c_hat = predict_abstract_one_step(z_c, tau_c, u_c_hat)

# Ramp:
alpha = min(1.0, (step - stage3_start) / c_cond_ramp_steps)
c_teacher = c_plus
c_pred = as_target(c_hat)
c_cond = (1 - alpha) * c_teacher + alpha * c_pred   # broadcast over tokens

# Fine loss uses c_cond
L = L_flow + L_e + lambda_var * variance_floor(c_t)   # no SIGReg
```

After ramp (alpha = 1): `c_cond = as_target(c_hat)` only.

### 7.4 EMA

Continue through Stages 2–3 on the **bottleneck only** (`B → B_EMA`). The encoder is frozen. Same
schedule: m ramps 0.996 → 0.9999 over 105k steps.

### 7.5 Checkpointing

Save every 5000 steps:
- `phase2_step{step}.pt` or unified naming `checkpoint_step{step}.pt`
- Must include `global_step` for stage detection

At step 105k: final Phase 2 checkpoint required for Phase 3.

### 7.6 CLI

```bash
# Continue from Phase 1 end through Stage 3
python train.py --data ssv2_tiny --steps 105000 --resume /workspace/checkpoints/phase1_step30000.pt

# Full pipeline latent-only on full data
python train.py --data ssv2 --steps 105000 --resume ...
```

---

## §8. Smoke verification

Before full 105k run:

1. `--steps 32000` from Phase 1 ckpt — 2k steps in Stage 2, L_e logged.
2. `--steps 57000` — 2k steps in Stage 3 with ramp active.
3. Confirm shuffled-c ratio > 1.0 trending upward (not required to hit 1.5 in 2k steps).

---

## §9. Expected runtime notes

From Phase 1 timing (~3 steps/sec A100 80GB bf16):
- Stage 2+3 add F_e forward/backward — estimate ~2–2.5 steps/sec.
- 75k new steps (30k→105k) ≈ 8–10 hours on `ssv2_tiny`.
- Full `ssv2` at 105k total steps is the production latent-training run.

---

## §10. Acceptance gates — Phase 2 complete when ALL pass

| Gate | Criterion |
|---|---|
| Resume | Loads Phase 1 ckpt at step 30k without shape errors |
| Stage 2 | L_e decreases; training stable 30k→55k |
| Teacher-forced | F_e val L_e with real `c_plus` beats nonsense condition |
| Shuffled-c @ 55k | ratio ≥ **1.5** on val |
| Stage 3 ramp | alpha reaches 1.0 at step 60k; no NaN |
| Shuffled-c @ 105k | ratio ≥ **2.0** on val |
| Zero-c ablation | zero/null condition hurts L_e vs real |
| Teacher vs predicted | gap narrows but predicted-coarse L_e not orders-of-magnitude worse than teacher at 105k |
| F_c baselines | still meet Phase 1 thresholds (don't regress) |
| Latent health | rank / std thresholds still met |
| L_e ↛ F_c | autograd check: grad from L_e w.r.t. F_c params is zero when using detached c_hat |
| Checkpoint @ 105k | loadable, correct global_step |

Report all metrics + checkpoint path to human.

---

## §11. README update

Add Phase 2 section:
- What F_e does (plain language).
- Teacher-forced vs predicted-coarse explained.
- What shuffled-c test proves.
- Resume command from Phase 1 checkpoint.
- Step schedule table (0–30k / 30–55k / 55–105k).

---

## §12. Files modified in Phase 2

```
config.py          # extended
models.py          # + FineFlow
losses.py          # + L_e helpers (if not inline in train)
diagnostics.py     # + shuffled-c, zero-c, teacher gap
train.py           # Stages 2–3, ramp, resume
README.md          # Phase 2 section
```

**Not in scope:** `FrameGenerator`, VAE, `L_frame`, Stage 4, `eval.py` (Phase 3).

---

## §13. Handoff to Phase 3

Deliver checkpoint at step **105000** with all latent modules (E, B, F_c, F_e, EMA). Phase 3 freezes these and trains `D` only. Continue with [`AGENT_FILES/PHASES/PHASE_3.md`](PHASE_3.md).
