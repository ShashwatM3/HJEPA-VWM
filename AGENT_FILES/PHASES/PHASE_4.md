# PHASE_4.md — Multi-Horizon Prediction (NEW; deferred)

> **Agent instruction:** Do **not** execute this phase until Phases 1–3 are healthy on the frozen
> encoder and the human explicitly says "execute Phase 4." This document specifies the supervisor's
> multi-horizon generalization of the coarse predictor. It extends the Phase 1 codebase; it does not
> rewrite the encoder, bottleneck, fine flow, or frame generator.
>
> **Prerequisites:** Phase 1 acceptance gates passed with the frozen encoder + variance floor.
> Read [`AGENT_FILES/KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md`](../KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md)
> §6 (multi-horizon) and [`BRIEF_V0_2.md`](../KNOWLEDGE/BRIEF_V0_2.md) §4.1.

---

## §1. What Phase 4 adds

Phases 1–3 predict a **single** horizon. Phase 4 generalizes the **coarse** predictor `F_c` to
predict the future abstract latent at **several horizons** `k`, using one shared predictor told which
horizon it is aiming at via a learned **horizon embedding** `h_k`.

- **Horizon set:** `k ∈ {4, 8, 16, 32}` frames.
- **Sampling probabilities:** `{0.30, 0.30, 0.25, 0.15}` (near horizons weighted slightly higher).
- **Per sample:** draw one `k`, build the target `c⁺_{t+k} = B_EMA(E(x_{≤t+k}))`, and condition the
  predictor on `h_k`.
- **Shared predictor:** `v_θ(z_τ, τ, c_t, h_k)` — same weights across horizons; only `h_k` changes.

Scope: Phase 4 extends the **coarse/abstract** path (and its metrics). Whether the fine flow `F_e`
and frame generator also become multi-horizon is a **separate later decision** and is out of scope
here unless the human extends it.

---

## §2. Config extensions

Add to `TrainConfig` / `ModelConfig`:

```python
# Multi-horizon (Phase 4)
horizons: tuple = (4, 8, 16, 32)
horizon_probs: tuple = (0.30, 0.30, 0.25, 0.15)
horizon_embed_dim: int = 256        # matches D_c; added to / concatenated with c-conditioning
phase4_steps: int = 0               # set when the phase is scheduled
```

Data pipeline must be able to sample a future clip ending at `t + k` for each chosen `k` (the video
must be long enough; skip/clip-pad short videos deterministically and log counts).

---

## §3. Horizon embedding `h_k`

- A small learnable embedding table with one vector per horizon value (4 entries for the set above),
  each of dim `horizon_embed_dim`.
- Injected into the **coarse predictor** alongside the `c_t` conditioning. Two acceptable wirings
  (pick one, document it):
  1. **Additive to the time vector:** add `h_k` to the adaLN-Zero time embedding input.
  2. **Extra conditioning token:** concatenate `h_k` as an extra token in `F_c`'s conditioning
     sequence.
- Initialize small; let training specialize per-horizon behavior.

---

## §4. Training step (multi-horizon coarse)

```
1. context = sample 8-frame clip ending at t
2. k ~ Categorical(horizons, horizon_probs)               # per example
3. future = sample clip ending at t+k
4. with no_grad: e_t = E(context); e_tk = E(future)        # frozen encoder
5. c_t = B(e_t)                                             # trainable
6. c_plus = as_target(B_EMA(e_tk))                          # stop-grad EMA target
7. z_0 ~ N(0,I); tau ~ U(0,1)
8. z_tau = (1-tau)*z_0 + tau*c_plus
9. v_target = c_plus - z_0
10. v_hat = F_c(z_tau, tau, c_t, h_k)
11. L_flow = mean_square(v_hat - v_target)
12. L = L_flow + 0.1 * variance_floor(c_t)
13. backward; clip; step; EMA update on B only
```

Notes:
- The variance floor is still on `c_t` only (one `c_t` per sample, independent of `k`).
- EMA on the bottleneck only (encoder frozen), same schedule as Phases 1–3.

---

## §5. Diagnostics (per horizon)

Extend `diagnostics.py`:

- **Per-horizon flow loss:** log `L_flow` separately for each `k` (so far horizons can be seen to
  be harder).
- **Per-horizon coarse baselines:** copy / batch-mean baselines per `k`; `F_c` should beat both for
  each horizon after warmup.
- **`c_t` health** (variance / cross-video cosine / effective rank): unchanged (on `c_t`).
- Optional: **target drift vs `k`** — how fast `c⁺_{t+k}` moves away from `c_t` as `k` grows (a
  sanity check that far horizons are genuinely harder).

---

## §6. Acceptance gates

| Gate | Criterion |
|---|---|
| All horizons train | `L_flow` decreases for every `k`; no NaN |
| Per-horizon baselines | `F_c` beats copy & batch-mean for each `k` after warmup |
| Near vs far | far horizons (16, 32) have higher loss than near (4, 8) but still beat baselines |
| `c_t` health | variance / cosine / effective-rank thresholds still met |
| Shared predictor | one `F_c` (with `h_k`) handles all horizons — no per-horizon weights |

---

## §7. Files modified in Phase 4

```
config.py          # horizons, probs, horizon_embed_dim
data.py            # sample future clip ending at t+k per chosen k
models.py          # horizon embedding table + F_c wiring for h_k
diagnostics.py     # per-horizon losses / baselines
train.py           # horizon sampling in the coarse step
README.md          # Phase 4 section
```

**Not in scope:** multi-horizon fine flow / frame generator (separate decision); any encoder change.
