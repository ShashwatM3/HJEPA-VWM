# SIGReg sweep — design, implementation, and λ rationale

The heavy design doc for [investigation_008](DESCRIPTION.md) (mirrors inv007's
[`SWEEP_PLAN_decoder_capacity.md`](../investigation_007/SWEEP_PLAN_decoder_capacity.md)).
Execution/runbook is in [`GUIDE.md`](GUIDE.md). Grounded in the code on
`phase1-v0.2-frozen-encoder` (`losses.py`, `train.py`, `diagnostics.py`, `config.py`)
and in live W&B (`smahalanobis-uc-davis/hjepa-vwm`).

---

## 1. Why SIGReg, and what it attacks

### 1.1 The open wound from investigation_007 Wave 1

Wave 1 ruled out reconstruction **weight** and **decoder size** as the lever on the
0.585 floor and produced the load-bearing finding (WAVE1_ANALYSIS §1.4): the floor is a
**utilization limit**, not a capacity one. `c_effective_rank` sat at **~13 / 256** in
*every* run, invariant to weight and decoder. The collapsed axis is **`d_c`** (per-slot
feature dim) — `c` leaves ~95% of its 256 dimensions unused. Wave 1's knobs don't touch
`d_c`, and **Wave 2's `n_c` knob doesn't either** (it adds *slots*, not *dims*). So the
one axis that is actually collapsed has never been attacked.

### 1.2 SIGReg is the principled tool for exactly this axis

`END_OF_WAVE_2.md` §2.4 already named the cure when surveying the literature:

> "Standard defense = an anti-collapse distribution regularizer (SIGReg /
> isotropic-Gaussian); **our per-dimension variance floor is a cousin of this** — but
> it guards *cross-sample* variance, not *temporal* informativeness."

The variance floor (`losses.variance_floor`) is a **one-sided hinge**:
`L_var = mean_j max(0, 1 − Std(c_j))`. It only forbids a dimension from going *constant*.
It says nothing about the *shape* of the distribution and nothing about *correlation*
between dims — so 256 dims can each clear the std≥1 bar while still living on a ~13-dim
correlated subspace. That is precisely the rank-13/256 symptom.

**SIGReg (Sketched Isotropic Gaussian Regularization, LeJEPA — Balestriero & LeCun,
2025)** replaces that hinge with a *two-sided distributional* objective: drive the latent
toward an **isotropic unit Gaussian `N(0, I_{d_c})`**, which is the provably
risk-optimal embedding law (minimizes worst-case downstream risk). Isotropy means *all*
eigenvalues of the `d_c` covariance are equal → effective rank is maximal by
construction. **This is a direct, mechanistic attack on the rank-13 ceiling that nothing
in inv006/007 could move.**

### 1.3 Honest scope — what SIGReg does NOT fix

`END_OF_WAVE_2.md` §2.3 argued the *deepest* disease is **temporal
under-informativeness** (`c` is distinct-per-video but nearly static in time, so the
copy baseline wins). SIGReg regularizes the *marginal/cross-sample* distribution of `c`;
it does **not** directly make `c` encode *change over time*. So inv008 is not a
substitute for the Tier-1 prediction pivot — it is the **missing utilization
experiment**, and it is decisive either way:

- If `c_effective_rank` rises toward 60+ **and** `coarse_vs_copy_ratio` falls → a richer,
  better-conditioned `c` was a real bottleneck → reconsider the pivot ordering.
- If `c_effective_rank` rises **but** `copy_ratio` stays >1 → the strongest possible
  evidence that the disease is temporal, not utilization → the Tier-1 prediction pivot
  becomes unimpeachable (we'd have driven rank up and prediction *still* fails).
- If `c_effective_rank` won't move even under hard SIGReg → the bottleneck `B` itself is
  the constraint (architectural), not the regularizer.

---

## 2. The implementation (code spec)

SIGReg is a new pure-tensor loss in `losses.py`, wired into `train.py` exactly where the
variance floor already is. It operates on the **same pooled distribution** that
`diagnostics.effective_rank` measures — `c` reshaped to `(N = B·N_c, d_c)` — so the loss
and the metric it targets are the same object.

### 2.1 The test (Cramér–Wold + BHEP/Epps–Pulley)

A distribution is `N(0, I)` **iff every 1-D projection is `N(0,1)`** (Cramér–Wold). So we
sketch many **random unit directions** in `d_c`-space, project, and penalize each
projection's deviation from a standard normal with the **BHEP / Epps–Pulley** normality
statistic (the differentiable, closed-form 1-D goodness-of-fit test LeJEPA uses).
Because directions are random and we test against `N(0,1)` (not a rescaled normal), the
loss enforces **unit variance + Gaussian shape + isotropy jointly** — i.e. it subsumes
the variance floor's job and adds the isotropy the floor lacks.

```python
# losses.py  (sketch — implementer to finalize + add a unit test)
import math

def sigreg_loss(
    abstract: Tensor,            # (B, N_c, D_c) online c_t  (NOT the EMA target)
    n_projections: int = 128,
    max_rows: int = 512,         # subsample pooled rows -> keeps the O(N^2) term cheap
    beta: float = 1.0,           # BHEP smoothing bandwidth
    eps: float = 1e-6,
) -> Tensor:
    """Push the pooled c_t distribution toward isotropic N(0, I) (LeJEPA SIGReg).

    Tests N(0, I_{d_c}) via the BHEP/Epps-Pulley 1-D normality statistic along random
    directions (Cramer-Wold). Random unit directions => isotropy + unit variance +
    Gaussian shape are enforced jointly, directly lifting the c_effective_rank ~13/256
    utilization ceiling that the one-sided variance floor cannot touch.
    """
    _require_torch()
    z = abstract.reshape(-1, abstract.shape[-1]).float()   # (N, d_c)
    n, d = z.shape
    if n < 2:
        return z.new_tensor(0.0)
    if n > max_rows:                                        # stochastic subsample of rows
        idx = torch.randperm(n, device=z.device)[:max_rows]
        z = z[idx]
        n = max_rows
    z = z - z.mean(dim=0, keepdim=True)                     # center: test vs N(0,1)
    v = torch.randn(d, n_projections, device=z.device, dtype=z.dtype)
    v = v / v.norm(dim=0, keepdim=True).clamp_min(eps)      # random unit directions
    y = z @ v                                               # (N, P) projected samples
    # --- BHEP statistic per projection vs N(0,1), closed form (>=0; 0 == perfect N(0,1)):
    #   T = mean_jk exp(-b^2 (y_j - y_k)^2 / 2)
    #       - 2/sqrt(1+b^2)   * mean_j exp(-b^2 y_j^2 / (2(1+b^2)))
    #       + 1/sqrt(1+2 b^2)
    b2 = beta * beta
    yt = y.t()                                              # (P, N)
    sq = yt * yt                                            # (P, N)
    pair = sq.unsqueeze(2) + sq.unsqueeze(1) - 2 * yt.unsqueeze(2) * yt.unsqueeze(1)
    term1 = torch.exp(-0.5 * b2 * pair).mean(dim=(1, 2))    # (P,)
    term2 = (2.0 / math.sqrt(1.0 + b2)) * torch.exp(
        -0.5 * b2 / (1.0 + b2) * sq
    ).mean(dim=1)                                           # (P,)
    term3 = 1.0 / math.sqrt(1.0 + 2.0 * b2)
    return (term1 - term2 + term3).mean()                  # average over projections
```

**Cost:** the pairwise term is `O(P · S²)`. With `P=128`, `S=512` → ~3.4e7 elems
(~0.13 GB fp32), freed each step — negligible on A100. Resample directions and the row
subset every step (stochastic sketching, as in LeJEPA). If throughput ever bites, chunk
over `P` or swap in the cheaper **sliced-Wasserstein-to-Gaussian** form (sort the
projected column, compare to `Φ⁻¹` of uniform quantiles; `O(S log S)`) — a valid
drop-in isotropy regularizer.

### 2.2 Wiring into `train.py` (mirror the variance-floor lines)

In `train_step` (currently `train.py:238,247`):

```python
var_loss    = variance_floor(abstract, cfg.train.var_floor_std_target)   # keep (safety)
sigreg_l    = sigreg_loss(abstract)                                       # NEW: always compute -> logs L_sigreg even on baseline
loss = flow_loss + cfg.train.lambda_var * var_loss
if cfg.train.lambda_sigreg > 0.0:                                         # add only when active -> lambda_sigreg=0 is byte-identical
    loss = loss + cfg.train.lambda_sigreg * sigreg_l
```

Follow the project's established pattern exactly (PROTOCOL "code hygiene", how
`lambda_cov`/`lambda_slot` are done): **compute always (log `L_sigreg`), add only when
`lambda_sigreg > 0`**, so `--lambda-sigreg 0` reproduces the baseline byte-for-byte.
Add `lambda_sigreg: float = 0.0` to `TrainConfig` (next to `lambda_var`,
`config.py:158`), a `--lambda-sigreg` CLI flag + override (mirror `--lambda-var`,
`train.py:605/731`), and `"L_sigreg"` to the metrics dict (`train.py:326`).

### 2.3 SIGReg vs the existing VICReg terms — no conflict

- **VICReg-V (variance floor, `lambda_var`)**: kept at **0.5** (its Wave-1 value) as a
  pure **safety net**. It is a one-sided hinge `max(0, 1−std)` that is *inactive* once
  `std ≥ 1`; SIGReg drives `std → 1`, so at the operating point the two never oppose
  (below 1 they push the *same* direction). Keeping it fixed means the **only changed
  variable is SIGReg** and the `λ_sigreg=0` control is *exactly* eager-plant-22.
- **VICReg-C (`lambda_cov`) and slot (`lambda_slot`)**: stay **off** (default 0.0) —
  these are the redundant terms SIGReg is meant to subsume.
- A **pure-replace arm** (`--lambda-var 0`, SIGReg only) is the natural Wave-2 follow-up
  if Wave 1 shows SIGReg's variance enforcement is sufficient on its own.

---

## 3. The sweep — values and rationale

**One factor at a time: `λ_sigreg`.** Everything else is pinned at the best-known-good
Wave-1 config (§4). 4× A100 → 4 runs in one wave.

| GPU | λ_sigreg | Role |
|---|---|---|
| 0 | **0.3**  | mild pressure — does a *light* isotropy push already move rank? |
| 1 | **1.0**  | comparable to the prediction loss scale (`L_flow` plateaus ~0.4) |
| 2 | **3.0**  | strong — expect clear rank lift; watch `L_flow` for task damage |
| 3 | **10.0** | saturation — force isotropy hard; the "can rank move *at all*" bookend |

**Why these values.** SIGReg's loss is `O(0.1–1)` (a bounded BHEP statistic) and `L_flow`
plateaus ~0.4, so the informative band for `λ_sigreg` is roughly `0.1–10`; below that the
term is inert, above it the latent is dragged to pure noise and `L_flow` blows up. The
sweep is **geometric over that band** (~×3 steps) to bracket the optimum without assuming
where it is — the same "cast a wide net when the scale is unknown" discipline inv007 used
for `λ_recon`. **Calibrate before trusting the values:** the GUIDE prints the step-0
loss components; if `λ_sigreg · L_sigreg` is ≪ `L_flow` even at λ=10 (or ≫ at λ=0.3),
re-center the geometric ladder and relaunch.

**The control is free:** `λ_sigreg=0` = eager-plant-22 (already on W&B, run `591mt31k`).
No GPU spent re-running it.

---

## 4. Fixed background config (held constant across all 4 runs)

Pinned at the **best-known-good Wave-1 setting = eager-plant-22**, so `λ_sigreg` is the
sole variable and the baseline is a real prior run:

| Knob | Value | Source / why |
|---|---|---|
| decoder | **512×4** (`--decoder-dim 512 --decoder-blocks 4`) | **best decoder of all Wave-1 runs on every metric** (§5); gives a richer `c` a decoder capable of expressing it, so the decoder can't become the confound |
| `λ_recon` | **0.05** (`--recon-warmup-steps 2000`) | recon anchor ON (the decoder must be live to matter); known **inert on rank** in Wave 1 → a clean, stable background, so any rank rise is attributable to SIGReg |
| `λ_recon_pred` | 0 | isolate (option-3 was inert at this floor) |
| `λ_var` | 0.5 | non-interfering safety net; makes the control = eager-plant-22 exactly (§2.3) |
| `n_c` | 32 | **not** the axis under test (Wave 2 owns `n_c`); SIGReg attacks `d_c` |
| `horizon_k` | 12 | matches Wave 1 |
| `lr_coarse_flow` | 1e-4 | matches Wave 1 |
| steps | 15000, cut at the `c_effective_rank` plateau | as Wave 1 (~8–9k) |

---

## 5. The decoder decision (live W&B, 2026-06-27)

The question: *use the original 256×2, or the "best decoder" from Wave 1?* The metric
that isolates the decoder's own job is **`L_recon_present`** (the decoder's direct
reconstruction quality). Live final-step (8500) pull of the two decoder-axis runs:

| Run (id) | decoder | L_recon_present | c_effective_rank | copy_ratio | L_flow |
|---|---|---|---|---|---|
| **eager-plant-22** (591mt31k) | **512×4** | **0.5847** | **12.72** | **1.60** | **0.396** |
| gallant-dew-22 (708jrel8) | 512×2 | 0.5895 | 12.51 | 1.74 | 0.435 |
| *(baseline easy-blaze-19)* | 256×2 | ~0.60 | ~13 | — | — |

**Decision: use 512×4 (eager-plant-22).** It is the best of *all* Wave-1 runs on the
decoder-isolating metric (and, in fact, on every secondary metric — rank, copy-ratio,
flow). Depth (512×4) beat width (512×2). Two reasons it's the right background for inv008:

1. It satisfies the stated criterion exactly — "best decoder by the decoder-focused
   metric," measured live, not assumed.
2. It **complements SIGReg**: if SIGReg succeeds in enriching `c` (higher rank), a thin
   decoder would become the new bottleneck and *confound* the reconstruction readout. A
   capable decoder ensures the recon floor reflects the *latent*, not the decoder.

**Honest caveat:** the absolute decoder gain is small (~0.005, "noise around a fixed
wall" — WAVE1_ANALYSIS §1.2), and VRAM is a non-issue on A100 (inv007 GUIDE §6), so the
cost of 512×4 over 256×2 is pure compute. If you'd rather keep the cheaper/original
256×2 for a leaner sweep, the SIGReg conclusion would very likely be unchanged — but
512×4 is the defensible OFAT choice (hold every non-swept knob at best-known-good).

---

## 6. Interpretation matrix (what each outcome means)

Read each run against the **decision triad** `c_effective_rank ∧ coarse_vs_copy_ratio ∧
L_recon_present`, plus health (`c_cross_video_cosine`, `L_flow`, grad).

| `c_effective_rank` | `copy_ratio` | Verdict |
|---|---|---|
| rises → 60+ | falls → <1 | **SIGReg wins both** — utilization *was* a real lever for prediction. Richest possible result; build on this config. |
| rises → 60+ | stays >1 | rank fixed, **prediction still fails** → disease is temporal, not utilization → the Tier-1 prediction pivot is now unimpeachable. (Most likely, ~55%.) |
| rises a little (20–30) | unchanged | partial — `B` resists full isotropy; note ceiling, pivot. |
| flat ~13 even at λ=10 | unchanged | the bottleneck `B` *architecture* binds rank, not the regularizer → architectural follow-up (revisit `d_c`/`B`). |
| any | — **but** `L_flow` blew up / `c_cross_video_cosine` climbed | λ too high (latent → noise) or var-floor net failed → drop λ / add a pure-replace safety check. |

**Decision metric, in priority order:** (1) does `c_effective_rank` break the 13/256
ceiling? (2) if so, does `coarse_vs_copy_ratio` fall toward <1? `L_recon_present` is a
corroborating diagnostic, no longer the objective (END_OF_WAVE_2 §2.6).

---

### Sources
- **LeJEPA: Provable and Scalable Self-Supervised Learning** (Balestriero & LeCun, 2025) —
  SIGReg / isotropic-Gaussian objective; the canonical reference for this implementation.
- Our own: [`investigation_007/WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md`](../investigation_007/WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md)
  §1.4 (rank-13 utilization limit on `d_c`), [`investigation_007/END_OF_WAVE_2.md`](../investigation_007/END_OF_WAVE_2.md)
  §2.3–2.4 (temporal-under-informativeness reframe + SIGReg named as the principled cousin).
