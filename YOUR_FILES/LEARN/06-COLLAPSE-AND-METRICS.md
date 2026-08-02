# 06 — Collapse and the Measurement System: How Self-Supervised Learning Fails Silently, and How We Catch It

> **What you'll understand after this file:** the taxonomy of collapse, the
> variance floor (our one anti-collapse loss) line by line, all seven
> diagnostic metric families with healthy/sick values, the acceptance
> baselines that define Phase 1 success, and why SIGReg was removed.

---

## 1. The central pathology: collapse

In supervised learning, a degenerate model gets a bad loss — labels don't
cooperate with laziness. In self-supervised latent prediction, the model
*makes its own targets*, so degenerate solutions can achieve **excellent
loss**. The canonical one:

> Map every input to the same vector. Then "predict the future latent" is
> trivially solved — the future latent is always that vector. Loss ≈ 0.
> Information content of the representation: zero.

That's **full collapse**. The insidious variants matter more in practice:

| Mode | What happens | What still looks fine | What catches it |
|---|---|---|---|
| **Full collapse** | `c_t` constant for all inputs | loss (it's great!) | per-dim std → 0 |
| **Dimensional collapse** | `c_t` varies, but only inside a low-dim subspace of the 256 dims | loss, per-dim std (partially) | effective rank low, dead-dim fraction high |
| **Directional collapse** | vectors vary in length but point the same way | per-dim std | cross-video cosine → 1 |
| **Conditioning bypass** | latents healthy, but predictor *ignores* `c_t` | loss, all latent stats | model-vs-batch-mean ratio ≈ 1 |

The design lesson: **loss is not a health metric in self-supervised
learning.** You need instruments that measure what the loss cannot see.
That's the entire purpose of `diagnostics.py`.

A v0.2 structural note: with the encoder frozen, `e_t` *cannot* collapse —
its statistics are fixed by Meta's pretraining. Only `c_t` (the trainable
bottleneck's output) is at risk, so only `c_t` is monitored. The
attack surface shrank with the architecture.

## 2. The one preventive loss: the variance floor

```85:107:losses.py
def variance_floor(abstract: Tensor, std_target: float = 1.0) -> Tensor:
    """Per-dimension variance floor on the abstract latent `c_t` (replaces SIGReg).

    `L_var = (1/d) Σ_j max(0, std_target - Std(c_j))`,  d = N_c * D_c.
    """
    _require_torch()
    flat = abstract.reshape(abstract.shape[0], -1).float()
    if flat.shape[0] < 2:
        return flat.new_tensor(0.0)
    std = flat.std(dim=0, unbiased=False)
    return torch.clamp(std_target - std, min=0.0).mean()
```

Mechanics, slowly:

1. Flatten each sample's `c_t` to 8,192 numbers; over a batch you get a
   `(B, 8192)` matrix.
2. For each of the 8,192 dimensions, compute the **std across the batch**
   — "how much does this coordinate differ between different videos?"
3. **Hinge** each dimension at 1.0: dimensions with std ≥ 1 contribute
   zero; dimensions below contribute `1 − std`.
4. Average. Weight in the total loss: `λ_var` — **`0.10` in `config.py`
   defaults; `0.50` in validated operating runs** (`cerulean-snow-13`,
   investigation 003). At 0.10, `c_std_mean` can sit ~0.45 and cosine
   stays high (~0.7–0.8); at 0.50 the floor actively fights collapse
   pressure and rank rises to ~13+.

Three properties to internalize:

- **It's a floor, not a force.** Above std=1, gradient is exactly zero (the
  hinge). It never rewards extra variance — it only forbids deadness. This
  is the difference from "maximize variance," which would invite noise.
- **It's per-dimension across the batch** — the same quantity
  `variance_stats` monitors. Loss and diagnostic look at the same object,
  so the metric can verify the loss is doing its job.
- **It's deliberately weak at the default (λ=0.10).** At that weight its
  only job is to fence off the constant solution. *Semantics come from
  L_flow.* Investigation 003 showed 0.10 was too weak for full SSv2 —
  operating weight **0.50** is required for healthy variance and rank.
  If you find yourself cranking λ_var beyond 0.5 to fix representation
  quality, you're treating the symptom — variance can be high and
  meaningless (noise has great variance).

Why std-across-batch and not within-sample? Collapse means "all *inputs*
map to the same output" — a between-inputs property. A constant-per-video
but video-specific `c_t` would be perfectly healthy.

### Why SIGReg/VICReg was removed (the v0.2 story)

v0.1 carried SIGReg — a covariance regularizer that also pushed
*off-diagonal* covariance toward zero (decorrelating dimensions, directly
boosting effective rank). The supervisor's v0.2 directive: **"only a
variance floor on c_t, no SIGReg, no full VICReg, no covariance loss
initially."** The reasoning chain:

1. The frozen encoder already removed the worst collapse risks; heavy
   anti-collapse machinery was sized for a threat that shrank.
2. Every extra loss term is an extra hyperparameter, an extra interaction
   to debug, an extra way to not know which component caused an effect.
   Phase 1 is a *minimal viability experiment* — the cleanest version of
   the question gets asked first.
3. A covariance loss can "fix" the effective-rank *metric* without fixing
   representation *quality* — Goodhart's law applied to diagnostics. If
   rank rises only because a loss term pushes the number, the metric stops
   being evidence.

The word *initially* matters: optional **`covariance_floor`** (VICReg-C,
`lambda_cov`) and **`slot_diversity_loss`** (`lambda_slot`) exist in code
and are always logged, but default off. Investigation **004** (VICReg-C)
is paused — not needed yet at `lambda_var=0.5`. SIGReg remains the
documented escalation if rank stays stuck after task/regularizer tuning
(file 03 §7). Removed ≠ refuted.

## 3. The instrument panel: every diagnostic, with healthy/sick readings

Diagnostics run every `diag_every = 500` steps on a **fixed validation
batch** — fixed so the numbers are comparable across time (changes mean
the *model* changed, not the data). All log to W&B with these names.

### 3a. `variance_stats` → `c_std_mean`, `c_std_median`, `c_dead_dim_frac`

Per-dim std across the batch (same object as the variance floor), plus the
fraction of dimensions with std < 10% of the median std ("dead").

- **Healthy:** std mean/median near ~1.0 (the floor target keeps them
  there); dead fraction ≈ 0.
- **Sick:** std drifting toward 0 (full collapse) or dead fraction
  climbing (dimensional collapse in progress, dimension by dimension).

### 3b. `cross_video_cosine` → `c_cross_video_cosine`

Flatten each video's `c_t`, normalize to unit length, take all pairwise
cosines between *different* videos in the batch, average the off-diagonal.

- **Healthy:** well below ~0.5 — different videos get visibly different
  latents.
- **Sick:** drifting to 1.0 — all latents point the same direction. This
  is the metric that catches **directional collapse**, which variance
  stats can miss completely (vectors can vary in magnitude along a single
  shared direction and still have healthy per-dim std).

### 3c. `effective_rank` → `c_effective_rank`

The most information-dense metric. Procedure: pool all tokens, center,
form the 256×256 covariance of `c_t`'s feature dims, take eigenvalues,
normalize them into a probability distribution, compute its entropy, and
report `exp(entropy)`.

Interpretation: **"how many dimensions is the latent *really* using?"** If
variance is spread evenly over k dims and zero elsewhere, effective rank ≈
k. It's the smooth version of counting nonzero eigenvalues.

- **Spec healthy:** > 60 (of 256).
- **Run 1 reality:** ~5. Not collapsed but using ~2% of capacity.
- **Current best (`cerulean-snow-13`):** ~13–14. Alive, beats copy,
  distinguishable — but still far below the >60 soft target.
- **NaN-robustness:** after Run 1, this function returns NaN instead of
  crashing when the covariance is non-finite. Design rule learned the hard
  way: **diagnostics must never be able to kill the training loop** —
  a NaN model is a finding to log, not an excuse to crash (the original
  `eigvalsh` crash destroyed the run's final state; file 10).

### 3d. `coarse_baselines` → the acceptance gates

The question "is L_flow = 1.05 good?" is unanswerable in isolation. The
baselines make it answerable by re-scoring the *same* noised inputs with
two cheap rival "models":

- **Copy baseline:** pretend the future equals the present — velocity
  `c_t − ε` instead of `F_c`'s output. Its loss is what a
  "nothing-ever-changes" world model would score.
- **Batch-mean baseline:** pretend every future is the batch-average
  future. Its loss is what an input-blind model would score.

Logged: the three losses plus two ratios.

| Metric | Gate (from PHASE_1 spec) | Meaning if failed |
|---|---|---|
| `coarse_vs_copy_ratio` | **≤ 0.70** | the model hasn't learned *dynamics* — copying the present beats it or nearly does |
| `coarse_vs_batch_mean_ratio` | **≤ 0.50** | predictions aren't *input-specific* — the conditioning isn't being used (conditioning-bypass failure) |

These ratios — not the loss — are the **definition of Phase 1 success**.
They embody the project's epistemic style: every claim ("the model learned
dynamics", "the model uses its conditioning") gets a falsifiable numeric
test against a null model.

### 3e. `gradient_health` → `grad_global_norm`, `grad_has_nan`, `grad_param_count`

Optimization-side vitals at diagnostic cadence: global gradient norm, any-NaN
flag, and the count of parameters carrying gradients. The last one is an
invariant check in disguise — if the count changes between steps, something
structural broke (a module silently detached, a param freeze leaked).

Plus, from the training step itself (every 50 steps, not just diagnostics):
**`grad_norm`** (pre-clip — shows intent) and **`grad_skipped`** (whether
the skip-guard fired — file 07). **`grad_skipped` must stay at 0** for the
entire run; sustained skips mean the optimizer is frozen while metrics may
still look fine (file 10 §7, run `elated-snowflake-15`).

### 3f. `slot_diversity_rank` → `c_slot_diversity_rank`

Within each video, treat the 32 slot outputs as 32 vectors of dim 256 and
compute effective rank across slots (same entropy formula as 3c, but over
slots not feature dims).

- **Healthy:** rank approaching 32 — each slot carries distinct information.
- **Sick:** rank near 1 — all slots collapse to the same summary; the
  bottleneck is wasting its query budget. Run A with aggressive
  `lambda_slot=0.25` moved this metric but hurt copy ratio (Goodhart).

### 3g. `attention_entropy` → `c_attn_entropy`, `c_attn_entropy_min`

Per-head entropy of the cross-attention weights (queries reading encoder
tokens). High entropy = diffuse attention (reading everything equally);
low entropy = peaked attention (specialized lookups).

- **Healthy:** moderate entropy, heads differ (`c_attn_entropy_min` not
  stuck at the floor).
- **Sick:** near-maximum entropy on all heads — queries not specializing;
  often co-occurs with slot collapse.

## 4. How to read the panel (triage runbook)

When you open W&B mid-run, scan in this order:

1. **`grad_norm` & `grad_skipped`** — is optimization stable? Spikes or
   any nonzero skip rate → stability problem (file 10), nothing else is
   trustworthy until resolved.
2. **`c_std_mean` & `c_dead_dim_frac`** — is the latent alive? std should
   sit near 1.0 within the first few hundred steps and stay; dead frac ≈ 0.
3. **`c_cross_video_cosine`** — are videos distinguishable? Should drop
   early and stay low.
4. **`coarse_vs_copy_ratio` then `coarse_vs_batch_mean_ratio`** — is it
   *learning* anything? These fall slowly; judge trend, not level, until
   late in the run.
5. **`c_effective_rank`** — how rich is the latent? Slowest-moving, most
   forward-looking number.

Mnemonic ordering: *stability → aliveness → distinctness → usefulness →
richness.* Each layer is meaningless unless the previous one is green.

And a pairing to remember: **L_var ≈ 0 + `c_std_mean` ≈ 1.0 together**
mean the variance floor is satisfied and inactive — the healthy steady
state. L_var rising mid-run means the floor is actively fighting a
collapse pressure — investigate even though nothing has "failed" yet.

## 5. Questions to test yourself

1. Why can loss be excellent during full collapse? *(Model controls its
   own targets; constant target = trivially predictable.)*
2. A run shows healthy `c_std_mean` but `c_cross_video_cosine` → 0.95.
   Diagnose. *(Directional collapse: variation along one shared direction;
   per-dim variance can't see it.)*
3. Why does the variance floor use a hinge instead of rewarding variance?
   *(Only the constant solution must be forbidden; rewarding variance
   invites high-variance noise — variance is necessary, not sufficient.)*
4. Both ratio gates pass but effective rank is 13. Did Phase 1 succeed?
   *(By the formal gates, dynamics may pass; the latent-quality bar (>60)
   still fails — acceptance criteria are conjunctive. Rank ~13 is the
   current reality, not Run 1's ~5; the question shifted from "is it
   collapsed?" to "can we enrich further?")*
5. Why are diagnostics computed on a fixed validation batch? *(Numbers
   must be comparable across steps; with a changing batch you can't tell
   model change from data change.)*
