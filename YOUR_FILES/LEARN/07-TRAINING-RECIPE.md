# 07 — The Training Recipe: Every Number in TrainConfig, and Why

> **What you'll understand after this file:** what AdamW actually computes,
> why learning rates are warmed up and cosine-decayed, what gradient
> clipping and the skip-guard do mechanically, what bf16 is and what it
> costs, and how batch size / steps / epochs relate. This is the file that
> turns `config.py` from a list of magic numbers into a set of arguments.

---

## 1. The optimizer: AdamW

```python
adam_betas: tuple = (0.9, 0.95)
weight_decay: float = 0.05
```

**SGD** updates `θ ← θ − lr·g`. **Adam** improves on it by maintaining,
per parameter, two running averages:

- `m` — EMA of the gradient (**momentum**, decay β₁ = 0.9): smooths the
  direction over the last ~10 steps.
- `v` — EMA of the *squared* gradient (decay β₂): tracks how big this
  parameter's gradients typically are.

Update: `θ ← θ − lr · m̂ / (√v̂ + ε)`. Dividing by `√v̂` gives each
parameter its own effective step size — parameters with consistently large
gradients get smaller steps, rarely-updated ones get larger steps. That
adaptivity is why Adam dominates for transformers, whose layers have
wildly different gradient scales.

**The W in AdamW:** weight decay (pull every weight toward 0, a complexity
penalty) is applied *directly to weights* rather than mixed into the
gradient. Mixed-in decay gets distorted by Adam's `√v̂` division; AdamW's
decoupling restores a clean, uniform regularizer. 0.05 is the standard
ViT/DiT-scale value.

**Why β₂ = 0.95 instead of the default 0.999:** β₂ sets the memory of the
gradient-magnitude estimate (~20 steps vs ~1000). Long memory means: after
a sudden gradient spike, `v` stays inflated/stale for hundreds of steps,
mis-scaling updates. Modern large-model practice (GPT-3 onward) uses
0.95 for faster adaptation to non-stationary gradient scales. The flip
side: a *shorter* memory also means `v` deflates quickly during calm
periods, so a spike after calm hits with less of a dampening buffer —
which is part of the Run 1 story (file 10).

There are exactly **two param groups** — bottleneck (lr 1e-4) and coarse
flow (lr 2e-4) — and the frozen encoder is in none.

## 2. The learning-rate schedule: warmup → cosine

```python
lr_bottleneck = 1e-4      # peak, halved after Run 1
lr_coarse_flow = 2e-4     # peak, halved after Run 1
warmup_steps = 1_500
stage1_steps = 15_000
```

The multiplier (from `train.py::lr_scale`):

```
lr_mult(step) = (step+1)/1500                          step < 1500   (linear warmup)
              = 0.5·(1 + cos(π·progress))              step ≥ 1500   (cosine decay to 0)
```

**Why warmup exists.** At step 0, two things are dangerously uncalibrated:
(a) the weights are random, so early gradients are large and erratic;
(b) Adam's `m, v` estimates are built from almost no data, so the
adaptive scaling is garbage for the first tens of steps. Full-size steps
under both conditions can fling weights into bad regions you never
recover from. Warmup ramps the LR from ~0 so the model takes baby steps
while gradients and optimizer statistics calibrate. It's the standard fix
for transformer training instability (every modern LLM/DiT uses it).

**Why cosine decay.** Early, you want big steps (far from any optimum);
late, you want tiny steps (so the model settles into a minimum instead of
bouncing around it — especially important for the EMA target's stability).
Cosine is the conventional smooth interpolation; the specific shape
matters less than "high → smoothly → ~0."

**Why the bottleneck's LR is half the flow's.** The bottleneck's output
feeds *both* the conditioning and (through EMA) the targets; moving it
fast destabilizes the whole self-referential system. The flow head is a
plain conditional regressor — it can afford faster learning. Asymmetric
LRs encode "change the representation slowly, fit the predictor quickly."

**The Run 1 connection (why these are the *halved* values).** The original
spec was 2e-4/4e-4 with a 10k warmup over a 30k run. Run 1 exploded at
step ~10550 — almost exactly *peak LR* (end of warmup, before decay bites).
Peak LR is the maximum-energy moment of any run: the largest steps the run
will ever take, applied to a model good enough to have sharp loss surfaces.
The fix was to halve the peaks and shorten warmup to 1.5k (10k warmup in a
15k run would leave almost no full-LR training). General heuristic worth
keeping: **if a run survives warmup but dies near peak LR, the peak is too
high — halve it.**

## 3. Gradient clipping and the skip-guard

```python
grad_clip = 0.5
grad_skip_threshold = 50.0
```

**Clipping (clip-by-norm):** after `backward()`, compute the global norm
of all gradients stacked into one vector: `‖g‖ = √(Σ gᵢ²)`. If it exceeds
0.5, multiply *all* gradients by `0.5/‖g‖` — same direction, capped
length. This bounds the worst-case step size; one weird batch (a corrupted
video, an unlucky τ draw) can't fling the weights.

Two subtleties:

- PyTorch's `clip_grad_norm_` *returns the pre-clip norm* — which is why
  we get the `grad_norm` metric for free, and why that metric can read 7.5
  while the applied gradient was at norm 0.5. Logging the **pre-clip**
  norm is the right choice: it shows you what the model *wanted* to do,
  which is the diagnostic signal (a healthy run's pre-clip norm trends
  down; a run trending up is storing trouble).
- Clipping rescues the step but **not Adam's statistics** in the way you
  might hope: `v` is updated from the (clipped) gradient, but a long
  sequence of at-the-cap steps still drags the run in whatever direction
  the explosion points.

**The skip-guard** (added post-Run 1) is the second line of defense:

```223:231:train.py
    grad_norm_f = float(grad_norm)
    grad_skipped = (
        not math.isfinite(grad_norm_f)
        or grad_norm_f > cfg.train.grad_skip_threshold
    )
    if grad_skipped:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()
```

If the pre-clip norm is non-finite (NaN/Inf) or above 50 — two orders of
magnitude beyond normal — the step is **discarded entirely**: no optimizer
update, no EMA update. Rationale: clipping a norm-10¹⁶ gradient still
applies a *direction* computed from garbage, and in bf16 the division
`g · (0.5/10¹⁶)` itself destroys precision. A gradient that absurd carries
no usable information; the correct amount to learn from it is *nothing*.
One bad batch then costs one update out of 15,000 instead of the run.

The threshold's logic: normal pre-clip norms run ~1–10; 50 is far above
anything legitimate and far below explosion territory (10⁴+). And the
operational contract: at current LRs this should fire **approximately
never** — `grad_skipped` is logged precisely so that a nonzero rate reads
as "stop and investigate," not "the guard is handling it." A guard that
fires regularly is masking a disease, not curing it.

## 4. Precision: bf16

```python
precision = "bf16"   # via torch.autocast on CUDA
```

Floating-point formats trade bits between **range** (exponent) and
**precision** (mantissa):

| Format | Bits | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~10³⁸ | ~7 decimal digits |
| fp16 | 16 | 5 | 10 | ~65,504 max | ~3 digits |
| **bf16** | 16 | 8 | **7** | ~10³⁸ (same as fp32) | **~2–3 digits** |

bf16 keeps fp32's full exponent and sacrifices mantissa. Consequences:

- **Why we use it:** half the memory and roughly 2× the throughput on
  modern GPU tensor cores; and unlike fp16, it essentially never
  *overflows*, so no loss-scaling machinery is needed. It's the default
  for modern large-model training.
- **What it costs:** with ~3 significant digits, small relative
  differences vanish. `1.001 + 0.0001 = 1.001` in bf16. Gradients that
  differ by a factor of 10⁻³ from the values they update can be rounded to
  no-ops or to noise.
- **The interaction that bit us:** autocast keeps master weights and the
  optimizer in fp32 (good), but activations and the backward pass run in
  bf16. During a gradient explosion, intermediate values span enormous
  scales; bf16 *represents* them (range is fine) but with so little
  mantissa that directions become garbage long before anything overflows.
  bf16 doesn't cause explosions — it converts "large but maybe recoverable"
  into "numerically meaningless" faster than fp32 would. Hence the
  tighter clip (0.5) and the skip-guard: in bf16 you must keep the
  *numbers small*, not just finite.

## 5. Batch, steps, epochs — and the overtraining trap

```python
global_batch = 64
stage1_steps = 15_000      # was 30_000
horizon_k = 4              # config default; operating value is 12 (see below)
frame_stride = 2           # context samples every 2nd frame
```

**Config default vs operating value:** `config.py` ships with
`horizon_k=4` and `lambda_var=0.10` (matching `UNDERSTANDING.md` §2.6).
Validated Phase 1 runs override via CLI: **`--horizon-k 12 --lambda-var 0.5`**
(investigation 003, run `cerulean-snow-13`). Always record which you used —
W&B snapshots the resolved config at launch.

Definitions, precisely:

- A **step** = one optimizer update = one batch of 64 (context, target)
  clip pairs = 128 encoder forwards.
- An **epoch** = one pass over the dataset. With ssv2_tiny (~4k clips) and
  batch 64: ~63 steps/epoch. So **15,000 steps ≈ 240 epochs**; the
  original 30k was ~480 epochs.
- The same step count on full SSv2 (~170k clips, ~2,650 steps/epoch) would
  be ~5.7 epochs.

The 30k spec was calibrated for *full SSv2*. Running it unchanged on
ssv2_tiny re-showed each clip ~480 times — deep **overtraining** territory
(and yes, in this context overtraining ≈ overfitting: the model starts
memorizing the 4k specific clips rather than learning general dynamics;
the augmentations — random temporal window, crop, jitter — soften but
don't eliminate it). The general law: **a step budget is only meaningful
relative to a dataset size.** Quote steps *and* epochs when discussing any
run.

Why per-step thinking dominates anyway: LR schedule, EMA schedule, warmup,
checkpoint cadence are all written in steps, because what matters to the
optimizer is the number of updates, not dataset passes.

`horizon_k` and `frame_stride` define the prediction problem itself:
context = 8 frames sampled every 2nd frame (≈ 1.3s of 12fps SSv2 video);
target window = same geometry shifted `horizon_k` raw frames ahead.

- **At `horizon_k=4` (config default):** target starts 4 frames later —
  ≈ 0.33s ahead, with heavy overlap between context and target windows
  (file 13). Too easy for a low-rank latent to survive.
- **At `horizon_k=12` (operating default):** target starts 12 frames later
  — ≈ 1.0s ahead, minimal overlap (frames 12 and 14 only). Harder task
  that forces richer representations. Phase 4 generalizes k further (file 11).

Recommended launch for acceptance runs:

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

## 6. The loss weights

```python
lambda_var = 0.10          # config default
lambda_cov = 0.0           # VICReg-C; logged always, added only if > 0
lambda_slot = 0.0          # slot diversity; logged always, added only if > 0
var_floor_std_target = 1.0
```

Total loss = `L_flow + λ_var · L_var + λ_cov · L_cov + λ_slot · L_slot`.

**Operating weight:** `λ_var = 0.50` (investigation 003). At 0.10 the
ratio is 10:1 flow-to-var — the floor is a light fence. At 0.50 the floor
actively fights collapse and rank rises to ~13+. The config default (0.10)
reproduces the v0.2 baseline for ablations; production runs use 0.50.

In a healthy run L_var ≈ 0 within a few hundred steps and the total is
effectively pure L_flow. If you ever see the two terms competing (L_var
persistently > 0 while L_flow falls), something is pushing `c_t` toward
deadness and the fence is holding it back — investigate rather than
retune λ blindly.

## 7. Cadences

```python
log_every = 50            # scalar metrics → stdout + W&B
diag_every = 500          # full diagnostic panel on the fixed val batch
checkpoint_every = 2_500  # ~6 checkpoints over a 15k run
```

The three-tier design is deliberate: cheap scalars often (you can't
diagnose what you didn't record — Run 1's autopsy was possible *only*
because grad_norm was logged every 50 steps); expensive diagnostics
(extra forwards, eigendecomposition) at 10× coarser grain; checkpoints at
the cost-of-loss granularity — a crash costs at most 2,500 steps ≈ 50
minutes of progress. When sizing these for any future run, that's the
question each answers: *what does a gap of this size cost me if things go
wrong?*

## 8. Questions to test yourself

1. Why does Adam need warmup more than plain SGD does? *(Its m/v
   statistics are uncalibrated early; adaptive scaling from garbage
   statistics makes early steps worse than un-adapted ones.)*
2. What does `grad_norm = 7.6` in the logs mean if `grad_clip = 0.5`?
   *(7.6 is the pre-clip norm; the applied update had norm 0.5; the metric
   shows intent, the clip shows action.)*
3. Why is skipping a step better than clipping when norm = 10¹⁶? *(The
   direction itself is numerical garbage; clipping preserves direction;
   bf16 rescaling by 10⁻¹⁶ adds rounding noise. No information → no
   update.)*
4. Why would fp16 (not bf16) have needed extra machinery here? *(fp16's
   max is 65k — gradients overflow to Inf routinely, requiring dynamic
   loss scaling; bf16's fp32-range exponent avoids that.)*
5. 15k steps is "short" on full SSv2 and "long" on ssv2_tiny — explain.
   *(~5.7 epochs vs ~240 epochs; step budgets only mean something relative
   to dataset size.)*
