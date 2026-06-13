# Postmortem — Run 1 (`peachy-terrain-5`, crashed step ~10750)

> **Status:** Postmortem written 2026-06-10. Run aborted by NaN propagation
> from a gradient explosion at the peak of the LR warmup.
>
> **TL;DR.** Phase 1 default hyperparameters were calibrated for the full SSv2
> training horizon (warmup=10k of 30k total). At step ~10550 the LR
> warmup completed (`lr_mult ≈ 1.0`), and within ~10 steps the pre-clip
> gradient magnitude jumped 4 orders of magnitude (864 → 2.6×10⁸ → 3.5×10¹⁶),
> overwhelmed bf16 precision, corrupted model weights with NaN, and
> propagated through to a `torch.linalg.eigvalsh` crash in the next
> diagnostic batch (step 11000). Root cause: peak `lr_coarse_flow=4e-4`
> combined with `grad_clip=1.0` (norm-clip; preserves direction but at
> 3.5×10¹⁶ the clipped direction is numerical noise in bf16) and AdamW
> `betas=(0.9, 0.95)` (low β2 = short variance memory; cannot dampen a
> single large gradient event).
>
> Fix landed in commit `<TBD>`: peak LR halved, clip tightened, NaN/Inf
> skip-step guard added, diagnostics made NaN-safe, total step count cut
> from 30k to 15k with shorter warmup. See `[#fix](#fix-applied)`.

---

## Timeline of the failed run

The run launched at ~17:50 PT 2026-06-09 from commit `5e78caa` on
`phase1-v0.2-frozen-encoder`, on pod `9bf9a825ee7b` (128-vCPU EPYC 7742,
2 TiB RAM, A100 80 GB). W&B run id: `peachy-terrain-5` (`1chv2608`).


| Step         | `L_flow` | `L_var` | `grad_norm`      | `coarse_vs_copy_ratio` | `c_effective_rank` | Notes                                                                          |
| ------------ | -------- | ------- | ---------------- | ---------------------- | ------------------ | ------------------------------------------------------------------------------ |
| 0            | 2.87     | 0.43    | 0.36             | 171.6                  | —                  | step-0 baseline (bit-identical to H1.2)                                        |
| 500          | 1.88     | 0.07    | 0.51             | 3.27                   | 5.29               | First diagnostic batch, healthy                                                |
| 1000         | 1.25     | 0.12    | 2.44             | **12.93**              | **10.48**          | Rank peaks, ratio jumps (noise)                                                |
| 8000         | 1.07     | 0.045   | 2.44             | 5.01                   | **4.38**           | Rank has compressed back; ratio falling                                        |
| **8500**     | **1.04** | 0.14    | 2.64             | **2.29**               | 4.97               | **Best `coarse_vs_copy_ratio` of the run**                                     |
| 8700–8900    | —        | —       | **24–27 spikes** | —                      | —                  | **First warning: pre-clip grad spikes**                                        |
| ~9500        | 1.09     | 0.05    | 0.97             | 7.90                   | 5.30               | Ratio regressed; grad calmed briefly                                           |
| 10500        | 1.07     | 0.04    | 3.72             | 7.94                   | 4.90               | Final clean diagnostic batch                                                   |
| **10550**    | 1.07     | 0.06    | **864**          | —                      | —                  | **Explosion begins** (lr_mult=0.998)                                           |
| 10600        | 1.11     | 0.07    | 406              | —                      | —                  | Brief recovery                                                                 |
| **10650**    | 1.11     | 0.05    | **2.6×10⁸**      | —                      | —                  | Catastrophic                                                                   |
| **10700**    | 1.11     | 0.08    | **3.5×10¹⁶**     | —                      | —                  | bf16 precision floor blown                                                     |
| 10750        | **NaN**  | **NaN** | **NaN**          | —                      | —                  | Model weights now NaN                                                          |
| 10750–10950  | NaN      | NaN     | NaN              | —                      | —                  | NaN kept propagating; EMA kept updating                                        |
| 11000 (diag) | —        | —       | —                | —                      | —                  | `torch.linalg.eigvalsh` raised `_LinAlgError` on NaN covariance → process exit |


Net: 3 hr 5 min of wall-clock, ~$8 of compute, no usable checkpoint
post-explosion (last clean checkpoint at step 10000, which is by definition
right before the failure mode and would re-fail on resume with the same
schedule).

## What we read correctly vs missed

### Read correctly (in real time)

1. **Variance stayed healthy throughout** — `c_dead_dim_frac=0` at every
  diagnostic batch; `c_std_mean` in `[0.74, 0.86]`. No representational
   collapse.
2. **Acceptance ratio's first downward trajectory** — `coarse_vs_copy_ratio`
  12.93 → 5.01 → 2.29 over steps 1k → 8k → 8.5k was correctly identified
   as the desired direction.
3. **Pre-clip grad spikes (24–27) at steps 8700–8900** were flagged in the
  step-8900 check-in as "watch — drifting up." This was the actual early
   warning of the eventual catastrophic failure.
4. `**c_effective_rank` stagnating at ~5** was flagged in the step-9500
  check-in as likely-soft-fail on the spec gate (`> 30 by step 12k`).

### Missed (in retrospect)

1. **The grad-norm spikes at step 8700–8900 should have been an
  abort-or-tighten-clip moment**, not a "watch." Pre-clip norm of 27 is  already 27× the clip threshold; in bf16 with low-β2 Adam, this is one  bad sample away from the explosion regime.
2. `**coarse_vs_copy_ratio` regressed 2.29 → 7.90 between step 8500 and
  step 9500.** This was attributed to "diagnostic-batch noise." It was
   partly noise, but it was *also* the optimizer beginning to oscillate
   around a poor minimum as Adam moments destabilized from the earlier
   grad spikes.
3. **No NaN/Inf skip guard in the training loop.** When pre-clip grad hit
  864 at step 10550, the optimizer should have been able to *skip* the
   update (zero the gradient, no parameter movement) rather than apply a
   nominally-clipped-but-numerically-garbage step. The model would still
   be alive.

## Root cause analysis

**Mechanism:**

1. AdamW's update is `θ ← θ − η · m̂ / (√v̂ + ε)`, where `m̂` and `v̂` are
  the bias-corrected first and second moment estimates.
2. With `β₂ = 0.95`, the effective variance window is ~20 steps. The
  grad-norm spikes around step 8700–8900 (norms 24–27) inflated `v̂`,
   but only briefly. By step 10500, `v̂` had decayed back toward "normal"
   magnitude (`grad_norm ≈ 3.7` at step 10500).
3. At step 10550, `lr_mult` was 0.998 — within 0.2% of peak. The effective
  per-step LR was `lr_coarse_flow × lr_mult = 4e-4 × 0.998 ≈ 4e-4`. For
   a 6-block transformer with variance-floor regularization, this is on
   the upper edge of stable.
4. A random batch produced an unusually large gradient (a single video
  with high optical-flow content, say). Pre-clip norm went to 864.
5. `clip_grad_norm_(params, 1.0)` divided every gradient component by
  864, preserving direction. But: the gradient is computed in bf16 (per
   `cfg.train.precision="bf16"`). Bf16 has 8 mantissa bits ≈ 3 decimal
   digits of precision. A divisor of 864 destroys all meaningful
   precision in the direction vector — the "clipped" gradient is
   numerically a quasi-random direction.
6. Adam's `v̂` is updated with `0.05 × g²` (where `g` is the unclipped
  gradient, magnitude 864). `v̂` jumps to ~0.05 × 864² ≈ 37k. `m̂` jumps
   to ~0.1 × 864 ≈ 86.
7. Next step: the optimizer applies the *previous* step's update using
  the numerical-garbage clipped direction. Some parameters move by
   `4e-4 × m̂ / √v̂ ≈ 4e-4 × 86 / 192 ≈ 1.8e-4` in a quasi-random
   direction. For most parameters this is fine; for a few in a
   high-curvature region of the loss surface, it isn't.
8. Step 10650: the model now has slightly corrupted parameters; some
  activation overflows in the next forward; backward produces grad
   magnitude 2.6×10⁸.
9. Within two more steps the gradient is 3.5×10¹⁶ — exceeds bf16 range
  (`~3.4×10³⁸`, but accumulation in fp32 reduces effective margin) —
   `Inf` enters the parameter tensor.
10. NaN = Inf - Inf. From step 10750 onward, every parameter is NaN.
11. Diagnostic batch at step 11000: covariance matrix has NaN entries;
  `eigvalsh` cannot iterate; process exits.

**What each contributing factor cost:**


| Factor                         | Contribution                | Counterfactual                                                                                                 |
| ------------------------------ | --------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `lr_coarse_flow = 4e-4` (peak) | Primary                     | If 2e-4: step-7 in the chain doesn't trigger (gradient stays bounded).                                         |
| `grad_clip = 1.0` (norm-clip)  | Secondary                   | If 0.5: divisor in step 5 is half, bf16 precision loss is meaningful but not catastrophic.                     |
| `bf16` precision               | Secondary                   | If fp32: ~7 decimal digits of mantissa — clipped direction would retain meaningful information at divisor 864. |
| AdamW `β₂ = 0.95`              | Contributor                 | If 0.999: `v̂` memory longer; the earlier 24–27 spikes would stay in `v̂` and dampen future updates.           |
| No NaN/Inf step-skip           | Survivability               | With skip: model would have lost 1 update at step 10550 instead of 200+ NaN steps + crash.                     |
| 30k step plan with 10k warmup  | Setting up the failure mode | Shorter warmup + lower peak: peak LR happens at lower magnitude, less time spent at maximum risk.              |


## Fix applied

Changes in commit `<TBD>`:

### `config.py`


| Setting               | Old   | New       | Why                                                                          |
| --------------------- | ----- | --------- | ---------------------------------------------------------------------------- |
| `lr_bottleneck`       | 2e-4  | **1e-4**  | 2× safety margin                                                             |
| `lr_coarse_flow`      | 4e-4  | **2e-4**  | 2× safety margin; the primary failure trigger                                |
| `warmup_steps`        | 10000 | **1500**  | Calibrated for the 15k run; reach peak quickly so we observe stability early |
| `stage1_steps`        | 30000 | **15000** | LR-schedule denominator; matches new total                                   |
| `max_steps`           | 30000 | **15000** | Total training budget                                                        |
| `grad_clip`           | 1.0   | **0.5**   | Tighter clip; less precision loss per clipping event                         |
| `grad_skip_threshold` | (new) | **50.0**  | Skip optimizer step if pre-clip grad exceeds this                            |


### `train.py`

- `train_step` now checks `torch.isfinite(grad_norm)` and
`grad_norm <= cfg.train.grad_skip_threshold` *before* `optimizer.step()`.
- If skipped: `optimizer.zero_grad`, no EMA update, returned metrics include
`grad_skipped=1.0`. Optimizer state is unchanged. Model parameters are
unchanged. The training loop moves on.
- This is a survivability mechanism, not a regularizer — at the new LR/clip
values it should fire rarely (target: 0 fires per run). If it fires
more than a handful of times, that itself is a stop-and-investigate
signal.

### `diagnostics.py`

- `effective_rank` now checks `torch.isfinite(cov).all()` before
`eigvalsh`. If non-finite, returns `{"c_effective_rank": float("nan")}`
instead of raising. The training loop sees NaN as a metric, logs it,
continues. (Combined with the train-loop skip guard, this means even
a future "model produces NaN bottleneck" scenario degrades gracefully
rather than killing the run mid-decay.)

### `parse_args`

- `--steps` default now `15000` (was `30000`).

## Why 15k steps with 1500-step warmup is the right total

The original 30k-step / 10k-warmup spec was calibrated for **full SSv2**
(~170k clips → ~11 epochs at batch-64). On **ssv2_tiny** (4k clips), 30k
steps = **480 epochs** — extreme overtraining for the dataset, and most
real signal in the diagnostic ratios should have stabilized long before
step 25k.

15k × 64 / 4002 ≈ **240 epochs** of ssv2_tiny, which is still very high but
buys us:


|                              | 30k+10k warmup  | 15k+1.5k warmup   |
| ---------------------------- | --------------- | ----------------- |
| Wall-clock (at ~1.25 s/step) | 10.4 h          | **5.2 h**         |
| Pod cost (A100, $2–3/hr)     | $24–32          | **$12–18**        |
| Time spent at near-peak LR   | 20k steps (66%) | 13.5k steps (90%) |
| Iterations per restart cycle | painful         | tolerable         |


Trade-off cost: the acceptance gates in `PHASES/PHASE_1.md` §12 were sized
for a 30k-step trajectory. On 15k they become "stretch goals" rather than
hard requirements. The retest of the design hypothesis ("does the model
learn coarse dynamics without diverging?") remains intact; the absolute
numerical thresholds need to be interpreted with a "30k-equivalent" lens
(scaling rule of thumb: rank may be ~half of the spec, ratio may not yet
reach 0.70 — see "milestone schedule" below).

## Recalibrated milestone schedule (15k run)


| Old (30k run) | New (15k run) | Step                                              |
| ------------- | ------------- | ------------------------------------------------- |
| Milestone A   | A             | ~step 1000 (post-warmup, first stable readings)   |
| Milestone B   | B             | ~step 7500 (mid-run, first acceptance-curve read) |
| Milestone C   | C             | ~step 12500 (late-run, last opportunity to abort) |
| Final         | Final         | step 15000 (acceptance report)                    |


Acceptance interpretation per gate, on a 15k trajectory:


| Gate (PHASE_1 §12)           | 30k threshold      | 15k interpretation                                                    |
| ---------------------------- | ------------------ | --------------------------------------------------------------------- |
| `coarse_vs_copy_ratio`       | ≤ 0.70 by step 25k | **trending under 1.5 by step 12500 = PASS-equivalent; <1.0 = strong** |
| `coarse_vs_batch_mean_ratio` | ≤ 0.50             | <0.80 by step 12500 = PASS-equivalent                                 |
| `c_effective_rank`           | > 60 at step 25k   | **> 30 at step 12500 = PASS-equivalent**                              |
| `c_dead_dim_frac`            | < 0.15             | unchanged                                                             |
| `c_cross_video_cosine`       | well < 0.5         | unchanged                                                             |
| `grad_has_nan`               | 0 throughout       | unchanged                                                             |
| `grad_skipped` (new)         | (new)              | **0 throughout; > 0 = stop and investigate**                          |


A 15k run that hits PASS-equivalent across the board justifies a 30k run
on full SSv2 (Plan Phase 04 territory). A 15k run that fails the
PASS-equivalent gates is informative *regardless*: it's a "the design has
a real problem" signal we get for 5 hours of compute instead of 10.

## What to never do again

1. **Never set `grad_clip` higher than necessary for the precision in
  use.** With bf16 and Adam, clip at 0.5–1.0 with a strict skip-step
   guard at ~10–50×.
2. **Never let pre-clip `grad_norm` spike above ~10× the clip threshold
  without aborting or tightening.** What looks like "noise" in one run
   can be the precursor to a single-event explosion in the next.
3. **Always have a NaN/Inf step-skip in any long-running training loop.**
  Cost: zero per step in the healthy regime. Benefit: survives one bad
   sample without losing the run.
4. **Match step count to dataset size.** 480 epochs of ssv2_tiny was a
  spec carryover from full-SSv2 planning, not a deliberate choice. The
   "smoke test on tiny" run should never have been 30k.

