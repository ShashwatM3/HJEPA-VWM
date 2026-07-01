# Problems, Metrics, and Experiments

This note answers one question:

> What problems have we faced, how did we recognize them in W&B/code metrics, and what did we try?

The emphasis is on the W&B dashboard view: which metric moved, what that movement meant, what code
or experiment we changed because of it, and what the next run told us.

## What We Are Trying To Solve

The research target is a compact abstract video latent `c_t` that is both:

1. **Informative**: it carries enough state to distinguish videos and preserve useful content.
2. **Predictive**: it supports forecasting the future abstract latent better than trivial baselines.

The central acceptance metric is not just whether losses go down. It is whether the model beats
simple baselines:

```text
coarse_vs_copy_ratio <= 0.70
coarse_vs_batch_mean_ratio <= 0.50
```

The current story is:

- Early runs failed because `c_t` collapsed or training exploded.
- Later runs fixed much of the representation health problem.
- The current bottleneck is prediction: `c_t` can look healthy, but `F_c` still often ties the
  copy / zero-residual baseline.

## Metric Glossary

### Prediction Metrics

#### Coarse metrics — read this first

**There is only one network: `F_c`.** Copy and batch-mean are **not** networks. They are **fixed
guesses** — wrong answers we write down by hand so we can ask: *how bad would this stupid guess
be?*

Every 500 steps, `run_diagnostics` (`train.py`) runs on a fixed validation batch. For each video
we already computed:

- **`c_t`** — abstract latent of the **present** clip (32 vectors × 256 numbers).
- **`c_plus`** — abstract latent of the **future** clip (same shape).

The true answer we want `F_c` to learn is basically **`c_plus`** (or the change from present to
future, in residual mode). The three loss numbers all use the **same scoring function** in
`diagnostics.coarse_baselines` — we are not comparing three networks. We are comparing **one
network's output** against **two hand-written wrong answers**, using the same ruler.

**CRITICAL: `coarse_copy_loss` is NOT a training loss. Nothing is optimized to make it go down.**

Search `train.py` — `coarse_copy_loss` never appears. It is logged only inside `run_diagnostics`
(`torch.no_grad`, no `backward`). The training loss is `L_flow`, which pushes **`F_c` to predict the
real future `c_plus`**, not to shrink `(c_t - c_plus)`.

| | Trained? | What it does |
|---|---|---|
| `L_flow` | **Yes** — minimized every step | `F_c` must match the true future target |
| `coarse_copy_loss` | **No** — read-only diagnostic | measures how far apart `c_t` and `c_plus` already are |
| `coarse_vs_copy_ratio` | **No** — pass/fail check | asks: is `F_c`'s error much smaller than the copy guess? |

**What we actually want:** `coarse_model_loss` **low**, and `coarse_model_loss / coarse_copy_loss`
**≤ 0.70**. That means `F_c` beats "future = present" by a wide margin.

**We do NOT want:** `coarse_copy_loss` → 0. If it went to zero, present and future would be
identical in latent space and copy would be unbeatable — but we are not training toward that. When
`coarse_copy_loss` **drops during a run**, that is a **side effect** of other things being trained
(variance floor, SIGReg, reconstruction, etc. shaping the bottleneck `B`). Those losses never mention
`coarse_copy_loss`; they can still make `c_t ≈ c_plus` accidentally, which is a known failure mode
(static `c`), not the goal.

**Analogy:** you train a weather model to predict tomorrow's temperature (`L_flow`). Each week you
also log: "how wrong would you be if you just repeated today's temperature?" (`coarse_copy_loss`).
You are **not** training the model to make today and tomorrow the same temperature. You are
checking whether your model beats the lazy guess. If your model's error ≈ the lazy guess's error,
training failed — even if both numbers are small.

---

#### `coarse_model_loss` — the network runs

1. The code runs **`F_c`** (the coarse flow network).
2. `F_c` sees the present `c_t` and outputs its best attempt at the flow target.
3. The code compares that output to the **correct** target (derived from `c_plus`).
4. `coarse_model_loss` = mean squared error of that comparison.

**Plain English:** how wrong is **`F_c`** right now?

This is the only metric where a neural network is involved.

---

#### `coarse_copy_loss` — no network, just a fixed guess

**The guess:** "the future latent is exactly the same as the present latent."

Concretely, for one video:

- true future = `c_plus` (what actually happened in the later frames)
- copy guess = `c_t` (just reuse the present tensor, change nothing)
- copy error = how far apart those two tensors are, squared and averaged

```text
coarse_copy_loss  ≈  average over all numbers in (c_t - c_plus)²
```

**No network runs for this.** Nobody calls `F_c`. The code does not "predict" anything. It takes
the tensor `c_t` that is already sitting there and asks: *if we insisted the answer were `c_t`
instead of `c_plus`, how much would we be wrong?*

That is all "nothing changes" means:

```text
my answer for the future  =  what I already have for the present
```

**Why does the code route this through `flow_matching_loss`?** Because `F_c` is trained with that
loss. Plugging the copy guess into the **same** formula gives an apples-to-apples number you can
divide against `coarse_model_loss`. Under the hood it sets `copy_velocity = c_t - eps` in
`diagnostics.coarse_baselines`; the `eps` cancels and you get the `(c_t - c_plus)²` distance above.
You do not need to think about velocities to interpret the metric.

**Residual mode** (`--predict-residual`): `F_c` is trained to predict the **change**
`Δ = c_plus - c_t` instead of the full future. The copy guess becomes **"the change is zero"**
(`Δ = 0`, i.e. future = present). Then:

```text
coarse_copy_loss  ≈  average over all numbers in Δ²
```

Still no network — just measuring how big the true change is.

**How to read the number:**

- **Low** → `c_t` and `c_plus` are already close. Saying "nothing changed" is a strong guess.
- **High** → present and future are farther apart. "Nothing changed" is a weak guess.

**This does not mean the model is copying.** The model is not involved. It only measures whether
the **data** makes "future = present" a good or bad guess.

**Again: we never backprop into this number.** Confusing it with a loss to minimize is the main
misread. Training minimizes `L_flow` (predict the future). Copy loss is a **yardstick** taped to
the wall, not a target on the dartboard.

---

#### `coarse_batch_mean_loss` — also no network

**The guess:** pick **one** future tensor — the average of every video's `c_plus` in the batch —
and use that same tensor as the answer for **every** video.

Still no network. Still just: guess vs true `c_plus`, squared distance, same scoring function.

**Plain English:** how wrong would you be if you ignored each video entirely and gave everyone the
batch average?

---

#### `coarse_vs_copy_ratio` — the only number that really matters

```text
coarse_vs_copy_ratio = coarse_model_loss / coarse_copy_loss
```

| Value | Meaning |
|---|---|
| `< 1.0` | `F_c` beats the "future = present" guess |
| `≈ 1.0` | `F_c` is no better than saying "nothing changed" |
| `> 1.0` | `F_c` is worse than saying "nothing changed" |
| `≤ 0.70` | Phase 1 pass gate |

If this sits at ~1.0, `F_c` learned roughly nothing beyond the copy guess — even if other metrics
look fine.

---

#### `coarse_vs_batch_mean_ratio`

```text
coarse_vs_batch_mean_ratio = coarse_model_loss / coarse_batch_mean_loss
```

Same idea, weaker bar: is `F_c` better than ignoring the video and using the batch average?
Pass gate: `≤ 0.50`.

---

#### `L_flow`

Same as `coarse_model_loss`, but measured during **training** on a random batch (not the fixed
diagnostic batch). Only `F_c` runs; no copy baseline involved in the training loss itself.

### Representation Health Metrics

#### `c_std_mean` and `c_std_median`

These measure per-dimension standard deviation of `c_t` across the batch.

Low values mean many dimensions are not moving much. A healthy value is near the variance-floor
target, usually around `1.0` in our successful runs.

These metrics diagnosed the early weak-variance-floor problem: `lambda_var=0.1` allowed low-std
representations; `lambda_var=0.5` pushed std toward the intended range.

#### `c_dead_dim_frac`

This is the fraction of dimensions whose std is much smaller than the median.

High values mean part of the latent is dead. Low values mean dimensions are at least numerically
active, but this does not guarantee they are semantically useful.

#### `c_cross_video_cosine`

This measures mean pairwise cosine similarity between different videos' flattened `c_t`.

Interpretation:

- Low, well below `0.5`: different videos get distinct `c_t`.
- High, drifting toward `1.0`: many videos point in the same direction; this is video-independent
  collapse.

This metric was critical because variance alone can look okay while every video still maps to a
similar direction.

#### `c_effective_rank`

This estimates how many feature directions are actually used by the `c_t` distribution.

The maximum meaningful feature dimension is `D_c = 256`. The current healthy target is roughly:

```text
c_effective_rank > 60
```

Low rank means the bottleneck is using only a small subspace, even if the latent is not literally
constant. This metric exposed the rank-13 ceiling.

#### `c_plus_effective_rank`

This is effective rank on the EMA target `c_plus`.

This matters because `F_c` predicts the target branch, not just the online branch. If online `c_t`
looks healthy but `c_plus` lags or stays lower-rank, the regression target is still weak or unstable.

#### `c_slot_diversity_rank`

This measures how many independent directions the bottleneck's query slots span within a video.

It answers:

```text
Are the 32 slots learning different things, or are they redundant?
```

It helped diagnose slot collapse, but later also taught us that optimizing a slot metric directly
can Goodhart: the slot metric improves while actual representation/prediction quality worsens.

#### `c_attn_entropy` and `c_attn_entropy_min`

These measure how uniform the bottleneck cross-attention is over V-JEPA tokens.

Values near `1.0` mean attention is close to uniform. Lower values mean more selective attention.

This is a weak supporting metric. It can be misleading if different heads specialize in different
ways, so we treat it as context rather than a primary gate.

### Reconstruction Metrics

#### `L_recon`

This is the training-time present reconstruction loss when `lambda_recon > 0`.

It trains:

```text
c_t -> D -> e_hat_t
compare e_hat_t to e_t
```

It tells us whether the online bottleneck and decoder can reconstruct present detailed frozen
features.

#### `L_recon_pred`

This is the training-time predicted-future reconstruction loss when `lambda_recon_pred > 0`.

It trains:

```text
c_hat -> D -> e_hat_plus
compare e_hat_plus to e_plus
```

This term routes reconstruction pressure through the predicted future abstract state.

#### `L_recon_present`

This is the diagnostic present reconstruction readout:

```text
D(c_t) vs e_t
```

It is the cleanest reconstruction metric for asking whether `c_t` carries enough information for
the decoder to recover frozen detailed features.

#### `L_recon_cplus`

This diagnostic decodes the true future abstract target:

```text
D(c_plus) vs e_plus
```

It tells us how well the decoder can reconstruct future detailed features when it receives the true
future abstract latent, not the model's prediction.

#### `L_recon_chat`

This diagnostic decodes the predicted future abstract state:

```text
D(c_hat) vs e_plus
```

The gap:

```text
L_recon_chat - L_recon_cplus
```

is important. If the gap is large, the predicted future `c_hat` lands somewhere worse than the true
future target. If the gap is tiny while `coarse_vs_copy_ratio` is bad, reconstruction is blind to
prediction quality.

That tiny gap is exactly what earlier reconstruction experiments showed.

#### `recon_scale`

This is the linear warmup multiplier for reconstruction losses.

Early in a run, `recon_scale` starts near zero and ramps to `1.0` over `recon_warmup_steps`.

### Regularization Losses

#### `L_var`

This is the variance-floor loss:

```text
mean(max(0, 1.0 - std(c_t_dim)))
```

It is high when dimensions have too little variance and low when dimensions meet the std floor.

Important detail: `L_var` can become small once std is healthy, but it does not ensure high rank,
good prediction, or useful temporal dynamics.

#### `L_sigreg`

SIGReg pushes the pooled `c_t` distribution toward an isotropic Gaussian.

It is designed to increase utilization and effective rank. It succeeded at raising rank, but by
itself it made prediction worse, which told us rank alone was not the final bottleneck.

#### `sigreg_scale`

This is the linear warmup multiplier for SIGReg.

It prevents the online bottleneck geometry from moving too fast relative to the EMA target branch.

#### `L_cov`

This is an optional covariance penalty. It targets correlated feature dimensions in `c_t`.

It is logged/available but not currently the main lever.

#### `L_slot`

This is an optional slot-diversity penalty. It targets redundancy among the 32 bottleneck slots.

We tried this direction and rejected aggressive slot loss because it improved the slot-specific
metric while hurting broader representation and prediction metrics.

### Stability Metrics

#### `grad_norm`

This is the true pre-clip gradient norm from the training step.

Large spikes showed the instability problem. This metric was central in diagnosing late training
explosions.

#### `grad_global_norm_postclip`

This is the gradient norm after AGC and global clipping.

It is often near the global clip value and should not be read as the true raw gradient magnitude.
Use `grad_norm` for that.

#### `grad_skipped`

This is `1` when a step is skipped because the gradient norm is too large or non-finite.

If this starts firing repeatedly, training is effectively frozen or unstable.

#### `grad_has_nan`

This flags NaNs in gradients.

Any sustained NaN signal is a hard failure.

#### `instability_warn`

This is an early warning based on a combination of high gradient norm and high flow loss.

It is meant to catch the pre-explosion regime before a run fully breaks.

#### `agc_B_*`, `agc_Fc_*`, `agc_D_*`

These show adaptive gradient clipping behavior by module:

- `B`: bottleneck.
- `Fc`: coarse flow.
- `D`: reconstruction decoder.

High clipping on one module tells us where gradient pressure is concentrated.

## Problem 1: Throughput Was Too Slow

### How We Recognized It

The signal was not a model metric. It was run throughput:

- seconds per step were too high;
- GPU utilization was not matching the available A100 capacity;
- the dataloader was decoding more video than the training step needed.

This was visible from logs and wall-clock time per W&B step.

### What We Changed

`data.py` was changed to decode only the context and target frames needed for the current sample,
instead of decoding full videos unnecessarily.

### Result

Throughput improved from roughly `1.66 s/step` to roughly `1.41 s/step`.

This did not solve learning, but it reduced the cost of each experiment.

## Problem 2: Training Exploded Late

### How We Recognized It

Early full runs showed:

- `grad_norm` spiking sharply;
- losses jumping after thousands of apparently healthy steps;
- NaNs appearing;
- runs dying or becoming unrecoverable.

`peachy-terrain-5` was the clearest early example: it looked healthy for a while, then hit a
catastrophic gradient explosion around the 10k-step region and produced NaNs.

### What We Tried

We changed the training mechanics:

- reduced peak learning rates;
- shortened the first acceptance run shape to 15k steps;
- changed warmup to 1500 steps;
- tightened global grad clipping to `0.5`;
- added `grad_skip_threshold`;
- added AGC by module;
- added instability metrics;
- later hardened optimizer parameter grouping so geometry parameters and zero-init gates were
  excluded from weight decay / AGC where appropriate.

### How We Judged It

We watched:

- `grad_norm`;
- `grad_skipped`;
- `grad_has_nan`;
- `instability_warn`;
- `agc_*_clipped`;
- loss continuity around the previous failure window.

### Result

Training became much more survivable. The latest clean runs, especially inv010, had:

- no NaNs;
- no skip spiral;
- stable AGC behavior;
- no catastrophic late gradient break.

This moved instability from the main blocker to a guarded risk.

## Problem 3: `c_t` Collapsed Or Became Video-Independent

### How We Recognized It

The dashboard showed:

- low `c_std_mean`;
- high `c_cross_video_cosine`, often around `0.7-0.8`;
- low `c_effective_rank`;
- poor copy/batch-mean ratios.

The key lesson was that variance alone was not enough. A representation can have nonzero variance
but still point in nearly the same direction for every video.

### What We Tried

We tried several fixes:

1. Orthogonal bottleneck query initialization.
2. Zero-init output MLP for a safer residual start.
3. Stronger variance floor.
4. Larger prediction horizon.
5. Slot-diversity losses.

### What Happened

Orthogonal queries and zero-init helped the initial geometry, but did not fully solve collapse.

Slot-diversity loss was a negative result. It improved the slot-specific metric but hurt the metrics
that actually mattered:

- `c_effective_rank` stayed low or worsened;
- `c_cross_video_cosine` rose;
- `coarse_vs_copy_ratio` stayed poor;
- gradients became less stable.

The first meaningful win came from:

```text
horizon_k = 12
lambda_var = 0.5
```

This moved `c_std_mean` toward `1.0`, lowered cross-video cosine, and produced the first healthier
representation runs.

### Result

Basic collapse became manageable. However, even after collapse improved, rank and prediction were
still not solved.

## Problem 4: The Rank-13 Utilization Ceiling

### How We Recognized It

Some runs had:

- reasonable `c_std_mean`;
- acceptable `c_cross_video_cosine`;
- stable gradients;
- but `c_effective_rank` stayed around `13 / 256`.

That meant `c_t` was not constant, but it was using only a small part of the feature space.

### What We Tried

We tried reconstruction anchoring:

```text
c_t -> D -> e_hat_t
compare e_hat_t to e_t
```

The hypothesis was:

```text
If c_t must reconstruct e_t, it may need to use more dimensions.
```

We also tried decoder and reconstruction-capacity sweeps:

- higher `lambda_recon`;
- larger decoder width/depth;
- different bottleneck capacity directions.

### How We Judged It

We watched:

- `L_recon_present`;
- `c_effective_rank`;
- `coarse_vs_copy_ratio`;
- stability metrics.

### Result

Reconstruction helped stability but did not break the rank ceiling by itself. `L_recon_present`
settled near a floor around the old relative-MSE range, while `c_effective_rank` remained near the
same low plateau.

This told us decoder size and reconstruction weight were not the main missing ingredient.

## Problem 5: Reconstruction Was Blind To Prediction Quality

### How We Recognized It

The dashboard showed this pattern:

```text
L_recon_present ~= L_recon_cplus ~= L_recon_chat
```

while:

```text
coarse_vs_copy_ratio was still bad
```

That means the decoder reconstructed present, true future, and predicted future abstract states to
almost the same loss level. The reconstruction channel could not tell a good future prediction from
a poor one.

The important diagnostic was:

```text
L_recon_chat - L_recon_cplus
```

The gap stayed tiny. That made the prediction-side reconstruction anchor weak as a prediction
improvement tool.

### What We Tried

We tried:

- present reconstruction only as an anchor;
- prediction-side reconstruction via `lambda_recon_pred`;
- higher reconstruction weight;
- larger decoder.

### Result

Prediction-side reconstruction did not improve copy ratio. `L_recon_pred` stayed pinned and
`L_recon_chat` did not separate enough from `L_recon_cplus`.

This is why the current code added a reconstruction-loss mode switch:

```text
--recon-loss-mode cosine
--recon-loss-mode relative_mse
```

The new `cosine` loss normalizes each detailed tubelet vector before comparison:

```text
L_recon = mean(1 - cos(e_hat, e))
```

This tests whether the old `MSE / Var(e)` loss let the decoder partially game feature magnitude
instead of learning useful angular alignment.

## Problem 6: SIGReg Raised Rank But Hurt Prediction

### How We Recognized It

The SIGReg sweep showed a clear split:

- `c_effective_rank` rose strongly as `lambda_sigreg` increased;
- `coarse_vs_copy_ratio` got worse;
- `L_flow` increased;
- `c_cross_video_cosine` could also rise at strong settings.

This was one of the most important results. It showed that:

```text
more rank != better prediction
```

SIGReg can fill dimensions with information that is static, appearance-heavy, or otherwise not
useful for forecasting.

### What We Tried

We swept `lambda_sigreg` across increasing strengths and kept the reconstruction background stable.

Approximate pattern:

```text
lambda_sigreg up -> rank up -> prediction worse
```

### Result

SIGReg became useful as a rank lever, but not as a standalone solution. It told us the disease was
not simply low utilization. The disease was temporal.

## Problem 7: `c_t` Was Too Static Over Time

### How We Recognized It

When `coarse_copy_loss` is low, the copy baseline is strong. That means:

```text
c_t and c_{t+k} are close
```

If `c_t` barely moves, `F_c` has little temporal structure to learn. A model can look bad against
copy even if it is doing something nontrivial, because copy is already very competitive.

The SIGReg-only run sharpened this diagnosis: rank improved, but copy loss suggested much of the
new information was static rather than predictive.

### What We Tried

We changed the prediction target to a residual:

```text
Delta = c_{t+k} - c_t
```

Both sides of the residual target come from `B_EMA`, so the target is detached and temporal.

The hypothesis was:

```text
Make F_c focus on the change, not on transporting the whole future latent.
```

### Result

Residual prediction was a real improvement diagnostically:

- rank became much healthier;
- `c_cross_video_cosine` stayed low;
- `coarse_copy_loss` rose, meaning the representation became more dynamic;
- `coarse_vs_copy_ratio` dropped dramatically compared with bad SIGReg-only runs.

But it still plateaued near copy. The model often tied the zero-residual baseline.

That moved the problem from:

```text
c_t is too static
```

to:

```text
c_t moves, but the movement is not being predicted well enough
```

## Problem 8: Residual Prediction Tied Zero-Residual Copy

### How We Recognized It

In residual mode, copy means:

```text
predict Delta_hat = 0
```

The key W&B pattern was:

- `coarse_vs_copy_ratio` hovered near `1.0`;
- `coarse_model_loss` tracked `coarse_copy_loss`;
- rank and cosine looked healthy;
- stability looked healthy.

The latest inv010-style result was the cleanest example:

```text
c_effective_rank ~= 61
c_cross_video_cosine ~= 0.16
c_std_mean ~= 1.0
coarse_vs_copy_ratio ~= 1.06
grad_skipped = 0
```

So the representation passed many health gates, but prediction still failed the copy-ratio gate.

### What We Tried

We hardened optimizer and regularization plumbing:

- better decay/no-decay parameter groups;
- AGC exclusions for geometry and zero-init gates;
- SIGReg warmup;
- clearer target-side rank logging.

### Result

Training became cleaner and rank passed the gate, but prediction did not. That means the remaining
problem is not just optimizer hygiene.

## Problem 9: We Need To Separate Reconstruction Geometry From Prediction Dynamics

### How We Recognized It

The combined evidence is:

- reconstruction helped stability but did not fix prediction;
- old reconstruction metrics were too similar across present, true future, and predicted future;
- SIGReg fixed rank but not prediction;
- residual prediction made `c_t` more dynamic but still did not decisively beat copy.

So we need to distinguish two questions:

1. Can reconstruction make `c_t` richer under a better loss geometry?
2. Does a richer `c_t` actually make future prediction easier?

### What We Are Testing Now

Investigation 011 separates those questions:

#### Run A

Full current residual/SIGReg/reconstruction recipe with:

```text
--recon-loss-mode cosine
```

This tests whether the new per-tubelet cosine reconstruction objective improves the full recipe.

#### Run B

Present reconstruction bottleneck test with:

```text
--present-recon-only
--recon-loss-mode relative_mse
```

This tests whether the original reconstruction objective alone can train a rich present `c_t` when
the target is simply:

```text
D(B(e_t)) -> e_t
```

Run B should be judged by reconstruction and representation metrics, not by copy-ratio gates.

## Summary Table

| Problem | W&B / metric symptom | What we tried | Result |
|---|---|---|---|
| Slow throughput | high seconds/step, weak GPU utilization | decode only needed frames | faster experiments, not a learning fix |
| Late explosion | `grad_norm` spike, NaNs, skipped/frozen steps | LR changes, warmup, clip, skip guard, AGC | stability much improved |
| Collapse | low std, high cross-video cosine, low rank | init fixes, stronger variance floor, horizon 12 | collapse mostly controlled |
| Slot redundancy | low slot-rank / uniform attention | slot-diversity loss | rejected; Goodharted slot metric |
| Rank ceiling | `c_effective_rank` around 13 | reconstruction, decoder/weight sweeps | recon stabilized but rank stayed low |
| Recon blindness | `L_recon_chat ~= L_recon_cplus` despite bad copy ratio | prediction-side recon, bigger D, higher weight | did not improve prediction |
| Low utilization | low rank with otherwise stable training | SIGReg sweep | rank improved, prediction worsened |
| Static `c` | low `coarse_copy_loss`; copy too strong | residual prediction | made `c` more dynamic, but model tied copy |
| Zero-residual tie | ratio near 1 while rank/cosine healthy | optimizer plumbing, SIGReg warmup | cleaner run, prediction still unsolved |
| Loss geometry uncertainty | old recon may allow norm games | cosine recon mode | active inv011 Run A test |

## Current State

Solved or mostly controlled:

- catastrophic instability;
- basic collapse;
- low std;
- rank gate in the best current recipe;
- logging/diagnostic visibility.

Still open:

- beating copy by a meaningful margin;
- making `c_t` not just rich, but predictably dynamic;
- making reconstruction pressure help prediction instead of only stabilizing representation;
- deciding whether cosine reconstruction changes the usefulness of the reconstruction channel.

The current bottleneck is therefore not simply:

```text
Can the model make a noncollapsed c_t?
```

It is:

```text
Can the model make a c_t whose future change is learnable enough that F_c beats copy?
```
