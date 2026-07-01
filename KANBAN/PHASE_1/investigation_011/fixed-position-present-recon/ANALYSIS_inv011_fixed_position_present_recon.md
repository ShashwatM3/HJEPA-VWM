# Investigation 011 Run D - final analysis: `inv011_fixed_position_present_recon`

**Status:** VALID RUN, W&B state `crashed` but no training-health failure observed  
**W&B:** `inv011_fixed_position_present_recon` (`hcr2qx19`)  
**Group:** `inv011_fixed_position_present_recon`  
**Created:** 2026-07-01 11:39 UTC  
**Pull / read date:** 2026-07-01  
**Commit:** `693c881` (`Add fixed-position present recon run docs`)  
**Primary source:** W&B run history via `run_history.py` and W&B metadata  
**Last logged training step:** 14400  
**Last diagnostic step:** 14000  

Diagnostic metrics (`c_*`, `L_recon_present`) log every 500 steps. Training metrics
(`L_recon`, `L_sigreg`, `L_var`, `grad_*`) log every 50 steps. This analysis uses the
present-reconstruction-only cycle from `KANBAN/README_for_reading_experiments.md`.

## 0. What this run tested

This is investigation 011 Run D. It is not a prediction run. It disables the future branch and asks
one narrower question:

```text
x_t -> E -> e_t -> B -> c_t
fixed-position D(c_t) -> e_hat_t
cosine(e_hat_t, e_t)
```

The run tests whether the current bottleneck plus the fixed-position decoder can produce a strong,
video-specific, decodable present representation before asking `F_c` to forecast it.

What is new relative to the original inv011 present-only run `original_recon_loss + no-pred`
(`kttd1fib`):

| Axis | Old present-only Run B | Run D |
|---|---|---|
| Decoder | learned-query decoder | fixed-position decoder |
| Recon loss | `relative_mse` | `cosine` |
| `lambda_var` | 0.0 | 0.5 |
| `lambda_sigreg` | 0.0 | 5.0 |
| `lambda_recon` | 0.05 | 0.05 |
| Prediction branch | off | off |

What is inactive:

- no target clip path;
- no `B_EMA(e_plus)`;
- no residual target;
- no `F_c` forward;
- no `L_flow`;
- no prediction-side reconstruction;
- no copy or batch-mean gate.

Therefore, this run can prove present-representation capacity. It cannot prove forecasting ability.

## 1. Config highlights

| Field | Value |
|---|---:|
| dataset | `ssv2` |
| requested steps | 15000 |
| last logged step | 14400 |
| `present_recon_only` | true |
| `prediction_active` | false |
| `predict_residual` | false |
| `horizon_k` | 12, logged only; no training effect in this mode |
| `lambda_var` | 0.5 |
| `lambda_sigreg` | 5.0 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0.0 |
| `recon_loss_mode` | `cosine` |
| `sigreg_warmup_steps` | 2000 |
| `recon_warmup_steps` | 2000 |
| `lr_bottleneck` | 1e-4 |
| `lr_decoder` | 1e-4 |
| decoder | fixed-position, 512 dim, 4 blocks |
| `n_c`, `d_c` | 32, 256 |
| AGC / global clip | enabled, `grad_clip=0.5` |

The W&B state is `crashed`, but there is no uploaded traceback or output log. W&B metadata only
contains the command, host, GPU, and commit. The run reached step 14400 with all health flags clean,
so the state is read as external/manual termination near the end, not as a training failure.

## 2. Protocol verdict table

| Q | Question | Result | Evidence |
|---|---|---:|---|
| Q1 | Training alive and correct mode? | PASS | `present_recon_only=1`, `prediction_active=0`, `L_flow=0`, `L_recon_pred=0` for all rows. `grad_skipped=0`, `grad_has_nan=0`, `instability_warn=0`; max `grad_norm=1.72`, final `0.166`. |
| Q2 | `c_t` alive / video-specific? | PASS | Last diag `c_std_mean=0.984`, `c_dead_dim_frac=0`, `c_cross_video_cosine=0.091`. |
| Q3 | Rich latent? | FAIL | Last diag `c_effective_rank=49.60`, plateau `49.60`, below the `>60` gate. `c_slot_diversity_rank=5.08/32` is low. |
| Q4 | Present reconstruction learned content? | PASS | `L_recon_present` fell `0.986 -> 0.345`; `recon_scale=1` after warmup; `agc_D_clipped=0` throughout. |
| Q5 | Geometry and reconstruction helping? | PARTIAL | Reconstruction and video-specificity are strong, but rank stalls below gate and slots are redundant. |
| Q6 | Verdict | LOW-RANK DECODABLE | Strong content reconstruction and healthy video identity, but representation geometry is still too compressed/redundant for the strong-present-representation label. |

Do not assign `Passing prediction` or `Healthy rep, no predictor`; those are full-prediction labels.

## 3. Final metrics

Final diagnostic step is 14000. Plateau means diagnostic steps 13000, 13500, and 14000.

| Metric | Final diag | Last-3 plateau | Interpretation |
|---|---:|---:|---|
| `L_recon_present` | 0.3452 | 0.3454 | Strong present reconstruction; slightly better than Run C's 0.3464. |
| `c_effective_rank` | 49.60 | 49.60 | Below `>60`; low-rank decodable. |
| `c_cross_video_cosine` | 0.0911 | 0.0917 | Excellent video-specificity. |
| `c_std_mean` | 0.9842 | 0.9838 | Healthy variance near target. |
| `c_std_median` | 0.9791 | 0.9789 | Healthy median spread. |
| `c_dead_dim_frac` | 0.0 | 0.0 | No dead dimensions by this metric. |
| `c_slot_diversity_rank` | 5.08 | 5.07 | Low slot diversity; slots are redundant. |
| `c_attn_entropy` | 0.7747 | 0.7746 | Attention became sharper than init, not uniform. |
| `c_attn_entropy_min` | ~0 | ~0 | At least one attention head/slot is extremely sharp. |
| `L_sigreg` | 0.00213 at 14000 / 0.00267 at 14400 | 0.00247 train last-3 | SIGReg active and minimized. |
| `L_var` | 0.0154 at 14000 / 0.0151 at 14400 | 0.0153 train last-3 | Variance floor mostly satisfied but still lightly active. |
| `grad_norm` | 0.1609 at 14000 / 0.1658 at 14400 | 0.1722 train last-3 | Small, stable gradients. |
| `grad_skipped` | 0 | 0 | No skip spiral. |
| `grad_has_nan` | 0 | 0 | No NaN signal. |
| `instability_warn` | 0 | 0 | No instability warning. |
| `agc_B_clipped` | 0 | 0 | Bottleneck gradients not clipped. |
| `agc_D_clipped` | 0 | 0 | Decoder gradients not clipped. |

## 4. Trajectory

Selected diagnostic points:

| step | `L_recon_present` | rank | cross-video cos | std mean | slot rank | attn entropy | attn min | grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9858 | 9.47 | 0.724 | 0.495 | 16.71 | 0.9999 | 0.9998 | 1.580 |
| 500 | 0.4483 | 11.36 | 0.148 | 0.884 | 19.87 | 1.0000 | 0.9999 | 0.608 |
| 1000 | 0.4358 | 12.01 | 0.095 | 0.925 | 20.74 | 1.0000 | 0.9999 | 0.533 |
| 2000 | 0.4285 | 12.35 | 0.099 | 0.957 | 13.22 | 0.9999 | 0.9995 | 0.505 |
| 3000 | 0.3810 | 13.60 | 0.100 | 0.967 | 2.07 | 0.9987 | 0.9508 | 0.400 |
| 4000 | 0.3592 | 19.34 | 0.123 | 0.962 | 1.61 | 0.9233 | 0.0089 | 0.326 |
| 6000 | 0.3530 | 31.36 | 0.077 | 0.991 | 3.29 | 0.8392 | ~0 | 0.217 |
| 8000 | 0.3499 | 42.53 | 0.089 | 0.985 | 4.60 | 0.7831 | ~0 | 0.192 |
| 10000 | 0.3471 | 46.56 | 0.100 | 0.979 | 4.84 | 0.7729 | ~0 | 0.190 |
| 12000 | 0.3458 | 48.94 | 0.097 | 0.981 | 5.03 | 0.7757 | ~0 | 0.173 |
| 13000 | 0.3455 | 49.59 | 0.093 | 0.983 | 5.05 | 0.7743 | ~0 | 0.169 |
| 13500 | 0.3454 | 49.63 | 0.091 | 0.984 | 5.07 | 0.7747 | ~0 | 0.185 |
| 14000 | 0.3452 | 49.60 | 0.091 | 0.984 | 5.08 | 0.7747 | ~0 | 0.161 |

Read:

1. The reconstruction objective works early and strongly: most of the improvement happens by
   step 4000, then it slowly polishes to 0.345.
2. Geometry improves much later than reconstruction: rank climbs from ~13 at 3000 to ~50 by the end.
3. The representation is not video-independent: cross-video cosine stays exceptionally low.
4. Slot rank collapses early and only recovers to ~5/32. The code uses many feature directions
   globally, but not many independent slots within each video.

## 5. Comparison to relevant runs

| Run | Mode | Decoder / recon | Rank | `L_recon_present` | Cross-video cos | Std mean | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| `original_recon_loss + no-pred` (`kttd1fib`) | present-only | learned-query, `relative_mse`, no var/SIGReg | 10.46 | 0.5149 (relative MSE) | 0.864 | 0.347 | Collapsed rep |
| `inv011_fixed_position_decoder` (`io74f32b`) | full prediction | fixed-position, cosine | 50.76 | 0.3464 | 0.165 | 1.005 | Low-rank rep + predictor fail |
| `new_recon_loss` (`1u69hpfm`) | full prediction | learned-query, cosine | 60.35 | 0.3459 | 0.147 | 1.013 | Healthy rep, no predictor |
| `soft-universe-37` (`2vbo6pbm`) | full prediction | learned-query, relative MSE | 61.08 | 0.5713 (relative MSE) | 0.162 | 1.005 | Healthy rep, no predictor |
| `inv011_fixed_position_present_recon` (`hcr2qx19`) | present-only | fixed-position, cosine, var+SIGReg | 49.60 | 0.3452 | 0.091 | 0.984 | Low-rank decodable |

Important comparisons:

- Versus old present-only Run B, Run D is a major success: std/cosine/rank all improve sharply.
- Versus Run C, removing prediction does not lift rank. Run C finished at rank ~50.8; Run D
  finishes at ~49.6. Therefore Run C's rank loss is not caused by `F_c`.
- Versus learned-query full runs, fixed-position decoding is associated with a roughly 10-rank-point
  lower plateau under the current `lambda_sigreg=5` / `lambda_recon=0.05` recipe.

## 6. What succeeded

### 6.1 Present reconstruction is real

`L_recon_present` falls from `0.986` to `0.345`, with `agc_D_clipped=0`. The decoder is not failing
as an optimizer site, and the cosine objective is learnable under fixed-position decoding.

### 6.2 The old present-only collapse is fixed

The old Run B collapsed into a weak, video-independent present representation:

```text
old: rank 10.46, std 0.347, cosine 0.864
new: rank 49.60, std 0.984, cosine 0.091
```

The combination of variance floor + SIGReg + cosine recon turns present-only reconstruction from
collapsed into usable.

### 6.3 The fixed-position decoder can train

The decoder has no learned output-token content table, yet reconstructs as well as Run C's full
fixed-position recipe. This means the fixed-position decoder is viable as a present feature decoder.

## 7. Failure modes

### Failure 1: low-rank decodable representation

The main failure is the protocol verdict:

```text
L_recon_present improves strongly, but c_effective_rank stays below >60.
```

The run ends at rank `49.60`, not a collapse and far above the old rank-13 ceiling, but not a
strong present representation by the current gate.

Architectural tie:

- This is tied primarily to the **Bottleneck `B` representation geometry**, not `F_c`.
- `F_c` is inactive (`prediction_active=0`, `L_flow=0`), so it cannot be the cause.
- The fixed-position `D` can decode the representation, but it does not force `B` to use enough
  independent feature directions.
- SIGReg at `lambda_sigreg=5` shapes the distribution but does not clear the rank gate in this
  fixed-position present-only setting.

### Failure 2: slot redundancy inside `B`

`c_slot_diversity_rank` ends around `5.08/32`, and `c_attn_entropy_min` collapses to near zero.
That means the code is video-specific and decodable, but the 32 slots are not acting like 32
independent carriers.

Architectural tie:

- This is tied to the **Bottleneck learned-query cross-attention** and slot organization.
- SIGReg is applied to pooled rows over `D_c`; it can improve global feature rank without ensuring
  independent per-video slots.
- This should not be "solved" by resurrecting the old slot-diversity loss. Investigation 003 showed
  that direct slot-diversity optimization Goodharted the slot metric while harming representation.

### Non-failure: training stability

Despite W&B state `crashed`, the training metrics do not show a training failure:

```text
grad_skipped=0
grad_has_nan=0
instability_warn=0
grad_norm small and finite
agc_B_clipped=0
agc_D_clipped=0
```

If exact completion to step 15000 matters, rerun or resume operationally. It is not a learning
diagnosis.

### Non-failure: prediction

There is no prediction result here. Copy ratio, batch-mean ratio, and `L_recon_chat` are intentionally
absent. Do not use this run to claim `F_c` improved or failed.

## 8. Diagnosis

The run proves that the bottleneck/decoder channel can carry useful present content:

```text
c_t is video-specific, spread, decodable, and stable.
```

But it also proves that this alone does not guarantee a fully rich abstract representation:

```text
c_t remains low-rank relative to the >60 gate and slot-redundant relative to N_c=32.
```

So the current failure is upstream of `F_c` but narrower than "B cannot represent content." The
more precise failure is:

```text
B can encode decodable present content, but it compresses that content into too few effective
directions and too few independent slots under the fixed-position D + cosine + SIGReg recipe.
```

## 9. Recommended fixes / next experiments

### 9.1 Add dependency diagnostics before changing the architecture

Add diagnostic-only readouts:

```text
L_recon_present_real      = D(c_t) vs e_t
L_recon_present_shuffled  = D(c_t shuffled across batch) vs e_t
L_recon_present_zero      = D(0) vs e_t
```

Interpretation:

- large real-vs-shuffled/zero gap: `D` genuinely uses video-specific `c_t`; focus on bottleneck
  geometry.
- small gap: even fixed-position `D` still reconstructs too much from shared decoder priors/biases;
  fix decoder dependence before trusting `L_recon_present`.

This is the cleanest next step because current `L_recon_present` alone cannot tell how much
information came from `c_t` versus shared decoder defaults.

### 9.2 If the dependency gap is strong: tune geometry, not prediction

Run a small present-only geometry ladder, keeping the same fixed-position decoder and cosine loss:

```text
lambda_sigreg in {7.5, 10.0}
lambda_recon = 0.05
lambda_var = 0.5
present_recon_only = true
```

Success criterion:

```text
rank > 60 while L_recon_present stays around 0.345 and cross-video cosine remains < 0.5
```

This is not a prediction recipe; it is a controlled test of whether the present bottleneck can clear
the rank gate under stronger isotropy pressure without losing content.

### 9.3 If slot redundancy remains: change slot routing, not the old slot loss

Do not restore the rejected slot-diversity training loss from investigation 003. Instead, investigate
architecture-level slot routing:

- fixed or semi-fixed spatiotemporal slot anchors in `B`;
- local or grouped cross-attention from slots to V-JEPA tubelets;
- diagnostics for per-slot token coverage and per-slot contribution to reconstruction.

The goal is to make slots specialize by construction or routing pressure, not by optimizing a
standalone slot-rank metric that can Goodhart.

### 9.4 If the dependency gap is weak: fix decoder bypass

If `D(shuffled c_t)` or `D(0)` scores close to `D(c_t)`, then fixed-position decoding still has a
template path through shared weights/biases. Possible fixes:

- add a shuffled-c reconstruction margin: real `c_t` must reconstruct better than shuffled `c_t`;
- residualize against a dataset-level feature mean and make `D(c_t)` predict only video-specific
  residual detail;
- reduce decoder bias/template capacity until the real-vs-shuffled gap is large.

## 10. Bottom line

Run D is a useful partial win, not a clean success.

It resolves one ambiguity: the present bottleneck/decoder path is not broken. With fixed-position
decoding, cosine reconstruction, SIGReg, and the variance floor, `c_t` becomes stable,
video-specific, and strongly decodable.

It also exposes the remaining present-side bottleneck: the representation is still too compressed
and slot-redundant. The clean measurement move would be to add shuffled/zero-c dependency
diagnostics, then either tune present-only geometry or adjust bottleneck slot routing depending on
whether `D` is actually using `c_t`. The operator has chosen to skip that diagnostic path and move
directly to active experiment changes.

## 11. 2026-07-02 follow-up: three active recommendations

The follow-up question was whether to run a broader `lambda_sigreg` sweep or make a more structural
change. I rechecked this run, the prior SIGReg sweep history, the present-only code path, and
external literature on variance/covariance regularization, slot representations, and masked
reconstruction. The result is: run a sweep, but make it a structured geometry sweep instead of
repeating the old low-value `lambda_sigreg` range.

### Recommendation 1 - Run a structured present-only geometry sweep

Run a present-only sweep over the active band:

| Axis | Values | Rationale |
| --- | --- | --- |
| `lambda_sigreg` | `5.0, 7.5, 10.0, 12.5` | Inv008 showed `0.3` and `1.0` were inert, `3.0` started moving rank, and `10.0` strongly lifted rank. Inv009 showed `5.0-6.0` is near the current rank gate. The current run at `5.0` reaches only rank `49.6`, so the useful range starts at the current value, not below it. |
| `lambda_cov` | `0.0, 0.003, 0.01` | `L_cov` is already logged and remains nontrivial at the end of this run (`~3.3`) but is not optimized. A coefficient of `0.003` contributes about `0.010` loss late; `0.01` contributes about `0.033`, roughly on the same scale as the active present-only loss terms. |
| `lambda_recon` | keep `0.05` first | Do not mix in a reconstruction-weight change until the geometry regularizers are isolated. |

This is a 12-run sweep. If runtime pressure appears later, the reduced 8-run version is
`lambda_sigreg={5,7.5,10,12.5}` crossed with `lambda_cov={0,0.01}`.

Success criteria:

- `c_effective_rank >= 60` sustained late, not just a transient.
- `c_cross_video_cosine < 0.30` preferred, `<0.50` hard ceiling.
- `c_std_mean` remains in the `0.8-1.2` range.
- `L_recon_present <= 0.36`, so the rank gain is not bought by losing present-token information.
- `c_slot_diversity_rank` should not regress below the current late value of `~5`; ideally it moves
  above `8`.

Why this is first:

The current failure is a geometry failure before it is a prediction failure. The bottleneck gets
non-static and reconstructive, but it stops at rank `~50` and uses only about five independent slot
directions. Prior inv008/inv009 evidence says SIGReg is the one regularizer that reliably moves
rank, and the present-only version removes the temporal-prediction confound that made high SIGReg
produce static-c in older runs. The new part is adding a small covariance axis, because this
repository already implements and logs `L_cov`, but the clean present-only substrate has not tested
it as an active term.

External support:

- [VICReg](https://arxiv.org/abs/2105.04906) frames variance and covariance regularization as
  complementary anti-collapse terms.
- [VCReg](https://arxiv.org/abs/2306.13292) supports variance-covariance regularization as a way to
  improve feature diversity and avoid collapsed or redundant features.

Implementation note:

Do not include `lambda_slot` in this sweep. Investigation 003 showed slot-diversity loss can
Goodhart the slot-rank diagnostic while damaging global rank, cosine geometry, and gradients.
`lambda_cov` is the safer redundancy lever because it acts on feature covariance rather than
directly rewarding the slot-rank diagnostic.

### Recommendation 2 - Build an anchored-slot bottleneck if slot rank stays low

If the sweep raises global rank but `c_slot_diversity_rank` remains near `5`, the next architectural
change should be an anchored-slot bottleneck.

Current bottleneck behavior:

- `Bottleneck` uses learned global query slots.
- Every slot can attend to the whole mixed token grid.
- There is no fixed coverage prior, local attention window, or competition mechanism forcing
  different slots to own different spatiotemporal regions.
- In this run, slot rank starts around `16-20`, collapses to `~1.6` by step `4000`, and only recovers
  to `~5.1` late.

That means the system has learned a useful global representation, but not a healthy multi-slot
representation. The bottleneck has 32 slots in shape, but only about five independent slot
directions in practice.

Proposed architecture:

- Add `bottleneck_slot_mode=global|anchored`.
- For `anchored`, initialize 32 slots as fixed spatiotemporal anchors over the detailed-token grid,
  for example 2 temporal groups x 4 x 4 spatial cells.
- Give each anchor a fixed 3D position code.
- Limit each anchor's first cross-attention to its local tubelet neighborhood, with an optional
  radius expansion.
- Add one lightweight slot self-attention mixer after local collection so slots can coordinate after
  they have distinct coverage.
- Log per-slot attention coverage entropy and per-anchor token mass, in addition to the existing
  rank metrics.

Why this is second:

This attacks the architectural element tied to the clearest remaining failure: global learned slots
are redundant. It avoids reactivating `lambda_slot`, which already failed historically, and instead
changes the routing prior that lets all slots collapse onto the same few global factors.

External support:

- [Slot Attention](https://arxiv.org/abs/2006.15055) shows that slot structure becomes useful when
  slots compete to explain different parts of the input.
- [SAVi](https://arxiv.org/abs/2111.12594) shows the value of conditioning slot state on spatial cues
  for video object representations.
- [SAVi++](https://arxiv.org/abs/2206.07764) extends object-centric video slot learning with stronger
  video cues.
- [SlotFormer](https://arxiv.org/abs/2210.05861) supports the broader direction that slots should
  become entity- or factor-like carriers before downstream dynamics are expected to work.
- [TokenLearner](https://arxiv.org/abs/2106.11297) is another compact-token precedent: adaptive
  token selection works best when the model is pressured to allocate a small set of tokens to
  distinct useful content.
- [Perceiver IO](https://arxiv.org/abs/2107.14795) supports the cross-attention latent-array shape
  already used here, but the current implementation needs stronger slot routing structure for this
  specific bottleneck problem.

Decision rule:

Run this if Recommendation 1 produces `c_effective_rank >= 60` but `c_slot_diversity_rank` remains
below `8-10`, or if rank improves only by inflating global covariance directions while slots remain
redundant.

### Recommendation 3 - Add masked feature reconstruction as the next objective change

If the sweep and anchored slots still leave the representation too low-rank or too slot-redundant,
change the present reconstruction task itself.

Current objective:

- `B` sees the full detailed representation for the present timestep.
- `D` reconstructs all present detailed tokens from the bottleneck.
- Fixed output positions prevent the old learned-query decoder shortcut, but the task can still be
  solved with a small number of broad, shared latent factors.

Proposed objective:

- Add `bottleneck_token_mask_ratio`, starting with `0.50`.
- Randomly mask or drop a subset of detailed tokens before `B`.
- Decode from `c_t` to the original unmasked detailed tokens.
- Compute reconstruction on the masked or held-out target positions first; optionally include a
  smaller full-token auxiliary reconstruction term.
- Sweep mask ratio later over `0.50, 0.65, 0.75` only after the basic path is stable.

Why this is third:

This changes the training problem rather than only the regularization. The model must use the
bottleneck to integrate context and recover missing feature-space content, instead of compressing a
fully visible detailed tensor into a few global factors. It is more invasive than Recommendation 1
and should come after the geometry sweep, but it is likely more impactful than simply increasing
slot count or reusing the old slot-diversity loss.

External support:

- [MAE](https://arxiv.org/abs/2111.06377) supports the idea that masked reconstruction prevents
  trivial dense autoencoding and encourages useful representation learning.
- [VideoMAE](https://arxiv.org/abs/2203.12602) shows that video representation learning can benefit
  from high masking ratios because temporal redundancy otherwise makes reconstruction too easy.

Success criteria:

- Rank improves without a large recon penalty: `c_effective_rank >= 60`, `L_recon_present <= 0.38`
  at first.
- Slot diversity improves relative to this run: `c_slot_diversity_rank > 8` preferred.
- No return to static-c behavior: cosine remains below the hard ceiling.
- No instability: `grad_skipped=0`, `grad_has_nan=0`, `instability_warn=0`.

## 12. Updated recommendation order

1. Run the structured present-only geometry sweep: `lambda_sigreg x lambda_cov`, holding
   `lambda_recon=0.05`.
2. If global rank improves but slot rank remains low, implement anchored-slot bottleneck routing.
3. If dense present reconstruction still admits low-rank solutions, add masked feature
   reconstruction.

The main failure in this run is not that the bottleneck is static or collapsed. It is that the
fixed-position present-reconstruction setup produces a usable but still undercomplete
representation: global rank plateaus below the target and 32 slots behave like roughly five
independent carriers. The recommendations above target that failure in increasing order of
invasiveness.
