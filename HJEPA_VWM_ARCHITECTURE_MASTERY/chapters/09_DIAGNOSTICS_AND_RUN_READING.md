# 09 — Diagnostics and how to read a run

Diagnostics are not decorative plots. They define independent failure gates for optimization,
representation geometry, temporal change, conditional prediction, and decoder honesty.

## Diagnostic execution contract

- ordinary train metrics log every 50 steps;
- expensive diagnostics log every 500 steps;
- the same fixed validation batch, at most 16 and greater than one, is reused;
- diagnostic RNG seed is `base*1,000,003+900,001`;
- `torch.random.fork_rng` restores the training stream;
- `F_c` receives an explicit all-false dropout mask;
- all code runs under no-gradient where appropriate;
- validation never updates the reconstruction mean.

Consequently, compare diagnostic values at the same step and on the same run type.

## Representation metrics

### `c_std_mean`, `c_std_median`, `c_dead_dim_frac`

Flatten each video to 8,192 coordinates and compute population standard deviation across the fixed
batch. A dimension is “dead” when its std is less than 10% of the median std.

Project reading guidance:

```text
c_std_mean ≈ 0.8–1.2
c_dead_dim_frac ≈ 0
```

The 1.0 target comes from the variance floor. These are heuristics/gates, not mathematical
guarantees.

### `c_cross_video_cosine`

Flatten and L2-normalize every video's full abstract tensor, then average off-diagonal pairwise
cosines. Below 0.5 is the project health guide; drift toward 0.7–1.0 signals video-independent
directional collapse.

This metric assumes examples correspond to different videos. On historical EGO fixed batches, the
lexically first 16 chunks could all share one source UID, weakening the “cross-video” interpretation.

### `c_effective_rank`

Pool batch and slots to `(B*N_c,D_c)`, center, form sample covariance, eigendecompose, normalize
eigenvalues to probabilities, and report:

```text
exp(-Σ p_i log(p_i+1e-8))
```

Maximum meaningful feature rank is `D_c=256`; current project gate is above 60. Non-finite covariance
returns NaN rather than crashing the run.

### `c_plus_*`

`c_plus_std_mean`, `c_plus_std_median`, and `c_plus_effective_rank` repeat key checks on the EMA
target. A healthy online representation with lagging target rank still gives `F_c` a weak moving
regression target.

### Slot diversity ranks

For each video, form a slot Gram matrix and calculate entropy effective rank:

- `c_slot_diversity_rank`: raw slot vectors;
- `c_slot_diversity_rank_centered`: subtract per-video mean slot first.

The centered value is the cleaner content-diversity readout. Fixed slot identities can mechanically
inflate raw rank.

### Attention entropy

On final bottleneck cross-attention, for every batch/head/slot:

```text
H = -Σ a_i log(a_i) / log(N_e)
```

`c_attn_entropy` is the mean; `c_attn_entropy_min` is the most selective head/slot. One means uniform
attention. It is measured per head because head-averaged attention can make several different sharp
heads look uniform.

Low entropy is not automatically better: indiscriminate sharpness can be as unhelpful as uniform
averaging.

## Coarse baselines

Diagnostics use the same sampled flow target/noise/time for model and fixed baselines.

### Model

```text
coarse_model_loss = MSE(F_c(z_τ,τ,c_t), target_velocity)
```

### Copy

Ordinary mode uses velocity `c_t-ε`. Residual mode uses `-ε`, equivalent to predicting zero temporal
residual. In both cases:

```text
coarse_copy_loss = mean((c_t-c⁺)^2) = mean(Δ²)
```

This contains no copy network and is never optimized. A decreasing copy loss means the learned
representation is making present and future more similar, not that the predictor improved.

### Batch mean

Replace every example's target with the fixed diagnostic batch's mean target:

```text
coarse_batch_mean_loss
```

It tests whether per-example prediction improves over a generic prototype.

### Ratios

```text
coarse_vs_copy_ratio       = model_loss / max(copy_loss,1e-8)
coarse_vs_batch_mean_ratio = model_loss / max(mean_loss,1e-8)
```

Phase 1 full-prediction acceptance gates:

```text
copy ratio       <= 0.70
batch-mean ratio <= 0.50
```

These must hold over a stable late window, not one lucky point.

## Reconstruction honesty

| Metric | Latent decoded | Target |
|---|---|---|
| `L_recon_present` | online `c_t` | present `e_t` |
| `L_recon_cplus` | true EMA `c⁺` | future `e⁺` |
| `L_recon_chat` | one-step predicted `c_hat` | future `e⁺` |
| `L_recon_shuffled_c` | another batch example's `c_t` | present `e_t` |
| `L_recon_video_gap` | `shuffled - present` | derived |

`torch.roll(...,shifts=1,dims=0)` creates deterministic shuffled pairs. Batch size must exceed one so
it is not identity.

Interpretation:

- large positive video gap: decoder uses video-specific latent information;
- near-zero gap: template collapse/reconstruction blindness;
- `L_recon_chat > L_recon_cplus`: prediction lands in a worse-decodable place than the teacher;
- `L_recon_chat ≈ L_recon_cplus` while copy ratio is bad: decoder cannot expose predictor error.

## Gradient metrics

| Metric | Meaning |
|---|---|
| `grad_norm` | post-AGC norm returned before global clip's rescale; primary spike signal |
| `grad_global_norm_postclip` | norm after AGC and global clipping; often near 0.5 |
| `grad_skipped` | optimizer and EMA transition was rejected |
| `grad_has_nan` | diagnostic scan found NaN gradients |
| `instability_warn` | `grad_norm>30` and `L_flow>1` |
| `agc_B/Fc/D_*` | eligible tensors clipped and worst bound ratio |

Repeated skips, any sustained NaNs, or a late spike followed by flat metrics invalidates the later
portion of a run.

## Eight-question cycle for full prediction

1. **Stability:** did it train to completion without NaNs, skip spirals, or frozen metrics?
2. **Collapse:** is std near one, dead fraction near zero, and cross-video cosine below 0.5?
3. **Rank:** is online effective rank above 60 and is EMA rank keeping up?
4. **Static `c`:** did copy loss rise, or did the representation make present/future nearly equal?
5. **Copy gate:** does `F_c` beat copy by at least 30%?
6. **Batch-mean gate:** does it beat the per-batch prototype by at least 50% at the same time?
7. **Reconstruction blindness:** do true and predicted future readouts meaningfully separate?
8. **Verdict:** assign one explicit label.

Verdict labels:

| Label | Pattern |
|---|---|
| Invalid | stability fails |
| Collapsed representation | Q2 fails |
| Low-rank representation | Q3 fails |
| Static-`c` trap | copy loss falls and ratio remains near one |
| Healthy representation, no predictor | representation passes; copy gate fails |
| Passing prediction | representation/stability pass; both prediction ratios pass |
| Amazing | passing prediction, honest reconstruction, no late instability |

## Static-versus-prediction matrix

| Copy-loss trend | Copy ratio | Rank | Story |
|---|---:|---:|---|
| falling | ~1 | rising | richer but static appearance; copy becomes unbeatable |
| rising | ~1 | healthy | latent dynamics exist; predictor did not learn them |
| rising | `<0.7` | healthy | desired dynamic, forecastable state |
| any | `>1` | any | network is worse than the lazy copy baseline |

## Present-only cycle

Do not apply copy or batch-mean gates when `present_recon_only=True`. Ask instead:

1. Is the run actually in present-only mode and stable?
2. Is `c_t` alive and video-specific?
3. Is rank above 60, with centered slot diversity and nonuniform attention interpreted cautiously?
4. Does present reconstruction fall?
5. Is the shuffled-video gap positive, proving video-specific decode?
6. Are geometry improvements purchased without collapse?
7. Does the experiment isolate its proposed axis?
8. Verdict: invalid, collapsed, pretty-geometry/weak-content, template reconstruction, or useful
   representation.

## Known EGO diagnostic limitation

EGO chunks are named by source UID plus chunk index. Lexical ordering can fill the fixed validation
batch with adjacent chunks from one source. Then:

- cross-video cosine is actually cross-chunk same-source cosine;
- shuffled reconstruction may pair adjacent chunks from the same source;
- temporal differences may be easier/more correlated than globally diverse videos.

The scientifically correct repair is a deterministic source-diverse fixed batch, not relabeling the
existing metrics.

## Never conclude from one curve

- Falling `L_flow` does not prove beating copy.
- High rank does not prove temporal dynamics.
- Low cross-video cosine does not prove good slots.
- Low reconstruction does not prove video specificity.
- Beating batch mean does not prove beating copy.
- A low copy loss is not a goal.
- AGC activity does not itself mean instability; inspect actual norm and skips.
