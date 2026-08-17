# Reading experiment runs (W&B)

> **Purpose:** A fixed **reading cycle** for HJEPA-VWM Phase 1 runs. Read the run in
> order, using at most four metric panels per question. When all questions are answered, you have a
> full end-to-end picture of the run, grounded in **our** failure modes rather than generic ML
> checklists.
>
> **Metric definitions:** [`GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`](PROBLEMS_METRICS_AND_EXPERIMENTS.md)
> (especially the coarse / copy section).
>
> **Agents:** use skill `[.agents/skills/read-wandb-run/SKILL.md](../.agents/skills/read-wandb-run/SKILL.md)`
> whenever analyzing one or more runs.

---

## How to use this

1. **Check run config first** (`present_recon_only`, `predict_residual`, `lambda_sigreg`,
  `lambda_recon`, `lambda_recon_pred`, `lambda_rollout`, `rollout_ramp_steps`, dataset,
  step count, horizon).
2. Choose the correct cycle:
  - if `present_recon_only=false`, use **Reading Cycle A — Full Prediction Runs**;
  - if `present_recon_only=true`, use **Reading Cycle B — Present Reconstruction Only Runs**.
3. Walk the selected cycle in order. Do not skip ahead to the metric you already care about.
4. If the first stability question fails, note it and continue only for diagnosis. Do not call the run healthy.
5. **One run:** answer each question, then assign the cycle's verdict label.
6. **Several runs:** complete the relevant cycle per run, then compare verdict labels and the
  metrics that belong to those run types. Do not average metrics across runs before labeling each
  run.

**Data sources:** W&B project `hjepa-vwm`, or locally:

```bash
python run_history.py --run <run_id> --report
python run_history.py --run <run_id> --format parse_logs -o logs/<name>/output.log
```

Core diagnostic metrics (`c_*`, prediction baselines, active `L_recon_*`) log every
`diag_every` steps (default 500). Training health and the fixed rollout schema
(`loss/rollout`, `rollout/*`) log every `log_every` steps (default 50). Internal
debug variants are not forwarded to W&B.

## How to use — humans

1. **Understand what your run was supposed to test** (hypothesis, config diff vs baseline, what
  would count as success or failure).
2. **Walk each question in order.** For each question:
  - note down what is off;
  - write your own understanding in plain language, not just the metric value.
3. **Synthesize the notes:**
  - **First → understand what happened** (one coherent story of the run end-to-end);
  - **Second → brainstorm** (what to try next, what lever to move, what to compare against).

---

## Reading Cycle A — Full Prediction Runs

Use this cycle when `present_recon_only=false`. These runs train the future-prediction path, so the
copy and batch-mean gates are part of the verdict.

### Q1 — Did the run actually train, or did it die / freeze?

This question checks whether optimization stayed alive for the whole run. The failure modes we care
about here are late explosions, grad-skip spirals, and NaN corruption of the weights.

| Metric | Healthy indication |
|---|---|
| `grad_skipped` | Stays at **0** for the entire run, meaning optimizer steps were not skipped. |
| `grad_has_nan` | Stays at **0** for the entire run, meaning gradients did not become NaN. |
| `grad_norm` | Stays finite with no late cliff into the hundreds; large spikes preceded past blow-ups. |
| `lr_mult` | Use this for context only; note whether a blow-up coincided with peak learning rate. |

**W&B panel search:**

```text
grad_skipped|grad_has_nan|grad_norm|lr_mult
```

If you see repeated skips, any sustained NaN signal, or a sharp late spike after which other metrics
flatline, treat the run as invalid from that point forward and do not trust later panels.

---

### Q2 — Is `c_t` alive and video-specific, or collapsed?

This question checks whether the abstract latent has healthy variance and whether different videos
land in different places in latent space. The failure modes are a weak variance floor and
video-independent collapse, where every clip points in nearly the same direction.

| Metric | Healthy indication |
|---|---|
| `c_std_mean` | Roughly **0.8–1.2**, near the variance-floor target of 1.0. |
| `c_dead_dim_frac` | Roughly **0**, meaning no large fraction of dimensions has near-zero std. |
| `c_cross_video_cosine` | **Below 0.5**; strong inv010-style runs are often around **0.15–0.3**. |

**W&B panel search:**

```text
c_std_mean|c_dead_dim_frac|c_cross_video_cosine
```

If standard deviation is far below 1.0, dead-dimension fraction climbs, or cross-video cosine drifts
toward **0.7+**, the representation has collapsed or become video-independent even if loss curves
look smooth.

---

### Q3 — Is `c_t` rich, or stuck in the rank-13 ceiling?

This question checks whether the bottleneck uses many directions in feature space, and whether the
EMA target branch that `F_c` regresses against keeps up with the online branch.

| Metric | Healthy indication |
|---|---|
| `c_effective_rank` | **Above 60** on online `c_t`; max meaningful feature dimension is 256. |
| `c_plus_effective_rank` | Close to online rank, meaning the EMA target is not lagging badly behind online `B`. |
| `c_std_mean` | Roughly **0.8–1.2** as a sanity check; rank can mislead if std is already dead. |

**W&B panel search:**

```text
c_effective_rank|c_plus_effective_rank|c_std_mean
```

If rank parks around **10–15** despite okay-looking std, you are hitting the historical rank-13
ceiling. If online rank rises but target rank lags badly, `F_c` may still be chasing a weak target.

---

### Q4 — Is `c_t` changing over time, or frozen so “future = present” is almost free?

This question checks whether present and future latents actually differ over time, and whether a
rising rank is real dynamics or static appearance detail. The failure mode is static `c`, where
`coarse_copy_loss` falls while the copy ratio stays near 1.

| Metric | Healthy indication |
|---|---|
| `coarse_copy_loss` | **Rising** over the run usually means more temporal separation. **Falling** while ratio is near 1 often means static `c`. |
| `coarse_vs_copy_ratio` | **Below 0.7** together with rising copy loss is the target pattern. Near 1 with falling copy loss is the static-`c` trap. |
| `c_effective_rank` | **Above 60**; rank alone does not prove temporal dynamics, so read it with copy-loss trend. |

**W&B panel search:**

```text
coarse_copy_loss|coarse_vs_copy_ratio|c_effective_rank
```

Read the three panels together:

| `coarse_copy_loss` trend | `coarse_vs_copy_ratio` | `c_effective_rank` | Story |
|---|---|---|---|
| falling | ~1 | rising | Static `c`: latents look richer but present ≈ future, so copy is unbeatable. |
| rising | ~1 | healthy | Dynamics without predictor: `c` moves but `F_c` did not learn it. |
| rising | < 0.7 | healthy | Target state: future differs from present and `F_c` beats copy. |

Remember that `coarse_copy_loss` is diagnostic only. Training never minimizes it, so a falling copy
loss is usually a warning sign, not evidence of progress.

---

### Q5 — Did `F_c` actually forecast, or only match the lazy guess?

This question is the main Phase 1 outcome: whether the coarse flow beats saying “future equals
present.” The failure modes are the zero-residual tie and `L_flow` dropping while the copy ratio
stays near 1.

| Metric | Healthy indication |
|---|---|
| `coarse_vs_copy_ratio` | **≤ 0.70** and stable across late diagnostic steps, not a single lucky point. |
| `coarse_model_loss` | Clearly **below** `coarse_copy_loss` when the ratio passes. |
| `coarse_copy_loss` | Compare it on the **same step** as model loss; near-equal values mean `F_c` tied copy. |

**W&B panel search:**

```text
coarse_vs_copy_ratio|coarse_model_loss|coarse_copy_loss
```

A ratio near **1.0** means `F_c` is only as good as doing nothing, which fails the Phase 1 gate even
when `L_flow` fell. A ratio above **1.0** means the network is worse than the lazy guess.

---

### Q6 — Does `F_c` use **this** video’s `c_t`, or ignore it?

This question checks whether the model uses video-specific conditioning, as opposed to a generic
batch-average answer. The trap is passing here while still failing Q5: some conditioning without
real forecasting.

| Metric | Healthy indication |
|---|---|
| `coarse_vs_batch_mean_ratio` | **≤ 0.50**, meaning the model beats predicting the batch-mean future for every clip. |
| `coarse_vs_copy_ratio` | Should also pass Q5 on the **same step**; good batch-mean with copy ≈ 1 is not enough. |

**W&B panel search:**

```text
coarse_vs_batch_mean_ratio|coarse_vs_copy_ratio
```

Beating the batch-mean baseline only shows that `F_c` uses per-video information. It does not by
itself prove that the model forecasts the future better than saying nothing changed.

---

### Q7 — Is reconstruction telling the truth about prediction, or blind?

This question checks whether reconstruction readouts distinguish a bad predicted future from the
true future. The failure mode is all three recon losses sitting nearly equal while the copy ratio is
still bad.

| Metric | Healthy indication |
|---|---|
| `L_recon_present` | Decreases when `lambda_recon > 0`, meaning the present path is learning. |
| `L_recon_cplus` | Serves as the benchmark for how well the decoder reconstructs from the **true** future `c_plus`. |
| `L_recon_chat` | Should sit **above** `L_recon_cplus` when prediction is poor and should drop as `F_c` improves. |

**W&B panel search:**

```text
L_recon_present|L_recon_cplus|L_recon_chat
```

If all three curves are nearly equal, the decoder cannot tell predicted from true future latents, so
reconstruction pressure will not fix a bad copy ratio. Only interpret these panels when
`lambda_recon` or `lambda_recon_pred` is active; they may still log at diagnostic cadence on
baselines.

---

### Q8 — Verdict: what kind of full prediction run was this?

This step has no new W&B panels. Combine Q1–Q7 and assign one label:

| Label | Pattern |
|---|---|
| **Invalid** | Q1 fail. |
| **Collapsed rep** | Q2 fail. |
| **Low-rank rep** | Q3 fail. |
| **Static-`c` trap** | Q4 shows copy loss falling while ratio stays near 1. |
| **Healthy rep, no predictor** | Q2–Q3 pass, but Q4/Q5 show ratio near 1. |
| **Passing prediction** | Q1–Q3 pass, Q5 ≤ 0.70, and Q6 ≤ 0.50. |
| **Amazing** | Passing prediction plus Q7 is not blind and no late instability appears. |

For full prediction runs, Phase 1 acceptance requires:

```text
coarse_vs_copy_ratio <= 0.70
coarse_vs_batch_mean_ratio <= 0.50
```

---

## Reading Cycle B — Present Reconstruction Only Runs

Use this cycle when `present_recon_only=true`. These runs train only the present branch:

```text
x_t -> E -> e_t -> B -> c_t
D(c_t) -> e_hat_t
compare e_hat_t with e_t
```

They do **not** train `F_c`, do **not** use the future clip, and do **not** log prediction baselines.
Success means a rich, video-specific, decodable present representation. It does not mean the model
can forecast.

### Q1 — Did the present-only run actually train, or did it die / freeze?

This question checks whether optimization stayed alive and whether the run really used the
present-only path. The main failure modes are a normal stability failure, a wrong config, or a run
that accidentally activated prediction.

| Metric | Healthy indication |
|---|---|
| `grad_skipped` | Stays at **0** for the entire run, meaning optimizer steps were not skipped. |
| `grad_has_nan` | Stays at **0**, meaning gradients did not become NaN. |
| `grad_norm` | Stays finite with no late cliff into the hundreds. |
| `present_recon_only` | Stays at **1**, confirming the present-only branch is active. |
| `prediction_active` | Stays at **0**, confirming the future-prediction branch is off. |
| `L_flow` | Stays at **0**, because this run should not train `F_c`. |
| `L_recon_pred` | Stays at **0**, because this run should not decode predicted future latents. |

**W&B panel search:**

```text
grad_skipped|grad_has_nan|grad_norm|present_recon_only|prediction_active|L_flow|L_recon_pred
```

If `prediction_active=1`, `L_flow` is nonzero, or `L_recon_pred` is nonzero, stop reading the run as
present-only. That run is a different experiment.

---

### Q2 — Is `c_t` alive and video-specific, or collapsed?

This question checks whether the present bottleneck output has healthy spread and assigns different
videos to different directions. Reconstruction loss can look smooth even when `c_t` becomes too
generic, so read these metrics before trusting `L_recon_present`.

| Metric | Healthy indication |
|---|---|
| `c_std_mean` | Roughly **0.8–1.2**, near the variance-floor target of 1.0. |
| `c_dead_dim_frac` | Roughly **0**, meaning no large fraction of dimensions has near-zero std. |
| `c_cross_video_cosine` | **Below 0.5**; strong runs are often around **0.15–0.3**. |

**W&B panel search:**

```text
c_std_mean|c_dead_dim_frac|c_cross_video_cosine
```

If cross-video cosine rises high, the bottleneck is making many videos point in the same direction.
That is still a bad present representation even if the decoder loss moves.

---

### Q3 — Is `c_t` rich enough, or did reconstruction find a low-rank code?

This question checks whether the bottleneck uses many feature directions while learning to
reconstruct present features. The failure mode is a code that decodes somewhat but stays cramped,
correlated, or slot-redundant.

| Metric | Healthy indication |
|---|---|
| `c_effective_rank` | **Above 60** on online `c_t`. |
| `c_std_mean` | Roughly **0.8–1.2** so rank is not being read on a weak or collapsed latent. |
| `c_slot_diversity_rank` | Higher is better; low values mean the 32 slots are redundant even if feature rank improves. |
| `c_attn_entropy` | Use for context; very high values mean bottleneck attention may still be broad or uniform. |

**W&B panel search:**

```text
c_effective_rank|c_std_mean|c_slot_diversity_rank|c_attn_entropy
```

If `L_recon_present` improves but rank stays below target, reconstruction found a decodable but
low-rank present code. If rank improves but slot rank stays very low, the representation may still
be redundant inside each video.

---

### Q4 — Is present reconstruction actually learning useful content?

This question checks whether the decoder and bottleneck are getting better at reconstructing frozen
present features from `c_t`. This is the main outcome for present-only runs.

| Metric | Healthy indication |
|---|---|
| `L_recon_present` | **Decreases** over training and reaches a materially better value than initialization. |
| `L_recon` | Tracks the active train-step reconstruction loss when `lambda_recon > 0`. |
| `recon_scale` | Reaches **1.0** after warmup, so the reconstruction objective is fully active. |
| `agc_D_clipped` | Should not show sustained heavy clipping; heavy clipping means decoder gradients may be too aggressive. |

**W&B panel search:**

```text
L_recon_present|L_recon|recon_scale|agc_D_clipped
```

Do not compare `L_recon_present` across incompatible loss modes without saying so. A cosine run and
a `relative_mse` run use different reconstruction geometry.

---

### Q5 — Are representation geometry and reconstruction helping each other, or fighting?

This question reads the content objective and the geometry regularizers together. A good present
representation should become more decodable without losing healthy rank, variance, or video
specificity.

| Metric | Healthy indication |
|---|---|
| `L_recon_present` | Falls while the representation metrics stay healthy. |
| `c_effective_rank` | Rises or stays above **60** rather than improving reconstruction through a cramped code. |
| `c_cross_video_cosine` | Stays below **0.5**, so reconstruction does not collapse video identity. |
| `L_sigreg` | Use for context when `lambda_sigreg > 0`; SIGReg should support geometry, not replace content learning. |
| `L_var` | Should shrink when the variance floor is satisfied; persistent high values mean spread is still weak. |

**W&B panel search:**

```text
L_recon_present|c_effective_rank|c_cross_video_cosine|L_sigreg|L_var
```

Read this pattern table:

| `L_recon_present` trend | `c_effective_rank` | `c_cross_video_cosine` | Story |
|---|---|---|---|
| falling | healthy | low | Target state: `c_t` is decodable and has healthy geometry. |
| falling | low | low | Low-rank decodable: reconstruction works, but the code space is still cramped. |
| flat | healthy | low | Pretty geometry, weak content: SIGReg/rank improved but reconstruction did not. |
| falling or flat | any | high | Video-independent collapse: do not trust reconstruction as a useful representation. |

This is the question that decides whether SIGReg, variance floor, and reconstruction are working
together or merely optimizing separate-looking metrics.

---

### Q6 — Verdict: what kind of present-reconstruction run was this?

This step has no new W&B panels. Combine Q1–Q5 and assign one label:

| Label | Pattern |
|---|---|
| **Invalid** | Q1 fail, wrong mode flags, skipped-step spiral, NaNs, or early crash. |
| **Collapsed rep** | Q2 fail: high cross-video cosine, dead dims, or weak std. |
| **Low-rank decodable** | `L_recon_present` improves, but rank stays below the target. |
| **Pretty geometry, weak content** | Rank/std/cosine look healthy, but `L_recon_present` stalls. |
| **Strong present representation** | Q1–Q5 pass: reconstruction improves and `c_t` stays high-rank, spread, and video-specific. |

For present-only runs, do **not** assign `Passing prediction` or `Healthy rep, no predictor`. Those
labels belong to the full prediction cycle. A present-only success means the bottleneck can carry
decodable present information; it does not prove `F_c` can forecast.

---

## Comparing multiple runs

Compare only metrics that belong to the run type.

For **full prediction runs**, compare:

1. `coarse_vs_copy_ratio` at the end or best stable late window.
2. `coarse_vs_batch_mean_ratio` at the same late window.
3. Q4 pattern: did `coarse_copy_loss` fall (static `c`) or rise (more dynamics)?
4. Stability: any Q1 failure disqualifies the run regardless of ratios.
5. Config diffs: `predict_residual`, SIGReg, recon weights, horizon, decoder architecture, dataset.

For **present reconstruction only runs**, compare:

1. `L_recon_present` at the end or best stable late window.
2. `c_effective_rank`, `c_std_mean`, `c_dead_dim_frac`, and `c_cross_video_cosine`.
3. The Q5 pattern: strong present representation, low-rank decodable, or pretty geometry with weak
  content.
4. Stability: any Q1 failure disqualifies the run.
5. Config diffs: `recon_loss_mode`, fixed-position vs learned-query decoder, SIGReg/variance
  weights, decoder size, dataset.

Do **not** declare a sweep winner from `L_flow` alone. Do **not** use copy-ratio gates on
present-only runs.

---

## Why this order

For full prediction runs:

| Step | Why before the next |
|---|---|
| Q1 stability | A blown run invalidates all later metrics. |
| Q2 collapse | Dead `c_t` makes prediction metrics meaningless. |
| Q3 rank | Rank without Q4/Q5 confused inv010. |
| Q4 static `c` | Explains why ratio can stay near 1 even when Q2–Q3 look fine. |
| Q5 copy gate | This is the actual prediction outcome. |
| Q6 batch mean | This separates “uses `c_t`” from “forecasts well.” |
| Q7 recon | This only matters once you know whether prediction failed or passed. |
| Q8 label | This gives one clear story for KANBAN / NEXT_STEPS. |

For present reconstruction only runs:

| Step | Why before the next |
|---|---|
| Q1 stability/mode | A wrong-mode or blown run is not evidence about present reconstruction. |
| Q2 collapse | A generic or collapsed `c_t` makes reconstruction success suspect. |
| Q3 rank | Present reconstruction should not win through a cramped or redundant code. |
| Q4 reconstruction | This is the actual present-branch outcome. |
| Q5 joint read | This tells whether geometry and content objectives helped or fought each other. |
| Q6 label | This gives one clear story for KANBAN / NEXT_STEPS. |

---

## What these cycles deliberately skip

Not required for a full run story unless debugging a **specific** lever:

- `L_cov`, `L_slot`, slot/attention entropy, unless the selected cycle points at representation geometry
  or slot redundancy;
- `agc_*` clips, unless the first stability question points at optimization instability;
- `grad_global_norm_postclip`, because `grad_norm` is the better true-magnitude readout;
- throughput / seconds per step, because it is operations context rather than learning quality.

Add those only when the selected cycle points at a specific subsystem.
