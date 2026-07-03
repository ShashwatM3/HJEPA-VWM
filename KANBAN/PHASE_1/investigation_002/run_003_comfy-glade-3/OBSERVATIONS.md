# Observations - run 003 `comfy-glade-3`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `0mgmqxxi`  
**State:** `finished`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=finished; step=199; grad_skipped max=n/a; grad_has_nan max=0; grad_norm last=0.5805 |
| Q2 | c_t alive / video-specific? | FAIL | std=0.4984; dead_dim=0; cross_video_cosine=0.7175 |
| Q3 | Rich latent? | FAIL | c_effective_rank=8.9986; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=flat; ratio=171.5862; story=insufficient temporal evidence or early run |
| Q5 | F_c beats copy? | FAIL | model_loss=3.3071; copy_loss=0.0193; ratio=171.5862; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=12.2494; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Smoke / inconclusive | The run is too short for Phase 1 learning gates; use it as infrastructure evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 2.9153 @ 0 | 3.1503 @ 199 | 0.235 | 2.5388 / 2.9952 / 3.462 | n/a |
| `L_flow` | 2.8726 @ 0 | 3.1362 @ 199 | 0.2636 | 2.5269 / 2.9681 / 3.4195 | n/a |
| `L_var` | 0.4267 @ 0 | 0.1407 @ 199 | -0.286 | 0.1138 / 0.2718 / 0.4491 | n/a |
| `c_std_mean` | 0.4984 @ 0 | 0.4984 @ 0 | 0 | 0.4984 / 0.4984 / 0.4984 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7175 @ 0 | 0.7175 @ 0 | 0 | 0.7175 / 0.7175 / 0.7175 | n/a |
| `c_effective_rank` | 8.9986 @ 0 | 8.9986 @ 0 | 0 | 8.9986 / 8.9986 / 8.9986 | n/a |
| `coarse_copy_loss` | 0.0193 @ 0 | 0.0193 @ 0 | 0 | 0.0193 / 0.0193 / 0.0193 | n/a |
| `coarse_model_loss` | 3.3071 @ 0 | 3.3071 @ 0 | 0 | 3.3071 / 3.3071 / 3.3071 | n/a |
| `coarse_batch_mean_loss` | 0.27 @ 0 | 0.27 @ 0 | 0 | 0.27 / 0.27 / 0.27 | n/a |
| `coarse_vs_copy_ratio` | 171.5862 @ 0 | 171.5862 @ 0 | 0 | 171.5862 / 171.5862 / 171.5862 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.2494 @ 0 | 12.2494 @ 0 | 0 | 12.2494 / 12.2494 / 12.2494 | n/a |
| `grad_norm` | 0.3553 @ 0 | 0.5805 @ 199 | 0.2252 | 0.2681 / 0.3576 / 0.5805 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 1.000e-04 @ 0 | 0.02 @ 199 | 0.0199 | 1.000e-04 / 0.01 / 0.02 | n/a |

## Last Key Metrics

`c_effective_rank`=8.9986 @ 0; `c_cross_video_cosine`=0.7175 @ 0; `c_std_mean`=0.4984 @ 0; `coarse_vs_copy_ratio`=171.5862 @ 0; `coarse_vs_batch_mean_ratio`=12.2494 @ 0

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 171.5862 and the latest batch-mean ratio is 12.2494; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 8.9986, cross-video cosine is 0.7175, and copy loss is 0.0193. The reading-cycle verdict is **Smoke / inconclusive** because The run is too short for Phase 1 learning gates; use it as infrastructure evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

It established infrastructure behavior. The result should be used to trust launch/logging mechanics, not to infer learning quality.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — comfy-glade-3

## Outcome

**Passed** as a dense-logged smoke. Both pre-fix (this run, `efficient-aardvark-2`)
and post-fix (`charmed-haze-4`) smokes produce bit-identical diagnostics on the
deterministic seed-42 batch, so this run carries no separate throughput conclusion —
its value was confirming the `--log-every` flag worked and the per-step trajectory
was sane.

## Evidence

- ~5m29s runtime for 200 steps, i.e. still the ~1.6 s/step pre-fix regime
  (selective decode `5e78caa` had not yet landed on 06-09).
- Step-0 diagnostics identical to `efficient-aardvark-2` (same seed-42 diag batch):
  `L_flow≈2.87`, `L_var≈0.43`, `c_effective_rank≈9.0`, `c_cross_video_cosine≈0.72`,
  `coarse_vs_copy_ratio≈171` at init.
- Per-step `loss`/`L_flow` jitter visible but no downward trend over 200 steps — as
  expected for a throughput/plumbing smoke, not a training run.

## Interpretation

Operational check only; the substantive throughput comparison is
[`efficient-aardvark-2`](../run_002_efficient-aardvark-2/) (pre-fix, 1.66 s/step) vs
[`charmed-haze-4`](../run_004_charmed-haze-4/) (post-fix, 1.41 s/step). The identical
diagnostics across all three smokes are the evidence that the dataloader refactor was
**behavior-preserving** — the diag batch is deterministic, so any change in model
inputs would have moved these numbers.
