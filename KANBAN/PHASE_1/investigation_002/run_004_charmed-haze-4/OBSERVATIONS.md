# Observations - run 004 `charmed-haze-4`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `gj8ypv0d`  
**State:** `finished`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=finished; step=150; grad_skipped max=n/a; grad_has_nan max=0; grad_norm last=0.2996 |
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
| `loss` | 2.9153 @ 0 | 3.0106 @ 150 | 0.0953 | 2.9153 / 3.0732 / 3.4206 | n/a |
| `L_flow` | 2.8726 @ 0 | 2.9962 @ 150 | 0.1235 | 2.8726 / 3.0417 / 3.3788 | n/a |
| `L_var` | 0.4267 @ 0 | 0.1441 @ 150 | -0.2826 | 0.1441 / 0.3149 / 0.4267 | n/a |
| `c_std_mean` | 0.4984 @ 0 | 0.4984 @ 0 | 0 | 0.4984 / 0.4984 / 0.4984 | n/a |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `c_cross_video_cosine` | 0.7175 @ 0 | 0.7175 @ 0 | 0 | 0.7175 / 0.7175 / 0.7175 | n/a |
| `c_effective_rank` | 8.9986 @ 0 | 8.9986 @ 0 | 0 | 8.9986 / 8.9986 / 8.9986 | n/a |
| `coarse_copy_loss` | 0.0193 @ 0 | 0.0193 @ 0 | 0 | 0.0193 / 0.0193 / 0.0193 | n/a |
| `coarse_model_loss` | 3.3071 @ 0 | 3.3071 @ 0 | 0 | 3.3071 / 3.3071 / 3.3071 | n/a |
| `coarse_batch_mean_loss` | 0.27 @ 0 | 0.27 @ 0 | 0 | 0.27 / 0.27 / 0.27 | n/a |
| `coarse_vs_copy_ratio` | 171.5862 @ 0 | 171.5862 @ 0 | 0 | 171.5862 / 171.5862 / 171.5862 | n/a |
| `coarse_vs_batch_mean_ratio` | 12.2494 @ 0 | 12.2494 @ 0 | 0 | 12.2494 / 12.2494 / 12.2494 | n/a |
| `grad_norm` | 0.3553 @ 0 | 0.2996 @ 150 | -0.0558 | 0.2996 / 0.3496 / 0.3895 | n/a |
| `grad_has_nan` | 0 @ 0 | 0 @ 0 | 0 | 0 / 0 / 0 | n/a |
| `lr_mult` | 1.000e-04 @ 0 | 0.0151 @ 150 | 0.015 | 1.000e-04 / 0.0076 / 0.0151 | n/a |

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

# Observations — charmed-haze-4

## Outcome

**Fix validated.** Throughput improved; metrics unchanged within batch noise.

## Evidence

- **`real 4m42.195s` for 200 steps = 1.41 s/step**, vs `efficient-aardvark-2`'s
  `5m31.226s = 1.66 s/step` pre-fix — a ~15% improvement, well short of the hoped ~3×
  that decoding 16/≈60 frames might have suggested.
- Step-0 `L_flow=2.8726348876953125`, `L_var=0.42669573426246643`, `grad_has_nan=0` —
  **bit-identical** to the pre-fix `efficient-aardvark-2` step-0 line (and step-50/100/150
  match within batch-sampling noise). This is the proof the refactor was
  behavior-preserving: decoding only needed frame indices did not change what the model sees.
- Remaining cost is decord **seeking through VP9 keyframes** (`.webm` at `num_threads=1`),
  not numpy array allocation — so the cheap win was already captured and the rest is
  Plan Phase 03 (CPU→GPU decode) territory.

## Interpretation

Selective decode was sufficient to proceed with full SSv2 runs. CPU→GPU offload
([`AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/`](../../../../AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD))
was **not** opened.
