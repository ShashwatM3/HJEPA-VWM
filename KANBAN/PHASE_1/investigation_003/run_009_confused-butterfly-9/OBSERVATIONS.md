# Observations - run 009 `confused-butterfly-9`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `m30jxiye`  
**State:** `killed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Smoke / inconclusive**  
**Verdict note:** The run is too short for Phase 1 learning gates; use it as infrastructure evidence.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=killed; step=-1; grad_skipped max=n/a; grad_has_nan max=n/a; grad_norm last=n/a |
| Q2 | c_t alive / video-specific? | FAIL | std=n/a; dead_dim=n/a; cross_video_cosine=n/a |
| Q3 | Rich latent? | FAIL | c_effective_rank=n/a; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=n/a; ratio=n/a; story=insufficient temporal evidence or early run |
| Q5 | F_c beats copy? | FAIL | model_loss=n/a; copy_loss=n/a; ratio=n/a; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=n/a; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Smoke / inconclusive | The run is too short for Phase 1 learning gates; use it as infrastructure evidence. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|

## Last Key Metrics

No key diagnostic metrics were present in the export.

## Interpretation

This is a full-prediction experiment. The latest copy ratio is n/a and the latest batch-mean ratio is n/a; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is n/a, cross-video cosine is n/a, and copy loss is n/a. The reading-cycle verdict is **Smoke / inconclusive** because The run is too short for Phase 1 learning gates; use it as infrastructure evidence. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — confused-butterfly-9

## Outcome

**Failed instantly.** No training signal.

## Interpretation

Discard for metrics. Likely a launch mistake between `sleek-leaf-7` and `serene-cloud-8`
in the W&B run sequence. Not part of BRIEF Runs 2–6 narrative.
