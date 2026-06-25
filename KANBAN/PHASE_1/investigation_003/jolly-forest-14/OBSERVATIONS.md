# Observations — jolly-forest-14

## Outcome

**Winning config reproduced — then crashed early at step 3900.** Over its short life it
re-traced cerulean-snow-13's healthy trajectory (variance climbing to the floor,
cross-video cosine low, model beating batch-mean), independently confirming the
`lambda_var=0.5` result before the crash ended it.

## Key numbers (verified vs W&B `8bkeeuio`, var=0.5, k=12, no slot/cov)

| Step | `c_std_mean` | `c_cross_video_cosine` | `c_effective_rank` | `coarse_vs_copy_ratio` | `coarse_vs_batch_mean_ratio` | `grad_skipped` |
|---|---|---|---|---|---|---|
| 0 | 0.50 | 0.72 | 9.47 | 62.1 | 12.6 | 0 |
| 500 | 0.87 | **0.18** | 10.13 | 4.78 | 1.73 | 0 |
| 1500 | 0.87 | 0.23 | 7.55 | 3.69 | 1.06 | 0 |
| 2500 | 0.92 | 0.21 | 6.69 | 2.03 | **0.50** | 0 |
| 3500 | 0.94 | 0.23 | 9.39 | 1.56 | **0.36** | 0 |

These tracks are nearly identical to [`cerulean-snow-13`](../cerulean-snow-13/) over the
same step range (e.g. cerulean at 3500: std 0.93, cosine 0.24, rank 9.27, batch-mean
0.37) — strong reproducibility of the winning config.

## Interpretation

- **Not exploratory or off-config.** Earlier KANBAN treated this as an uncertain
  "intermediate" run; W&B confirms it is the winning `lambda_var=0.5` k=12 config,
  reproducing cerulean. It belongs to the winning-config lineage, bridging
  investigation_003 → investigation_005.
- `c_std_mean` is already climbing toward 1.0 and the model already **beats batch-mean**
  (ratio < 1 from step 2500) — the same reversal cerulean showed.
- It crashed at 3900 (state: crashed), so it never reached 15k. The clean 15k push moved
  to [`elated-snowflake-15`](../../investigation_005/elated-snowflake-15/).

## Chronology note

Created 06-20, after cerulean-snow-13 (06-19) and before elated-snowflake-15 (06-20).
Treat cerulean as the canonical "Run 6 win" and this as its short confirmation; neither
is the completed acceptance run.
