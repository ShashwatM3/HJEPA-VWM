# Observations — drawn-elevator-16

## Outcome

**TBD — verify on W&B.** Runtime (~3.5h) suggests substantial training post-resume.

## Context

Follows [`elated-snowflake-15`](elated-snowflake-15/) failure mode (sustained
`grad_skipped` after step 8500). Intended fix: resume from **pre-spike** checkpoint
(~7500) with halved coarse-flow LR.

## What to check on W&B

- Did `grad_skipped` stay at 0?
- Did `coarse_vs_copy_ratio` hold < 1?
- Final step count vs 15k target
- Whether investigation 005 acceptance gates were met
