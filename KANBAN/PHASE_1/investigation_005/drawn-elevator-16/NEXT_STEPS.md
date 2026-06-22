# Next steps — drawn-elevator-16

## Context

Resume after [`elated-snowflake-15`](elated-snowflake-15/) grad-skip freeze — see that
run's `NEXT_STEPS.md` for why checkpoint 7500 + `--lr-coarse-flow 1e-4`.

## On success (gates met)

1. Record final metrics vs [`AGENT_FILES/PHASES/PHASE_1.md`](../../../AGENT_FILES/PHASES/PHASE_1.md) §12.
2. **Close** [investigation_005](../DESCRIPTION.md).
3. Re-evaluate [investigation_004](../investigation_004/) only if rank too low.
4. Proceed toward Phase 2 per [`AGENT_FILES/PHASES/PHASE_2.md`](../../../AGENT_FILES/PHASES/PHASE_2.md).

## On failure (grad-skip recurs)

1. Do **not** use post-spike checkpoints from `drawn-elevator-16` or late `elated-snowflake-15` steps.
2. Retry resume from 7500 with lower `--lr-coarse-flow` (e.g. 5e-5) or reset Adam state
   on resume — new run folder if launched.
3. Do **not** revert `lambda_var=0.5` / `horizon_k=12` without new evidence.

## Immediate

Pull final metrics from W&B (`0n5mx3qf`) and update this file's `OBSERVATIONS.md`.
