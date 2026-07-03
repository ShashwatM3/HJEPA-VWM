# Next Steps - investigation_003

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

The next branch tried longer acceptance-style runs and optimizer/EMA hardening, because basic collapse control was still not Phase 1 success.

## Closure / Carry-Forward Status

- Status: **CLOSED**.
- Conclusion to carry forward: The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.
- Runs covered: 006, 007, 008, 009, 010, 011, 012, 013, 014.

## Follow-Up Chain

This investigation feeds into `investigation_004`: standalone covariance/VICReg-C branch, later superseded. The reason is: Covariance was a plausible answer to low-dimensional c_t use, but the branch was paused before it became the main evidence path.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.

## Original Notes Preserved

# Next steps — Investigation 003

Investigation **closed** — winning config from [`cerulean-snow-13`](run_013_cerulean-snow-13/).

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

## Why follow-up is not “more collapse runs”

Runs 2–5 showed init, data scale, slot loss, and low-weight VICReg-C adjunct do not
fix collapse. **`lambda_var=0.5`** does. Slot loss rejected (Goodhart).

## Spawned

**Next investigation:** [investigation_005](../investigation_005/) — finish 15k and
hit PHASE_1 §12 gates. First run: [`elated-snowflake-15`](../investigation_005/run_015_elated-snowflake-15/).
(Note: [`jolly-forest-14`](run_014_jolly-forest-14/) already reproduced the winning config and
crashed at 3900 — the 15k arc is a continuation, not a fresh hypothesis.)

**Conditional:** [investigation_004](../investigation_004/) — isolated VICReg-C A/B
only if rank plateaus low **after** a complete stable 15k attempt.

Do not re-enable `lambda_slot` without new evidence.
