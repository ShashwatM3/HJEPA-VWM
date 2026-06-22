# Next steps — Investigation 003

Investigation **closed** — winning config from [`cerulean-snow-13`](cerulean-snow-13/).

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

## Why follow-up is not “more collapse runs”

Runs 2–5 showed init, data scale, slot loss, and low-weight VICReg-C adjunct do not
fix collapse. **`lambda_var=0.5`** does. Slot loss rejected (Goodhart).

## Spawned

**Next investigation:** [investigation_005](../investigation_005/) — finish 15k and
hit PHASE_1 §12 gates. First run: [`elated-snowflake-15`](../investigation_005/elated-snowflake-15/).

**Conditional:** [investigation_004](../investigation_004/) — isolated VICReg-C A/B
only if rank plateaus low **after** a complete stable 15k attempt.

Do not re-enable `lambda_slot` without new evidence.
