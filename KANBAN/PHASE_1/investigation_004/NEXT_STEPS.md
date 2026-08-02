# Next Steps - investigation_004

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

Keep this as historical context; use investigation_011 for covariance evidence rather than treating this paused branch as an experimental result.

## Closure / Carry-Forward Status

- Status: **PAUSED**.
- Conclusion to carry forward: No W&B run is assigned to this investigation in the canonical run sequence. Later present-only geometry sweeps revisited covariance in a better-controlled setting after SIGReg and fixed-position reconstruction clarified the failure mode.
- Runs covered: none.

## Follow-Up Chain

This investigation feeds into `investigation_005`: 15k acceptance attempts, resume behavior, AGC and skip control. The reason is: Shorter runs could look stable while longer runs exposed late spikes, rank collapse, or copy-baseline failure. This branch tested the run length and optimizer safety needed for an acceptance attempt.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.

## Original Notes Preserved

# Next steps — Investigation 004

**Status: PAUSED** — VICReg-C never tested as the **primary isolated** lever.

## Why paused

[`sleek-leaf-7`](../investigation_003/run_007_sleek-leaf-7/) (Run A, `lambda_cov=0`) calibrated
weight; Runs 3–5 tried `lambda_cov=0.0027` only as **adjunct** with slot loss.
[`cerulean-snow-13`](../investigation_003/run_013_cerulean-snow-13/) won with `lambda_cov=0` and
`lambda_var=0.5` alone.

## What must happen before this reopens

1. Complete [investigation_005](../investigation_005/) with `lambda_var=0.5`, `lambda_cov=0`.
2. If rank ≥ ~30 and copy ratio holds → **CLOSE** this investigation as unnecessary.
3. If rank stuck ~10–15 with healthy cosine/std → run isolated A/B:

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 \
  --lambda-cov <calibrated>
```

Require tech-lead sign-off before locking `lambda_cov` default.

Do not run VICReg-C before confirming variance-only path on a **complete** stable 15k.
