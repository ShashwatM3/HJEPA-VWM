# Next steps — Investigation 004

**Status: PAUSED** — VICReg-C never tested as the **primary isolated** lever.

## Why paused

[`sleek-leaf-7`](../investigation_003/sleek-leaf-7/) (Run A, `lambda_cov=0`) calibrated
weight; Runs 3–5 tried `lambda_cov=0.0027` only as **adjunct** with slot loss.
[`cerulean-snow-13`](../investigation_003/cerulean-snow-13/) won with `lambda_cov=0` and
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
