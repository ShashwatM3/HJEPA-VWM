# Run — init-fixes-full-ssv2-run-2

**Slug:** no W&B display name in chat (BRIEF_V0_3 "Run 2" / VICReg **Run A**).

Same run as VICReg Run A stopped ~step 3500 for data-confound verdict.

## What this run tested

First full-SSv2 run after Run 1 stability retune **and** bottleneck init fixes
(orthogonal queries, zero-init `out_mlp`). Isolated: does init + full data fix rank ~5?

## Command (approximate)

```bash
python train.py --data ssv2 --steps 15000
```

Post-retune defaults; init fixes on; `lambda_var=0.10`, `horizon_k=4`, no slot/cov.

## Config delta vs peachy-terrain-5

- Full SSv2 (not tiny)
- Halved LRs, 1.5k warmup, 15k steps, grad clip 0.5, skip guard
- Init fixes baked into `models.py`

## W&B

Name not recorded in repo. If found on W&B, rename this folder to match.

## Parent

[investigation_003](../DESCRIPTION.md)
