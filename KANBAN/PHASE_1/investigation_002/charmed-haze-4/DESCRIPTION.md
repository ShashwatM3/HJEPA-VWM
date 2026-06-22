# Run — charmed-haze-4

## What this run tested

200-step timed smoke **after** decode-only-needed-frames fix (`commit 5e78caa`).
Validated bit-identical inputs with improved throughput.

## Command

```bash
time python train.py --data ssv2_tiny --steps 200 --log-every 50
```

## Config delta vs efficient-aardvark-2

- Selective frame decode in `data.py`
- `num_threads=1` kept for VP9 `.webm` reliability

## W&B

- Run name: `charmed-haze-4`
- Run id: `gj8ypv0d`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/gj8ypv0d

## Parent

[investigation_002](../DESCRIPTION.md)
