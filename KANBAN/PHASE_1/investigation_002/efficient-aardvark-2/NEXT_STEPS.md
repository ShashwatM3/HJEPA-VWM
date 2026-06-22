# Next steps — efficient-aardvark-2

## Why

~1.66 s/step confirmed CPU-bound decode (all frames decoded, then sliced). Fix
required before full SSv2.

## Code change (before next smoke)

Selective decode only needed frame indices (`commit 5e78caa`).

## Spawned

**Next smoke:** [`charmed-haze-4`](charmed-haze-4/) — re-time 200 steps post-fix;
metrics must stay bit-identical to this run.

```bash
time python train.py --data ssv2_tiny --steps 200 --log-every 50
```
