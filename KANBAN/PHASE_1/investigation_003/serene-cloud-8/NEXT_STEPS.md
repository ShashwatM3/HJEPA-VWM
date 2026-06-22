# Next steps — serene-cloud-8

## Why

Goodhart: slot metric moved but `c_cross_video_cosine` ~0.84 and rank worsened.
Aggressive `lambda_slot=0.25` rejected. Easy horizon k=4 made copy baseline strong.

## Spawned

**Next run:** [`skilled-waterfall-10`](../skilled-waterfall-10/) (BRIEF Run 4) — harder
horizon + gentler slot loss; keep `lambda_cov=0.0027` adjunct.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

**Watch:** whether `L_slot` actually moves (loss/metric alignment was suspect).
