# Next steps — olive-terrain-11

## Why

Intermediate run between Run 4 (SSH drop) and Run 5 (centered slot). Config on W&B
may differ — verify `horizon_k` / `lambda_slot` on dashboard.

## Spawned

**Next run in collapse arc:** [`copper-sky-12`](../copper-sky-12/) (BRIEF Run 5) —
centered slot loss; see skilled-waterfall `NEXT_STEPS` for command.

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```
