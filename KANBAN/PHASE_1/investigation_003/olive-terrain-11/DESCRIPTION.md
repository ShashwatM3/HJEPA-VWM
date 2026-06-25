# Run — olive-terrain-11

**The first centered-slot, k=12 run.** Previously logged as config-TBD; W&B has since
resolved it.

## What this run tested

First launch after the slot loss/metric centering fix (`ffc33ed`, 06-17 12:55 UTC): now
that the slot penalty actually bites, does **centered** slot loss at a mild weight lift
slot diversity and rank without Goodhart — at the harder **k=12** horizon? This is the
run [`skilled-waterfall-10`](../skilled-waterfall-10/) was *meant* to be (centered loss,
k=12) before its bug + k=4 launch drift + crash.

## Command (reconstructed from W&B config `q40nq0l3`)

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 \
  --lambda-slot 0.05 --lambda-cov 0.0027
```

Config confirmed: `horizon_k=12`, `lambda_slot=0.05`, `lambda_cov=0.0027`,
`lambda_var=0.10`. Created 06-17 13:01 UTC — minutes after the centering commit, and the
**first** run whose `L_slot` starts well below 1.0 (~0.01–0.04), i.e. the first run where
the slot penalty is real.

## Config delta vs skilled-waterfall-10

- Code: **centered** `slot_diversity_loss` (`ffc33ed`) — loss now matches the metric
- `horizon_k`: 4 (actual) → **12**
- Same weights otherwise (`lambda_slot=0.05`, `lambda_cov=0.0027`, `lambda_var=0.10`)

## W&B

- Run name: `olive-terrain-11`
- Run id: `q40nq0l3`
- State: killed at `_step=3900` (~1h58m)
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/q40nq0l3

## Parent

[investigation_003](../DESCRIPTION.md)

## Mapping note

Earlier KANBAN marked this "config TBD / mapping uncertain." Resolved via MCP: it is a
**centered-slot k=12 slot=0.05 cov=0.0027 run** — the first of the two post-centering
slot runs (olive-terrain-11 then [`copper-sky-12`](../copper-sky-12/)). copper-sky-12
(06-19) is the re-run after this one was killed at 3900; the two together are the
combined "Run 5" slot evidence.
