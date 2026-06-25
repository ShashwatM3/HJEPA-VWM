# Run — comfy-glade-3

## What this run tested

The **dense-logged** pre-fix smoke. After `efficient-aardvark-2` established the
~1.66 s/step baseline with the default `log_every=50`, this run re-ran the same
200-step `ssv2_tiny` smoke with **per-step logging** (`--log-every 1`) so every
step's `loss`/`L_flow`/`L_var`/`grad_norm`/`ema_m`/`lr_mult` was visible — the user
had asked why the loop only printed every 50 steps, which motivated wiring the
`--log-every` CLI flag (commit `4c1abb3`).

## Command

```bash
python train.py --data ssv2_tiny --steps 200 --log-every 1
```

W&B config confirms `log_every=1` (the only run in the project with that value),
`diag_every=500`, `max_steps=200`, `dataset=ssv2_tiny` — i.e. the proposed
`--diag-every 100` from the chat draft was **not** passed at launch.

## Config delta

Pre-fix dataloader (commit `4c1abb3` era, before selective decode `5e78caa`). Same
model/optimizer config as `efficient-aardvark-2`; only `log_every` differs.

## W&B

- Run name: `comfy-glade-3`
- Run id: `0mgmqxxi`
- Created: 2026-06-09 02:52 UTC (after `efficient-aardvark-2` 02:41, before `charmed-haze-4` 06-10)
- Runtime: ~5m 29s
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0mgmqxxi

## Parent

[investigation_002](../DESCRIPTION.md)
