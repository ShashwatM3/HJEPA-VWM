# Run — exalted-lion-6

## What this run tested

Plan Phase 04 **P1 instrumented diagnostic** (~500 steps on `ssv2_tiny`, ~13 min):
confirm the new collapse instrumentation works on the real pipeline at low cost, and
check whether the `c_t` collapse seen in [`peachy-terrain-5`](../../investigation_001/peachy-terrain-5/)
is an **init/early transient** or **persistent**. It doubled as an **init-knob
ablation**: this run was launched while bottleneck init was still **flag-configurable**.

## Command

```bash
python train.py --data ssv2_tiny --steps 500 --log-every 50 --diag-every 100
```

(`--lambda-cov`, `--lambda-slot`, `--lambda-var` all default / off.)

## Config delta (verified vs W&B config `wv69n7n5`)

- **Init was flag-based, NOT the later orthogonal bake.** W&B config shows
  `bottleneck_query_init="small_gaussian"`, `bottleneck_query_init_scale=0.1`,
  `bottleneck_zero_init_out_mlp=false`. This run **predates** commit `62b94dd`
  (06-13 18:40 UTC, "bake init fixes into the model; reserve flags for empirical
  knobs") — it ran at 14:35 UTC. So the earlier KANBAN claim that "orthogonal queries
  + zero-init `out_mlp` were baked in" is **wrong for this run**: exalted-lion-6 is
  precisely the run whose result *justified* baking a default and deleting the init
  flags (it proved init knobs are a non-lever).
- `lambda_var=0.10` (default), `horizon_k=4`, `lambda_cov`/`lambda_slot` off
- Post–Run-1 stability retune (`611f2cd`: halved LRs, `grad_clip=0.5`, skip guard)
- **Head-averaged** `c_attn_entropy` only — the per-head `c_attn_entropy_min` fix
  (`a96c0d6`, 06-13 18:17 UTC) landed **after** this run, so the entropy reading here
  is the flawed head-averaged metric (see OBSERVATIONS).

## W&B

- Run name: `exalted-lion-6`
- Run id: `wv69n7n5`
- State: finished; ~13 min; `ssv2_tiny`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/wv69n7n5

## Parent

[investigation_003](../DESCRIPTION.md) — chronologically first run in this investigation
