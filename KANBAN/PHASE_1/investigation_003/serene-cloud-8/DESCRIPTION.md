# Run — serene-cloud-8

**BRIEF Run 3.** Aggressive slot-diversity loss at easy horizon.

## What this run tested

Hypothesis: within-video **slot-diversity loss** breaks redundant slots and lifts rank.

## Why this run, not VICReg-C (the planned next step)

The plan after [`sleek-leaf-7`](../sleek-leaf-7/) was "Run B = VICReg-C / `lambda_cov`."
The team deliberately **diverted** from that. sleek-leaf-7's per-head metrics showed the
dominant collapse was **within-video slot redundancy** (`c_slot_diversity_rank` 1.62/32,
attention near-uniform), whereas VICReg-C only decorrelates the **256 feature dims**
(the cross-video axis, `c_effective_rank` 8.7). Per the user's rule — *"unless the next
step directly combats the problem(s) arisen, make the needed changes first"* — running
VICReg-C alone would have spent hours on the secondary axis. The agent's overnight
research (VICReg; Perceiver / Slot-Attention "prototype/slot collapse"; the "Trap of
Mediocrity" uniform-attention work) found the standard remedy for learned-query collapse
is a **slot-diversity / orthogonality penalty**, which attacks the root (uniform
attention) through the only available lever because `out_mlp` is shared. So a new
`slot_diversity_loss` (`losses.py`, flag-gated `--lambda-slot`, default 0, always logged
as `L_slot`) was built and tested here, with VICReg-C kept on gently as the secondary
term. A **governance flag** was raised: this exercises the supervisor's "variance floor
only, no covariance loss initially" directive — both new knobs stay flag-gated,
default-off, reversible. Decision gate set in advance: slot-rank ↑ **and** copy-ratio
holds → win; slot-rank ↑ but copy-ratio worsens → Goodhart → escalate to task difficulty.

## Command

```bash
python train.py --data ssv2 --steps 5000 --log-every 50 --diag-every 250 \
  --lambda-slot 0.25 --lambda-cov 0.0027
```

(`horizon_k=4` default; `lambda_var=0.10` default.)

## Config delta vs sleek-leaf-7

- `lambda_slot=0.25` (active)
- `lambda_cov=0.0027` (active — not logging-only)

## W&B

- Run name: `serene-cloud-8`
- Run id: `dhp1i3fk`
- Runtime: ~1h 32m
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/dhp1i3fk

## Parent

[investigation_003](../DESCRIPTION.md)
