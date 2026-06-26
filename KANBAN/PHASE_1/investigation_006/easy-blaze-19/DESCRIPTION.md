# Run — easy-blaze-19

## What this run tested

**First option-3 run: the predicted-latent reconstruction anchor (VITA-style joint
objective).** On top of the present anchor (`lambda_recon=0.05`), it adds the prediction
anchor `lambda_recon_pred=0.05`: decode the **predicted** future latent `c_hat` through the
shared decoder `D` and penalize MSE against the **true future** detailed features `e_{t+k}`,
with the gradient flowing **through `F_c`** (and into `B` via the `F_c` conditioning on
`c_t`, intentionally not detached). This is the tech-lead's (Arbab) joint-objective design.

A clean A/B vs [`fanciful-lake-18`](../fanciful-lake-18/) (option 1 alone): **same seed (42)
and identical config except `lambda_recon_pred` 0 → 0.05**, so any divergence is attributable
to the prediction anchor.

## Hypothesis

Option 1 left the copy gate failing (`F_c` loses to copy-forward) because present-anchored
recon is blind to prediction error. Routing reconstruction through `F_c` on the *predicted*
latent should give `F_c` a richer training signal — `coarse_vs_copy_ratio` should fall toward
<1 and the diag readout `L_recon_chat` should drop (the gradient now acts on it).

## Command (actual — matches logged config)

```bash
python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 \
  --lr-coarse-flow 1e-4 \
  --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-warmup-steps 2000 \
  --log-every 50 --diag-every 500
```

Launched in `tmux` on the RunPod pod. Code: commit `6805c75` (option-3 branch).

## Config delta vs `fanciful-lake-18` (W&B `3syv6wp2`)

| Knob | fanciful-lake-18 | easy-blaze-19 |
|---|---|---|
| `lambda_recon_pred` | 0.0 (branch not run) | **0.05** (decode `c_hat` → `e_{t+k}`, grad through `F_c`) |
| everything else | — | identical (`lambda_recon=0.05`, `lambda_var=0.5`, `horizon_k=12`, `lr_coarse_flow=1e-4`, seed 42, AGC) |

Implementation note: `c_hat` is the **cheap** rectified-flow one-step endpoint estimate
(`z_c + (1-τ)·u_c_hat`, reusing the flow-loss `u_c_hat`), at random τ — leaks the true future
via `z_c` at high τ. Chosen deliberately for run 1 (consistent with the `L_recon_chat` readout,
no extra `F_c` forward). The clean from-noise variant was deferred.

## W&B

- Run name: `easy-blaze-19`
- Run id: `3syv6wp2`
- State: **finished** — completed the full 15k (`_step` 14950)
- Runtime: 22212s (~6h 10m wall-clock logged)
- Created 2026-06-25 22:10 UTC
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3syv6wp2

## Parent

[investigation_006](../DESCRIPTION.md) — does a reconstruction anchor break the rank ceiling?
