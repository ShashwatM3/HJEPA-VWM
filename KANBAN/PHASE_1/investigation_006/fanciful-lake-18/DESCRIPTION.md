# Run — fanciful-lake-18

## What this run tested

**First active option-1 reconstruction-anchor run.** A small cross-attention decoder
`D` rebuilds the frozen detailed features from the abstract latent, trained with a
scale-free relative MSE whose gradient reaches **`D` and `B` only — never `F_c`, never
the EMA targets** (we decode the online `c_t`, not the predicted `c_hat`). The anchor
is meant to force `c` to stay information-rich. Same regime as the
[`royal-cherry-17`](../../investigation_005/royal-cherry-17/DESCRIPTION.md) acceptance
attempt (full SSv2, `horizon_k=12`, `lambda_var=0.5`, halved flow LR `1e-4`) so the
comparison isolates the recon term.

This is the first run that mounts the decoder built in commit `91da83e`.

## Hypotheses

- **Primary (rank ceiling).** The reconstruction MSE is a direct information-richness
  floor on `c`, so `c_effective_rank` should climb past its ~13/256 ceiling toward the
  `>60` spec gate (`PHASE_1.md` §9.2).
- **Secondary (cliff immunity).** A richer / less-sharp `c` should sit in a flatter
  region of the flow-matching landscape and so resist the Mode-B optimization cliff
  that destroyed `royal-cherry-17` at step 8600.
- **Prediction must not regress.** `L_flow` and `coarse_vs_copy_ratio` must not degrade
  versus `royal-cherry-17`; the anchor is a regularizer, not the objective.

## Command (actual — matches logged config)

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4 \
  --lambda-recon 0.05 \
  --recon-warmup-steps 2000 \
  --log-every 50 \
  --diag-every 500
```

Launched in `tmux` on the RunPod pod (`/workspace/hierarchal-jepa-flow-world-model`).

## Config delta vs `royal-cherry-17` (W&B `yd5958s6`)

| Knob | royal-cherry-17 | fanciful-lake-18 | Note |
|---|---|---|---|
| `lambda_recon` | 0.0 (no decoder) | **0.05** | the experiment — scale-free relative MSE |
| `recon_warmup_steps` | — | **2000** | linear ramp of `recon_scale` 0→1 |
| `lr_decoder` | — | 1e-4 | decoder `D` peak LR |
| `agc_lambda_decoder` | — | 0.20 | AGC λ for `D` (mirrors `B`) |
| `lr_coarse_flow` | 2e-4 | **1e-4** | already halved in royal's option A; kept |
| everything else | — | identical | `lambda_var=0.5`, `horizon_k=12`, AGC, seed 42 |

Decoder geometry: `decoder_dim=256`, `decoder_blocks=2`, `decoder_heads=8`
(`c_t` `32×256` → `e_hat` `1024×1024` via orthogonal-init learned queries).

## W&B

- Run name: `fanciful-lake-18`
- Run id: `yd5958s6`
- State: **killed** at step **14400** (target 15000) — stopped by operator after the
  trajectory was clear, not a crash.
- Runtime: 23326s (~6h 29m)
- Created 2026-06-25 12:50 UTC
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/yd5958s6

## Parent

[investigation_006](../DESCRIPTION.md) — does a reconstruction anchor break the rank ceiling?
