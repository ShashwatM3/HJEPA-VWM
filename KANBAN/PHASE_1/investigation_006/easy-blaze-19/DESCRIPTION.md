# Run — easy-blaze-19

## What this run tested

**Option 3: predicted-latent reconstruction through `F_c`, added on top of the
present reconstruction anchor.** The decoder `D` still reconstructs frozen detailed
features from the online present latent (`lambda_recon=0.05`), and this run also
trains `D(c_hat) -> e_{t+k}` with `lambda_recon_pred=0.05`, where `c_hat` is the
coarse flow prediction. Gradient is intentionally allowed to reach `D`, `F_c`, and
`B` through the `F_c` conditioning path.

This is the clean A/B against
[`fanciful-lake-18`](../fanciful-lake-18/DESCRIPTION.md): same seed and same royal /
fanciful regime, with the only intended behavioral delta being
`lambda_recon_pred=0.05`.

## Hypotheses

- **Primary (copy gate).** Putting reconstruction on the predicted latent should give
  `F_c` a prediction-quality signal that the present anchor could not provide, so
  `coarse_vs_copy_ratio` should fall below the `fanciful-lake-18` curve and ideally
  approach the <=0.70 Phase 1 gate.
- **Success-signal check.** `L_recon_pred` should descend if the decoder channel can
  distinguish better from worse predictions.
- **Guardrail.** The extra gradient through `F_c` must not reintroduce the Mode-B
  optimization cliff or destabilize the decoder / bottleneck.
- **Secondary watch.** Rank was not expected to improve much, but the added objective
  should not push `c` toward representational collapse.

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
  --lambda-recon-pred 0.05 \
  --log-every 50 \
  --diag-every 500
```

## Config delta vs `fanciful-lake-18`

| Knob | fanciful-lake-18 | easy-blaze-19 | Note |
|---|---:|---:|---|
| `lambda_recon` | 0.05 | 0.05 | present anchor retained |
| `lambda_recon_pred` | 0.0 | **0.05** | experiment: predicted-latent anchor through `F_c` |
| `recon_warmup_steps` | 2000 | 2000 | shared ramp for reconstruction terms |
| `lr_coarse_flow` | 1e-4 | 1e-4 | kept from the stabilized regime |
| `lambda_var` | 0.5 | 0.5 | variance floor unchanged |
| everything else | — | identical | same seed/config family for a clean A/B |

## W&B

- Run name: `easy-blaze-19`
- Run id: `3syv6wp2`
- State: **finished**
- Steps: **15000** target completed (full run; unlike `fanciful-lake-18`, not
  operator-killed at 14400)
- Project: `smahalanobis-uc-davis/hjepa-vwm`
- https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3syv6wp2

## Data sources for this record

- W&B MCP-derived run analysis supplied with this update: full trajectory, run
  summary/config, and matched-step comparison to `fanciful-lake-18`.
- Uploaded local launch log parse: `logs/1.json` / `logs/1.txt`, which covers only
  steps 0-199 and therefore confirms early launch behavior but not final conclusions.

## Parent

[investigation_006](../DESCRIPTION.md) — does a reconstruction anchor break the rank
ceiling, and can the predicted-latent anchor fix the copy gate?
