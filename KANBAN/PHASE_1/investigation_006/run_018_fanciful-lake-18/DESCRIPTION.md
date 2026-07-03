# Run 018 - `fanciful-lake-18`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_006](../)  
**W&B:** `yd5958s6` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/yd5958s6  
**State:** `killed`  
**Created:** 2026-06-25T12:50:29Z  
**Last history step:** 14400  
**Runtime in W&B export:** 6.48h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

Present reconstruction anchor run; tests whether decoding pressure breaks rank collapse.

This run sits inside **feature reconstruction anchors on present and predicted abstract latents**. The parent investigation question is: **Can reconstruction pressure make the abstract bottleneck information-rich enough for prediction?**

## Hypothesis Being Tested

The controlled question for this run is whether the configuration below changes the bottleneck or predictor failure mode observed in the preceding run sequence. In this project, a lower `L_flow` is never enough by itself: full-prediction runs must beat the copy and batch-mean baselines, while present-only runs must show a decodable, spread, video-specific, high-rank `c_t`.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2 |
| `steps` | 15000 |
| `stage1_steps` | 15000 |
| `horizon_k` | 12 |
| `frame_stride` | 2 |
| `lambda_var` | 0.5 |
| `lambda_cov` | 0 |
| `lambda_slot` | 0 |
| `lambda_sigreg` | 0 |
| `sigreg_warmup_steps` | n/a |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | n/a |
| `present_recon_only` | false |
| `predict_residual` | false |
| `n_c` | 32 |
| `decoder_dim` | 256 |
| `decoder_blocks` | 2 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/checkpoints |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_recon` | 0 | 0.05 |
| `decoder_dim` | n/a | 256 |
| `decoder_blocks` | n/a | 2 |
| `lr_coarse_flow` | 2.000e-04 | 1.000e-04 |
| `lr_decoder` | n/a | 1.000e-04 |

## Chronological Linkage

- Previous W&B run: Run 017 [`royal-cherry-17`](../../investigation_005/run_017_royal-cherry-17/).
- Next W&B run: Run 019 [`easy-blaze-19`](../run_019_easy-blaze-19/).
- Parent investigation conclusion: Reconstruction improved stability/readouts but did not break the rank ceiling or make F_c beat copy. Prediction-side reconstruction also showed that the decoder could be blind to whether c_hat was actually a good future latent.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3041380275/wandb_manifest.json` (0 bytes), `config.yaml` (4375 bytes), `output.log` (204729 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (1390 bytes), `wandb-summary.json` (1349 bytes) |
| Logged artifacts | `run-yd5958s6-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — fanciful-lake-18

## What this run tested

**First active option-1 reconstruction-anchor run.** A small cross-attention decoder
`D` rebuilds the frozen detailed features from the abstract latent, trained with a
scale-free relative MSE whose gradient reaches **`D` and `B` only — never `F_c`, never
the EMA targets** (we decode the online `c_t`, not the predicted `c_hat`). The anchor
is meant to force `c` to stay information-rich. Same regime as the
[`royal-cherry-17`](../../investigation_005/run_017_royal-cherry-17/DESCRIPTION.md) acceptance
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
