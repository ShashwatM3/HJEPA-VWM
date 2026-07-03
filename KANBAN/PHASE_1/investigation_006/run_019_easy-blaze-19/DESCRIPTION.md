# Run 019 - `easy-blaze-19`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_006](../)  
**W&B:** `3syv6wp2` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3syv6wp2  
**State:** `finished`  
**Created:** 2026-06-25T22:10:05Z  
**Last history step:** 14950  
**Runtime in W&B export:** 6.17h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

Prediction-side reconstruction anchor; tests whether decoding c_hat toward e_plus helps prediction.

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
| `lambda_recon_pred` | 0.05 |
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
| `lambda_recon_pred` | 0 | 0.05 |

## Chronological Linkage

- Previous W&B run: Run 018 [`fanciful-lake-18`](../run_018_fanciful-lake-18/).
- Next W&B run: Run 020 [`jolly-glade-20`](../../investigation_007/wave_1/run_020_jolly-glade-20/).
- Parent investigation conclusion: Reconstruction improved stability/readouts but did not break the rank ceiling or make F_c beat copy. Prediction-side reconstruction also showed that the decoder could be blind to whether c_hat was actually a good future latent.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3044271892/wandb_manifest.json` (0 bytes), `config.yaml` (4478 bytes), `output.log` (222407 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (1429 bytes), `wandb-summary.json` (1381 bytes) |
| Logged artifacts | `run-3syv6wp2-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# Run — easy-blaze-19

## What this run tested

**First option-3 run: the predicted-latent reconstruction anchor (VITA-style joint
objective).** On top of the present anchor (`lambda_recon=0.05`), it adds the prediction
anchor `lambda_recon_pred=0.05`: decode the **predicted** future latent `c_hat` through the
shared decoder `D` and penalize MSE against the **true future** detailed features `e_{t+k}`,
with the gradient flowing **through `F_c`** (and into `B` via the `F_c` conditioning on
`c_t`, intentionally not detached). This is the tech-lead's (Arbab) joint-objective design.

A clean A/B vs [`fanciful-lake-18`](../run_018_fanciful-lake-18/) (option 1 alone): **same seed (42)
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
