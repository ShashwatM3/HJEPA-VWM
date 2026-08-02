# Run 040 - `inv011_fixed_position_decoder`

**Current W&B run name:** `Investigation 11 · Fixed-position decoder · Full prediction`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_011](../)  
**W&B:** `io74f32b` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/io74f32b  
**State:** `finished`  
**Created:** 2026-06-30T18:48:31Z  
**Last history step:** 14950  
**Runtime in W&B export:** 6.14h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

Fixed-position decoder full-prediction run; tests decoder geometry while keeping prediction active.

This run sits inside **cosine reconstruction, fixed-position decoder, present-only geometry sweeps**. The parent investigation question is: **Can reconstruction geometry produce a strong, decodable present bottleneck, and can that evidence be separated from prediction failure?**

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
| `lambda_sigreg` | 5 |
| `sigreg_warmup_steps` | 2000 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0.05 |
| `recon_loss_mode` | cosine |
| `present_recon_only` | false |
| `predict_residual` | true |
| `n_c` | 32 |
| `decoder_dim` | 512 |
| `decoder_blocks` | 4 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/inv011_fixed_position_decoder |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_var` | 0 | 0.5 |
| `lambda_sigreg` | 0 | 5 |
| `lambda_recon_pred` | 0 | 0.05 |
| `recon_loss_mode` | relative_mse | cosine |
| `present_recon_only` | true | false |
| `predict_residual` | false | true |
| `lr_coarse_flow` | 2.000e-04 | 1.000e-04 |
| `checkpoint_dir` | /workspace/ckpt/inv011_present_recon_only | /workspace/ckpt/inv011_fixed_position_decoder |

## Chronological Linkage

- Previous W&B run: Run 039 [`original_recon_loss + no-pred`](../run_039_original_recon_loss-no-pred/).
- Next W&B run: Run 041 [`inv011_fixed_position_present_recon`](../run_041_inv011_fixed_position_present_recon/).
- Parent investigation conclusion: Present reconstruction can be strong, and SIGReg plus small covariance can raise c_effective_rank above 100 while preserving video specificity. But those are present-side wins; no full-prediction run has inherited them and passed the Phase 1 copy/batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3079653913/wandb_manifest.json` (0 bytes), `config.yaml` (5285 bytes), `output.log` (259593 bytes), `requirements.txt` (3821 bytes), `wandb-metadata.json` (1769 bytes), `wandb-summary.json` (1605 bytes) |
| Logged artifacts | `run-io74f32b-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# fixed-position-decoder - Run C: full residual recipe with fixed-position D

**Investigation:** [investigation_011](../DESCRIPTION.md)
**Position:** Run C, after Run A (`new_recon_loss`) and Run B (`original_recon_loss + no-pred`)
**Status:** READY TO LAUNCH
**Code requirement:** commit `0a9ff46` or later
**W&B group:** `inv011_fixed_position_decoder`

## Question

Run A in this investigation changed the reconstruction objective to cosine distance while keeping the
full residual/SIGReg recipe. It produced healthier reconstruction numbers but still tied the
zero-residual/copy baseline. This run keeps that same training recipe and changes the decoder
architecture: `D` no longer owns learned per-output-token queries.

There is no CLI flag for this feature. The fixed-position decoder is now the default `models.Decoder`
implementation, so the run must start from commit `0a9ff46` or later and must not resume an old
learned-query decoder checkpoint.

## Config Delta

Compared with Run A (`new_recon_loss`):

- same full residual recipe;
- same cosine reconstruction objective;
- same decoder width/depth knobs: `decoder_dim=512`, `decoder_blocks=4`;
- changed only by code: learned output queries replaced by fixed tubelet position codes.

The intended decoder rule is:

```text
position tells D where to write;
c tells D what to write.
```

## Command Shape

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 \
  --steps 15000 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 5.0 \
  --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.05 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --predict-residual \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv011_fixed_position_decoder \
  --log-every 50 \
  --diag-every 500
```

See [GUIDE.md](GUIDE.md) for smoke checks, launch steps, and tripwires.

## Main Comparison

Compare primarily against:

- Run A `new_recon_loss` (`1u69hpfm`): same recipe, learned-query decoder, cosine recon.
- investigation_010 `soft-universe-37` (`2vbo6pbm`): previous best residual/SIGReg run.

## Main Readouts

- `coarse_vs_copy_ratio`: does the fixed-position decoder help prediction beat copy?
- `L_recon_present`: does reconstruction become more honest, even if numerically harder?
- `L_recon_cplus` and `L_recon_chat`: does the true-future vs predicted-future gap widen?
- `c_effective_rank`, `c_plus_effective_rank`, `c_cross_video_cosine`, `c_std_mean`: representation
  health must remain intact.
- `grad_skipped`, `grad_norm`, `agc_D_*`: decoder change must not destabilize training.

## Interpretation

- **Win:** copy ratio drops below Run A while rank/cosine/stability stay healthy, and
  `L_recon_chat - L_recon_cplus` becomes meaningfully larger when `c_hat` is poor.
- **Useful neutral:** copy ratio remains near 1, but reconstruction readouts become more diagnostic;
  this confirms the old decoder was masking prediction quality without solving dynamics.
- **Fail:** reconstruction returns to the same floor with tiny `chat-cplus` gap, or representation
  health degrades without exposing new predictive signal.
