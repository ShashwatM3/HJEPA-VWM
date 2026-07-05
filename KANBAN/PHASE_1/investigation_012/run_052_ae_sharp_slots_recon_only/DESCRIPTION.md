# Run 052 - `ae_sharp_slots_recon_only`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_012](../)  
**W&B:** `662hfy3c` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/662hfy3c  
**State:** `running`  
**Created:** 2026-07-02T15:32:02Z  
**Last history step:** 5600  
**Runtime in W&B export:** 2.09h  
**Mode:** present-reconstruction-only  
**Reading-cycle verdict:** **Collapsed rep** - Present reconstruction may improve, but c_t is still weakly spread or video-independent.

## Research Role

Sharp-slot, reconstruction-only run with geometry regularizers off; tests architecture-only bottleneck health.

This run sits inside **sharp-slot bottleneck reconstruction-only test without geometry regularizers**. The parent investigation question is: **Does sharpened slot attention let pure reconstruction train a healthy bottleneck without variance, SIGReg, or covariance regularizers?**

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
| `lambda_var` | 0 |
| `lambda_cov` | 0 |
| `lambda_slot` | 0 |
| `lambda_sigreg` | 0 |
| `sigreg_warmup_steps` | 2000 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | cosine |
| `present_recon_only` | true |
| `predict_residual` | false |
| `n_c` | 32 |
| `decoder_dim` | 512 |
| `decoder_blocks` | 4 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/inv012_sharp_slot_recon_only |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_var` | 0.5 | 0 |
| `lambda_cov` | 0.003 | 0 |
| `lambda_sigreg` | 12.5 | 0 |
| `checkpoint_dir` | /workspace/ckpt/inv011_present_only_geometry_sweep/po_geom_sig12p5_cov0p003 | /workspace/ckpt/inv012_sharp_slot_recon_only |

## Chronological Linkage

- Previous W&B run: Run 051 [`po_geom_sig12p5_cov0p003`](../../investigation_011/present-only-geometry-sweep/wave_2/run_051_po_geom_sig12p5_cov0p003/).
- Next W&B run: None in the current W&B project export.
- Parent investigation conclusion: Live W&B through step 5600 shows strong reconstruction progress but not healthy representation geometry: c_std_mean is still far below 1, cross-video cosine remains high, and rank has fallen into the low 20s. The run is still active, so the final verdict remains provisional.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Present Reconstruction Only** cycle. Do not apply copy-ratio gates to this run because the future-prediction branch is off.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `requirements.txt` (3821 bytes), `wandb-metadata.json` (1815 bytes) |
| Logged artifacts | none listed in export |
| API/group fetch caveats | none |
| Live refresh | `tmp/wandb_evidence/052_ae_sharp_slots_recon_only_662hfy3c_live.json`; refreshed with `run_history.py --run 662hfy3c --report` |

## Original Notes Preserved

**Command** (sharpened-slot bottleneck, present-recon-only, cosine loss, EVERY geometry regularizer off):

```bash
python train.py --data ssv2 --steps 15000 --seed 42 --horizon-k 12 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 \
  --lambda-var 0 --lambda-sigreg 0 --lambda-cov 0 --lambda-slot 0 \
  --lambda-recon 0.05 --lambda-recon-pred 0 --recon-loss-mode cosine --recon-warmup-steps 2000 \
  --present-recon-only --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv012_sharp_slot_recon_only --log-every 50 --diag-every 500
```

**What it tested / config delta:** with the new sharpened-slot bottleneck (orthogonal query slots,
sharp cosine cross-attention, zero-init residual output), test whether reconstruction ALONE — every
geometry regularizer at 0 — now makes c healthy, i.e. whether the architecture can replace the
regularizers that the investigation_011 sweep needed.
