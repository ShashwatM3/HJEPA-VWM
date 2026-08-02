# Run 035 - `sigreg-recon-residual`

**Current W&B run name:** `Investigation 09 · Residual prediction · With reconstruction`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_009](../)  
**W&B:** `jsh6uo7p` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/jsh6uo7p  
**State:** `crashed`  
**Created:** 2026-06-28T03:34:51Z  
**Last history step:** 14050  
**Runtime in W&B export:** 6.16h  
**Mode:** full-prediction  
**Reading-cycle verdict:** **Low-rank rep** - The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Research Role

Residual plus reconstruction run; tests whether residual prediction and reconstruction combine.

This run sits inside **residual prediction and zero-change baseline diagnosis**. The parent investigation question is: **If F_c predicts the future residual instead of the full future latent, does it finally beat the zero-change/copy baseline?**

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
| `sigreg_warmup_steps` | n/a |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0.05 |
| `recon_loss_mode` | n/a |
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
| `checkpoint_dir` | /workspace/ckpt/inv009_run2_sigreg5_residual |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_sigreg` | 6 | 5 |
| `lambda_recon` | 0 | 0.05 |
| `lambda_recon_pred` | 0 | 0.05 |
| `predict_residual` | false | true |
| `checkpoint_dir` | /workspace/ckpt/inv009_run1_sigreg6_full | /workspace/ckpt/inv009_run2_sigreg5_residual |

## Chronological Linkage

- Previous W&B run: Run 034 [`sigreg-only`](../run_034_sigreg-only/).
- Next W&B run: Run 036 [`upbeat-frog-36`](../../investigation_010/run_036_upbeat-frog-36/).
- Parent investigation conclusion: Residual prediction made c_t more dynamic and healthier, but F_c mostly tied the zero-residual baseline. The failure moved from representation collapse toward predictor learning.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Full Prediction** cycle. Prediction success requires `coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50` at a stable late diagnostic point.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3064582489/wandb_manifest.json` (0 bytes), `requirements.txt` (3820 bytes), `wandb-metadata.json` (1855 bytes) |
| Logged artifacts | `run-jsh6uo7p-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# graceful-river-35 — Run 2: residual prediction + recon (λ_sigreg=5, --predict-residual)

**Wave:** [investigation_009](../DESCRIPTION.md) · **Position:** 2 of 2 (the residual-prediction arm)
**Pair:** [fine-meadow-34](../run_034_sigreg-only/) (SIGReg substrate arm)
**W&B:** `jsh6uo7p` · https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/jsh6uo7p
**Commit:** `bc77db6` · **Status:** COMPLETE (manually ended at step 14050, ~6h10m)

## Hypothesis

Keep the full stack (SIGReg + recon) but change `F_c`'s task from predicting the full future latent
`c_{t+k}` to predicting the **temporal residual** Δ = `c_{t+k} − c_t`, reconstructing the future from
`ĉ = c_t + Δ̂`. Two questions: (1) does focusing `F_c` on the *change* extract predictable motion the
full-latent flow wasted capacity copying through — i.e. does `coarse_vs_copy_ratio` fall toward < 1 —
and (2) does reconstruction-with-residuals help (does `L_recon_chat` finally drop below the inv007
~0.585 floor)?

Load-bearing caveat ([../DESCRIPTION.md](../DESCRIPTION.md)): predicting Δ is a reparametrization —
the copy baseline stays ‖Δ‖², so the ratio is numerically comparable across both arms and is **not**
weakened by the residual. The lazy shortcut relocates from "parrot `c_t`" to "output 0".

Registered predictions ([../OBSERVATIONS.md](../OBSERVATIONS.md) §2026-06-28): rank ~45–50 (R2-P1);
ratio most likely ≥ 1, watch for a dip below the full-latent baseline (R2-P2); `L_recon_chat` likely
stays ~0.585, a drop would be the surprising hypothesis-supporting result (R2-P3); stability fine,
collapse-watch on `Δ̂ → 0` and a *climbing* `c_cross_video_cosine` (R2-P4).

## Config delta (vs the inv008 full-latent control)

`λ_sigreg`: **5.0**. `λ_var`: 0.5. `λ_recon`: **0.05** (present anchor) + `λ_recon_pred`: **0.05**
(residual-future decode), `recon_warmup_steps`=2000. `predict_residual`: **true**. Decoder 512×4,
`n_c`=32, `k`=12, `lr_coarse_flow`=1e-4.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-sigreg 5.0 --lambda-var 0.5 \
  --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-warmup-steps 2000 \
  --predict-residual \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv009_run2_sigreg5_residual \
  --log-every 50 --diag-every 500
```

Result + interpretation → [OBSERVATIONS.md](OBSERVATIONS.md).
