# fine-meadow-34 — Run 1: SIGReg substrate (λ_sigreg=6, no recon, full-latent)

**Wave:** [investigation_009](../DESCRIPTION.md) · **Position:** 1 of 2 (the substrate arm)
**Pair:** [graceful-river-35](../graceful-river-35/) (residual prediction arm)
**W&B:** `xz3nabr9` · https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/xz3nabr9
**Commit:** `bc77db6` · **Status:** COMPLETE (manually ended at step 14050, ~6h10m)

## Hypothesis

Isolate SIGReg as the sole rank lever with the reconstruction machinery removed (`λ_recon=0`,
`λ_recon_pred=0`) and full-latent prediction (`F_c` predicts `c_{t+k}`, not Δ). Two questions:
(1) does the inv008 "rank↑ ⟺ prediction↓" law reproduce **without any recon confound**, and (2) is a
strong-SIGReg `c` a clean substrate to build prediction on?

Registered predictions ([../OBSERVATIONS.md](../OBSERVATIONS.md) §2026-06-28): rank ~55 (R1-P1);
`coarse_vs_copy_ratio` > 1, ~6–7, worse than the no-SIGReg baseline (R1-P2); `L_flow` ~0.85–0.90,
`c_std_mean` ~0.90–0.93 with the var floor active, no collapse (R1-P3); `L_recon_*` meaningless
because the decoder is never trained (R1-P4).

## Config delta (vs the inv008 full-latent control)

`λ_sigreg`: **6.0**. `λ_var`: 0.5. `λ_recon`: **0**, `λ_recon_pred`: **0** (recon stripped).
`predict_residual`: **false**. Decoder 512×4, `n_c`=32, `k`=12, `lr_coarse_flow`=1e-4.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-sigreg 6.0 --lambda-var 0.5 \
  --lambda-recon 0 --lambda-recon-pred 0 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv009_run1_sigreg6_full \
  --log-every 50 --diag-every 500
```

Result + interpretation → [OBSERVATIONS.md](OBSERVATIONS.md).
