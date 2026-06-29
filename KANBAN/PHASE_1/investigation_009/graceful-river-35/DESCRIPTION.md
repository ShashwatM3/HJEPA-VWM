# graceful-river-35 — Run 2: residual prediction + recon (λ_sigreg=5, --predict-residual)

**Wave:** [investigation_009](../DESCRIPTION.md) · **Position:** 2 of 2 (the residual-prediction arm)
**Pair:** [fine-meadow-34](../fine-meadow-34/) (SIGReg substrate arm)
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
