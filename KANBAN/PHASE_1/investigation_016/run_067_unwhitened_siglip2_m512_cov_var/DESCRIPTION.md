# Run 67 — Investigation 16 · Standard-ViT geometry · SigLIP2-B M=512

## Status

STOPPED BY USER — intentionally ended at step 11,000/15,000 on 2026-07-19 after the
scientific question was redirected to the cleaner zero-regularizer SigLIP2 control. The run was
healthy at stop time; this status is not a crash or a failed gate. A durable step-10,000 checkpoint
and resolved provenance remain on the RunPod volume.

## W&B

- Display name: `Investigation 16 · Standard-ViT geometry · SigLIP2-B M=512 covariance plus variance`
- Group: `inv016_unwhitened_m512_geometry`
- ID: `ufbeokj2`
- URL: <https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ufbeokj2>

## Question

How does the exact Run 66 raw-feature geometry recipe behave when the frozen V-JEPA2 ViT-L/16
encoder is replaced by the repository's implemented standard-ViT lane, SigLIP 2 ViT-B/16?

This is the contemporaneous encoder companion to
[`../run_066_unwhitened_internal_memory_m512_cov_var/`](../run_066_unwhitened_internal_memory_m512_cov_var/).
It starts from scratch and holds full EGO4D, seed 42, batch 64, M=512, `N_c=32`, external
`D_c=256`, three late-projection latent blocks, the 512-by-4 decoder, present-only absolute
raw-feature cosine reconstruction, `lambda_recon=1`, `lambda_var=0.5`, `lambda_cov=0.01`, and the
15,000-step schedule fixed.

The selected encoder is the only experimental intervention:

```text
encoder:          vjepa2_vitl16 -> siglip2_vitb16
revision:         b3c1679...     -> 3f9f96c...
normalization:    ImageNet       -> SigLIP [-1, 1]
detailed tensor:  1024 x 1024    -> 2048 x 768
```

The normalization and detailed-token geometry are inseparable consequences of selecting the
frozen encoder. Whitening, residual reconstruction, SIGReg, slot loss, future prediction, and
checkpoint resume remain off. The final abstract rate remains exactly `32 x 256`.

## Interpretation boundary

Raw reconstruction losses live in different frozen feature spaces and are not directly scaled
quality scores across encoders. Compare training health, abstract geometry, correct-versus-shuffled
gap, and conditioned share; describe raw-loss differences as feature-space compressibility. The
fixed EGO4D diagnostic batch remains within one source UID, so cross-example metrics do not prove
global cross-source behavior.

## Actual execution record

- Published commit: `4c402556e0b85684499f5a111630b7c8bc20ca89`
- Encoder revision: `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`
- Feature fingerprint: `2340dea66764e6f8800443add7f02dd5586ae4e017bf81b824121b3256bce848`
- Dataset fingerprint: `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c`
- Last logged step: `11,000`
- Last durable checkpoint: `/workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var/phase1_step10000.pt`
- Provenance: `/workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var/run_provenance.json`
- Failure tripwires through stop: zero gradient skips, instability warnings, NaNs, OOMs, and tracebacks

The direct zero-regularizer successor is
[`../run_068_unwhitened_siglip2_m512_no_geometry_regularizers/`](../run_068_unwhitened_siglip2_m512_no_geometry_regularizers/).
