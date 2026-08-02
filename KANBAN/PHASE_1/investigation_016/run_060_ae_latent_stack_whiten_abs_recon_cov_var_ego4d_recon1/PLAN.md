# Plan — run 060 EGO4D AE weight extreme (`lambda_recon=1.0`)

## Scope

CLI-only ablation of
[run 058](../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/). Same present-only whitened
absolute-target autoencoder recipe on full EGO4D; sole deliberate change:

```text
lambda_recon: 0.05 → 1.0
```

No production-code change. Reuse the existing V-JEPA/EGO4D whitening file from run 058.

## Why this arm

Run 058 was stable but template-dominated (video gap ~0.018). The historical project has only
one incomplete `lambda_recon=1.0` attempt (inv007 run 029, full-prediction, external crash).
This run asks whether absolute-target recon on EGO4D simply needs a much larger weight against
`lambda_var=0.5` / `lambda_cov=0.01`.

## Matched recipe

Exact command in [GUIDE.md](GUIDE.md). Fields identical to run 058 except `lambda_recon`:

- encoder: default `vjepa2_vitl16`
- `--data ego4d`, seed 42, 15,000 steps
- `present_recon_only`, absolute target, whitening ON
- `lambda_var=0.5`, `lambda_cov=0.01`, SIGReg/slot OFF
- decoder 512×4, `n_c=32`, horizon_k 12
- unique checkpoint / log / W&B identity; train from scratch

## Abort gates

- Missing or wrong whitening file (`dataset != ego4d`, wrong `d_e`, SSv2 path).
- Stage 0: `whiten_active!=1`, residual target on, prediction active, NaNs.
- Launch with `lambda_recon=0.05` by mistake, or resume from run 058 checkpoints.

## Verdict method

Reading Cycle B (present-only). Success requires nontrivial `L_recon_video_gap` **and**
non-collapsing abstract geometry relative to run 058's failure. Raw recon drop alone is not
enough. Overlay geometry curves against `mvbx96nv` in W&B group `inv016_ego4d_transfer`.
