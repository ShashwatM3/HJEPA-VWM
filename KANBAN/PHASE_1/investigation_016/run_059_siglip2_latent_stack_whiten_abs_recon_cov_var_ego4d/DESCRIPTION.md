# Run 059 — `siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d` (inv016)

**Planned W&B display name:**
`Investigation 16 · EGO4D encoder substrate · SigLIP 2 absolute target`

**W&B run:** not launched yet

## Status

READY AFTER RUNPOD PREFLIGHT — the required code is integrated, but real CUDA, complete-EGO4D,
SigLIP whitening, and resource evidence must pass before the 15,000-step launch.

## Question

Does replacing run 058's frozen V-JEPA2 ViT-L/16 substrate with the implemented
SigLIP 2 ViT-B/16 patch tower prevent the EGO4D absolute-target autoencoder from learning the
same collapsed, video-independent template shortcut?

This is an encoder-substrate control of run 058, not the residual-target follow-up already
pre-registered by that run. The objective, dataset, trainable architecture, seed, schedules, and
loss weights remain fixed.

## Hypothesis

SigLIP 2's still-image, language-aligned dense features may change the balance between shared
spatial template and video-specific content enough to improve reconstruction honesty and abstract
geometry. The counter-hypothesis is that the shortcut belongs to the absolute-target objective on
EGO4D and therefore survives the encoder replacement.

## Control and intended delta

Control: [run 058](../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/), W&B
[`mvbx96nv`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/mvbx96nv).

| Contract | Run 058 | Run 059 |
|---|---|---|
| frozen encoder | `vjepa2_vitl16` | `siglip2_vitb16` |
| immutable Hub revision | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` |
| encoder normalization | ImageNet | SigLIP `(x - 0.5) / 0.5` |
| detailed layout | 4x16x16 tubelets | 8x16x16 frames, time-major |
| detailed tensor | `(B,1024,1024)` | `(B,2048,768)` |
| whitening | V-JEPA/EGO4D artifact | newly fit SigLIP/EGO4D artifact |

Those geometry, normalization, and whitening changes are consequences of selecting the encoder;
they are not separately tunable interventions. The common pipeline still consumes only the
resolved `EncoderSpec` and returns the same abstract tensor `(B,32,256)`.

Everything below is held fixed:

```text
data = ego4d                  present_recon_only = true
steps = 15000                seed = 42
recon_residual_target = false
whiten_features = true       whiten_expected_clips = 12800
lambda_recon = 0.05          lambda_recon_pred = 0.0
lambda_var = 0.5             lambda_cov = 0.01
lambda_sigreg = 0.0          lambda_slot = 0.0
recon_loss_mode = cosine     recon_warmup_steps = 2000
horizon_k = 12               n_c = 32
bottleneck_latent_blocks = 3
decoder_dim = 512            decoder_blocks = 4
lr_bottleneck = 1e-4         lr_coarse_flow = 1e-4
lr_decoder = 1e-4            physical batch = 64 (required control target)
```

## Interpretation constraint

Raw reconstruction loss and raw frozen-feature rank are not directly comparable across V-JEPA
and SigLIP feature spaces. The decisive shared-space readouts are abstract geometry, stability,
`L_recon_video_gap`, and the fraction of reconstruction improvement that disappears when the
latent is shuffled across videos. Follow Reading Cycle B.

Exact setup and launch commands: [GUIDE.md](GUIDE.md). Execution design and abort rules:
[PLAN.md](PLAN.md).
