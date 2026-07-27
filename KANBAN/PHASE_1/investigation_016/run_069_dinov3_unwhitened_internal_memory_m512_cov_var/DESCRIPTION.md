# Run 69 — Investigation 16 · Encoder substrate · DINOv3 unwhitened memory M=512 covariance plus variance

## Status

COMPLETED — 15,000-step run finished successfully; W&B identity and final checkpoint checksum verified.

## Identity

- Display name: `Investigation 16 · Encoder substrate · DINOv3 unwhitened memory M=512 covariance plus variance`
- Group: `inv016_encoder_substrate_unwhitened_memory`
- Entity / project: `smahalanobis-uc-davis/hjepa-vwm`
- W&B: [`fiactcw6`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/fiactcw6)
- Final checkpoint: `/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/phase1_step15000.pt`
- Final checkpoint SHA-256: `fbd4bda4c75780c06b35c837a1291218c4525c1549baf978ef3561eaf81e7cb8`
- Uploaded checkpoint directory: `s3://hjepa-volume/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/`
  (`7` objects, `2.9 GiB`)
- Uploaded preflight directory: `s3://hjepa-volume/preflight/inv016_dinov3_unwhitened_memory_m512_cov_var/`
  (`1` object)

## Question and result boundary

Run 69 is the controlled unwhitened ablation of Run 68. It disables whitening while retaining the late `M -> D_c` projection and `lambda_var=0.5`, `lambda_cov=0.01` geometry pressure.

Final active reconstruction was `L_recon=0.11770`; fixed-batch present reconstruction was `L_recon_present=0.13343`. This is consistent with whitening contributing to the reconstruction degradation qualitatively reported for Run 68. The conclusion remains qualified because Run 68 has no verified exact metric readout in the repository; do not treat the comparison as a precise effect estimate.

## Locked configuration

Full EGO4D; 15,000 steps; seed 42; DINOv3 `dinov3_vitb16` at `5931719e67bbdb9737e363e781fb0c67687896bc`; bf16; frame microbatch 32; SDPA; `/workspace/hf_cache`; batch 64; horizon 12; present-only absolute cosine reconstruction; `lambda_recon=1.0`; `lambda_recon_pred=0`; warmup 2,000; `lambda_var=0.5`; `lambda_cov=0.01`; SIGReg and slot weights zero; whitening off; `M=512`; three latent blocks; `N_c=32`; late `abstract_proj`; decoder width 512 with four blocks; all three learning rates `1e-4`; no resume.

See [`OBSERVATIONS.md`](OBSERVATIONS.md) for the verified final metric snapshot and [`GUIDE.md`](GUIDE.md) for the retained execution recipe.
