# Observations — Run 67, SigLIP2-B standard-ViT companion

## Execution outcome

Run 67 passed its real CUDA adapter smoke, exact Stage 0, and batch-64 resource preflight, then ran
stably from scratch on the full EGO4D dataset. Its W&B ID is [`ufbeokj2`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ufbeokj2).
The user intentionally stopped it at step 11,000/15,000 to replace it with the cleaner SigLIP2
zero-geometry-regularizer control. No operational tripwire fired before the stop.

The resource preflight measured a CUDA peak of `19,783,331,840` bytes with approximately `59.0`
examples/second at physical batch 64. The paid run kept whitening, prediction, residual targets,
SIGReg, and slot loss inactive. It used `lambda_var=0.5` and `lambda_cov=0.01` exactly as planned.
The last durable checkpoint is `phase1_step10000.pt` (approximately 488 MB), and provenance is
resolved in the run's checkpoint root.

## Scientific trajectory before stop

The run was not collapsed or reconstruction-blind. From step 500 to step 11,000, the fixed-batch
correct reconstruction fell from `0.46256` to `0.22494`, while the rolled-code gap increased from
`0.00046` to `0.05956`. Over the same interval, cross-example cosine improved from `0.98148` to
`0.75727` and effective rank rose from `33.23` to `62.21`. At step 11,000, mean code std was
`0.56499`, centered slot rank was `28.72/32`, dead-dimension fraction was zero, and all stability
flags remained zero.

These values establish that covariance and variance pressure materially opened the SigLIP2 code
and made reconstruction increasingly code-dependent. They do not constitute a completed
15,000-step endpoint, and raw reconstruction values cannot be compared as universal quality scores
against V-JEPA because the two encoders define different feature spaces. The fixed diagnostic batch
also contains adjacent chunks from one EGO4D source UID, so its cosine is a within-source
cross-example measure rather than a global cross-source collapse test.
