# Run 057 — `ae_latent_stack_whiten_abs_recon_cov_var`

**W&B:** [`cdvp6hou`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/cdvp6hou)  
**State:** FINISHED — full 15,000-step schedule; final training row at step 14,950 and final
diagnostic at step 14,500.  
**Mode:** present-reconstruction-only; prediction was inactive.

## Question

Is SIGReg necessary once covariance and the variance floor already protect the whitened
latent-stack code, or does its full-isotropy pressure tax reconstruction and attention without
providing additional rank?

Run 057 is the exact run-056 recipe with one delta:

```text
lambda_sigreg: 5.0 -> 0.0
```

It keeps absolute whitened-feature reconstruction, the three-block latent-stack bottleneck,
`lambda_cov=0.01`, `lambda_var=0.5`, `lambda_recon=0.05`, 32 slots, the 512x4 fixed-position
decoder, seed 42, and the same whitening statistics.

## Result

Removing SIGReg did not cost geometry. The run ended at effective rank `208.22`, centered slot
rank `30.71`, std `1.117`, cross-video cosine `0.0585`, and zero dead dimensions. Reconstruction
also improved relative to run 056: present loss `0.71285`, shuffled-code loss `0.93966`, and video
gap `0.22681` (about `78.0%` of learned improvement conditioned on the correct code). Attention
remained much broader and gradients were calmer.

The controlled conclusion is that covariance is the operative rank lever. SIGReg caused a small
decodability and attention-specialization tax without supplying necessary rank. Run 057 is the
settled absolute-target SSv2 control for that geometry bundle, not a final residual-target recipe
or a prediction success.

## Chronological linkage

- Previous: run 056
  [`ae_latent_stack_whiten_abs_recon_geom`](../run_056_ae_latent_stack_whiten_abs_recon_geom/)
  (`tl5dh73c`), identical except for active SIGReg.
- Next science run: investigation 016 run 058
  [`ae_latent_stack_whiten_abs_recon_cov_var_ego4d`](../../investigation_016/run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/)
  (`mvbx96nv`), the intended EGO4D transfer. Later runtime changes and a single-source fixed
  EGO4D diagnostic batch limit causal and cross-source interpretation of that transfer.

Full W&B-backed read: [`ANALYSIS.md`](ANALYSIS.md). Compact cycle:
[`OBSERVATIONS.md`](OBSERVATIONS.md). Carry-forward: [`NEXT_STEPS.md`](NEXT_STEPS.md).
