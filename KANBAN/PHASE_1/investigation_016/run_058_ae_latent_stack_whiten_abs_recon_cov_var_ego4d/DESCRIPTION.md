# Run 058 — `ae_latent_stack_whiten_abs_recon_cov_var_ego4d` (inv016)

**Current W&B run name:** `ae_latent_stack_whiten_abs_recon_cov_var_ego4d`
**W&B:** [`mvbx96nv`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/mvbx96nv)

## Status

FINISHED — all 15,000 steps completed. Reading Cycle B verdict: **Collapsed rep / template
shortcut**. Full result and the reconstruction-weight audit: [`ANALYSIS.md`](ANALYSIS.md).

## Hypothesis

Run 057's exact recipe on `--data ego4d` (plus EGO4D whitening stats) reproduces the
same qualitative geometry + honesty pattern that SSv2 run 057 settled:
high `c_effective_rank` under cov+var alone, healthy centered slot rank, video-specific
recon (nontrivial `L_recon_video_gap`), stable training.

If it does, the inv015 present-only AE recipe is substrate-portable and EGO4D becomes
the next place to push prediction. If it does not, the failure localizes to dataset
statistics (whitening / motion / egocentric distribution), not to the recipe logic.

## Control (byte-identical except dataset)

| | SSv2 run 057 | This run (058) |
|---|---|---|
| W&B display name | `ae_latent_stack_whiten_abs_recon_cov_var` | `ae_latent_stack_whiten_abs_recon_cov_var_ego4d` |
| W&B id | `cdvp6hou` | *(assigned at launch)* |
| KANBAN | [`../investigation_015/run_057_...`](../../investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/) | this folder |
| `--data` | `ssv2` | `ego4d` |
| whitening stats | `logs/whiten/whiten_stats_ssv2_train_seed42.pt` | `logs/whiten/whiten_stats_ego4d_train_seed42.pt` |

Everything else matches run 057 / W&B config of `cdvp6hou`:

```text
present_recon_only = true
recon_residual_target = false
whiten_features = true
lambda_recon = 0.05          lambda_recon_pred = 0.0
lambda_var = 0.5             lambda_cov = 0.01
lambda_sigreg = 0.0          lambda_slot = 0.0
horizon_k = 12               n_c = 32
decoder_dim = 512            decoder_blocks = 4
seed = 42                    steps = 15000
lr_bottleneck = 1e-4         lr_coarse_flow = 1e-4
```

## Parent

[`../DESCRIPTION.md`](../DESCRIPTION.md). Analysis of the SSv2 control:
[`../../investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/ANALYSIS.md`](../../investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/ANALYSIS.md).
