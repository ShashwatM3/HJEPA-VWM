# Run 060 — `ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1` (inv016)

**Current W&B display name:**
`Investigation 16 · Whitened latent stack EGO4D · Reconstruction weight 1.00`

**W&B run:** [`2423b84g`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2423b84g)

## Status

FINISHED — full 15,000-step schedule completed; final training row at step 14,950 and final
diagnostic at step 14,500. Reading Cycle B verdict: **stable, geometrically healthy present-only
autoencoder with weak exact-chunk conditioning and source-confounded diagnostics**. Full result:
[`ANALYSIS.md`](ANALYSIS.md).

## Hypothesis

Run 058's present-only whitened absolute-target AE recipe on EGO4D (cov+var geometry,
`lambda_recon=0.05`) failed the honesty and geometry gates with a near-zero video gap. Raising
only the reconstruction weight to `lambda_recon=1.0` (20×) tests whether the absolute-target
objective was simply under-weighted relative to the geometry terms on this substrate.

If honesty opens and geometry holds, the EGO4D failure was a relative-weight issue. If raw
recon improves while `L_recon_video_gap` stays near zero / rank keeps collapsing, stronger
weight alone does not fix the template shortcut (consistent with the run-058 audit).

## Intended control and causal limitation

| | Run 058 | This run (060) |
|---|---|---|
| W&B display name | `ae_latent_stack_whiten_abs_recon_cov_var_ego4d` | `Investigation 16 · Whitened latent stack EGO4D · Reconstruction weight 1.00` |
| W&B id | `mvbx96nv` | `2423b84g` |
| KANBAN | [`../run_058_...`](../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/) | this folder |
| `--data` | `ego4d` | `ego4d` |
| whitening stats | `logs/whiten/whiten_stats_ego4d_train_seed42.pt` | **same file** (reuse) |
| `lambda_recon` | **0.05** | **1.0** |

Everything else matches run 058:

```text
present_recon_only = true
recon_residual_target = false
whiten_features = true
lambda_recon = 1.0           lambda_recon_pred = 0.0
lambda_var = 0.5             lambda_cov = 0.01
lambda_sigreg = 0.0          lambda_slot = 0.0
horizon_k = 12               n_c = 32
decoder_dim = 512            decoder_blocks = 4
seed = 42                    steps = 15000
lr_bottleneck = 1e-4         lr_coarse_flow = 1e-4
encoder = vjepa2_vitl16 (default)
```

The planned config delta was only `lambda_recon: 0.05 -> 1.0`, but the executed runs are not a
byte-identical one-variable pair. Run 058 used commit `21d2aa8...`; run 060 used commit
`a27cd84dd67783f7ee8e68bb68e7c1a38a309534`, after deterministic initialization/data-order,
encoder pinning, whitening-envelope, and provenance changes. Treat the numerical comparison as
observational until a same-commit `0.05` companion is run.

## Result

The 20x reconstruction weight did not materially lower the whitened reconstruction floor:
`L_recon_present` ended at `0.67091` versus historical run 058's `0.67825`. Late geometry was much
healthier (`c_effective_rank=84.36`, `c_std_mean=0.800`, recorded cross-sample cosine `0.479`,
centered slot rank `30.81`), but correct-code dependence remained weak: shuffled-code loss
`0.69736`, gap `0.02644`, or 7.66% of the learned improvement conditioned on the exact chunk.

All 16 fixed validation chunks share one EGO4D source UID, so the last two values are within-source
adjacent-chunk measurements, not global cross-video honesty. The result rejects simple scalar
underweighting as the leading explanation, but it does not establish global template collapse or
license prediction transfer.

## Note on priority vs audit

Run 058's [`ANALYSIS.md`](../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md)
prefers residual-target before a weight sweep. This arm is an explicit human request for the
weight extreme on the absolute-target recipe; it does not replace the residual-target control.

## Parent

[`../DESCRIPTION.md`](../DESCRIPTION.md). Exact launch: [`GUIDE.md`](GUIDE.md). Execution design:
[`PLAN.md`](PLAN.md).
