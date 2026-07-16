# Next Steps — run 058

1. ~~Execute the 15,000-step EGO4D transfer and analyze it.~~ Completed as W&B `mvbx96nv`; see
   [`ANALYSIS.md`](ANALYSIS.md).
2. ~~Execute the human-requested absolute-target weight extreme at
   `lambda_recon=1.0`.~~ Completed as W&B `2423b84g`; see
   [`../run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/ANALYSIS.md`](../run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/ANALYSIS.md).
   It improved reconstruction by only `0.0073` versus run 058 and did not establish a
   meaningful video-conditioning gain. The residual-target control below remains necessary.
3. Run the exact EGO4D recipe with `recon_residual_target=true` and keep
   `lambda_recon=0.05`. This remains the pre-registered response to the measured near-zero
   video gap.
4. Before treating a full weight ladder as decisive, add per-loss/per-module pre-clipping
   gradient and update-norm diagnostics. The current combined `grad_norm` cannot test whether B's
   reconstruction gradient is small, conflicting with geometry, or normalized away by Adam.
5. If honest residual-target reconstruction still plateaus, complete the weight ladder
   `{0.05, 0.20, 1.00}` under residual target. Rank arms by video gap and representation
   gates before raw reconstruction.
