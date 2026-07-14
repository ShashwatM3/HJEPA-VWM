# Next Steps — investigation_016

1. ~~Launch and analyze the exact EGO4D twin of run 057.~~ Completed as run 058 (`mvbx96nv`);
   verdict **Collapsed rep / template shortcut**.
2. Open the single-delta EGO4D residual-target arm: same whitening stats, seed, architecture,
   `lambda_var=0.5`, `lambda_cov=0.01`, and `lambda_recon=0.05`; change only
   `recon_residual_target=false -> true`.
3. In parallel, execute the planned
   [`run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d`](run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d/)
   recipe analogue. Treat run 058 only as historical context; add a same-commit V-JEPA companion
   before attributing a difference to encoder choice.
4. Treat `L_recon_video_gap` and its conditioned share as the primary content gate. Do not transfer
   this absolute-target checkpoint into prediction.
5. Defer a high reconstruction-weight arm until the residual target makes reconstruction honest
   and gradient-source diagnostics can distinguish scale from objective conflict.
