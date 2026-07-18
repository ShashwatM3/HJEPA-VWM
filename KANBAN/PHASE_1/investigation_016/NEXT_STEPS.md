# Next Steps — investigation_016

1. ~~Launch and analyze the exact EGO4D twin of run 057.~~ Completed as run 058 (`mvbx96nv`);
   verdict **Collapsed rep / template shortcut**.
2. ~~Launch and analyze
   [`run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1`](run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/)
   — the weight-1 absolute-target EGO4D arm.~~ Completed as W&B `2423b84g`; geometry finished
   healthy on the recorded batch, but reconstruction moved only 0.007 and exact-chunk conditioned
   share remained 7.66%. Scalar underweighting is not the leading explanation.
3. Keep the single-delta EGO4D residual-target arm queued: same whitening stats, seed,
   architecture, `lambda_var=0.5`, `lambda_cov=0.01`; change only
   `recon_residual_target=false -> true` (run 058 audit priority; do not drop it after 060).
4. In parallel, execute the planned
   [`run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d`](run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d/)
   recipe analogue. Treat run 058 only as historical context; add a same-commit V-JEPA companion
   before attributing a difference to encoder choice.
5. Treat `L_recon_video_gap` and its conditioned share as the primary content gate. Do not transfer
   an absolute-target EGO4D checkpoint into prediction until honesty opens.

## 2026-07-16 priority correction

Before using item 5 or assigning a final global-collapse verdict to run 060, repair the source
contract described in
[`reconstruction_floor_architecture_audit/NEXT_STEPS.md`](reconstruction_floor_architecture_audit/NEXT_STEPS.md):

1. use one fixed validation clip per EGO4D source UID and assert source uniqueness;
2. derange reconstruction codes across different source UIDs;
3. add zero/mean/B-zero-input template baselines;
4. run the cached-feature PCA/constant-predictor oracles and fixed-batch overfit ladder;
5. only then choose between residual-target, capacity, whitening-strength, and optimizer arms.

The weight-1 arm finished. Interpret it as an intended config-weight arm on a newer runtime, not a
byte-identical one-variable replay of historical run 058. Measurement repair and the cached-feature
oracle/overfit ladder now come before items 3–4; use the run-060 folder's
[`NEXT_STEPS.md`](run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/NEXT_STEPS.md)
for the full ordered sequence.

## 2026-07-17 — colleague-directed slot-capacity sweep complete

The three-arm EGO4D
[`bottleneck_slot_capacity_sweep/`](bottleneck_slot_capacity_sweep/) completed as W&B `x03xlpyl`,
`evyokqrm`, and `7pmvxrxi`. The modest monotonic training-loss improvement fails both the material
effect threshold and the recorded-batch geometry guard. More learned query slots are not the
leading reconstruction-floor fix.

Ordered action from this result:

1. do **not** launch 256 slots;
2. launch the 32-slot no-whitening EGO4D reconstruction arm as the next clean single delta;
3. design a separate channel-width/`D_c` sweep for the 1,024-to-256 squeeze rather than relabeling
   another `N_c` run as general bottleneck capacity;
4. keep alternate encoders orthogonal and retain source-diverse diagnostic repair before global
   conditioning claims.

## 2026-07-18 — active execution bundle

The human selected the combined no-whitening plus late-projection implementation and a two-arm
complete-width sweep. Execute
[`internal_memory_width_sweep/GUIDE.md`](internal_memory_width_sweep/GUIDE.md): first `M=512`, then
`M=1024`, sequentially on the one A100, with `lambda_var=lambda_cov=lambda_sigreg=lambda_slot=0`.

This supersedes the earlier recommendation for an unwhitened 256 arm in this paid bundle. Interpret
only the 512-versus-1,024 delta causally, and require correct-versus-shuffled code separation before
calling a lower raw-feature loss improved preservation.
