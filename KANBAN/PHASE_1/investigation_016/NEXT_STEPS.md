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

## 2026-07-16 — colleague-directed slot-capacity sweep

The next planned paid branch is now the six-arm
[`bottleneck_slot_capacity_sweep/`](bottleneck_slot_capacity_sweep/): same-commit
`N_c=32/64/128` ladders on EGO4D and SSv2 with `D_c=256`, V-JEPA, full whitening, and
`lambda_recon=1.0` fixed. This deliberately acts on the bottleneck-capacity intuition without
requiring the audit's full probe ladder first. Only launch-safety resource/provenance checks remain
prerequisites; no-whitening and encoder-substrate experiments stay orthogonal.
