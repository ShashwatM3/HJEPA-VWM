# Next Steps — investigation_016

## Current disposition (2026-07-26)

Investigation 016 is closed. The numbered and dated sections below are preserved decision history,
not a simultaneous active queue.

1. Continue the external `N_c×D_c` question only in
   [investigation 017](../investigation_017/). Reuse the exact tried `32×256` scientific
   configurations `guiduvjp`, `ufbeokj2`, and `fiactcw6`; do not pay for duplicate centers.
2. Keep source-unique fixed validation and cross-source code derangement as an explicit
   measurement prerequisite before making any global EGO4D collapse/preservation claim.
3. Do not treat the old run-059 SigLIP proposal or the residual-target arm below as current launch
   instructions. They remain unlaunched historical proposals unless a new investigation
   re-registers them.
4. Do not infer DINO prediction readiness from runs 069–071. All are present-only.

## Historical queue and decisions

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

## 2026-07-18/19 — execution bundle complete

The human selected the combined no-whitening plus late-projection implementation and a two-arm
complete-width sweep. The exact [`GUIDE.md`](GUIDE.md) completed successfully: first
[`M=512`](run_064_unwhitened_internal_memory_m512/), then
[`M=1024`](run_065_unwhitened_internal_memory_m1024/), sequentially on the one A100, with
`lambda_var=lambda_cov=lambda_sigreg=lambda_slot=0`.

This supersedes the earlier recommendation for an unwhitened 256 arm in this paid bundle. Interpret
only the 512-versus-1,024 delta causally, and require correct-versus-shuffled code separation before
calling a lower raw-feature loss improved preservation. The 1,024 arm missed the registered loss
and gap thresholds, so keep M=512.

Ordered action from this result:

1. carry the 512-wide late-projection bottleneck as the practical default;
2. repair source-unique validation and cross-source derangement before any global preservation or
   collapse claim;
3. do not use the historical whitened `~0.67` versus raw `~0.26` loss difference as a causal or
   numerically equivalent comparison;
4. isolate the remaining external-rate (`D_c`/final code size), decoder, and objective-honesty
   hypotheses one at a time;
5. restore geometry pressure in a dedicated arm if the goal is a healthy representation rather
   than reconstruction-only low-rank compression.

## 2026-07-21 — current handoff

Runs 66–69 completed the immediate raw-geometry and encoder-substrate controls. Keep `M=512`, the
late projection, three latent blocks, raw features, and the pinned encoder adapters. The next paid
question is no longer another internal-width or no-geometry arm: execute
[investigation 017](../investigation_017/), which sweeps `N_c × D_c` independently for V-JEPA2,
SigLIP 2, and DINOv3 with covariance plus variance enabled throughout.
