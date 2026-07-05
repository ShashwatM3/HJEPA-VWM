# Next Steps - run 052 `ae_sharp_slots_recon_only`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Collapsed rep** - Present reconstruction may improve, but c_t is still weakly spread or video-independent.

## Immediate Consequence

Do not transfer this config directly into prediction. First add or restore geometry pressure, or change bottleneck architecture, until c_t is spread and video-specific.

## Linkage To The Research Chain

- This run belongs to `investigation_012`: sharp-slot bottleneck reconstruction-only test without geometry regularizers.
- Parent investigation next direction: Continue monitoring the late diagnostic window. If the same pattern holds, sharpened slots alone are not enough; geometry regularization or a stronger bottleneck design remains necessary.
- Next chronological W&B run: No later W&B run exists in the live project export.

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

**Connection:** run 052 is a clean negative on "sharp slots + reconstruction-only, no geometry, is
enough" and, more importantly, it isolated a specific, fixable failure — the decoder-side TEMPLATE
shortcut. That directly spawned investigation_013 run 053: subtract the per-position feature mean
from the reconstruction target so the shared template earns zero loss and all recon pressure must
carry video-specific content through c. Run 053 fixed the honesty (video gap +0.433, ~77%
video-conditioned) but geometry still collapsed, which then motivated the feature-space attack
(whitening, investigation_014/015). Do NOT transfer run 052's config into prediction — it would
test F_c on a collapsed present code. The post-mortem also added the honesty diagnostics
(`L_recon_shuffled_c`, `L_recon_video_gap`) used from run 053 onward.
