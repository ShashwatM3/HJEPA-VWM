# Next Steps - investigation_012

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

Continue monitoring the late diagnostic window. If the same pattern holds, sharpened slots alone are not enough; geometry regularization or a stronger bottleneck design remains necessary.

## Closure / Carry-Forward Status

- Status: **RUNNING**.
- Conclusion to carry forward: Live W&B through step 5600 shows strong reconstruction progress but not healthy representation geometry: c_std_mean is still far below 1, cross-video cosine remains high, and rank has fallen into the low 20s. The run is still active, so the final verdict remains provisional.
- Runs covered: 052.

## Follow-Up Chain

This is the current frontier in the canonical KANBAN sequence.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.

## Original Notes Preserved

Final follow-up (run finished): sharp slots alone are NOT enough; geometry pressure is required. Two
threads spawned. (1) The narrow fixable failure — the decoder template shortcut — went to
investigation_013 run 053 (residual reconstruction target), which fixed honesty but not geometry.
(2) The broader "recon cannot hold geometry" (H2) confirmation drove the feature-space attack:
investigation_014 (measure the encoder rank budget) and investigation_015 run 054 (whitening +
latent-stack bottleneck on the residual recipe). The clean controlled question for any sharp-slot
follow-up is "does the sharp bottleneck IMPROVE the already-successful geometry recipe (runs 043-051)?",
not "can it replace it?".
