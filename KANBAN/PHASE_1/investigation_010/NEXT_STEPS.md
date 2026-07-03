# Next Steps - investigation_010

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

The research moved to reconstruction geometry and decoder honesty, because prediction failure could no longer be blamed only on rank collapse.

## Closure / Carry-Forward Status

- Status: **CLOSED**.
- Conclusion to carry forward: Run 037 proved a key negative: c_t can be high-rank, video-specific, and stable while F_c still fails the copy gate. That is the canonical healthy-representation/no-predictor result.
- Runs covered: 036, 037.

## Follow-Up Chain

This investigation feeds into `investigation_011`: cosine reconstruction, fixed-position decoder, present-only geometry sweeps. The reason is: The residual branch showed a healthy c_t was still not enough for prediction. This branch isolated the decoder/readout side, changed the reconstruction geometry, and ran present-only sweeps to understand whether B can carry present information at all.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.
