# Next Steps - investigation_011

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

The next question is whether sharper slot attention or a different bottleneck geometry can retain content without needing external geometry regularizers, then transfer that geometry back into full prediction.

## Closure / Carry-Forward Status

- Status: **CLOSED**.
- Conclusion to carry forward: Present reconstruction can be strong, and SIGReg plus small covariance can raise c_effective_rank above 100 while preserving video specificity. But those are present-side wins; no full-prediction run has inherited them and passed the Phase 1 copy/batch-mean gates.
- Runs covered: 038, 039, 040, 041, 042, 043, 044, 045, 046, 047, 048, 049, 050, 051.

## Follow-Up Chain

This investigation feeds into `investigation_012`: sharp-slot bottleneck reconstruction-only test without geometry regularizers. The reason is: Investigation 011 produced good present-only geometry with active regularizers. The sharp-slot run tests whether an architectural attention change alone can make c_t decodable, spread, and video-specific.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.

## Original Notes Preserved

The open question this investigation handed forward: can a sharper bottleneck ARCHITECTURE retain
this rich present geometry WITHOUT the external SIGReg/covariance regularizers, so it could later be
transferred cheaply into prediction? That is exactly investigation_012's run 052 (sharpened slot
attention, reconstruction-only, all geometry regularizers off). The two durable code artifacts from
this investigation — the cosine reconstruction loss and the fixed-position decoder — carry into every
run afterward (052, 053, 054). The standing transfer goal remains: take a geometry-holding present-only
bottleneck into a full-prediction run and beat the copy/batch-mean gates.
