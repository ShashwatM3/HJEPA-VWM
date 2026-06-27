# Next steps — helpful-snow-25

**DROPPED from the 4-GPU re-run (2026-06-27).** With Wave 1's clean −0.005/doubling weight slope,
the λ=1.0 outcome (~0.581, no break) is near-certain, and the live W&B pull confirmed `L_flow`
*didn't even degrade* at λ=0.5 (≈0.42, same as λ=0.1) — so the one thing this run might have shown
won't appear either. It is the single most predictable run in the wave (zero expected information),
so with only 4 GPUs it's the cut: the slots go to the full `n_c` ladder (64/128/256) + the combined
run instead. Rationale: [Wave 2 NEXT_STEPS](../NEXT_STEPS.md), [`../../GUIDE.md`](../../GUIDE.md) §3c.

Resurrect only if a later result makes the weight extreme worth closing on the record, or to observe
the recon-vs-prediction trade-off directly — neither is currently warranted.

Its lasting value is forensic, not experimental: it is the architectural control proving the Wave-2
death was an external pod event, not a code/`n_c` bug (see [OBSERVATIONS.md](OBSERVATIONS.md)).
