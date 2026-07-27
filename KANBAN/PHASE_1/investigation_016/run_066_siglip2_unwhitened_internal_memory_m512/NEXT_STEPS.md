# Next steps — Run 66, SigLIP 2 unwhitened internal-memory M=512

Before launch:

1. Complete `AGENT_FILES/SETUPS/NEW_POD.md` Sections 1–5 on the pod (through W&B login).
2. Execute [`GUIDE.md`](GUIDE.md) in order; stop at the first failed gate.

After the run completes:

1. Fill [`OBSERVATIONS.md`](OBSERVATIONS.md) with the W&B ID/URL, final SHA-256, and the present-only
   Reading Cycle B table.
2. Compare within-run diagnostics against the V-JEPA M=512 control
   ([`../run_064_unwhitened_internal_memory_m512`](../run_064_unwhitened_internal_memory_m512/),
   W&B `4biwq87o`). Do not compare raw cross-encoder cosine values as though they share a target
   space.
3. Update the parent investigation
   [`../OBSERVATIONS.md`](../OBSERVATIONS.md) / [`../DESCRIPTION.md`](../DESCRIPTION.md) with the
   encoder-substrate synthesis, and add the run row to
   [`../../README.md`](../../README.md) with the verified W&B ID/state/verdict.
