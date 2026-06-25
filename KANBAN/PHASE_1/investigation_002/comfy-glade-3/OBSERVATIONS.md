# Observations — comfy-glade-3

## Outcome

**Passed** as a dense-logged smoke. Both pre-fix (this run, `efficient-aardvark-2`)
and post-fix (`charmed-haze-4`) smokes produce bit-identical diagnostics on the
deterministic seed-42 batch, so this run carries no separate throughput conclusion —
its value was confirming the `--log-every` flag worked and the per-step trajectory
was sane.

## Evidence

- ~5m29s runtime for 200 steps, i.e. still the ~1.6 s/step pre-fix regime
  (selective decode `5e78caa` had not yet landed on 06-09).
- Step-0 diagnostics identical to `efficient-aardvark-2` (same seed-42 diag batch):
  `L_flow≈2.87`, `L_var≈0.43`, `c_effective_rank≈9.0`, `c_cross_video_cosine≈0.72`,
  `coarse_vs_copy_ratio≈171` at init.
- Per-step `loss`/`L_flow` jitter visible but no downward trend over 200 steps — as
  expected for a throughput/plumbing smoke, not a training run.

## Interpretation

Operational check only; the substantive throughput comparison is
[`efficient-aardvark-2`](../efficient-aardvark-2/) (pre-fix, 1.66 s/step) vs
[`charmed-haze-4`](../charmed-haze-4/) (post-fix, 1.41 s/step). The identical
diagnostics across all three smokes are the evidence that the dataloader refactor was
**behavior-preserving** — the diag batch is deterministic, so any change in model
inputs would have moved these numbers.
