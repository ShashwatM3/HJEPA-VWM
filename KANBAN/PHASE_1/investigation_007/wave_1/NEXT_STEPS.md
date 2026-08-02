# Next steps — Wave 1

## What this wave spawned

Wave 1 ruled out the weight and decoder axes and left the **latent axis (`n_c`)** as the only
untested binding-constraint candidate. That directly defined **[Wave 2](../wave_2/DESCRIPTION.md)**:
- the latent ladder `n_c ∈ {64, 128, 256}` (the real experiment), plus
- two **saturation extremes** added to fill the otherwise-idle GPUs and close the OFAT blind spot:
  `lambda_recon=1.0` (does the weight axis bite at the extreme?) and `n_c=256` (8×).

The pre-registered Wave-2 prediction (with odds) lives in
[`../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md`](../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md) §Section 2.

## The doubt Wave 1 carries into Wave 2

Wave 1's finding #3 (utilization-limited on `d_c`, an axis `n_c` doesn't touch) + finding #2
(blindness gap ~0.01 ≪ floor 0.585) means **even a "successful" n_c that lowers the floor probably
won't fix prediction.** So Wave 2 is framed as *mostly confirmatory*: most likely it confirms the
pivot away from reconstruction.

## Decision gate handed to Wave 2

- **n_c lowers floor AND `coarse_vs_copy_ratio` → <1 AND `c_effective_rank` rises** → latent
  capacity is the lever; go to a Stage-2 refinement.
- **anything else** (floor flat, or floor down but copy-ratio still >1) → reconstruction is the
  wrong lever → pivot to a horizon/temporal objective (see
  [`../END_OF_WAVE_2.md`](../END_OF_WAVE_2.md) §2.6).

→ Continue at [Wave 2](../wave_2/DESCRIPTION.md).
