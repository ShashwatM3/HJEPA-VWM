# Observations — centered-slot-k12-run-5

## Metrics (BRIEF_V0_3)

- Slot metric: improved (loss now bites)
- `c_effective_rank`: **~4.8** (worse)
- `c_cross_video_cosine`: **~0.84**
- `grad_norm`: spikes to **~10⁵** territory

## Interpretation

Centering fixed the instrumentation bug but **confirmed slot loss is the wrong lever** —
Goodhart + instability. Team pivoted to **`lambda_var=0.5`**.

## Surprises

Slot metric can look better while semantics get worse — always pair with
`c_cross_video_cosine` and copy ratio.
