# Observations — copper-sky-12

## Outcome

**Stopped ~step 4300.** Slot loss now bites mechanically but predictive quality
did not improve — Goodhart + grad spikes.

## Key numbers (chat log, June 16)

| Step | `c_effective_rank` | `c_slot_diversity_rank` | `c_cross_video_cosine` | `coarse_vs_copy_ratio` | `grad_skipped` |
|---|---|---|---|---|---|
| 3500 | 13.5 | 7.8 | 0.76 | 7.55 | 0 |
| 4000 | 15.1 | 8.0 | 0.71 | 6.96 | 0 |
| 4300 | — | — | — | 11.46 | **1** (grad ~232) |

Earlier in run: `L_slot` dropped from ~1.0 to ~0.17–0.22; `c_attn_entropy_min`
fell to ~0.86–0.90 (attention less uniform).

## Interpretation

Centering fixed the instrumentation bug but **confirmed slot loss is the wrong lever** —
rank could rise while copy ratio stayed bad (~5–11) and cross-video cosine stayed high
(~0.7+). Team pivoted to **`lambda_var=0.5`**.

## Surprises

Slot metric can look better while semantics get worse — always pair with
`c_cross_video_cosine` and copy ratio. Rank 14–18 here vs Run A's 8.7 is **not**
a clean win when copy ratio degrades.

Source: `COMPLETE_FULL_CHAT` lines ~6620–6672.
