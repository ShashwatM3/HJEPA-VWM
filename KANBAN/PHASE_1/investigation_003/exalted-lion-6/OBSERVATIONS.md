# Observations — exalted-lion-6

## Outcome

**Diagnostic success.** Training healthy; **rank collapse reproduced** at small scale.

## Key numbers (step 0 → 400)

| Metric | step 0 | step 400 | Reading |
|---|---|---|---|
| `L_flow` | 2.87 | 1.40 | learning |
| `coarse_vs_copy_ratio` | 171 | ~3.0 | high at init (EMA≈online artifact), then falling |
| `c_std_mean` | 0.50 | 0.83 | alive, below floor target |
| `c_cross_video_cosine` | 0.72 | 0.25 | videos distinguishable |
| `c_effective_rank` | 9.0 | ~10.3 | **bouncing ~8, not climbing** |
| `c_slot_diversity_rank` | 16.2 | 16.6 | slots still differentiate (pre per-head fix) |
| `grad_skipped` | 0 | 0 | clean |

## Instrumentation caveat (corrected later)

`c_attn_entropy` read ~1.0 (uniform) at every tick, but `c_slot_diversity_rank ≈ 16`
contradicted truly uniform attention. Cause: head-averaged attention weights masked
per-head selectivity. Fixed in commit `a96c0d6` before Run A on full SSv2.

## Interpretation

- Collapse is **persistent objective failure**, not init transient
- Init knobs confirmed non-lever at this scale
- Spawned VICReg-C path ([investigation_004](../investigation_004/))

Source: chat step logs at lines ~5615–5644 in `COMPLETE_FULL_CHAT`.
