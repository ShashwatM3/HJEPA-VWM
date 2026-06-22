# Observations — init-fixes-full-ssv2-run-2

## Outcome

**Negative for collapse fix.** Init + full data helped but did not solve dimensional
collapse. Run stopped early (~step 3500) once the data-confound and calibration
questions were answered (VICReg path **Run A**).

## Step-3500 snapshot (chat record)

Full SSv2, `lambda_cov=0` (logs `L_cov` only), per-head entropy fix active:

| Metric | Value | Healthy target |
|---|---|---|
| `c_effective_rank` | **8.7** / 256 | >60 |
| `c_slot_diversity_rank` | **1.62** / 32 | →32 |
| `c_attn_entropy_min` | **0.96** | low |
| `c_cross_video_cosine` | 0.25 | <0.5 |
| `c_std_mean` | 0.84 | ~1.0 |
| `coarse_vs_copy_ratio` | **7.42** | <1.0 |
| `L_flow` / `L_cov` | 1.10 / 20.18 | — (calibration) |

## Interpretation

- **Not data-limited:** 42× more data (tiny → full 169k) moved rank ~5 → ~8.7 only
- **Dominant failure:** slot redundancy + uniform attention (per pre-registered diagnostic rule)
- VICReg-C targets cross-video correlation (rank 8.7) but **not** within-video slot collapse
- `λ_cov` calibration ready: start **0.0027** (5% of `L_flow`)

Same physical run as BRIEF "Run 2" and VICReg "Run A". No W&B display name in chat.

Source: `COMPLETE_FULL_CHAT` lines ~6200–6241, ~9268–9271.

## Surprises

BRIEF summary cited `slot_diversity_rank ~16` — that reflected pre–per-head-fix
readings (`exalted-lion-6` era). With fixed metrics on full data, slot collapse
(1.62) was the louder signal.
