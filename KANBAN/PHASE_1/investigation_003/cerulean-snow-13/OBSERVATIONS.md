# Observations — cerulean-snow-13

## Key numbers (from local log + BRIEF_V0_3)

| Step | `c_std_mean` | `c_cross_video_cosine` | `c_effective_rank` | `coarse_vs_copy_ratio` | `grad_skipped` |
|---|---|---|---|---|---|
| 0 | 0.50 | 0.72 | 9.47 | 62.1 | 0 |
| 4500 | 0.97 | 0.26 | 12.11 | **0.97** | 0 |
| 5000 | 1.00 | 0.22 | 12.64 | 1.37 | 0 |
| 6500 | 1.04 | 0.24 | 13.67 | **0.95** | 0 |
| 6900 | — | — | — | — | 0 (log ends) |

Throughout: `grad_norm` ~2–3, no sustained skips, `c_dead_dim_frac=0`.

## What worked

- Variance floor at 0.5 **wins over** `L_flow` collapse pressure
- Video-specific codes (cosine ~0.20–0.26)
- Rank rising to **~13.7+**
- **Beats copy baseline** (ratio < 1 in best windows)
- Training stable under post-Run-1 hyperparams

## What surprised us

Copy ratio can wobble batch-to-batch (e.g. 1.37 at 5000) — use diagnostic cadence
and trends, not single steps.

## Interpretation

**Validates investigation 003 conclusion.** This is the Phase 1 Stage 1 config to
carry forward. Open question: rank plateau level over full 15k (feeds investigation 005).

## Note on log length

Workspace `output.log` ends at step 6900; chat/W&B record run as healthy further.
Treat as **best known checkpoint window**, not necessarily final step 15000.
