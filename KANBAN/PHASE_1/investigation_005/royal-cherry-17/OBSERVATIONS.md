# Observations — royal-cherry-17

**W&B:** [`0xv4upvb`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0xv4upvb) · state **killed** · runtime ~4.9h · logged steps **0–11350** (228 rows)

## Outcome

**Partial success on stability; hard failure on representation quality.**

AGC achieved its narrow goal: **zero `grad_skipped` steps** (0/228) through step 11350,
including a clean pass through the historical break window (8500). Training never froze.

But the run **actively collapsed** `c_t` after step ~8600 while weights kept updating.
Investigation 005 acceptance gates are **not** met. Do **not** resume from royal-cherry
checkpoints.

## What we intended (from DESCRIPTION)

Resume `phase1_step7500.pt` with AGC (λ_B=0.20, λ_Fc=0.10, skip threshold 150) and
**full** coarse-flow LR 2e-4. Hypothesis: clip the 30–100 grad band, skip only tail
catastrophes, complete 15k with learning intact.

## Timeline

| Phase | Steps | What happened |
|---|---|---|
| Resume healthy | 7500–8500 | `grad_skipped=0`, `grad_norm` ~2.0–2.6, `L_flow` ~0.26–0.37, copy ratio **0.85–1.44**, rank **~13.8**, std ~1.04 |
| **Pass 8500** | 8500 | `grad_norm=2.05`, `L_flow=0.26`, copy **0.85**, rank **13.89** — **better than elated at same step** |
| Loss spike | 8550–8700 | `L_flow` **0.29 → 2.77**; `agc_Fc_max_ratio` **6 → 1284**; `grad_norm` post-AGC ~8–12 |
| Rank collapse | 9000–11000 | rank **12.2 → 5.8**; copy ratio **4.3 → 12.7**; `c_std_mean` **1.04 → 0.94**; cosine **0.26 → 0.35** |
| Killed | 11350 | Pod stopped run before 15k target |

## Key numbers vs prior runs

| Metric | royal @8500 | elated @8500 | drawn @8500 |
|---|---|---|---|
| `grad_skipped` | 0 | 1 | 0 |
| `grad_norm` | 2.05 | **169.6** | 42.8 |
| `L_flow` | 0.26 | 1.83 | — |
| `coarse_vs_copy_ratio` | **0.85** | 5.72 | 4.35 |
| `c_effective_rank` | **13.89** | 13.64 | 13.67 |

| Metric | royal final (11k diag) | elated final (13.5k diag) | drawn final (14.5k diag) |
|---|---|---|---|
| `grad_skipped` rate | **0%** | 39% (frozen post-8450) | **85%** |
| `coarse_vs_copy_ratio` | **12.73** | 3.47 | 3.33 |
| `c_effective_rank` | **5.84** | 13.69 | 13.62 |
| `c_std_mean` | 0.94 | 1.10 | 1.10 |
| `c_cross_video_cosine` | 0.35 | 0.17 | 0.17 |

Post-10k gate aggregates (PHASE_1 §12):

| Gate | threshold | royal | elated | drawn |
|---|---|---|---|---|
| `coarse_vs_copy_ratio` | ≤ 0.70 | med **5.88** | med 4.35 | med 2.94 |
| `coarse_vs_batch_mean_ratio` | ≤ 0.50 | med **1.85** | med 1.30 | med 1.21 |
| `c_effective_rank` | > 60 | med **6.85** | med 13.67 | med 13.62 |

## AGC behavior

- `agc_active=1` and `agc_Fc_any_clipped=1` on **every** logged step (228/228).
- Post-8500: `agc_Fc_max_ratio` median **111**, max **1284** (33/58 steps > 100).
- Post-AGC `grad_norm` capped ~12 (global clip 0.5 on already-clipped tensors).
- `instability_warn=0` throughout — warn threshold (grad>30 ∧ L_flow>1) never fired because
  post-AGC grad stayed < 12 even when raw landscape was extreme.

## Belief evolution

### Confirmed

- **AGC prevents the grad-skip death spiral.** The elated/drawn failure mode (sustained
  `grad_skipped` with frozen weights) did not recur.
- **The underlying instability is real and step-local.** Something in the 8550–8700 window
  still blows up `L_flow` (~0.3 → ~3.0) on this resume trajectory — same era as elated's
  first skip (8450, `L_flow≈1.9`).

### New (this run)

- **Skip-free ≠ healthy training.** Elated/drawn kept **frozen** weights in a region where
  forward latents looked fine (rank ~13.6, misleading). Royal **kept updating** through the
  spike under extreme F_c AGC → **active rank collapse** (13.9 → 5.8) and copy ratio blow-up.
- **Heavy per-step F_c clipping may be harmful.** λ_Fc=0.10 with ratios up to 1284 means F_c
  gradients are crushed while B still moves; the stack can diverge in representation space
  without triggering skip or warn guards.
- **Warn guard is blind to post-AGC grad.** Need a signal on raw/pre-AGC norm, `L_flow`
  level, or `agc_Fc_max_ratio` — not post-AGC `grad_norm`.

### Wrong hypothesis (for this run)

"AGC alone + full 2e-4 flow LR on resume from 7500 completes 15k acceptably" — **false**.
Stability guard passed; acceptance gates failed catastrophically after step 8600.

## Comparison to siblings

**[`elated-snowflake-15`](../elated-snowflake-15/):** pre-8500 twin; broke on skip at 8450.
Latents post-break looked healthy but weights never moved — **illusory health**.

**[`drawn-elevator-16`](../drawn-elevator-16/):** same resume ckpt, halved LR, no AGC;
85% skips from 8550. Same illusory latent health, no learning.

**royal-cherry-17:** AGC swaps freeze for **destructive learning** — worse for gates, better
for diagnosing that the spike region must be **avoided or exited**, not merely clipped through.

## Conclusion

AGC is **necessary but not sufficient** for investigation 005. Next attempt must combine
AGC with at least one of: halved flow LR, tighter λ_Fc, optimizer reset on resume, and/or
abort when `L_flow` or `agc_Fc_max_ratio` crosses sustained danger bands. Fresh 15k from init
with AGC is also on the table if resume keeps re-entering the same basin.
