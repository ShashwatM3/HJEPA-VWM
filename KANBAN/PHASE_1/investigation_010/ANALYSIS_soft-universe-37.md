# Investigation 010 — Final analysis: `soft-universe-37`

**Status:** **FINISHED** (W&B state `finished`, pulled 2026-06-29)  
**W&B:** [`soft-universe-37`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2vbo6pbm) (`2vbo6pbm`) · group `inv010_residual_plumbing`  
**Commit on pod:** `443c304` (includes `230096d` optimizer plumbing)  
**Baseline:** investigation_009 Run 2 — [`graceful-river-35`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/jsh6uo7p) (`jsh6uo7p`, commit `bc77db6`)

**Run facts:** step **14950** / 15000 target · runtime **24,453 s (~6.8 h)** · manually ended (same as inv009) · `grad_skipped = instability_warn = grad_has_nan = c_dead_dim_frac = 0` at **every** logged step.

Data source: W&B MCP `query_wandb_tool` + `get_run_history_tool` — **full diag cadence** (30 points, every 500 steps, steps 0–14500). All numbers measured unless marked *derived*.

---

## 0. Executive summary

Investigation_010 reran the investigation_009 residual-prediction configuration on optimizer plumbing commit `230096d`. The run **completed the full 15k-step budget** with the **best representation health in the project** and the **same prediction failure** as inv009.

| outcome | result |
|---|---|
| **Representation** | `c_effective_rank` **61.1** — **first run to clear the Phase 1 rank gate (>60)** at step 11000; plateaued ~61 |
| **Prediction** | `coarse_vs_copy_ratio` plateau **1.077** (last value **1.063**) — still tie-by-zero; gate needs ≤0.70 |
| **vs inv009** | ratio **slightly better** (1.063 vs 1.088 final); rank **+3.1**; min beat-copy dip **worse** (0.933 vs 0.915 @3500) |
| **Plumbing hypothesis** | **Partial pass** — stability/geometry improved; prediction unchanged |
| **Verdict** | Keep `230096d` plumbing; **do not** iterate on it; pivot to prediction-side fixes (see [`NEXT_STEPS.md`](NEXT_STEPS.md)) |

---

## 1. Final master table

**Plateau** = mean of last 3 diag points (steps 13500, 14000, 14500).

| metric | inv010 final | inv009 final | Phase 1 gate | status |
|---|---|---|---|---|
| **coarse_vs_copy_ratio** | **1.063** (last) / **1.077** (plateau) | 1.088 / 1.047 | ≤ 0.70 | ❌ |
| ↳ min (excl init) | **0.933 @3500** | **0.915 @3500** | < 1 transient | brief only |
| ↳ best post-10k | **1.036 @12500** | **1.014 @12500** | < 1 sustained | neither held |
| **coarse_vs_batch_mean_ratio** | **1.140** | ~1.17 | ≤ 0.50 | ❌ |
| **c_effective_rank** | **61.08** | 57.98 | > 60 | ✅ |
| c_plus_effective_rank | 60.41 | — | EMA aligned | ✅ |
| coarse_copy_loss | 1.562 | 1.560 | ↑ = `c` moves | — |
| coarse_model_loss | 1.661 | 1.697 | ≈ copy | tie-by-zero |
| model − copy gap | **0.099** | 0.137 | negative = skill | still positive |
| *implied ρ* | **~0.23** | ~0.23 | sweet spot ~0.7–0.8 | overshot |
| c_cross_video_cosine | **0.162** | 0.170 | < 0.5 | ✅ |
| c_std_mean | 1.005 | 1.005 | ≈ 1.0 | ✅ |
| L_recon_present | 0.571 | 0.569 | floor ~0.585 | flat |
| L_recon_chat − cplus | 0.012 | 0.015 | recon blind | unchanged |
| L_flow | 1.620 | 1.639 | Δ-scale | — |
| grad_skipped | 0 | 0 | 0 | ✅ |
| agc_Fc_clipped (final) | 1 | — | observability | healthy |

---

## 2. Full trajectory (all 30 diag points)

| step | ratio | copy | model | rank | rank_plus | ρ* | cosine |
|---|---|---|---|---|---|---|---|
| 0 | 22.94 | 0.053 | 1.223 | 9.5 | 9.2 | — | 0.724 |
| 500 | 5.50 | 0.154 | 0.846 | 10.9 | 10.2 | 0.90 | 0.168 |
| 1000 | 3.88 | 0.216 | 0.836 | 11.9 | 11.8 | 0.88 | 0.138 |
| 1500 | 2.18 | 0.295 | 0.642 | 11.9 | 12.4 | 0.84 | 0.124 |
| 2000 | 1.63 | 0.359 | 0.583 | 11.4 | 12.4 | 0.80 | 0.127 |
| 2500 | 1.19 | 0.384 | 0.459 | 12.0 | 12.5 | 0.79 | 0.136 |
| 3000 | 1.06 | 0.400 | 0.425 | 12.3 | 12.8 | 0.79 | 0.113 |
| **3500** | **0.933** | 0.408 | 0.381 | 14.3 | 14.0 | **0.78** | 0.137 |
| 4000 | 1.10 | 0.509 | 0.560 | 21.9 | 19.0 | 0.75 | 0.149 |
| 4500 | 1.11 | 0.769 | 0.854 | 31.5 | 27.1 | 0.61 | 0.142 |
| 5000 | 1.20 | 0.975 | 1.173 | 38.1 | 33.7 | 0.51 | 0.137 |
| 5500 | 1.12 | 1.109 | 1.245 | 42.6 | 38.8 | 0.45 | 0.140 |
| 6000 | 1.16 | 1.227 | 1.429 | 45.6 | 42.6 | 0.39 | 0.153 |
| 6500 | 1.13 | 1.292 | 1.453 | 48.7 | 45.0 | 0.36 | 0.150 |
| 7000 | 1.12 | 1.360 | 1.528 | 50.9 | 47.3 | 0.33 | 0.138 |
| 7500 | 1.12 | 1.398 | 1.566 | 52.9 | 49.5 | 0.31 | 0.150 |
| 8000 | 1.04 | 1.431 | 1.483 | 54.4 | 51.3 | 0.29 | 0.155 |
| 8500 | 1.06 | 1.449 | 1.537 | 56.0 | 53.4 | 0.28 | 0.157 |
| 9000 | 1.07 | 1.465 | 1.568 | 56.8 | 54.8 | 0.28 | 0.156 |
| 9500 | 1.09 | 1.482 | 1.622 | 57.8 | 56.2 | 0.27 | 0.155 |
| 10000 | 1.08 | 1.491 | 1.610 | 58.7 | 57.0 | 0.26 | 0.155 |
| 10500 | 1.06 | 1.508 | 1.597 | 59.8 | 57.9 | 0.25 | 0.160 |
| **11000** | 1.10 | 1.526 | 1.679 | **60.2** | 58.6 | 0.24 | 0.159 |
| 11500 | 1.11 | 1.532 | 1.695 | 60.4 | 59.2 | 0.24 | 0.162 |
| 12000 | 1.07 | 1.546 | 1.661 | 60.6 | 59.6 | 0.23 | 0.157 |
| **12500** | **1.036** | 1.552 | 1.608 | 60.8 | 59.9 | 0.23 | 0.161 |
| 13000 | 1.087 | 1.552 | 1.688 | 61.0 | 60.2 | 0.23 | 0.161 |
| 13500 | 1.066 | 1.557 | 1.659 | 61.0 | 60.3 | 0.23 | 0.163 |
| 14000 | 1.102 | 1.562 | 1.721 | 61.1 | 60.3 | 0.23 | 0.162 |
| **14500** | **1.063** | 1.562 | 1.661 | **61.1** | 60.4 | **0.23** | 0.162 |

*ρ derived: `1 − copy/(2·std²)` using logged `c_std_mean`.*

---

## 3. Phase structure (complete run)

### Phase A — Ratio collapse (steps 0–2500)
Init spike → ~1.2. SIGReg warmup (sigreg_scale 0→1 by step 2000) causes marginally slower early descent vs inv009 but converges by 2500.

### Phase B — Predictive window (steps 2500–4000)
**Step 3500: ratio 0.933** — `F_c` beats zero-residual copy. ρ ~0.78. Rank still low (14.3). This is the only sustained sub-1 region.

### Phase C — Rank surge + decorrelation (steps 4000–5500)
Rank 14 → 43; copy_loss 0.41 → 1.11; ρ 0.78 → 0.45; ratio returns above 1.0 permanently after step 4000.

### Phase D — Tie-by-zero plateau (steps 5500–14950)
- Ratio oscillates **1.04–1.11**, plateau **1.077**
- `coarse_model_loss` tracks `coarse_copy_loss` (gap ~0.08–0.14)
- Rank climbs **43 → 61.1**, crossing **60 at step 11000**
- ρ settles at **~0.23** (same as inv009 terminal)
- Brief dips at 8000 (1.037) and 12500 (1.036) — noise, not sustained skill
- inv009 had a deeper late dip at 12500 (**1.014**) — inv010 never got as close post-10k

---

## 4. Comparison vs `graceful-river-35` (matched steps)

| step | inv010 ratio | inv009 ratio | inv010 rank | inv009 rank |
|---|---|---|---|---|
| 3500 | 0.933 | **0.915** | 14.3 | 14.1 |
| 5000 | 1.200 | 1.228 | 38.1 | 37.2 |
| 9000 | 1.070 | 1.073 | 56.8 | 55.0 |
| 11000 | 1.100 | 1.098 | **60.2** | 56.8 |
| 12500 | 1.036 | **1.014** | **60.8** | 57.6 |
| 14000 | 1.102 | 1.088 | **61.1** | 58.0 |
| final | **1.063** | 1.088 | **61.1** | 58.0 |

**Net:** Parallel trajectories through step 9000. inv010 pulls ahead on rank after step 10k (+3 rank points at end) and ends with a slightly lower ratio. inv009 had the better beat-copy moments (3500 and 12500).

---

## 5. What plumbing (`230096d`) changed — final assessment

| change | final-run evidence |
|---|---|
| SIGReg warmup | Smooth early training; `c_plus_effective_rank` within ~1 of online at end |
| Decay/no-decay groups | No instability; `grad_skipped=0` |
| AGC exclusions | `agc_Fc_clipped` 3–6 mid-run, **1** at end; `agc_B_clipped=0` throughout |
| F_c slot/type embeddings | Cannot isolate; trajectory matches inv009 |
| Online+EMA rank logging | Gap ≤1.4 rank points in final 5k steps |

**Helped:** rank gate (+3 vs inv009), cross-video cosine (−0.008), training stability, EMA alignment.  
**Did not help:** sustained ratio < 1, ρ band, Δ̂ → 0 tie, recon blindness.

---

## 6. Verdict vs investigation_010 hypotheses

| GUIDE criterion | Final result | Verdict |
|---|---|---|
| **Win:** ratio stably < 1, rank > 60, cosine low | ratio 1.077 plateau; rank 61.1; cosine 0.162 | **Not met** (rank ✅, ratio ❌) |
| **Partial:** ratio ~1, cleaner grads/rank/cosine | yes — all three | **Met** |
| **Same failure:** ratio ~1.05–1.10, model ≈ copy | ratio 1.077; gap 0.099 | **Met** |
| **Regression** | none | **Not met** |

**Answer:** Plumbing hardened the loop and cleared the rank gate. It did **not** make `F_c` skillfully beat zero-residual copy.

---

## 7. Phase 1 acceptance after full run

| gate | inv010 value | pass? |
|---|---|---|
| `c_effective_rank > 60` | **61.1** @ step 11000+ | ✅ |
| `coarse_vs_copy_ratio ≤ 0.70` | **1.077** plateau | ❌ (need ~35% reduction) |
| `coarse_vs_batch_mean_ratio ≤ 0.50` | **1.140** | ❌ |
| No collapse | cosine 0.16, dead_dim 0, std 1.0 | ✅ |
| Stability | grad_skipped 0 all steps | ✅ |

**One of three Phase 1 prediction/health gates cleared.** Do not start Phase 2.

---

## 8. What changed from the mid-run (step 9100) analysis

The earlier live analysis (at ~61% progress) projected inv010 would mirror inv009 through 15k. The **completed run confirms that projection** with one positive surprise:

| projection @9100 | actual @14950 |
|---|---|
| ratio ~1.05–1.10 | plateau **1.077**, final **1.063** ✓ |
| rank → 57–59 | rank **61.1** — **exceeded** (cleared 60 gate) |
| ρ → 0.23 | ρ **0.23** ✓ |
| no sustained ratio < 1 | confirmed — best post-10k **1.036** @12500 |

---

## 9. Recommended next steps

See [`NEXT_STEPS.md`](NEXT_STEPS.md) for the full investigation_011 roadmap. Headline priorities:

1. **Anti-collapse on Δ̂** (forbid zero-residual shortcut)
2. **Govern ρ** — drop/reduce `λ_recon_pred` or recon entirely
3. **Horizon `k` sweep** with residual prediction
4. **Do not** re-run plumbing or SIGReg/recon sweeps

---

*Final W&B pull: 2026-06-29. Run state: `finished`, step 14950, runtime 6.8 h.*
