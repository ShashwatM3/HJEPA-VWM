# Observations — royal-cherry-17

**W&B:** [`0xv4upvb`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/0xv4upvb)  
**State:** killed · **Runtime:** 17549s (~4h 52m) · **228 logged rows** · steps **0–11350**

Data source: full unsampled history via `run_history.py` (`scan_history`).

---

## Run identity (corrected)

| Claim | Actual |
|---|---|
| Resume from `phase1_step7500.pt` | **No** — fresh run from step 0 |
| Evidence | `lr_mult=0.00067` at step 0 (warmup); step-0 `loss`/`L_flow` identical to elated; history starts at step 0, not 7500 |
| Implication | AGC was active **every step from init**, not only post-resume; weights diverged slightly from elated throughout |

---

## Outcome (one sentence)

**AGC eliminated all grad skips and cleanly passed step 8500, then the run fell off a cliff at step 8600 and actively destroyed `c_t` rank/slot structure over the next ~2500 steps while the optimizer kept updating.**

Investigation 005 gates: **not met**. Do **not** use royal-cherry checkpoints.

---

## Phase timeline

| Phase | Steps | n rows | What happened |
|---|---|---|---|
| P0 Warmup | 0–1499 | 30 | `L_flow` 2.9→0.9; copy ratio 62→3.6; rank ~9–10 |
| P1 Early train | 1500–4499 | 60 | `L_flow` med 0.43; rank climbing 7.9→11.4 |
| P2 Rank climb | 4500–7499 | 60 | `L_flow` med **0.31**; rank → **13.8**; copy ~0.9–1.4 |
| P3 Pre-break plateau | 7500–8350 | 18 | Healthy: `L_flow` med 0.28, grad med 2.2, agcFc med 7.3 |
| P4 Transition | 8400–8550 | 4 | Still healthy: copy **0.85**, rank **13.89** @8500 |
| **P5 Cliff** | **8600–8950** | 8 | `L_flow` **0.29→3.08** in 350 steps; agcFc max **1284** @8750 |
| P6 Collapse | 9000–11350 | 48 | Rank **13.9→5.8**; copy **4.3→12.7**; slot rank **19.3→5.7** |

---

## The cliff (steps 8450–9050, every 50 steps)

This is the critical window. One batch era destroys the run.

| Step | L_flow | grad (post-AGC) | agc_Fc ratio | agc_B ratio | copy ratio | rank |
|---|---|---|---|---|---|---|
| 8450 | 0.324 | 2.30 | 8 | 7 | — | — |
| 8500 | **0.262** | 2.05 | 6 | 9 | **0.851** | **13.89** |
| 8550 | 0.290 | 2.28 | 6 | 10 | — | — |
| **8600** | **1.111** | 7.67 | 19 | 149 | — | — |
| 8650 | 1.944 | 7.92 | 120 | 96 | — | — |
| 8700 | 2.774 | 11.93 | 125 | **2231** | — | — |
| **8750** | 2.819 | 11.60 | **1284** | 1292 | — | — |
| 8800 | 3.025 | 9.70 | 208 | 450 | — | — |
| 9000 | 3.107 | 11.70 | 119 | **3229** | 4.340 | 12.18 |
| 9050 | 3.019 | 10.56 | 193 | 958 | — | — |

**First L_flow jump >0.5:** step **8600** (+0.82 from 0.290).  
Not step 8500 — that step was the **best** of the entire run.

---

## Contrast with elated at the same era (no AGC, same seed)

Weights differ (AGC from step 0), so same global step ≠ same forward response:

| Step | royal L_flow | elated L_flow | royal grad | elated grad | elated skip |
|---|---|---|---|---|---|
| 8350 | 0.282 | 0.294 | 2.11 | 3.24 | 0 |
| 8400 | **0.290** | **1.345** | 2.14 | **30.66** | 0 |
| 8450 | 0.324 | 1.932 | 2.30 | 64.72 | **1** |
| 8500 | **0.262** | 1.830 | **2.05** | **169.62** | 1 |
| 8600 | 1.111 | 1.829 | 7.67 | 92.66 | 1 |
| 8700 | 2.774 | 2.030 | 11.93 | 222.82 | 1 |
| 8750 | 2.819 | 1.815 | 11.60 | 109.79 | 1 |

**Reading this table:**

1. **8400–8500:** elated hits the famous spike; royal **does not** (AGC-shaped weights still healthy on those batches).
2. **8600:** royal hits **its own** cliff — elated is already in a skip spiral.
3. **Post-8600:** royal keeps **learning** (0% skips); elated **freezes** (skip every step).

---

## What AGC actually did

### Grad stability — complete success

| Metric | royal | elated | drawn |
|---|---|---|---|
| `grad_skipped` | **0 / 228 (0%)** | 109 / 278 (39%) | 127 / 150 (85%) |
| grad p95 (all steps) | **10.1** | 214.0 | 221.6 |
| grad max | **12.35** | 467.15 | 366.53 |
| post-8500 grad med | **9.76** | 122.26 | 86.77 |

### AGC clipping intensity

| Metric | Pre-8500 (n=170) | Post-8500 (n=58) |
|---|---|---|
| `agc_Fc_max_ratio` med | 7.6 | **111.2** |
| `agc_Fc_max_ratio` max | 11.4 | **1284.1** |
| `agc_B_max_ratio` med | 9.5 | **239.3** |
| `agc_B_max_ratio` max | 34.0 | **3228.9** |
| Steps with agc_Fc > 100 | 0 | **33** |
| Steps with agc_B > 500 | 0 | **6** |

`agc_Fc_any_clipped=1` on **every** logged step (228/228).  
Top spike: step **8750**, agc_Fc=**1284**, agc_B=**1292**.

### Guards that failed to fire

| Guard | Threshold | Why it never triggered |
|---|---|---|
| `grad_skipped` | post-AGC norm > 150 | post-AGC norm capped ~12 |
| `instability_warn` | grad > 30 **and** L_flow > 1 | post-AGC grad always < 12 |

**L_flow > 1 from step 8600 onward** — but warn requires grad > 30 on the **post-AGC** norm.

---

## Representation collapse (the real failure)

### Coarse flow head — forward blow-up

At step 8500 (healthy): `coarse_model_loss=0.204`, `coarse_copy_loss=0.240`, ratio **0.85**.  
At step 9000 (diag): `coarse_model_loss=**3.106**`, `coarse_copy_loss=0.716`, ratio **4.34**.  
At step 11000: `coarse_model_loss=1.822`, `coarse_copy_loss=0.143`, ratio **12.73**.

The copy baseline stayed easy (~0.14–0.24 pre-break); **F_c's prediction** blew up.

### Rank and slot structure

| Step | `c_effective_rank` | `c_slot_diversity_rank` | `c_std_mean` | `c_cross_video_cosine` |
|---|---|---|---|---|
| 8500 | **13.89** | **19.27** | 1.044 | 0.262 |
| 9000 | 12.18 | 13.49 | 0.925 | 0.412 |
| 9500 | 8.25 | 9.12 | 0.973 | 0.335 |
| 10000 | 7.77 | 7.18 | 0.943 | 0.374 |
| 11000 | **5.84** | **5.73** | 0.945 | 0.352 |

Rank and slot-diversity rank track together — **slot collapse**, not just a scalar std issue.  
`c_dead_dim_frac=0` throughout (variance floor holding per-dim).  
Attention entropy stays ~0.92–0.93 (not the near-1.0 uniform-attention failure from inv. 003).

### L_var spike during cliff

`L_var` jumps at step 9200 (**0.138**) while `c_std_mean` still ~0.94 — variance floor
engaging on a batch where abstract spread contracted.

---

## PHASE_1 §12 gates — never passed post-10k

Best copy ratio entire run: **0.851 @ step 8500** (gate ≤ 0.70 — close but not passed).

| Gate | threshold | @8500 | @11000 | post-10k median |
|---|---|---|---|---|
| `coarse_vs_copy_ratio` | ≤ 0.70 | 0.85 | **12.73** | 5.88 |
| `coarse_vs_batch_mean_ratio` | ≤ 0.50 | 0.18 | 1.81 | 1.85 |
| `c_effective_rank` | > 60 | 13.9 | **5.8** | 6.85 |

---

## Mechanism (best current explanation)

1. **AGC from step 0** shapes a slightly different weight trajectory than elated. Royal
   **dodges** the 8400–8500 spike that triggers elated's skip spiral.

2. **Step 8600:** a hard batch (or batch sequence) hits the AGC-shaped weights. Forward
   `L_flow` jumps 0.29→1.11. Backward produces extreme raw grads → AGC clips heavily
   (Fc ratio 19→1284 over 150 steps).

3. **Optimizer still steps** (post-AGC norm ~8–12). Updates are dominated by **heavily
   clipped F_c and B gradients** — the stack walks into a basin where F_c predicts badly
   (model loss 0.2→3.1) while copy baseline stays easy.

4. **Slot/rank collapse** follows over ~2500 steps of continued training at `L_flow≈2.5–3.0`.
   This is **active destruction**, not freeze — the opposite of elated's illusory latent health.

5. **Guards are blind:** they watch post-AGC grad norm, not `L_flow` level or `agc_*_max_ratio`.

---

## What worked

- Zero grad skips through 11350 steps.
- Clean pass through step **8500** (elated's death step).
- Pre-8600 training matched cerulean/elated trajectory (rank ~13.9, `L_flow` ~0.26–0.32).
- Best copy ratio **0.851** @8500 — closest to gate of any point in the run.

## What broke

- Cliff at **8600** (not 8500).
- `F_c` forward prediction destroyed; rank 13.9→5.8; copy ratio 0.85→12.7.
- No recovery over 2750 steps of continued training.
- Run killed externally at 11350 before 15k target.

---

## Conclusion

AGC solves the **grad-skip death spiral** but introduces a **worse failure mode**: training
continues through an unstable forward-loss region with mutilated gradients, collapsing slots.

Next interventions should target **forward-loss / AGC-ratio signals**, not post-AGC grad alone.

See `NEXT_STEPS.md`.
