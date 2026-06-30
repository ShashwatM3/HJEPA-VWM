# Next steps — investigation_010 (optimizer plumbing + residual rerun)

**Status:** investigation_010 is **CLOSED**. Run [`soft-universe-37`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2vbo6pbm) (`2vbo6pbm`) — W&B state **`finished`**, step **14950**, runtime **24,453 s (~6.8 h)**. Full analysis:
[`ANALYSIS_soft-universe-37.md`](ANALYSIS_soft-universe-37.md).

**W&B pull:** 2026-06-29 via `get_run_history_tool` — 30 diag points (steps 0–14500, every 500 steps), plus terminal summary at step 14950.

**Synthesis in one sentence:** commit `230096d` hardened training and **cleared the rank gate for the first time** (61.1 at step 11000+, plateau 61.1), but **did not move prediction** — `coarse_vs_copy_ratio` plateau **1.077** (final 1.063), the same tie-by-zero failure as investigation_009 Run 2 (`graceful-river-35`, plateau 1.047, final 1.088). inv010 beat inv009 on rank (+3.1) and final ratio (−0.025) but inv009 had deeper beat-copy dips (0.915 @3500, 1.014 @12500 vs inv010's 0.933 and 1.036). The bottleneck is unchanged: **`F_c` predicts Δ̂ ≈ 0 once ρ overshoots the predictive band (~0.23 at end).**

This document is the prioritized roadmap for **investigation_011** and beyond. It synthesizes inv007–010 W&B evidence (including the **complete** inv010 run), the codebase surface area, Phase 1 acceptance gates, and external literature.

---

## 0. What investigation_010 proved (final numbers)

Pulled from W&B `get_run_history` on 2026-06-29 — **complete run**, 30 diag points (steps 0–14500).

| metric | inv010 final (`soft-universe-37` @14.95k) | inv009 final (`graceful-river-35` @14.05k) | Phase 1 gate |
|---|---|---|---|
| **coarse_vs_copy_ratio** (plateau, last 3 diag) | **1.077** | 1.047 | **≤ 0.70** ❌ |
| ↳ last value | **1.063** | 1.088 | — |
| ↳ min (excl init) | **0.933 @3500** | **0.915 @3500** | < 1 briefly ✓ |
| ↳ best post-10k | **1.036 @12500** | **1.014 @12500** | neither sustained |
| **coarse_vs_batch_mean_ratio** | **1.140** | ~1.17 | ≤ 0.50 ❌ |
| **c_effective_rank** | **61.1** (crossed 60 @11k) | 58.0 | **> 60** ✅ |
| c_plus_effective_rank | 60.4 | *(not logged)* | EMA aligned |
| coarse_copy_loss | 1.562 | 1.560 | ↑ = `c` moves |
| coarse_model_loss | 1.661 | 1.697 | ≈ copy → **Δ̂ ≈ 0** |
| model − copy gap | **0.099** | 0.137 | positive = tie |
| *implied ρ(c_t, c_{t+k})* | **~0.23** | ~0.23 | sweet spot ~0.7–0.8 |
| c_cross_video_cosine | **0.162** | 0.170 | < 0.5 ✅ |
| c_std_mean | 1.005 | 1.005 | ≈ 1.0 ✅ |
| L_recon_chat − cplus | 0.012 | 0.015 | recon blind |
| grad_skipped | 0 (all 30 diag steps) | 0 | stable ✅ |
| runtime | 6.8 h | 6.1 h | — |

**Phase 1 scorecard after inv010:**

| Gate | Status |
|---|---|
| `c_effective_rank > 60` | **PASS** (first time in project) |
| `coarse_vs_copy_ratio ≤ 0.70` | **FAIL** (~1.06, need ~35% relative improvement) |
| `coarse_vs_batch_mean_ratio ≤ 0.50` | **FAIL** (~1.14) |
| No collapse (cosine, std, dead dims) | **PASS** |
| Training stability | **PASS** |

**Verdict on the inv010 hypothesis:** plumbing is **necessary infrastructure** (keep it), not a **prediction lever** (do not iterate on it). Close investigation_010 as: *plumbing validated; prediction hypothesis not advanced.*

---

## 1. The diagnosis — what is actually broken

Three investigations (008 → 009 → 010) have converged on a single mechanistic picture. This is the constraint every next experiment must respect.

### 1.1 The master variable: ρ(c_t, c_{t+k})

Derived lens (not a logged metric): `1 − ρ ≈ coarse_copy_loss / (2 · c_std_mean²)`.

| regime | ρ | what happens | which run |
|---|---|---|---|
| Too static | ρ → 0.9+ | copy baseline trivially strong; `F_c` loses badly | inv008/009 Run 1 (SIGReg substrate) |
| **Predictive sweet spot** | **ρ ≈ 0.7–0.8** | ratio **< 1**; `F_c` beats zero-residual | inv009/010 **@step ~3500 only** |
| Too dynamic | ρ → 0.2–0.3 | future nearly independent of present; `F_c` ties copy via Δ̂ → 0 | inv009/010 **plateau** |

Both residual runs pass through the sweet spot early (~step 3500), then **recon + SIGReg drive rank up and ρ down** through step ~5000. After that, `coarse_model_loss` tracks `coarse_copy_loss` — the tie-by-zero signature.

### 1.2 What is NOT broken (stop optimizing these)

| area | evidence | implication |
|---|---|---|
| Representation collapse | `c_dead_dim_frac=0`, cosine ~0.16, std ~1.0 | variance floor + SIGReg working |
| Rank utilization | rank 61/256, climbing | SIGReg + residual + recon **solved** the inv007 rank-13 ceiling |
| Optimizer / AGC plumbing | `grad_skipped=0`, EMA gap ~1 rank point at end | commit `230096d` is done; do not re-sweep |
| Reconstruction floor | `L_recon_present` ~0.57, flat | inv007 blind to prediction; not the lever |
| Encoder | frozen V-JEPA 2; SSv2-pretrained | temporal structure exists in `e`; problem is in `B`/`F_c` objective |

### 1.3 What IS broken (optimize these)

| priority | metric / symptom | root cause |
|---|---|---|
| **P0** | `coarse_vs_copy_ratio ≈ 1.06` (need ≤ 0.70) | `F_c` predicts Δ̂ ≈ 0 — optimal tie, not skill |
| **P0** | ρ ~0.23 at plateau | recon-driven **appearance decorrelation**, not predictable dynamics |
| **P1** | Sweet spot lost after step ~4000 | no mechanism holds ρ in 0.7–0.8 band |
| **P2** | `c_slot_diversity_rank` ~7/32 | slot redundancy (secondary) |
| **P2** | `L_recon_chat ≈ L_recon_cplus` | recon cannot supervise prediction quality |

### 1.4 Why reconstruction makes prediction worse (mechanism)

From inv009 §5 and inv007 Wave 1, corroborated by external work:

- **Option 1** (`D(c_t) → e_t`) anchors `c` to **present** appearance.
- **Option 3** (`D(c_t + Δ̂) → e_{t+k}`) pulls `c` toward **future** appearance.
- With `k=12` frames of real SSv2 motion, these two anchors drive `c_t` and `c_{t+k}` to encode **different static appearance**, collapsing ρ — but that appearance change is **high-dimensional and not predictable** from `c_t` alone.

External parallel: [*What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction*](https://arxiv.org/html/2606.07687v1) — reconstruction spends capacity on static scene fidelity; temporal prediction preserves transition-relevant information. Highest pixel fidelity often has near-zero action recoverability.

---

## 2. What NOT to do next (evidence-backed deprioritization)

| do not | why |
|---|---|
| Re-run inv010 / more optimizer plumbing | A/B vs inv009 shows prediction unchanged; rank/cosine marginally better only |
| SIGReg λ sweep (another inv008) | rank↑ ⟺ prediction↓ under full-latent; residual already gets rank 61 with λ=5 |
| Recon weight / decoder sweep (another inv007 Wave 1) | floor inert to weight/decoder; blind to prediction (gap 0.01 vs floor 0.585) |
| Full-latent prediction without residual | inv008/009 showed this is strictly worse on ratio |
| Resume inv010 checkpoint with new loss terms | optimizer param-group count changed at `230096d`; fresh start per GUIDE |
| Treat total `loss` rising as divergence | expected when `L_flow` scales with ‖Δ‖²; `grad_skipped=0` is the real health signal |

**Still open but low priority:** inv007 Wave 2 `n_c` sweep (64 vs 256). Wave 1 blindness finding predicts floor flat and ratio > 1 even if `n_c` helps rank cosmetically. Run only if you need to formally close the capacity axis before a major architectural pivot.

---

## 3. Prioritized next steps → investigation_011

Ordered by **expected information per GPU-hour**, then implementation cost. Every experiment uses the **inv010 base config** unless noted:

```
predict_residual=True, λ_sigreg=5, λ_var=0.5, decoder 512×4, n_c=32,
lr_coarse_flow=1e-4, commit ≥ 230096d, fresh start, WANDB_RUN_GROUP=inv011_<arm>
```

**Decision metric (unchanged):** `coarse_vs_copy_ratio` **stably < 1** (target ≤ 0.70 for Phase 1 gate), with ρ in 0.7–0.8 and rank > 60 as corroboration. Reconstruction readouts are health checks only.

---

### Tier 0 — Close investigation_010 (housekeeping, ~0 GPU)

1. Append final measured results to `OBSERVATIONS.md` (mirror inv008/009 discipline).
2. Add `DESCRIPTION.md` with run table and status → **CLOSED**.
3. Update [`PHASE_1/README.md`](../README.md) investigation table.
4. **Checkpoint to keep:** `/workspace/ckpt/inv010_residual_230096d` — best representation in project (rank 61, healthy `c`). Useful as init for Tier 1b/c **only if** optimizer group layout matches (post-`230096d`).

---

### Tier 1a — Horizon `k` sweep (cheapest test, **no new code**)

**Hypothesis:** At `k=12`, residual + recon overshoots ρ. Smaller `k` keeps ‖Δ‖² in the predictable band longer; larger `k` weakens copy mechanically but may overshoot faster.

| arm | `--horizon-k` | rationale |
|---|---|---|
| A | **4** | SSv2 tubelet scale; smaller Δ, higher ρ; copy weaker than static-`c` regime |
| B | **8** | midpoint |
| C | **24** | copy baseline grows (√k effect on motion); tests if larger gap helps or hurts |

**Config:** residual + full recon stack (same as inv010), 15k steps, 1 GPU each (or 3-wide on 1 pod).

**Read:**
- Does ratio stay < 1 **past step 5000** at any `k`?
- ρ trajectory vs inv010 at matched steps
- rank vs `k` (may trade off)

**Literature:** V-JEPA 2 notes multi-timescale prediction as future work; SkyJEPA / rollout papers use longer horizons with **rollout consistency**, not single-step copy — so large `k` alone may fail without Tier 1d.

**Success criterion:** any `k` with plateau ratio **< 0.95** and ρ **0.5–0.8**.

---

### Tier 1b — Govern ρ: recon ablation (minimal code, high signal)

**Hypothesis:** `λ_recon_pred` (option 3) is the engine of runaway decorrelation. Removing or reducing it holds ρ in the sweet spot.

| arm | λ_recon | λ_recon_pred | predict_residual |
|---|---|---|---|
| **B1 — residual only** | **0** | **0** | true |
| **B2 — present recon only** | 0.05 | **0** | true |
| **B3 — reduced future anchor** | 0.05 | **0.01** | true |
| control | 0.05 | 0.05 | true (= inv010) |

**Why B1 first:** cleanest test of whether residual flow alone can predict without recon pulling `c` toward appearance change. inv009 Run 1 (no recon, full-latent) had ρ↑ — but residual may behave differently.

**Read:** ρ at 3500, 7000, 14000; ratio plateau; rank (may fall without recon — acceptable if ratio wins).

**Success criterion:** ratio plateau **< 1.0** with ρ **> 0.5** and rank **> 50**.

---

### Tier 1c — Anti-collapse on Δ̂ (**new code**, highest direct leverage)

**Hypothesis:** `F_c` ties copy because predicting Δ̂ = 0 is the path of least resistance once ρ is low. Forbid the zero shortcut explicitly.

**Proposed mechanism** (mirror variance floor on `c_t`):

```text
L_delta = hinge( std(Δ̂) , target=std(Δ) )   # or norm-matching: (‖Δ̂‖ - ‖Δ‖)²
loss += lambda_delta * L_delta
```

**Design choices to lock before coding:**

| choice | recommendation | rationale |
|---|---|---|
| Target for scale | **batch std of detached Δ** (EMA target residual) | matches what copy measures |
| Where to apply | on **Δ̂** before add-back `ĉ = c_t + Δ̂` | directly penalizes zero prediction |
| λ_delta sweep | {0.1, 0.5, 1.0} × λ_var=0.5 | start small; log `L_delta`, `delta_hat_std`, `delta_std` |
| Interaction with recon | run **with B1 (no recon)** and **with inv010 recon** | isolates whether anti-collapse alone suffices |

**Precedent:** inv009 DESCRIPTION deliberately **omitted** this (change #7) so tie-by-zero would be observable — it was observed in both inv009 and inv010. This is the single most direct follow-up the data supports.

**Literature parallel:** FLAM / latent-action models use capacity bottlenecks + KL on latent actions to prevent "action copies next state" ([Factored Latent Action World Models](https://arxiv.org/pdf/2602.16229)). Norm-matching on Δ̂ is the residual-prediction analogue.

**Success criterion:** `coarse_model_loss` **diverges below** `coarse_copy_loss` (ratio < 1) for **≥ 3000 consecutive steps** after step 5000.

---

### Tier 1d — Multi-step rollout loss on `F_c` (moderate code)

**Hypothesis:** Single-step training permits Δ̂ → 0 because one-step MSE is satisfied by zero motion. Rolling `F_c` forward 2–3 steps and matching rolled latents to true future targets punishes shortcuts that compound.

**Sketch:**

```text
ĉ₁ = c_t + Δ̂₁                           # step 1
ĉ₂ = ĉ₁ + Δ̂₂(c_t, ĉ₁, ...)             # step 2 (condition on rolled state)
L_rollout = MSE(ĉ_k, c_{t+k})           # k ∈ {1, 2} initially
loss += lambda_rollout * L_rollout
```

**Literature:** V-JEPA 2-AC uses teacher-forced next-step + two-step rollout ([arxiv 2506.09985](https://arxiv.org/abs/2506.09985)); SkyJEPA ([2606.23444](https://arxiv.org/html/2606.23444)) and Persistent Robot World Models ([2603.25685](https://arxiv.org/html/2603.25685)) extend this.

**Pair with:** Tier 1b B1 (no recon) to avoid ρ overshoot while testing rollout.

**Risk:** error compounding if `F_c` is weak; start with `k_roll=2` only.

---

### Tier 2a — Label-free inverse dynamics auxiliary (new module)

**Hypothesis:** SSv2 has no robot actions, but the **transition** `e_t → e_{t+k}` is a free supervisory signal. An inverse-dynamics head `h(c_t, c_{t+k}) → û` forces `c` to retain transition-relevant information and rules out static/collapsed latents.

**SSv2-appropriate formulation (no action labels):**

```text
û = h_ψ( c_t , stopgrad(c_{t+k}) )     # predict Δe or PCA(Δe) in encoder space
L_inv = MSE( û , stopgrad(e_{t+k} - e_t) )
```

Or predict **Δ** directly from `(c_t, c_{t+k})` and match EMA residual — redundant with forward path but **gradients into B** differ.

**Literature:**
- [*Prediction over Reconstruction*](https://arxiv.org/html/2606.07687v1) — inverse-dynamics auxiliary amplifies temporal structure (+0.29–0.45 R² on video-pretrained encoders).
- [Sensorimotor World Models](https://arxiv.org/html/2606.20104) — `L = L_fwd + λ L_inv`; inverse head prevents collapse by requiring action recoverability from consecutive embeddings.

**Caveat:** With frozen `e`, inverse dynamics on `e` only gradients through `B`. The head should predict something **low-dimensional** (e.g. pooled Δe, 64-dim) to avoid reconstructing all of `e`.

---

### Tier 2b — Copy-margin / anti-copy penalty (higher risk)

**Hypothesis:** Directly optimize margin against the copy baseline:

```text
L_margin = relu( coarse_model_loss - alpha * coarse_copy_loss + margin )
```

**Risk:** fights variance floor and SIGReg; can destabilize. Only try after Tier 1c fails.

---

### Tier 3 — Architectural (defer until Tier 1 exhausted)

| change | when | note |
|---|---|---|
| `n_c` sweep (64, 256) | after Tier 1 | inv007 Wave 2 never ran; low expected payoff on prediction |
| Increase `d_c` (256 → 512) | if rank plateaus < 80 with healthy ratio | invasive; ripples through B, F_c, D |
| Phase 4 multi-horizon `h_k` embedding | after single-k wins | already stubbed in CoarseFlow slot_pos comment |
| Unfreeze encoder layers | last resort | contradicts v0.2 design; supervisor escalation |

---

## 4. Recommended investigation_011 wave plan

**Goal:** break the tie-by-zero with **minimal simultaneous changes** (OFAT discipline).

### Wave 1 — 3-wide on 1× A100 or 3-wide on 3 GPUs (parallel, independent)

| GPU | arm | change vs inv010 | primary read |
|---|---|---|---|
| 0 | **1b-B1** | `λ_recon=0, λ_recon_pred=0`, residual | Does ρ stay high? ratio? |
| 1 | **1c** | `λ_delta=0.5` (new), inv010 recon | Does Δ̂ → 0 break? |
| 2 | **1a-k4** | `--horizon-k 4`, else inv010 | Does smaller k hold sweet spot? |

**Shared:** 15k steps, `predict_residual`, λ_sigreg=5, λ_var=0.5, plumbing from `230096d`, fresh ckpt dirs, group `inv011_wave1`.

**Tripwires:** step-0 residual scale check; step-600 stall check; chart `coarse_vs_copy_ratio`, `coarse_copy_loss`, `coarse_model_loss`, `c_effective_rank`, implied ρ.

### Wave 2 — conditional (only if Wave 1 shows signal)

| if Wave 1 shows… | then… |
|---|---|
| B1 (no recon) ratio < 1 but rank drops | combine B1 + 1c anti-collapse |
| 1c ratio < 1 | sweep λ_delta; add rollout (1d) |
| k=4 holds ρ but ratio ~1 | k=4 + 1c |
| all three still ~1.05 | escalate to Tier 2a inverse dynamics + consider stopping recon permanently |

---

## 5. Metrics to add before investigation_011 (logging only, low cost)

These are **not logged today** but would sharpen diagnosis:

| metric | definition | why |
|---|---|---|
| `delta_hat_norm` / `delta_norm` | `‖Δ̂‖` / `‖Δ‖` per batch | direct tie-by-zero detector |
| `delta_hat_std` / `delta_std` | batch std of residual magnitudes | feeds Tier 1c |
| `implied_rho` | `1 - copy/(2·std²)` | stop deriving by hand |
| `model_minus_copy` | `coarse_model_loss - coarse_copy_loss` | negative = beating copy |

Add to `diagnostics.coarse_baselines` or `run_diagnostics` — no architecture change.

---

## 6. Phase 1 acceptance — honest path forward

Current best (`soft-universe-37`):

| gate | value | gap to pass |
|---|---|---|
| rank > 60 | **61.1** | ✅ cleared |
| ratio ≤ 0.70 | **~1.06** | need **~34%** relative reduction |
| batch-mean ratio ≤ 0.50 | **~1.14** | need **~56%** reduction |

**Realistic assessment:** inv010 improved rank and stability but **did not materially close the prediction gap**. Beating copy stably (< 1) looks reachable (glimpsed at step 3500). Reaching **0.70** may require **combining** Tier 1b + 1c + tuned `k`, or Tier 2a — not more regularization plumbing.

**Do not start Phase 2** until `coarse_vs_copy_ratio` is decisively below 1 on val diagnostics — shuffled-c bypass is meaningless if `F_c` does not predict.

---

## 7. External research map (how it connects to our data)

| paper / system | relevant idea | our status |
|---|---|---|
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | latent prediction, rollout, action-conditioning | we have encoder; lack rollout + action |
| [Prediction over Reconstruction](https://arxiv.org/html/2606.07687v1) | recon hurts action-relevant latents; use inverse dynamics | **matches inv007/009 findings exactly** |
| [LeWorldModel / SIGReg](https://arxiv.org/pdf/2603.19312) | isotropic Gaussian anti-collapse | we use SIGReg; solved rank, not temporal ρ |
| [Sensorimotor WM](https://arxiv.org/html/2606.20104) | inverse dynamics as anti-collapse | candidate Tier 2a for SSv2 |
| [SkyJEPA](https://arxiv.org/html/2606.23444) | long-horizon + rollout consistency | candidate Tier 1d |
| [V-JEPA 2 blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | hierarchical multi-timescale (Phase 4) | deferred until single-horizon works |

**Key insight from literature + our runs:** distributional regularizers (SIGReg, variance floor) fix **cross-sample** collapse but not **temporal** informativeness. The field's move is toward **prediction-side** objectives: residual/rollout, inverse dynamics, and horizon tuning — not more reconstruction.

---

## 8. Decision tree (summary)

```text
inv010 done → ratio ~1.06, rank 61, ρ ~0.23, tie-by-zero confirmed
    │
    ├─ Tier 1a: sweep k ∈ {4, 8, 24}          [no code, 3 runs]
    ├─ Tier 1b: drop/reduce recon on residual  [no code, 1–3 runs]
    ├─ Tier 1c: anti-collapse on Δ̂            [small code change, 1–3 runs]
    │
    ├─ if any ratio < 1 stable → combine winners + sweep λ
    │
    └─ if all still ~1.05 → Tier 1d rollout + Tier 2a inverse dynamics
            │
            └─ if still stuck → escalate d_c / n_c / supervisor
```

---

## 9. Close investigation_010 — conclusion text (for DESCRIPTION.md)

> Investigation_010 reran the investigation_009 residual configuration on optimizer plumbing commit `230096d`. Run `soft-universe-37` finished 15k steps with **the healthiest `c` in the project** (`c_effective_rank` 61.1, `c_cross_video_cosine` 0.16, `grad_skipped` 0 throughout) but **the same prediction failure** as `graceful-river-35` (`coarse_vs_copy_ratio` plateau 1.08 vs 1.06). Plumbing is validated and retained. The bottleneck is **`F_c` tie-by-zero after recon-driven ρ overshoot**, not optimizer instability or rank collapse. Next: investigation_011 — govern ρ (recon ablation), forbid Δ̂ → 0 (anti-collapse), and sweep horizon `k`.

---

### Sources

- Final run: W&B [`soft-universe-37`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2vbo6pbm) — pulled 2026-06-29
- Baseline: [`graceful-river-35`](../investigation_009/RESULTS_ANALYSIS.md) (`jsh6uo7p`)
- Analysis: [`ANALYSIS_soft-universe-37.md`](ANALYSIS_soft-universe-37.md)
- Parent next steps: [`investigation_009/RESULTS_ANALYSIS.md`](../investigation_009/RESULTS_ANALYSIS.md) §7
- Temporal reframe: [`investigation_007/END_OF_WAVE_2.md`](../investigation_007/END_OF_WAVE_2.md) §2.3–2.6
- Phase 1 gates: [`AGENT_FILES/PHASES/PHASE_1.md`](../../AGENT_FILES/PHASES/PHASE_1.md) §12
- Code surface: `losses.residual_target`, `train.py` `--predict-residual`, `diagnostics.coarse_baselines`
