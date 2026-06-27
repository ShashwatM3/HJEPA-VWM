# Observations — investigation 007

No run data yet. Pre-run hypotheses below; metrics + interpretation are appended as dated
sections once the Stage-1 wave lands. Reasoning in
[`SWEEP_PLAN_decoder_capacity.md`](SWEEP_PLAN_decoder_capacity.md).

---

## 2026-06-27 — Pre-run hypotheses (the binding-constraint candidates)

Carried in from `easy-blaze-19`: `L_recon_present ≈ L_recon_cplus ≈ L_recon_chat ≈
L_recon_pred ≈ 0.60` in **both** option-1 and option-3 runs. Three candidate binding constraints:

1. **Weight-bound** (raise `lambda_recon` and the floor drops): we under-ask at 5%. *Cheapest
   to fix.* Watch for `L_flow` degradation as the cost.
2. **Decoder-bound** (bigger `D` drops the floor): `D` is too thin to expand `c → e` and serve
   two reconstruction targets. The tech-lead's intuition.
3. **Capacity-bound** (bigger `n_c` drops the floor): `c`'s 8,192-number bandwidth is the true
   limit (128:1). *Prior bet* — a bigger `D` can't recover information `c` never kept.

**What a result means** (interpretation matrix in SWEEP_PLAN §4): the axis that moves
`L_recon_present` below ~0.55 identifies the binding constraint. **If none move it**,
reconstruction is the wrong lever for the prediction problem → pivot to horizon/task (the copy
baseline is strong because `c` barely moves over horizon-12: `‖Δc‖/‖c‖` ~0.38 and falling).

**Second-level test (necessary AND sufficient?):** even if the floor drops, `coarse_vs_copy_ratio`
must fall toward <1 for reconstruction to be the right lever. Floor down but copy gate still
failed ⇒ prediction is not reconstruction-bound.

**Safety prior (full analysis in SWEEP_PLAN §4b):** low risk — recon is protective against both
collapse modes. Watch `n_c=128` (slot diversity / cross-video cosine) and high-`lambda_recon`
(`L_flow`).

---

## 2026-06-27 — Wave 1 results (weight × decoder): floor inert on both axes

Cross-wave detail: [`wave_1/OBSERVATIONS.md`](wave_1/OBSERVATIONS.md); narrative + plots:
[`WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md`](WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md). All numbers from
W&B. Three findings:

1. **Neither weight nor decoder moves the floor.** `L_recon_present` = **0.585 ± 0.01** across a 5×
   weight range and a 6× decoder-param range; none reached the 0.55 gate. → **hypotheses #1
   (weight-bound) and #2 (decoder-bound) are REJECTED.**
2. **Reconstruction is structurally blind to prediction.** `L_recon_chat − L_recon_cplus` ≈
   0.006–0.011 while the floor is 0.585 — a perfect vs a predicted future latent reconstruct almost
   identically. *No reconstruction objective can supervise prediction at this floor* (the mechanistic
   why of `easy-blaze-19`).
3. **The floor is a utilization limit, not capacity.** `c_effective_rank` ≈ 13/256 in every run,
   invariant to weight and decoder. The under-used axis is `d_c` (per-slot dim) — which `n_c` does
   **not** touch. This pre-weakens hypothesis #3.

## 2026-06-27 — Wave 2 (latent axis): FAILED TO RUN — hypothesis #3 still untested

All 5 latent-axis + saturation runs **died at step 200 (~350 s), synchronized whole-pod death** — no
usable data (only step-0 init logged). Forensics + re-run prescription:
[`wave_2/OBSERVATIONS.md`](wave_2/OBSERVATIONS.md), [`END_OF_WAVE_2.md`](END_OF_WAVE_2.md) §1. So
**hypothesis #3 (capacity-bound) is neither confirmed nor rejected** — it needs a re-run (reduced to
`n_c=64` + `n_c=256`).

**Current belief.** Two of three binding-constraint candidates are dead, and the blindness finding
suggests #3 — even if it nudges the floor — almost certainly won't fix prediction (you'd need the
floor near the ~0.01 prediction-gap scale, unreachable by `n_c`). The deeper reframe (corroborated by
external literature, [`END_OF_WAVE_2.md`](END_OF_WAVE_2.md) §2.4): `c` is **distinct-per-video but
nearly static in time** — it encodes *appearance*, not *dynamics* — and reconstruction *reinforces*
appearance. The "pivot to horizon/task" branch of the pre-run interpretation matrix is now the most
likely end state. Final call deferred to the Wave-2 re-run.
