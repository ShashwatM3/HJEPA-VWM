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
