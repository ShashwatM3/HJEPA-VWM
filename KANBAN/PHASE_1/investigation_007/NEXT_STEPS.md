# Next Steps - investigation_007

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Closure / Carry-Forward Status

- Status: **CLOSED**.
- Conclusion to carry forward: Decoder width/depth and reconstruction weight did not explain the floor. The important finding was a utilization ceiling: c_effective_rank stayed near the historical low-rank band unless a geometry regularizer directly attacked d_c usage.
- Runs covered: 020, 021, 022, 023, 024, 025, 026, 027, 028, 029.

## Follow-Up Chain

This investigation feeds into `investigation_008`: SIGReg ladder on top of reconstruction-capacity recipe. The reason is: The capacity branch pointed at underused feature dimensions. SIGReg was introduced to push pooled c_t toward an isotropic distribution and raise effective rank.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.

## Original Notes Preserved

# Next steps — investigation 007

## Status (updated 2026-06-27)

**Wave 1 complete** — weight and decoder ruled out (floor 0.585 ± 0.01). **Wave 2 failed** — all 5
latent-axis runs died at step 200 (synchronized whole-pod death), so hypothesis #3 (`n_c`
capacity-bound) is **untested**. See [`wave_1/`](wave_1/DESCRIPTION.md), [`wave_2/`](wave_2/DESCRIPTION.md),
[`END_OF_WAVE_2.md`](END_OF_WAVE_2.md).

## Immediate next action — re-run the failed latent wave (Tier 0)

1. **Diagnose the death:** `tail -n 50 logs/*.log` on the pod (traceback vs. bare `Killed`); check
   RunPod pod events for a stop/reclaim; `dmesg | grep -i oom`.
2. **Re-launch a reduced wave: `n_c=64` + `n_c=256` only** — the two bookends that decide the
   latent-capacity question ([wave_2/pious-mountain-28](wave_2/run_026_pious-mountain-28/),
   [wave_2/earnest-dragon-25](wave_2/run_025_earnest-dragon-25/)). Commands in those DESCRIPTIONs.
3. **Harden the launch:** confirm tmux detach before SSH disconnect; add a step-600 tripwire
   (`grep -L "step.*500" logs/*.log` after ~15 min) so a silent early death can't recur.
4. Read **`L_recon_present` ∧ `c_effective_rank` ∧ `coarse_vs_copy_ratio`** against SWEEP_PLAN §4.

## Then — the likely pivot (the real next investigation)

Per [`END_OF_WAVE_2.md`](END_OF_WAVE_2.md) §2 + external literature ("Prediction over
Reconstruction", V-JEPA 2-AC): the disease is **temporal under-informativeness** (`c` barely changes
in time → copy baseline wins), which reconstruction can't fix. Open **investigation_008** on the
prediction side: larger horizon `k`, inverse-dynamics/transition auxiliary, multi-step rollout loss,
or an explicit anti-copy term. Decision metric for everything downstream: **`coarse_vs_copy_ratio`
→ <1**.

## KANBAN hygiene (mostly done)

- ✅ Per-run + per-wave triads created under [`wave_1/`](wave_1/) and [`wave_2/`](wave_2/).
- ✅ `DESCRIPTION.md` Runs table + `OBSERVATIONS.md` synthesis updated; README row refreshed.
- ⏳ On the n_c re-run: fill the two run folders' `OBSERVATIONS.md` with real data and set the
  investigation status to CLOSED, pointing at investigation_008.

## Stage 2 (conditional on Stage-1 result)

- **An axis moved the floor** → push that axis further AND re-introduce option 3
  (`--lambda-recon-pred 0.05`) on the now-unsaturated decoder — does the prediction branch
  finally bite?
- **Nothing moved the floor** → reconstruction is the wrong lever. Close the reconstruction line;
  open a new investigation on the **horizon/task** (the copy baseline is strong because `c`
  barely moves over horizon-12). Possibly also revisit whether the `>60` rank gate is right at
  128:1 compression.

## Watch / abort (per run; inv_005 rules + sweep-specific)

- `L_flow > 1.5` for 200 consecutive steps; `c_effective_rank` drops >3 in 500 steps;
  `agc_Fc_max_ratio` median >200 over any 500-step window.
- Sweep-specific: on `n_c=128` watch `c_slot_diversity_rank` / `c_cross_video_cosine` (slot
  collapse); on `lambda_recon=0.5` watch `L_flow` (recon over-competing with prediction).
