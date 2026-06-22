# Observations — Investigation 003 (collapse)

## Hypothesis timeline

### H1: Init is the root cause (orthogonal queries + zero-init `out_mlp`)

**Evidence:** External review + code audit. Near-uniform attention from tiny random queries.

**Result:** **Partially true but insufficient.** Init fixes (baked into model, commit `62b94dd`)
improved starting rank (~9 vs ~5) and attention entropy. Rank still stalled without stronger
variance pressure. Init is **hygiene**, not the cure ([`ANALYSIS_AND_DECISIONS.md`](../../AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/ANALYSIS_AND_DECISIONS.md)).

### H2: Missing decorrelation term (VICReg-C / SIGReg)

**Evidence:** `L_var` only prevents constant dims, not correlated subspaces. Effective rank ~5
with healthy dead-dim fraction.

**Result:** VICReg-C implemented (flag-gated) but **deprioritized** after H4 won. See [investigation_004](investigation_004/). SIGReg held as escalation, not tried.

### H3: Slot-diversity loss fixes redundant slots

**Evidence:** `c_slot_diversity_rank` low in early runs; uniform attention.

**Result:** **Rejected as training objective.** `slot-loss-k4-run-3` and `centered-slot-k12-run-5`:
slot metric improved while `c_cross_video_cosine` worsened (~0.84) and rank fell — classic
Goodhart. Centering fix (`ffc33ed`) aligned loss with metric but did not fix the objective.

### H4: Variance floor too weak (`lambda_var=0.1`)

**Evidence:** Runs 2–5 showed `c_std_mean ~0.45` with `L_flow` dominating; cosine ~0.7–0.84.

**Result:** **Confirmed.** `cerulean-snow-13` with `lambda_var=0.5`: `c_std_mean ~1.0`,
`c_cross_video_cosine ~0.20–0.26`, `c_effective_rank ~12–13.7`, `coarse_vs_copy_ratio < 1`,
stable grads.

### H5: Task too easy (`horizon_k=4`)

**Evidence:** Copy baseline strong at k=4; overlap-heavy target.

**Result:** **Confirmed as necessary adjunct.** `horizon_k=12` adopted for runs 4–6; harder
prediction reduces trivial copy solutions.

## Cross-run synthesis

| Run | Rank | Cross-video cos | Copy ratio | Verdict |
|---|---|---|---|---|
| peachy-terrain-5 (ref) | ~5 | healthy | improved then regressed | collapse + crash |
| exalted-lion-6 | ~8–10 | 0.25 | falling | tiny-scale collapse confirmed |
| init-fixes-full-ssv2-run-2 (Run A @3500) | **8.7** | 0.25 | **7.42** | architectural, not data-limited |
| slot-loss-k4-run-3 | worse | ~0.84 | unstable | reject slot loss |
| slot-loss-k12-low-run-4 | — | — | slot inert | fix loss centering |
| centered-slot-k12-run-5 | ~4.8 | ~0.84 | grad 10⁵ | reject slot loss |
| **cerulean-snow-13** | **~13.7** | **~0.22** | **< 1** | **winning config** |

## Conclusion

**Primary collapse lever: raise `lambda_var` to 0.5.** Init fixes stay baked in.
**Do not train with slot-diversity loss.** **Use `horizon_k=12`.** VICReg-C remains optional
if rank plateaus low after a stable long run ([investigation_004](investigation_004/)).

Wrong turns retained in record: slot loss (H3) looked promising on metrics alone.

## Links

- Brief run table: [`AGENT_FILES/KNOWLEDGE/BRIEF_V0_3.md`](../../AGENT_FILES/KNOWLEDGE/BRIEF_V0_3.md) §13
- Local log: [`logs/cerulean-snow-13/output.log`](../../../logs/cerulean-snow-13/output.log)
