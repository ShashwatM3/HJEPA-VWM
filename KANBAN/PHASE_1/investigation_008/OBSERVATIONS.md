# Observations — investigation_008 (SIGReg sweep)

No run data yet (code not implemented; sweep not launched). This file holds the
**evidence basis** and **pre-registered predictions** so the read-out is honest when data
lands. Full design: [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md).

## 2026-06-27 — evidence basis (live W&B, group `inv007_capacity_floor`)

**The rank ceiling SIGReg targets is real and config-invariant.** Across both inv007
Wave-1 decoder runs, `c_effective_rank` converges to ~12.5–12.7 / 256 and never moves:

| Run (id) | decoder | L_recon_present | c_effective_rank | copy_ratio | L_flow |
|---|---|---|---|---|---|
| eager-plant-22 (591mt31k) | 512×4 | 0.5847 | **12.72** | 1.60 | 0.396 |
| gallant-dew-22 (708jrel8) | 512×2 | 0.5895 | **12.51** | 1.74 | 0.435 |

(Both pulled at final step 8500.) `c_effective_rank` rose only from ~9.5 at init to ~12.7
by 8.5k and plateaued — the ceiling is ~13 regardless of decoder. `coarse_vs_copy_ratio`
ends ~1.6–1.7 (still >1, losing to copy). This is the wall inv008 exists to break.

**Decoder decision recorded here for the audit trail:** 512×4 (eager-plant-22) is the
best decoder on the decoder-isolating metric `L_recon_present` *and* every secondary
metric → adopted as the fixed background (rationale: [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md) §5).

## Pre-registered predictions (before any inv008 run)

From the Wave-1 reframe (END_OF_WAVE_2 §2.3: `c` is distinct-per-video but nearly static
in time) and the literature (SIGReg guards cross-sample variance, not temporal
informativeness):

1. **`c_effective_rank` WILL rise** under SIGReg — isotropy maximizes rank by
   construction. Expect a clear lift above 13 at λ≥1 (confidence ~75%). If it *doesn't*
   move even at λ=10, the bottleneck `B` architecture (not the regularizer) binds rank.
2. **`coarse_vs_copy_ratio` will most likely STAY >1** (~55%) — the copy baseline is a
   *temporal* failure that a *marginal-distribution* regularizer doesn't address. This
   would be the strongest possible evidence for the Tier-1 prediction pivot.
3. **`L_recon_present` may drop modestly** if a richer `c` carries more reconstructable
   content — but per Wave-1's blindness finding it cannot reach the ~0.01 scale where the
   prediction gap becomes visible, so a floor drop alone does not fix prediction.
4. **Risk to watch:** at high λ, `c` is dragged toward Gaussian noise → `L_flow` rises and
   `c_cross_video_cosine` should *fall* (more spread); a *rise* in `c_cross_video_cosine`
   instead means collapse (var-floor net failed) — abort that run.

*(Conclusions go here once the sweep lands; do not rewrite this section — add dated
follow-ups per PROTOCOL.)*
