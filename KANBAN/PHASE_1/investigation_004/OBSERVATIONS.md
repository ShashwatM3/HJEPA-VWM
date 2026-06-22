# Observations — Investigation 004 (VICReg-C)

## Hypothesis

Decorrelation penalty directly targets low effective rank (correlated feature dims).

## Evidence without dedicated run

- VICReg-C implemented and logged (`L_cov` ~8–20 at `lambda_cov=0` on cerulean) for future calibration.
- **`cerulean-snow-13`** achieved rank ~13.7 with **`lambda_cov=0`** only.
- Slot-loss path showed **Goodhart risk** for auxiliary metrics — any future VICReg-C trial must
  watch `coarse_vs_copy_ratio` jointly with rank.

## Belief (current)

VICReg-C is **reasonable escalation** if rank plateaus too low after a **complete** stable 15k run
([investigation_005](investigation_005/)). It is **not** the first lever; strong variance floor was
underestimated in original `lambda_var=0.1` default.

## SIGReg

Held as second escalation per ANALYSIS_AND_DECISIONS — not tried. Deferred.

## Conclusion

**Paused, not closed.** Re-open only if investigation 005 shows rank stall with otherwise healthy metrics.
