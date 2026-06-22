# Observations — Investigation 004 (VICReg-C)

## Hypothesis

Decorrelation penalty directly targets low effective rank (correlated feature dims).

## Run A executed (via sleek-leaf-7)

Step-3500 verdict on full SSv2, `lambda_cov=0`:

- `c_effective_rank` **8.7** — architectural collapse, not data-limited (42× data → +3.7 rank)
- `c_slot_diversity_rank` **1.62** — dominant axis is slot/attention collapse
- `L_cov` **20.18** at λ=0 → calibrated start **λ_cov=0.0027**
- VICReg-C would address rank 8.7 axis; **would not** fix slot redundancy directly

**Run B never launched** — team pivoted to slot-diversity loss (runs 3–5), then to
`lambda_var=0.5` after slot path failed.

## Evidence without dedicated VICReg run

- VICReg-C implemented and logged (`L_cov` ~8–20 at `lambda_cov=0` on later runs)
- **`cerulean-snow-13`** achieved rank ~13.7 with **`lambda_cov=0`** only
- Slot-loss path showed **Goodhart risk** for auxiliary metrics

## Belief (current)

VICReg-C is **reasonable escalation** if rank plateaus too low after a **complete** stable 15k run
([investigation_005](investigation_005/)). It is **not** the first lever; strong variance floor was
underestimated in original `lambda_var=0.1` default.

## SIGReg

Held as second escalation per ANALYSIS_AND_DECISIONS — not tried. Deferred.

## Conclusion

**Paused, not closed.** Re-open only if investigation 005 shows rank stall with otherwise healthy metrics.

Source: `AGENT_FILES/COMPLETE_FULL_CHAT` lines ~5389–6241, ~9268–9303.
