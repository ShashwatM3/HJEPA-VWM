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

## Evidence without dedicated VICReg A/B

VICReg-C (`lambda_cov`) was **never tested as the primary isolated lever**. It appeared
only as a low-weight **adjunct** alongside slot loss on Runs 3–5
([`serene-cloud-8`](../investigation_003/serene-cloud-8/),
[`skilled-waterfall-10`](../investigation_003/skilled-waterfall-10/),
[`copper-sky-12`](../investigation_003/copper-sky-12/)) at **`lambda_cov=0.0027`** —
not at `lambda_cov=0` for calibration.

- **`sleek-leaf-7`** (Run A): `lambda_cov=0` — logs `L_cov` magnitude only; used to
  calibrate the 0.0027 weight used later
- **Runs 3–5**: `lambda_cov=0.0027` active with slot loss — VICReg-C tried as adjunct,
  not isolated; slot path rejected for Goodhart
- **`cerulean-snow-13`**: first run with **`lambda_cov=0`** and no slot loss — rank
  ~13.7 via `lambda_var=0.5` alone

## Belief (current)

VICReg-C is **reasonable escalation** if rank plateaus too low after a **complete** stable 15k run
([investigation_005](investigation_005/)). It is **not** the first lever; strong variance floor was
underestimated in original `lambda_var=0.1` default.

## SIGReg

Held as second escalation per ANALYSIS_AND_DECISIONS — not tried. Deferred.

## Conclusion

**Paused, not closed.** Re-open only if investigation 005 shows rank stall with otherwise healthy metrics.

Source: `AGENT_FILES/COMPLETE_FULL_CHAT` lines ~5389–6241, ~9268–9303.
