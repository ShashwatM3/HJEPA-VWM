# Investigation 008 — Does SIGReg break the `d_c` utilization ceiling (rank 13/256)?

**Status:** OPEN — design complete; **SIGReg code implemented** (`losses.sigreg_loss`,
`--lambda-sigreg`, `L_sigreg` metric, unit tests — uncommitted in the working tree);
no runs launched.
**Opened:** 2026-06-27 (spun out of [investigation_007](../investigation_007/DESCRIPTION.md))
**Closed:** —

## Question

investigation_007 Wave 1 established that the 0.585 reconstruction floor is a
**utilization** limit, not a capacity one: `c_effective_rank` sits at **~13 / 256** in
every run, invariant to reconstruction weight and decoder size — `c` leaves ~95% of its
`d_c` dimensions unused (WAVE1_ANALYSIS §1.4). Neither Wave 1's knobs nor Wave 2's `n_c`
touch the collapsed axis (`d_c`). The variance floor we run today
(`losses.variance_floor`, VICReg-V) is a one-sided hinge that only forbids *constant*
dims — it cannot enforce the *isotropy* that would fill `d_c`.

> **Can a proper isotropic-Gaussian regularizer (SIGReg, LeJEPA) break the rank-13
> ceiling — and if `c` becomes richer, does prediction (`coarse_vs_copy_ratio`) finally
> improve, or does it confirm the disease is temporal, not utilization?**

## How we answer it

A **one-factor sweep on `λ_sigreg`** (4 values, 4-wide on 4× A100), holding every other
knob at the best-known-good Wave-1 config so the `λ_sigreg=0` control is *exactly*
eager-plant-22 (already on W&B). SIGReg pushes the pooled `c` distribution toward
`N(0, I_{d_c})` via the BHEP/Epps–Pulley normality test along random projections
(Cramér–Wold) — isotropy by construction maximizes effective rank.

- **Primary readout:** does `c_effective_rank` break ~13 and climb toward 60+?
- **Secondary:** if rank rises, does `coarse_vs_copy_ratio` fall toward <1, and/or
  `L_recon_present` drop below ~0.55?
- **Health gates:** `c_cross_video_cosine` (no collapse), `L_flow` (task not destroyed).

## Full design, implementation, values

- **Design + SIGReg code spec + λ rationale + decoder decision + interpretation matrix:**
  [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md)
- **End-to-end execution (4× A100 pod → launch → monitor):** [`GUIDE.md`](GUIDE.md)
- **Predictions + W&B-grounded evidence:** [`OBSERVATIONS.md`](OBSERVATIONS.md)
- **Plan / tiers / close conditions:** [`NEXT_STEPS.md`](NEXT_STEPS.md)

## Key decisions (answers to the standing questions)

- **Decoder size:** **512×4** (eager-plant-22), the best decoder of all Wave-1 runs on
  the decoder-isolating metric `L_recon_present` (0.5847) and on every secondary metric —
  confirmed live from W&B. Rationale + caveat in [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md) §5.
- **VICReg:** the redundant terms (cov/slot) stay off; the VICReg-V variance-floor hinge
  stays at 0.5 as a *non-interfering safety net* (it's inactive once std≥1, the regime
  SIGReg targets), so SIGReg is the only changed variable. Pure-replace (`λ_var=0`) is a
  Wave-2 follow-up. [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md) §2.3.
- **Recon anchor stays ON** at `λ_recon=0.05` (the decoder must be live to matter); it's
  known inert on rank → a clean stable background.

## Parent / sibling context

- Spun out of [investigation_007](../investigation_007/DESCRIPTION.md): the rank-13
  utilization finding ([WAVE1_ANALYSIS](../investigation_007/WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md) §1.4)
  and the SIGReg recommendation ([END_OF_WAVE_2](../investigation_007/END_OF_WAVE_2.md) §2.4).
- inv007 **Wave 2** (`n_c` latent-slot sweep) is **ON HOLD** in favor of this — see
  [wave_2/NEXT_STEPS.md](../investigation_007/wave_2/NEXT_STEPS.md). SIGReg attacks `d_c`
  (the *actually* collapsed axis); `n_c` adds slots, not dims.
- Code: `losses.variance_floor` (the hinge SIGReg's role replaces), `train.py:238/247`
  (loss assembly), `diagnostics.effective_rank` (the metric SIGReg targets).

## Runs

**Wave 1 — `λ_sigreg` sweep (4× A100). Control (`λ=0`) = eager-plant-22 (`591mt31k`).**

| GPU | λ_sigreg · decoder · n_c | Role | W&B id |
|---|---|---|---|
| 0 | 0.3 · 512×4 · 32 | mild isotropy push | TBD |
| 1 | 1.0 · 512×4 · 32 | scale-matched to `L_flow` | TBD |
| 2 | 3.0 · 512×4 · 32 | strong push | TBD |
| 3 | 10.0 · 512×4 · 32 | saturation bookend (can rank move at all?) | TBD |

**Bottom line (pre-data):** decisive either way — break the rank ceiling and improve
prediction (build on it), break the ceiling but prediction still fails (the temporal
pivot becomes unimpeachable), or fail to move rank at all (the bottleneck `B` is the
architectural constraint).
