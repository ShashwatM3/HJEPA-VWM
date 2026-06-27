# Wave 2 — latent axis + saturation extremes

**Status:** ⏸️ ON HOLD (2026-06-27) — re-run deferred in favor of
[investigation_008](../../investigation_008/DESCRIPTION.md) (SIGReg). The latent-capacity
(`n_c`) question stays **open but deprioritized**: Wave 1 showed the collapsed axis is
`d_c` (per-slot dim, rank 13/256), and `n_c` adds *slots*, not *dims* — so SIGReg, which
attacks `d_c` directly via an isotropic-Gaussian objective, is the higher-value next
experiment. Revisit this `n_c` wave only if inv008 implicates latent *capacity* (not
utilization). The 4-GPU re-run was prepped (below) but **not launched**.
**Attempt 1 (failed):** launched ~03:41Z · died ~03:47:43Z (all five, same heartbeat
second — whole-pod death; step-0 init only, zero signal; runs being deleted from W&B).
**Attempt 2 (prepped, NOT launched):** 4 GPUs, λ=1.0 dropped — full `n_c` ladder +
combined run. Launch + hardening kept for if/when revived: [`../GUIDE.md`](../GUIDE.md) §3c.
**Parent:** [investigation_007](../DESCRIPTION.md) · **Predecessor:** [Wave 1](../wave_1/DESCRIPTION.md)
**Successor (chosen instead):** [investigation_008](../../investigation_008/DESCRIPTION.md)
**Full forensics + pivot:** [`../END_OF_WAVE_2.md`](../END_OF_WAVE_2.md)

## The re-run (4 GPUs — what's actually running now)

Only `helpful-snow-25` (λ=1.0 weight saturation) is dropped — it's the most predictable run in the
wave (Wave 1 has 3 clean weight points on a flat line, and `L_flow` didn't even degrade at λ=0.5).
The 4 kept are the full latent ladder (the only untested axis) + the one interaction check:

| GPU | Run folder | λ_recon · decoder · n_c | Role | W&B id |
|---|---|---|---|---|
| 0 | [pious-mountain-28](pious-mountain-28/) | 0.05 · 256×2 · 64  | latent bookend (low) ⭐ | new (TBD) |
| 1 | [classic-yogurt-29](classic-yogurt-29/) | 0.05 · 256×2 · 128 | latent interpolation | new (TBD) |
| 2 | [earnest-dragon-25](earnest-dragon-25/) | 0.05 · 256×2 · 256 | latent bookend (high) ⭐ | new (TBD) |
| 3 | [quiet-firebrand-25](quiet-firebrand-25/) | 0.2 · 512×2 · 64   | combined "all bigger" | new (TBD) |
| — | [helpful-snow-25](helpful-snow-25/) | ~~1.0 · 256×2 · 32~~ | **DROPPED** | — |

> Run folders keep their original (attempt-1) names; the W&B ids below are from the **deleted**
> failed runs and are now dead. Update each folder's `OBSERVATIONS.md` with the **new** run name +
> id once the re-run logs data.

## What this wave was meant to be

The second and final planned wave of the OFAT sweep. [Wave 1](../wave_1/OBSERVATIONS.md) had
eliminated the weight and decoder axes, leaving the **latent axis (`n_c`)** as the only untested
binding-constraint candidate. This wave was the latent ladder plus two saturation extremes (added
to fill the 5 GPUs and close the OFAT "flat vs not-pushed-hard-enough" blind spot):

| # | Run | λ_recon | Decoder | n_c | Intended probe | W&B id |
|---|---|---|---|---|---|---|
| 1 | [pious-mountain-28](pious-mountain-28/) | 0.05 | 256×2 | 64 | latent 2× | `7u5zkw6t` |
| 2 | [classic-yogurt-29](classic-yogurt-29/) | 0.05 | 256×2 | 128 | latent 4× | `ryuh8cpr` |
| 3 | [earnest-dragon-25](earnest-dragon-25/) | 0.05 | 256×2 | 256 | latent 8× (saturation) | `2xsd5jwr` |
| 4 | [helpful-snow-25](helpful-snow-25/) | 1.0 | 256×2 | 32 | weight saturation | `tw685b5g` |
| 5 | [quiet-firebrand-25](quiet-firebrand-25/) | 0.2 | 512×2 | 64 | combined "all bigger" | `bbrrydax` |

## What actually happened

All five processes stopped reporting at the **identical second (03:47:43Z)** after ~350 s, at
**step 200** — before the first diagnostic readout (step 500) and at 10% of the recon warmup
(2000). Only step-0 (initialization) was logged, so there is **zero signal** on any config. Init
was healthy (loss ~3.1, grad_norm ~1.5, no NaN, no instability), and the standard-architecture
`helpful-snow-25` (n_c=32) died alongside the n_c=256 run — so this is **not** a code/shape bug
but a whole-pod / tmux-session / network-volume death. Forensics + re-run prescription:
[OBSERVATIONS.md](OBSERVATIONS.md) and [`../END_OF_WAVE_2.md`](../END_OF_WAVE_2.md) §1.

## Pre-registered prediction (still untested)

From [Wave 1](../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md): the n_c runs were the only real
experiments; most likely (~90%) the floor stays flat or drops only cosmetically while
`coarse_vs_copy_ratio` stays >1 → pivot. **This prediction now awaits a re-run** (reduced to
`n_c=64` + `n_c=256`) — see [NEXT_STEPS.md](NEXT_STEPS.md).
