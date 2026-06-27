# Wave 1 — weight × decoder axes (5 runs)

**Status:** COMPLETE (all 5 cut at ~9k after the floor plateaued; healthy throughout)
**Launched:** 2026-06-26 ~23:37Z · **Read:** 2026-06-27
**Parent:** [investigation_007](../DESCRIPTION.md) · **Baseline it perturbs from:** [`easy-blaze-19`](../../investigation_006/easy-blaze-19/DESCRIPTION.md) (λ=0.05, 256×2, n_c=32 → floor ~0.60)
**Leads to:** [Wave 2](../wave_2/DESCRIPTION.md) (latent axis)

## What this wave is

The first wave of the OFAT capacity-floor sweep. It perturbs **two of the three** candidate
binding constraints one factor at a time from the `easy-blaze-19` baseline, holding
`lambda_recon_pred = 0`:

- **Weight axis** (`lambda_recon`): 0.1 → 0.2 → 0.5 — *is the floor weight-bound?*
- **Decoder axis** (`decoder_dim × blocks`): 512×2 (width), 512×4 (depth) — *decoder-bound?*

The third axis (latent `n_c`) was deferred to [Wave 2](../wave_2/DESCRIPTION.md). Run as 5
independent processes, one per GPU (5× H100), `WANDB_RUN_GROUP=inv007_capacity_floor`.

## The 5 runs (logical OFAT order)

| # | Run | λ_recon | Decoder | n_c | Probe | W&B id |
|---|---|---|---|---|---|---|
| 1 | [toasty-donkey-21](toasty-donkey-21/) | 0.1 | 256×2 | 32 | weight (low) | `a2trqp9c` |
| 2 | [jolly-glade-20](jolly-glade-20/) | 0.2 | 256×2 | 32 | weight (mid) | `5x7aoxnn` |
| 3 | [light-universe-24](light-universe-24/) | 0.5 | 256×2 | 32 | weight (aggressive) | `rju7xsh2` |
| 4 | [gallant-dew-22](gallant-dew-22/) | 0.05 | 512×2 | 32 | decoder width | `708jrel8` |
| 5 | [eager-plant-22](eager-plant-22/) | 0.05 | 512×4 | 32 | decoder depth | `591mt31k` |

*(Runs were launched in parallel; the numbering is the OFAT ladder, not wall-clock order — they
share a `createdAt`. Read each in sequence 1→5 for the cleanest story.)*

## Headline result (one line)

**Neither axis moved the floor** — `L_recon_present` = 0.585 ± 0.01 across all 5 — so the wave
*eliminated* weight and decoder as the binding constraint and motivated the latent-axis Wave 2.
Full cross-run synthesis: [OBSERVATIONS.md](OBSERVATIONS.md). Narrative writeup:
[`../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md`](../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md).
