# Investigation 007 — What binds the reconstruction capacity floor?

**Status:** ACTIVE — Wave 1 complete (weight/decoder ruled out); Wave 2 (latent `n_c` axis) **failed to run** and is now **ON HOLD**, superseded by [investigation_008](../investigation_008/DESCRIPTION.md) (SIGReg attacks the `d_c` utilization ceiling Wave 1 actually identified; `n_c` adds slots, not dims). Pivot to a prediction objective still likely after.
**Opened:** 2026-06-27 (after `easy-blaze-19` / investigation_006 capacity-floor finding)
**Closed:** —

> **Structure note (this investigation runs in *waves*).** The OFAT sweep executed as two 5-wide
> parallel waves on a 5× H100 pod. Run records are grouped under
> [`wave_1/`](wave_1/DESCRIPTION.md) (weight × decoder — **complete**) and
> [`wave_2/`](wave_2/DESCRIPTION.md) (latent axis + saturation extremes — **failed at step 200,
> re-run pending**). Each wave and each run carries its own triad. Narrative writeups:
> [`WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md`](WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md),
> [`END_OF_WAVE_2.md`](END_OF_WAVE_2.md).

## Question

Every reconstruction readout in `fanciful-lake-18` (option 1) and `easy-blaze-19` (option 3)
is pinned at **`L_recon_* ≈ 0.60`** (relative MSE). That floor is *why* the reconstruction
anchor (and option 3 in particular) failed to improve prediction — a good and a bad prediction
reconstruct identically, so the objective is blind to prediction quality. **What binds the
floor — the loss weight, the decoder size, or the latent `c` size?**

## How we answer it

An **8-run, one-factor-at-a-time (OFAT) sweep** over three axes, run 8-wide in parallel on a
6–8 GPU pod (≈ one run's wall-clock):
- `lambda_recon` ∈ {0.1, 0.2, 0.5} — is the floor **weight-bound**?
- decoder size (`decoder_dim × decoder_blocks`) ∈ {256×2, 512×2, 512×4} — **decoder-bound**?
- latent size `n_c` ∈ {32, 64, 128} — **capacity-bound**? (my prior bet)
- + 1 combined "all bigger" run. `lambda_recon_pred = 0` throughout (isolate the floor; option 3
  was inert at it).

**Primary readout:** does `L_recon_present` drop below ~0.55? **Secondary:** if it drops, does
`coarse_vs_copy_ratio` fall toward <1 (is prediction actually reconstruction-bound)?

## Full design, values, safety analysis, execution

- **Design + reasoning + interpretation matrix:** [`SWEEP_PLAN_decoder_capacity.md`](SWEEP_PLAN_decoder_capacity.md)
- **End-to-end execution (pod → parallel launch → monitor):** [`GUIDE.md`](GUIDE.md)
- **Per-run records:** grouped by wave — [`wave_1/<wandb-name>/`](wave_1/DESCRIPTION.md) and
  [`wave_2/<wandb-name>/`](wave_2/DESCRIPTION.md), each a triad. (The original 8-run plan grew to
  **10 runs in two 5-wide waves**: Wave 2 added the `lambda_recon=1.0` and `n_c=256` saturation
  extremes to fill the 5th GPU and close the OFAT blind spot — see `WAVE1_ANALYSIS...` §2.)

## Parent context

- The capacity-floor finding + option-3 negative result: [investigation_006](../investigation_006/) (`easy-blaze-19`)
- The reconstruction-anchor design taxonomy (option 1/2/3): [investigation_006/DESCRIPTION.md](../investigation_006/DESCRIPTION.md)
- Code: flags `--lambda-recon`, `--decoder-dim`, `--decoder-blocks`, `--n-c`, `--checkpoint-dir`
  (all on `phase1-v0.2-frozen-encoder`); `models.Decoder`, `losses.reconstruction_loss`.

## Runs

**[Wave 1](wave_1/DESCRIPTION.md) — weight × decoder (COMPLETE, cut ~9k):**

| Run | λ_recon · decoder · n_c | Outcome |
|---|---|---|
| [toasty-donkey-21](wave_1/toasty-donkey-21/) | 0.1 · 256×2 · 32 | floor 0.596 — weight inert |
| [jolly-glade-20](wave_1/jolly-glade-20/) | 0.2 · 256×2 · 32 | floor 0.592 — weight inert |
| [light-universe-24](wave_1/light-universe-24/) | 0.5 · 256×2 · 32 | floor 0.586 — weight inert |
| [gallant-dew-22](wave_1/gallant-dew-22/) | 0.05 · 512×2 · 32 | floor 0.590 — decoder inert |
| [eager-plant-22](wave_1/eager-plant-22/) | 0.05 · 512×4 · 32 | floor 0.585 — decoder inert |

**[Wave 2](wave_2/DESCRIPTION.md) — latent axis + extremes (FAILED at step 200 — re-run pending):**

| Run | λ_recon · decoder · n_c | Outcome |
|---|---|---|
| [pious-mountain-28](wave_2/pious-mountain-28/) | 0.05 · 256×2 · 64 | ⚠️ died step 200 — **re-run (Tier 0)** |
| [classic-yogurt-29](wave_2/classic-yogurt-29/) | 0.05 · 256×2 · 128 | ⚠️ died step 200 — re-run optional |
| [earnest-dragon-25](wave_2/earnest-dragon-25/) | 0.05 · 256×2 · 256 | ⚠️ died step 200 — **re-run (Tier 0)** |
| [helpful-snow-25](wave_2/helpful-snow-25/) | 1.0 · 256×2 · 32 | ⚠️ died step 200 — re-run low-prio |
| [quiet-firebrand-25](wave_2/quiet-firebrand-25/) | 0.2 · 512×2 · 64 | ⚠️ died step 200 — re-run low-prio |

**Bottom line so far:** weight and decoder are ruled out (floor pinned at 0.585 ± 0.01);
reconstruction is structurally blind to prediction; the floor is utilization-limited. The latent
axis is untested (Wave 2 died). Most likely end state: pivot to a temporal/prediction objective.
