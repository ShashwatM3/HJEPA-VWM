# investigation_007 - What binds the reconstruction-capacity floor: decoder size, reconstruction weight, number of slots, or latent utilization?

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Status:** CLOSED  
**Runs covered:** 020, 021, 022, 023, 024, 025, 026, 027, 028, 029  
**Theme:** decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis

## Question

What binds the reconstruction-capacity floor: decoder size, reconstruction weight, number of slots, or latent utilization?

## Why This Investigation Exists

Investigation 006 showed reconstruction was helpful but capped. This branch tested whether simply changing decoder/reconstruction capacity could lower the floor or reveal a richer c_t.

## W&B-Validated Run Coverage

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 20 | [`jolly-glade-20`](wave_1/run_020_jolly-glade-20/) | `5x7aoxnn` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.2/0; residual=false; present_only=false; n_c=32; D=256x2 | Low-rank rep | c_effective_rank=12.9849; c_cross_video_cosine=0.2506; c_std_mean=1.0427; coarse_vs_copy_ratio=1.5638; coarse_vs_batch_mean_ratio=0.3383; L_recon_present=0.5915 |
| 21 | [`eager-plant-22`](wave_1/run_021_eager-plant-22/) | `591mt31k` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=32; D=512x4 | Low-rank rep | c_effective_rank=12.9484; c_cross_video_cosine=0.2468; c_std_mean=1.037; coarse_vs_copy_ratio=1.5278; coarse_vs_batch_mean_ratio=0.3246; L_recon_present=0.5845 |
| 22 | [`gallant-dew-22`](wave_1/run_022_gallant-dew-22/) | `708jrel8` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=32; D=512x2 | Low-rank rep | c_effective_rank=12.5094; c_cross_video_cosine=0.3045; c_std_mean=0.9893; coarse_vs_copy_ratio=1.7402; coarse_vs_batch_mean_ratio=0.3723; L_recon_present=0.5895 |
| 23 | [`toasty-donkey-21`](wave_1/run_023_toasty-donkey-21/) | `a2trqp9c` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.1/0; residual=false; present_only=false; n_c=32; D=256x2 | Low-rank rep | c_effective_rank=12.9731; c_cross_video_cosine=0.2487; c_std_mean=1.0345; coarse_vs_copy_ratio=1.5857; coarse_vs_batch_mean_ratio=0.3422; L_recon_present=0.5959 |
| 24 | [`light-universe-24`](wave_1/run_024_light-universe-24/) | `rju7xsh2` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.5/0; residual=false; present_only=false; n_c=32; D=256x2 | Low-rank rep | c_effective_rank=12.6552; c_cross_video_cosine=0.2824; c_std_mean=1.0185; coarse_vs_copy_ratio=1.6644; coarse_vs_batch_mean_ratio=0.3594; L_recon_present=0.5855 |
| 25 | [`earnest-dragon-25`](wave_2/run_025_earnest-dragon-25/) | `2xsd5jwr` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=256; D=256x2 | Smoke / inconclusive | c_effective_rank=9.4052; c_cross_video_cosine=0.6885; c_std_mean=0.5243; coarse_vs_copy_ratio=47.1775; coarse_vs_batch_mean_ratio=9.839; L_recon_present=1.0324 |
| 26 | [`pious-mountain-28`](wave_2/run_026_pious-mountain-28/) | `7u5zkw6t` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=64; D=256x2 | Smoke / inconclusive | c_effective_rank=8.8219; c_cross_video_cosine=0.7359; c_std_mean=0.4799; coarse_vs_copy_ratio=62.9974; coarse_vs_batch_mean_ratio=12.9475; L_recon_present=1.0299 |
| 27 | [`quiet-firebrand-25`](wave_2/run_027_quiet-firebrand-25/) | `bbrrydax` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.2/0; residual=false; present_only=false; n_c=64; D=512x2 | Smoke / inconclusive | c_effective_rank=8.8218; c_cross_video_cosine=0.7359; c_std_mean=0.4799; coarse_vs_copy_ratio=62.9978; coarse_vs_batch_mean_ratio=12.9475; L_recon_present=1.0447 |
| 28 | [`classic-yogurt-29`](wave_2/run_028_classic-yogurt-29/) | `ryuh8cpr` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=128; D=256x2 | Smoke / inconclusive | c_effective_rank=9.6446; c_cross_video_cosine=0.7349; c_std_mean=0.4843; coarse_vs_copy_ratio=66.8338; coarse_vs_batch_mean_ratio=14.055; L_recon_present=1.028 |
| 29 | [`helpful-snow-25`](wave_2/run_029_helpful-snow-25/) | `tw685b5g` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=1/0; residual=false; present_only=false; n_c=32; D=256x2 | Smoke / inconclusive | c_effective_rank=9.4729; c_cross_video_cosine=0.7239; c_std_mean=0.4952; coarse_vs_copy_ratio=62.0671; coarse_vs_batch_mean_ratio=12.6011; L_recon_present=1.0246 |

## Current Conclusion

Decoder width/depth and reconstruction weight did not explain the floor. The important finding was a utilization ceiling: c_effective_rank stayed near the historical low-rank band unless a geometry regularizer directly attacked d_c usage.

## Evidence Standard

The run entries above were reconciled against W&B API/export data on 2026-07-02. For run 052, the newest active run, the evidence was additionally refreshed live with `python run_history.py --run 662hfy3c --report`, which showed the run still `running` through step 5600.

## Original Notes Preserved

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
| [toasty-donkey-21](wave_1/run_023_toasty-donkey-21/) | 0.1 · 256×2 · 32 | floor 0.596 — weight inert |
| [jolly-glade-20](wave_1/run_020_jolly-glade-20/) | 0.2 · 256×2 · 32 | floor 0.592 — weight inert |
| [light-universe-24](wave_1/run_024_light-universe-24/) | 0.5 · 256×2 · 32 | floor 0.586 — weight inert |
| [gallant-dew-22](wave_1/run_022_gallant-dew-22/) | 0.05 · 512×2 · 32 | floor 0.590 — decoder inert |
| [eager-plant-22](wave_1/run_021_eager-plant-22/) | 0.05 · 512×4 · 32 | floor 0.585 — decoder inert |

**[Wave 2](wave_2/DESCRIPTION.md) — latent axis + extremes (FAILED at step 200 — re-run pending):**

| Run | λ_recon · decoder · n_c | Outcome |
|---|---|---|
| [pious-mountain-28](wave_2/run_026_pious-mountain-28/) | 0.05 · 256×2 · 64 | ⚠️ died step 200 — **re-run (Tier 0)** |
| [classic-yogurt-29](wave_2/run_028_classic-yogurt-29/) | 0.05 · 256×2 · 128 | ⚠️ died step 200 — re-run optional |
| [earnest-dragon-25](wave_2/run_025_earnest-dragon-25/) | 0.05 · 256×2 · 256 | ⚠️ died step 200 — **re-run (Tier 0)** |
| [helpful-snow-25](wave_2/run_029_helpful-snow-25/) | 1.0 · 256×2 · 32 | ⚠️ died step 200 — re-run low-prio |
| [quiet-firebrand-25](wave_2/run_027_quiet-firebrand-25/) | 0.2 · 512×2 · 64 | ⚠️ died step 200 — re-run low-prio |

**Bottom line so far:** weight and decoder are ruled out (floor pinned at 0.585 ± 0.01);
reconstruction is structurally blind to prediction; the floor is utilization-limited. The latent
axis is untested (Wave 2 died). Most likely end state: pivot to a temporal/prediction objective.
