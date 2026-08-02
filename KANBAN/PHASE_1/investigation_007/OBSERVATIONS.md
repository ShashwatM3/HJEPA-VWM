# Observations - investigation_007

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Cross-Run Synthesis

Decoder width/depth and reconstruction weight did not explain the floor. The important finding was a utilization ceiling: c_effective_rank stayed near the historical low-rank band unless a geometry regularizer directly attacked d_c usage.

## Run-by-Run Evidence

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

## Pattern Across The Branch

Best copy ratio in this branch was run 021 at 1.5278; best batch-mean ratio was run 021 at 0.3246. None should be read as a full Phase 1 pass unless both gates pass together.

Verdict distribution: Low-rank rep=5, Smoke / inconclusive=5.

## What Changed The Research Direction

That result spawned the SIGReg sweep in investigation_008, because SIGReg targets d_c utilization directly while n_c only adds slots.

## Original Notes Preserved

# Observations — investigation 007

No run data yet. Pre-run hypotheses below; metrics + interpretation are appended as dated
sections once the Stage-1 wave lands. Reasoning in
[`SWEEP_PLAN_decoder_capacity.md`](SWEEP_PLAN_decoder_capacity.md).

---

## 2026-06-27 — Pre-run hypotheses (the binding-constraint candidates)

Carried in from `easy-blaze-19`: `L_recon_present ≈ L_recon_cplus ≈ L_recon_chat ≈
L_recon_pred ≈ 0.60` in **both** option-1 and option-3 runs. Three candidate binding constraints:

1. **Weight-bound** (raise `lambda_recon` and the floor drops): we under-ask at 5%. *Cheapest
   to fix.* Watch for `L_flow` degradation as the cost.
2. **Decoder-bound** (bigger `D` drops the floor): `D` is too thin to expand `c → e` and serve
   two reconstruction targets. The tech-lead's intuition.
3. **Capacity-bound** (bigger `n_c` drops the floor): `c`'s 8,192-number bandwidth is the true
   limit (128:1). *Prior bet* — a bigger `D` can't recover information `c` never kept.

**What a result means** (interpretation matrix in SWEEP_PLAN §4): the axis that moves
`L_recon_present` below ~0.55 identifies the binding constraint. **If none move it**,
reconstruction is the wrong lever for the prediction problem → pivot to horizon/task (the copy
baseline is strong because `c` barely moves over horizon-12: `‖Δc‖/‖c‖` ~0.38 and falling).

**Second-level test (necessary AND sufficient?):** even if the floor drops, `coarse_vs_copy_ratio`
must fall toward <1 for reconstruction to be the right lever. Floor down but copy gate still
failed ⇒ prediction is not reconstruction-bound.

**Safety prior (full analysis in SWEEP_PLAN §4b):** low risk — recon is protective against both
collapse modes. Watch `n_c=128` (slot diversity / cross-video cosine) and high-`lambda_recon`
(`L_flow`).

---

## 2026-06-27 — Wave 1 results (weight × decoder): floor inert on both axes

Cross-wave detail: [`wave_1/OBSERVATIONS.md`](wave_1/OBSERVATIONS.md); narrative + plots:
[`WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md`](WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md). All numbers from
W&B. Three findings:

1. **Neither weight nor decoder moves the floor.** `L_recon_present` = **0.585 ± 0.01** across a 5×
   weight range and a 6× decoder-param range; none reached the 0.55 gate. → **hypotheses #1
   (weight-bound) and #2 (decoder-bound) are REJECTED.**
2. **Reconstruction is structurally blind to prediction.** `L_recon_chat − L_recon_cplus` ≈
   0.006–0.011 while the floor is 0.585 — a perfect vs a predicted future latent reconstruct almost
   identically. *No reconstruction objective can supervise prediction at this floor* (the mechanistic
   why of `easy-blaze-19`).
3. **The floor is a utilization limit, not capacity.** `c_effective_rank` ≈ 13/256 in every run,
   invariant to weight and decoder. The under-used axis is `d_c` (per-slot dim) — which `n_c` does
   **not** touch. This pre-weakens hypothesis #3.

## 2026-06-27 — Wave 2 (latent axis): FAILED TO RUN — hypothesis #3 still untested

All 5 latent-axis + saturation runs **died at step 200 (~350 s), synchronized whole-pod death** — no
usable data (only step-0 init logged). Forensics + re-run prescription:
[`wave_2/OBSERVATIONS.md`](wave_2/OBSERVATIONS.md), [`END_OF_WAVE_2.md`](END_OF_WAVE_2.md) §1. So
**hypothesis #3 (capacity-bound) is neither confirmed nor rejected** — it needs a re-run (reduced to
`n_c=64` + `n_c=256`).

**Current belief.** Two of three binding-constraint candidates are dead, and the blindness finding
suggests #3 — even if it nudges the floor — almost certainly won't fix prediction (you'd need the
floor near the ~0.01 prediction-gap scale, unreachable by `n_c`). The deeper reframe (corroborated by
external literature, [`END_OF_WAVE_2.md`](END_OF_WAVE_2.md) §2.4): `c` is **distinct-per-video but
nearly static in time** — it encodes *appearance*, not *dynamics* — and reconstruction *reinforces*
appearance. The "pivot to horizon/task" branch of the pre-run interpretation matrix is now the most
likely end state. Final call deferred to the Wave-2 re-run.
