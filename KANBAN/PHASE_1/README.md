# Phase 1 - Research KANBAN

<!-- AUTO-GENERATED-WANDB-KANBAN -->

Phase 1 currently covers coarse dynamics only: a pinned frozen encoder (V-JEPA2, SigLIP 2, or
DINOv3), trainable bottleneck `B`, EMA bottleneck `B_EMA`, coarse flow `F_c`, optional feature
reconstruction decoder `D`, and diagnostic/regularization knobs used to understand collapse, rank,
temporal dynamics, reconstruction honesty, and prediction baselines.

## Historical W&B Snapshot (verified 2026-07-02)

This generated snapshot and its run table are preserved for chronology. They stop at run 052;
use the dated updates below—especially the 2026-07-16 reconciliation—for current state.

- W&B project: `smahalanobis-uc-davis/hjepa-vwm`.
- Live W&B project query found **52 runs** in creation order.
- Current newest run: run 052 `ae_sharp_slots_recon_only` (`662hfy3c`), still `running` when refreshed live through step 5600.
- No full-prediction run has passed both Phase 1 gates (`coarse_vs_copy_ratio <= 0.70` and `coarse_vs_batch_mean_ratio <= 0.50`).
- The canonical negative result is run 037 `soft-universe-37`: high-rank, spread, video-specific c_t, but copy ratio around 1.06.
- Present-only geometry runs in investigation 011 prove the bottleneck can be made high-rank and decodable, but they are not prediction successes until transferred into a full-prediction run.
- Run 052 currently shows the opposite tradeoff: reconstruction improves strongly without geometry regularizers, but c_t remains weakly spread and video-independent.

## Investigation Index

| Investigation | Status | Runs | Current conclusion |
|---|---|---|---|
| [investigation_001](investigation_001/) | CLOSED | 005 | The long baseline was not a success signal. It exposed late instability and weak prediction, which made optimizer hardening, restart discipline, and clearer acceptance metrics necessary before interpreting longer runs. |
| [investigation_002](investigation_002/) | CLOSED | 001, 002, 003, 004 | The smoke runs validated execution and logging. They also showed that early untrained representations were collapsed or low-rank, so later work had to separate infrastructure success from learning success. |
| [investigation_003](investigation_003/) | CLOSED | 006, 007, 008, 009, 010, 011, 012, 013, 014 | The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates. |
| [investigation_004](investigation_004/) | PAUSED | none | No W&B run is assigned to this investigation in the canonical run sequence. Later present-only geometry sweeps revisited covariance in a better-controlled setting after SIGReg and fixed-position reconstruction clarified the failure mode. |
| [investigation_005](investigation_005/) | CLOSED | 015, 016, 017 | Longer training exposed that nonzero variance was not enough. The runs either became unstable or stayed far from the prediction gates; rank was still low or collapsed, and copy remained competitive. |
| [investigation_006](investigation_006/) | CLOSED | 018, 019 | Reconstruction improved stability/readouts but did not break the rank ceiling or make F_c beat copy. Prediction-side reconstruction also showed that the decoder could be blind to whether c_hat was actually a good future latent. |
| [investigation_007](investigation_007/) | CLOSED | 020, 021, 022, 023, 024, 025, 026, 027, 028, 029 | Decoder width/depth and reconstruction weight did not explain the floor. The important finding was a utilization ceiling: c_effective_rank stayed near the historical low-rank band unless a geometry regularizer directly attacked d_c usage. |
| [investigation_008](investigation_008/) | CLOSED | 030, 031, 032, 033 | SIGReg is a real rank lever, especially at high weights. However, higher rank alone made prediction worse or left copy unbeaten, so geometry alone was not enough. |
| [investigation_009](investigation_009/) | CLOSED | 034, 035 | Residual prediction made c_t more dynamic and healthier, but F_c mostly tied the zero-residual baseline. The failure moved from representation collapse toward predictor learning. |
| [investigation_010](investigation_010/) | CLOSED | 036, 037 | Run 037 proved a key negative: c_t can be high-rank, video-specific, and stable while F_c still fails the copy gate. That is the canonical healthy-representation/no-predictor result. |
| [investigation_011](investigation_011/) | CLOSED | 038, 039, 040, 041, 042, 043, 044, 045, 046, 047, 048, 049, 050, 051 | Present reconstruction can be strong, and SIGReg plus small covariance can raise c_effective_rank above 100 while preserving video specificity. But those are present-side wins; no full-prediction run has inherited them and passed the Phase 1 copy/batch-mean gates. |
| [investigation_012](investigation_012/) | RUNNING | 052 | Live W&B through step 5600 shows strong reconstruction progress but not healthy representation geometry: c_std_mean is still far below 1, cross-video cosine remains high, and rank has fallen into the low 20s. The run is still active, so the final verdict remains provisional. |
| [investigation_014](investigation_014/) | OPEN | none | Offline rank probe shows frozen V-JEPA `e` is already anisotropic: pooled-token entropy rank is 192.7/1024, with within-video rank around 75.7 and a long low-energy tail. This reframes `e -> c` as selective compression/denoising rather than simple full-rank preservation. |
| [investigation_015](investigation_015/) | OPEN | 054 (planned) | Run-053 AE-only recipe on the Perceiver latent-stack bottleneck with fixed offline feature whitening (the inv014-motivated hypothesis). Single combined run; per-change attribution framework in the investigation README. Launch guide includes the one-time `whiten_stats.py` prerequisite. |

## Historical W&B Run Index (entries 001–052)

| # | Run | ID | Created | State | Investigation | Mode | Verdict |
|---:|---|---|---|---|---|---|---|
| 1 | [`youthful-pond-1`](investigation_002/run_001_youthful-pond-1/) | `x4pwz33d` | 2026-06-09T02:34:34Z | `finished` | [investigation_002](investigation_002/) | full-prediction | Smoke / inconclusive |
| 2 | [`efficient-aardvark-2`](investigation_002/run_002_efficient-aardvark-2/) | `fz7ztfc8` | 2026-06-09T02:41:14Z | `finished` | [investigation_002](investigation_002/) | full-prediction | Smoke / inconclusive |
| 3 | [`comfy-glade-3`](investigation_002/run_003_comfy-glade-3/) | `0mgmqxxi` | 2026-06-09T02:52:33Z | `finished` | [investigation_002](investigation_002/) | full-prediction | Smoke / inconclusive |
| 4 | [`charmed-haze-4`](investigation_002/run_004_charmed-haze-4/) | `gj8ypv0d` | 2026-06-10T00:21:43Z | `finished` | [investigation_002](investigation_002/) | full-prediction | Smoke / inconclusive |
| 5 | [`peachy-terrain-5`](investigation_001/run_005_peachy-terrain-5/) | `1chv2608` | 2026-06-10T00:42:46Z | `failed` | [investigation_001](investigation_001/) | full-prediction | Low-rank rep |
| 6 | [`exalted-lion-6`](investigation_003/run_006_exalted-lion-6/) | `wv69n7n5` | 2026-06-13T14:35:43Z | `finished` | [investigation_003](investigation_003/) | full-prediction | Smoke / inconclusive |
| 7 | [`sleek-leaf-7`](investigation_003/run_007_sleek-leaf-7/) | `rpxyg9qt` | 2026-06-13T18:52:06Z | `crashed` | [investigation_003](investigation_003/) | full-prediction | Invalid |
| 8 | [`serene-cloud-8`](investigation_003/run_008_serene-cloud-8/) | `dhp1i3fk` | 2026-06-16T04:01:47Z | `killed` | [investigation_003](investigation_003/) | full-prediction | Invalid |
| 9 | [`confused-butterfly-9`](investigation_003/run_009_confused-butterfly-9/) | `m30jxiye` | 2026-06-16T18:15:37Z | `killed` | [investigation_003](investigation_003/) | full-prediction | Smoke / inconclusive |
| 10 | [`skilled-waterfall-10`](investigation_003/run_010_skilled-waterfall-10/) | `27i1r9qi` | 2026-06-16T18:16:19Z | `crashed` | [investigation_003](investigation_003/) | full-prediction | Invalid |
| 11 | [`olive-terrain-11`](investigation_003/run_011_olive-terrain-11/) | `q40nq0l3` | 2026-06-17T13:01:03Z | `killed` | [investigation_003](investigation_003/) | full-prediction | Invalid |
| 12 | [`copper-sky-12`](investigation_003/run_012_copper-sky-12/) | `ejror834` | 2026-06-19T09:39:56Z | `killed` | [investigation_003](investigation_003/) | full-prediction | Invalid |
| 13 | [`cerulean-snow-13`](investigation_003/run_013_cerulean-snow-13/) | `4lo4j7qb` | 2026-06-19T12:51:57Z | `killed` | [investigation_003](investigation_003/) | full-prediction | Low-rank rep |
| 14 | [`jolly-forest-14`](investigation_003/run_014_jolly-forest-14/) | `8bkeeuio` | 2026-06-20T07:13:55Z | `crashed` | [investigation_003](investigation_003/) | full-prediction | Low-rank rep |
| 15 | [`elated-snowflake-15`](investigation_005/run_015_elated-snowflake-15/) | `jhodg49x` | 2026-06-20T15:23:36Z | `crashed` | [investigation_005](investigation_005/) | full-prediction | Invalid |
| 16 | [`drawn-elevator-16`](investigation_005/run_016_drawn-elevator-16/) | `0n5mx3qf` | 2026-06-22T04:01:20Z | `finished` | [investigation_005](investigation_005/) | full-prediction | Invalid |
| 17 | [`royal-cherry-17`](investigation_005/run_017_royal-cherry-17/) | `0xv4upvb` | 2026-06-24T12:03:39Z | `killed` | [investigation_005](investigation_005/) | full-prediction | Low-rank rep |
| 18 | [`fanciful-lake-18`](investigation_006/run_018_fanciful-lake-18/) | `yd5958s6` | 2026-06-25T12:50:29Z | `killed` | [investigation_006](investigation_006/) | full-prediction | Low-rank rep |
| 19 | [`easy-blaze-19`](investigation_006/run_019_easy-blaze-19/) | `3syv6wp2` | 2026-06-25T22:10:05Z | `finished` | [investigation_006](investigation_006/) | full-prediction | Low-rank rep |
| 20 | [`jolly-glade-20`](investigation_007/wave_1/run_020_jolly-glade-20/) | `5x7aoxnn` | 2026-06-26T23:37:41Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Low-rank rep |
| 21 | [`eager-plant-22`](investigation_007/wave_1/run_021_eager-plant-22/) | `591mt31k` | 2026-06-26T23:37:43Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Low-rank rep |
| 22 | [`gallant-dew-22`](investigation_007/wave_1/run_022_gallant-dew-22/) | `708jrel8` | 2026-06-26T23:37:43Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Low-rank rep |
| 23 | [`toasty-donkey-21`](investigation_007/wave_1/run_023_toasty-donkey-21/) | `a2trqp9c` | 2026-06-26T23:37:43Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Low-rank rep |
| 24 | [`light-universe-24`](investigation_007/wave_1/run_024_light-universe-24/) | `rju7xsh2` | 2026-06-26T23:37:43Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Low-rank rep |
| 25 | [`earnest-dragon-25`](investigation_007/wave_2/run_025_earnest-dragon-25/) | `2xsd5jwr` | 2026-06-27T03:41:07Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Smoke / inconclusive |
| 26 | [`pious-mountain-28`](investigation_007/wave_2/run_026_pious-mountain-28/) | `7u5zkw6t` | 2026-06-27T03:41:07Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Smoke / inconclusive |
| 27 | [`quiet-firebrand-25`](investigation_007/wave_2/run_027_quiet-firebrand-25/) | `bbrrydax` | 2026-06-27T03:41:07Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Smoke / inconclusive |
| 28 | [`classic-yogurt-29`](investigation_007/wave_2/run_028_classic-yogurt-29/) | `ryuh8cpr` | 2026-06-27T03:41:07Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Smoke / inconclusive |
| 29 | [`helpful-snow-25`](investigation_007/wave_2/run_029_helpful-snow-25/) | `tw685b5g` | 2026-06-27T03:41:07Z | `crashed` | [investigation_007](investigation_007/) | full-prediction | Smoke / inconclusive |
| 30 | [`lambda_sigreg_3.0`](investigation_008/run_030_lambda_sigreg_3.0/) | `9jxc8i1q` | 2026-06-27T17:44:35Z | `crashed` | [investigation_008](investigation_008/) | full-prediction | Low-rank rep |
| 31 | [`lambda_sigreg_1.0`](investigation_008/run_031_lambda_sigreg_1.0/) | `jk8kj7h7` | 2026-06-27T17:44:36Z | `crashed` | [investigation_008](investigation_008/) | full-prediction | Low-rank rep |
| 32 | [`lambda_sigreg_0.3`](investigation_008/run_032_lambda_sigreg_0.3/) | `x7z6e0ah` | 2026-06-27T17:44:36Z | `crashed` | [investigation_008](investigation_008/) | full-prediction | Low-rank rep |
| 33 | [`lambda_sigreg_10.0`](investigation_008/run_033_lambda_sigreg_10.0/) | `fbqgix1x` | 2026-06-27T17:44:38Z | `crashed` | [investigation_008](investigation_008/) | full-prediction | Healthy rep, no predictor |
| 34 | [`sigreg-only`](investigation_009/run_034_sigreg-only/) | `xz3nabr9` | 2026-06-28T03:34:48Z | `crashed` | [investigation_009](investigation_009/) | full-prediction | Low-rank rep |
| 35 | [`sigreg-recon-residual`](investigation_009/run_035_sigreg-recon-residual/) | `jsh6uo7p` | 2026-06-28T03:34:51Z | `crashed` | [investigation_009](investigation_009/) | full-prediction | Low-rank rep |
| 36 | [`upbeat-frog-36`](investigation_010/run_036_upbeat-frog-36/) | `b4lf89if` | 2026-06-29T11:38:22Z | `killed` | [investigation_010](investigation_010/) | full-prediction | Smoke / inconclusive |
| 37 | [`soft-universe-37`](investigation_010/run_037_soft-universe-37/) | `2vbo6pbm` | 2026-06-29T11:38:25Z | `finished` | [investigation_010](investigation_010/) | full-prediction | Healthy rep, no predictor |
| 38 | [`new_recon_loss`](investigation_011/run_038_new_recon_loss/) | `1u69hpfm` | 2026-06-30T10:17:00Z | `finished` | [investigation_011](investigation_011/) | full-prediction | Healthy rep, no predictor |
| 39 | [`original_recon_loss + no-pred`](investigation_011/run_039_original_recon_loss-no-pred/) | `kttd1fib` | 2026-06-30T10:22:24Z | `finished` | [investigation_011](investigation_011/) | present-only | Collapsed rep |
| 40 | [`inv011_fixed_position_decoder`](investigation_011/run_040_inv011_fixed_position_decoder/) | `io74f32b` | 2026-06-30T18:48:31Z | `finished` | [investigation_011](investigation_011/) | full-prediction | Low-rank rep |
| 41 | [`inv011_fixed_position_present_recon`](investigation_011/run_041_inv011_fixed_position_present_recon/) | `hcr2qx19` | 2026-07-01T11:39:16Z | `crashed` | [investigation_011](investigation_011/) | present-only | Low-rank decodable |
| 42 | [`po_geom_sig7p5_cov0`](investigation_011/present-only-geometry-sweep/wave_1/run_042_po_geom_sig7p5_cov0/) | `fq0crddc` | 2026-07-01T20:55:01Z | `finished` | [investigation_011](investigation_011/) | present-only | Low-rank decodable |
| 43 | [`po_geom_sig5_cov0p003`](investigation_011/present-only-geometry-sweep/wave_1/run_043_po_geom_sig5_cov0p003/) | `4f2p1e7b` | 2026-07-01T20:55:02Z | `finished` | [investigation_011](investigation_011/) | present-only | Strong present representation |
| 44 | [`po_geom_sig10_cov0p003`](investigation_011/present-only-geometry-sweep/wave_1/run_044_po_geom_sig10_cov0p003/) | `5xockdbh` | 2026-07-01T20:55:02Z | `finished` | [investigation_011](investigation_011/) | present-only | Strong present representation |
| 45 | [`po_geom_sig12p5_cov0`](investigation_011/present-only-geometry-sweep/wave_1/run_045_po_geom_sig12p5_cov0/) | `76d6o8d2` | 2026-07-01T20:55:02Z | `finished` | [investigation_011](investigation_011/) | present-only | Strong present representation |
| 46 | [`po_geom_sig10_cov0`](investigation_011/present-only-geometry-sweep/wave_1/run_046_po_geom_sig10_cov0/) | `9ap28tbw` | 2026-07-01T20:55:03Z | `finished` | [investigation_011](investigation_011/) | present-only | Strong present representation |
| 47 | [`po_geom_sig5_cov0p01`](investigation_011/present-only-geometry-sweep/wave_2/run_047_po_geom_sig5_cov0p01/) | `az60m6mx` | 2026-07-02T03:48:45Z | `crashed` | [investigation_011](investigation_011/) | present-only | Strong present representation |
| 48 | [`po_geom_sig7p5_cov0p003`](investigation_011/present-only-geometry-sweep/wave_2/run_048_po_geom_sig7p5_cov0p003/) | `bg7ennr5` | 2026-07-02T03:48:46Z | `crashed` | [investigation_011](investigation_011/) | present-only | Strong present representation |
| 49 | [`po_geom_sig10_cov0p01`](investigation_011/present-only-geometry-sweep/wave_2/run_049_po_geom_sig10_cov0p01/) | `bttexglp` | 2026-07-02T03:48:46Z | `crashed` | [investigation_011](investigation_011/) | present-only | Strong present representation |
| 50 | [`po_geom_sig7p5_cov0p01`](investigation_011/present-only-geometry-sweep/wave_2/run_050_po_geom_sig7p5_cov0p01/) | `h5t89ezx` | 2026-07-02T03:48:46Z | `crashed` | [investigation_011](investigation_011/) | present-only | Strong present representation |
| 51 | [`po_geom_sig12p5_cov0p003`](investigation_011/present-only-geometry-sweep/wave_2/run_051_po_geom_sig12p5_cov0p003/) | `mtrviiab` | 2026-07-02T03:48:46Z | `crashed` | [investigation_011](investigation_011/) | present-only | Strong present representation |
| 52 | [`ae_sharp_slots_recon_only`](investigation_012/run_052_ae_sharp_slots_recon_only/) | `662hfy3c` | 2026-07-02T15:32:02Z | `running` | [investigation_012](investigation_012/) | present-only | Collapsed rep |

## Reading Rule

Use [`GUIDES/READING_EXPERIMENTS.md`](../../GUIDES/READING_EXPERIMENTS.md) for every run. Full-prediction and present-only runs use different cycles and must not be compared with the same gates.

---

## Update (2026-07-05, post-auto-generation) — investigations 013-015, runs 053-055

The auto-generated status block and run table above were produced when run 052 was the newest
run and still running. State at the time of this dated update:

- **55 runs** total in the project (`smahalanobis-uc-davis/hjepa-vwm`). Run 052 finished; runs
  053 and 054 completed after the auto-generation; run 055 is planned/not launched.
- No full-prediction run has passed both Phase 1 gates. The canonical negative remains run 037
  (`soft-universe-37`): rank 61, cross-video cosine 0.16, std 1.0, yet copy ratio 1.06.
- The frontier has moved to the PRESENT-ONLY autoencoder arc, whose settled conclusion is that
  reconstruction pressure alone cannot hold representation geometry — an explicit anti-collapse
  term is required.

Newer investigation index (append to the table above):

| Investigation | Status | Runs | Current conclusion |
|---|---|---|---|
| [investigation_012](investigation_012/) | CLOSED | 052 | Sharp-slot reconstruction-only (no geometry regularizers) collapses to a decoder-side video-independent template: reconstruction excellent (0.293) but rank 13.4, cross-video cosine 0.906, std 0.295. Sharp attention cannot replace geometry regularizers. |
| [investigation_013](investigation_013/) | CLOSED | 053 | The residual reconstruction target (reconstruct e - mean) fixes the template shortcut (video gap +0.433, ~77% video-conditioned) but geometry still collapses (rank 10.5). H1 solved, H2 confirmed: an explicit anti-collapse force is required. |
| [investigation_014](investigation_014/) | OPEN | none | Offline rank probe: frozen V-JEPA `e` has pooled entropy rank ~193/1024 with a long low-energy tail (rank@90% 333, rank@99% 785). Reframes e->c as selective denoising and motivates whitening. |
| [investigation_015](investigation_015/) | OPEN | 054, 055 | Whitening + Perceiver latent-stack bottleneck on the residual recipe: strongest honesty yet (run 054, ~92% video-conditioned, shuffled-c pinned 0.975) at a much better geometry equilibrium (rank 21.9 vs 053's 10.5), but geometry still contracts — "neither delta sufficient." Run 055 (absolute-target ablation of 054) shows whitening alone reaches ~86% honesty with identical geometry, so the residual target buys ~6 points of honesty for zero geometric cost — it stays in the recipe. Bottleneck-only whitening-vs-architecture control still open. |
| [investigation_016](investigation_016/) | OPEN | 058, 060-063; 059 planned | The clean-commit EGO4D `N_c=32/64/128` sweep completed. Late training reconstruction improves only `0.67703 -> 0.67396 -> 0.66298`, while recorded-batch geometry is healthy only at 32 (`std/cos=0.806/0.473`) and fails at 64/128. More query slots are not a healthy capacity win; skip 256 and move to no-whitening plus channel-width/`D_c` work. |

Newer run index rows (append to the historical W&B snapshot above):

| # | Run | ID | State | Investigation | Mode | Verdict |
|---:|---|---|---|---|---|---|
| 53 | [`ae_sharp_slots_residual_recon`](investigation_013/run_053_ae_sharp_slots_residual_recon/) | `7teohhwc` | crashed (external) | investigation_013 | present-only | Low-rank decodable |
| 54 | [`ae_latent_stack_whiten_recon_only`](investigation_015/run_054_ae_latent_stack_whiten_recon_only/) | `lx1b6gw2` | finished | investigation_015 | present-only | Low-rank decodable |
| 55 | [`ae_latent_stack_whiten_abs_recon`](investigation_015/run_055_ae_latent_stack_whiten_abs_recon/) | `nzz64pl6` | finished | investigation_015 | present-only | Low-rank decodable |
| 56 | [`ae_latent_stack_whiten_abs_recon_geom`](investigation_015/run_056_ae_latent_stack_whiten_abs_recon_geom/) | `tl5dh73c` | finished | investigation_015 | present-only | Strong present representation |
| 57 | [`ae_latent_stack_whiten_abs_recon_cov_var`](investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/) | `cdvp6hou` | finished | investigation_015 | present-only | Strong present representation |
| 58 | [`ae_latent_stack_whiten_abs_recon_cov_var_ego4d`](investigation_016/run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/) | `mvbx96nv` | finished | investigation_016 | present-only | Historical label: collapsed/template; later single-source correction makes global collapse indeterminate |
| 60 | [`Investigation 16 · Whitened latent stack EGO4D · Reconstruction weight 1.00`](investigation_016/run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/) | `2423b84g` | finished | investigation_016 | present-only | Healthy recorded-batch geometry; weak/source-confounded conditioning |
| 61 | [`Bottleneck capacity · EGO4D 32 slots`](investigation_016/bottleneck_slot_capacity_sweep/) | `x03xlpyl` | finished | investigation_016 | present-only | Strong recorded-batch representation; sweep control |
| 62 | [`Bottleneck capacity · EGO4D 64 slots`](investigation_016/bottleneck_slot_capacity_sweep/) | `evyokqrm` | finished | investigation_016 | present-only | Recorded-batch collapsed / source-chunk invariant |
| 63 | [`Bottleneck capacity · EGO4D 128 slots`](investigation_016/bottleneck_slot_capacity_sweep/) | `7pmvxrxi` | finished | investigation_016 | present-only | Weak recorded-batch geometry; not a capacity win |

## Update (2026-07-14) — investigation_016 (EGO4D transfer of run 057)

- Source control: run 057 `ae_latent_stack_whiten_abs_recon_cov_var` (`cdvp6hou`), latest proper
  experiment before the two EGO4D GUIDE smoke runs.
- New investigation:
  [`investigation_016`](investigation_016/) — EGO4D twin of that recipe.
- Completed run:
  [`run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d`](investigation_016/run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/)
  — W&B `mvbx96nv`; full
  [`ANALYSIS.md`](investigation_016/run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md).
- Planned parallel branch:
  [`run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d`](investigation_016/run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d/)
  — same recipe on the current strict pipeline with SigLIP 2. Run 058 is historical context;
  causal encoder attribution requires a same-commit V-JEPA companion.
- Live W&B has 60 entries because two Investigation-16 switchability/regression smokes precede
  the final science run. The local scientific sequence keeps the planned `run_058` label.
- Result: stable execution but failed transfer (rank 52.9, cosine 0.863, std 0.419, video gap
  0.018). Residual-target control remains queued.
- At that date, the 2026-07-15 planned weight arm was:
  [`run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1`](investigation_016/run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/)
  — run 058 absolute-target AE recipe with `lambda_recon=1.0` only.

## Update (2026-07-16) — run 060 completed and live project reconciled

- Live W&B now has **61 entries**. Run 060 `2423b84g` is the newest and finished all 15,000
  steps; run 059 has no W&B counterpart and remains genuinely unlaunched.
- Run 060 ended with rank `84.36`, std `0.800`, recorded pair cosine `0.479`, present reconstruction
  `0.67091`, rolled-code reconstruction `0.69736`, and gap `0.02644` (7.66% exact-chunk
  conditioned share). It is stable and geometrically healthy on the recorded batch, but not
  prediction-ready.
- The 16 EGO4D diagnostic chunks all share one source UID. Treat the cosine/gap as within-source
  adjacent-chunk measurements until the source-aware validation contract is repaired.
- Full run record:
  [`run_060.../ANALYSIS.md`](investigation_016/run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/ANALYSIS.md);
  ordered follow-up:
  [`reconstruction_floor_architecture_audit/NEXT_STEPS.md`](investigation_016/reconstruction_floor_architecture_audit/NEXT_STEPS.md).

## Update (2026-07-17) — investigation_016 slot-capacity sweep completed

- Live W&B now has **64 entries**. The same-commit sweep arms are 32 slots `x03xlpyl`, 64 slots
  `evyokqrm`, and 128 slots `7pmvxrxi`; all finished 15,000 steps with zero skipped/NaN updates.
- Every arm shares clean commit `820a5b5`, EGO4D/data order, fixed validation batch, pinned V-JEPA
  features, whitening payload, schedule, and seed. Only `N_c` and derived initialization/output
  identities differ.
- Late training `L_recon` medians are `0.67703/0.67396/0.66298`. The 32-to-128 gain is `0.01405`
  (2.08%), below the preregistered `0.02`/3% support threshold; the late fixed-batch diagnostic gain
  is only `0.00546`.
- Late fixed-batch `std/cosine` is `0.806/0.473`, `0.326/0.914`, and `0.649/0.677`. Added slots
  therefore do not produce a healthy content code even when pooled rank rises.
- Do not launch 256. Next paid axes are no whitening and the 1,024-to-256 channel/`D_c` squeeze,
  kept separate from alternate-encoder work. Full analysis:
  [`bottleneck_slot_capacity_sweep/ANALYSIS.md`](investigation_016/bottleneck_slot_capacity_sweep/ANALYSIS.md).

## Update (2026-07-19) — investigation_016 internal-memory width pair completed

- Live W&B now has **66 entries**. Run 64 M=512 `4biwq87o` and Run 65 M=1024 `8gr3je5b`
  both finished 15,000 steps on clean commit `9522008` with zero skipped/nonfinite/warned updates.
- Both arms use full EGO4D, raw features with whitening disabled, `N_c=32`, external `D_c=256`,
  one final projection after three latent blocks, `lambda_recon=1`, and zero auxiliary geometry
  weights. Only complete internal memory/query/latent width differs.
- Late active reconstruction is `0.260785/0.259879`; fixed correct-code loss is
  `0.305063/0.303584`; shuffled-code gap is `0.161563/0.163592`. M=1024 improves loss by only
  `0.000906` and gap by only `0.002029`, below the preregistered `0.01`/`0.005` thresholds.
- M=1024 raises recorded-batch effective rank `10.75 -> 15.38`, but both arms remain low-rank
  decodable and the fixed batch is single-source. Keep M=512; global collapse/preservation remains
  indeterminate until source-diverse diagnostics are repaired.
- Full paired evidence:
  [`run_065.../OBSERVATIONS.md`](investigation_016/run_065_unwhitened_internal_memory_m1024/OBSERVATIONS.md).

Newest run index rows:

| # | Run | ID | State | Investigation | Mode | Verdict |
|---:|---|---|---|---|---|---|
| 64 | [`Investigation 16 · Internal memory width · EGO4D M=512`](investigation_016/run_064_unwhitened_internal_memory_m512/) | `4biwq87o` | finished | investigation_016 | present-only | Low-rank decodable; global collapse indeterminate; selected width |
| 65 | [`Investigation 16 · Internal memory width · EGO4D M=1024`](investigation_016/run_065_unwhitened_internal_memory_m1024/) | `8gr3je5b` | finished | investigation_016 | present-only | Low-rank decodable; global collapse indeterminate; not cost-justified |

## Update (2026-07-21) — raw encoder controls reconciled; investigation 017 expanded

- Live W&B has **70 entries**. Runs 66 (`guiduvjp`) and 67 (`ufbeokj2`) were intentionally stopped
  while healthy; Run 68 (`j7a3tzj5`) ended externally at step 14,350 after stable training; Run 69
  (`it7sq8nz`) completed all 15,000 steps.
- Run 66 shows covariance plus variance can keep the raw V-JEPA2 `M=512` code substantially open.
  Run 67 shows the same terms also open SigLIP 2 relative to its no-geometry successor, although its
  partial endpoint remains less spread and more aligned than V-JEPA2.
- Raw SigLIP 2 Run 68 and DINOv3 Run 69 both learn positive correct-versus-shuffled reconstruction
  gaps without geometry pressure, but both contract to low-rank, highly aligned codes. This
  validates the revised bottleneck's input dependence and re-establishes the need for explicit
  geometry pressure.
- DINOv3 is now a pinned implemented encoder at revision
  `5931719e67bbdb9737e363e781fb0c67687896bc`; Run 69 proved its `(B,2048,768)` token contract and
  complete EGO4D present-reconstruction training path. It did not exercise prediction.
- [Investigation 017](investigation_017/) is OPEN with three encoder-specific sweep bundles. Each
  runs the full `N_c={16,32,64}` by `D_c={128,256,512}` grid at fixed `M=512` with variance and
  covariance enabled: 27 planned runs total, none launched.

Current investigation rows:

| Investigation | Status | Runs | Current conclusion |
|---|---|---|---|
| [investigation_016](investigation_016/) | OPEN | 058, 060–069; 059 unlaunched | `M=512` is the practical internal width. Raw no-geometry codes remain low-rank across V-JEPA2, SigLIP 2, and DINOv3; covariance plus variance is the carried geometry bundle. |
| [investigation_017](investigation_017/) | OPEN | 27 planned | Sweep external slot count and slot width independently within V-JEPA2, SigLIP 2, and DINOv3. Compare raw losses only within encoder; synthesize normalized geometry, dependence, stability, and compute across encoders. |

Newest scientific run rows:

| # | Run | ID | State | Investigation | Mode | Verdict |
|---:|---|---|---|---|---|---|
| 66 | [`Investigation 16 · Raw-feature geometry · M=512 covariance plus variance`](investigation_016/run_066_unwhitened_internal_memory_m512_cov_var/) | `guiduvjp` | killed (intentional) | investigation_016 | present-only | Strong present representation; partial endpoint |
| 67 | [`Investigation 16 · Standard-ViT geometry · SigLIP2-B M=512 covariance plus variance`](investigation_016/run_067_unwhitened_siglip2_m512_cov_var/) | `ufbeokj2` | killed (intentional) | investigation_016 | present-only | Low-rank decodable; partial endpoint |
| 68 | [`Investigation 16 · Standard-ViT geometry · SigLIP2-B M=512 no covariance plus variance`](investigation_016/run_068_unwhitened_siglip2_m512_no_geometry_regularizers/) | `j7a3tzj5` | crashed (external) | investigation_016 | present-only | Low-rank decodable; global collapse indeterminate |
| 69 | [`Investigation 16 · Encoder substrate · DINOv3 unwhitened memory M=512`](investigation_016/run_069_unwhitened_dinov3_m512_no_geometry_regularizers/) | `it7sq8nz` | finished | investigation_016 | present-only | Low-rank decodable; global collapse indeterminate |

## Canonical live reconciliation (2026-07-26)

This section is the current source of truth. All earlier tables and dated updates above are
preserved snapshots and may contain states that were true only when written.

- Historical source-branch folders named
  `run_067_dinov3_unwhitened_internal_memory_m512` and
  `run_068_dinov3_whitened_internal_memory_m512_cov_var` preserve pre-reconciliation labels.
  Their canonical local records are Run 069 (`it7sq8nz`) and Run 070 (`qqozribu`),
  respectively; do not count the historical folders as additional scientific runs.
- Live project: `smahalanobis-uc-davis/hjepa-vwm`.
- W&B inventory: **75 entries** — 34 finished, 29 crashed, 11 killed, 1 failed, and **0 running**.
- Documentation snapshot: the 75-entry reconciliation was published in Git history through merge
  commit `b73bfa918c77da2574bf152f99935010c878b963`. Investigation 018 and local scientific run 072
  are newer pre-launch registrations and do not yet have a W&B entry. Individual W&B runs retain
  their own clean runtime commits.
- Investigation 016 is closed after recording the two completed DINO geometry arms.
- Investigation 017 remains open but has **no active queue**. Its only unique V-JEPA2 arm,
  `ihiuptdp` (`16×512`), is crashed at step 7,100; the remaining arms are unlaunched.
- No W&B entry in this project demonstrates DINOv3 full-prediction execution. Runs 069–071 and
  the DINO smoke are present-only: `prediction_active=0`, `L_flow=0`, and `L_recon_pred=0`.

### Canonical interpretation boundaries

For EGO4D runs in this ledger, the fixed diagnostic batch contains 16 adjacent chunks from one
source UID. Consequently, `c_cross_video_cosine` is a within-source cross-chunk statistic and the
rolled-code gap proves exact-chunk dependence only. A historical phrase such as “global template
collapse,” “cross-video separation,” or “video-conditioned share” is not a cross-source result
unless a later record explicitly supplies source-diverse evidence.

For DINOv3 and SigLIP 2, the normal W&B `model` block retains legacy V-JEPA compatibility fields.
The runtime authority is `resolved_provenance.encoder_spec`: DINOv3 resolves to 768 channels and an
`8×16×16` frame lattice even though `model.d_e=1024` remains serialized.

### Current investigation index

| Investigation | Current status | Current conclusion |
|---|---|---|
| [investigation_016](investigation_016/) | CLOSED | The EGO4D transfer, bottleneck, and three-encoder substrate arc is recorded through local scientific runs 071. DINO is validated only for present reconstruction. |
| [investigation_017](investigation_017/) | OPEN — no active runs | Three within-encoder `N_c×D_c` sweeps remain incomplete. Exact `32×256` recipe evidence now exists for all three encoder lanes; `ihiuptdp` is an invalid partial V-JEPA2 arm, not an active process. |
| [investigation_018](investigation_018/) | OPEN — run 072 registered | Test pure SIGReg on the current raw V-JEPA2 `M=512`, `32×256` present-reconstruction recipe with variance and covariance disabled. |

### Complete current W&B inventory

“W&B ordinal” is creation order and includes smokes, duplicates, and interrupted launches. “Local
label” is the scientific-folder sequence. They diverge after entry 057; the W&B ID is the
authoritative join key.

| W&B ordinal | W&B ID | State | Last logged step | Local label / role |
|---:|---|---|---:|---|
| 001 | `x4pwz33d` | finished | 50 | local 001, investigation 002 launch smoke |
| 002 | `fz7ztfc8` | finished | 150 | local 002, investigation 002 throughput before frame fix |
| 003 | `0mgmqxxi` | finished | 199 | local 003, investigation 002 dense logging before frame fix |
| 004 | `gj8ypv0d` | finished | 150 | local 004, investigation 002 throughput after frame fix |
| 005 | `1chv2608` | failed | 10,950 | local 005, investigation 001 long baseline |
| 006 | `wv69n7n5` | finished | 450 | local 006, investigation 003 collapse probe |
| 007 | `rpxyg9qt` | crashed | 4,450 | local 007, investigation 003 full-data baseline |
| 008 | `dhp1i3fk` | killed | 4,300 | local 008, investigation 003 strong slot loss |
| 009 | `m30jxiye` | killed | -1 | local 009, investigation 003 empty launch |
| 010 | `27i1r9qi` | crashed | 2,550 | local 010, investigation 003 mild slot loss |
| 011 | `q40nq0l3` | killed | 3,900 | local 011, investigation 003 centered slot loss |
| 012 | `ejror834` | killed | 5,650 | local 012, investigation 003 centered-slot repeat |
| 013 | `4lo4j7qb` | killed | 6,900 | local 013, investigation 003 strong variance |
| 014 | `8bkeeuio` | crashed | 3,900 | local 014, investigation 003 strong-variance repeat |
| 015 | `jhodg49x` | crashed | 13,850 | local 015, investigation 005 pre-AGC acceptance |
| 016 | `0n5mx3qf` | finished | 14,950 | local 016, investigation 005 checkpoint resume |
| 017 | `0xv4upvb` | killed | 11,350 | local 017, investigation 005 adaptive clipping |
| 018 | `yd5958s6` | killed | 14,400 | local 018, investigation 006 present reconstruction |
| 019 | `3syv6wp2` | finished | 14,950 | local 019, investigation 006 predicted reconstruction |
| 020 | `5x7aoxnn` | crashed | 9,050 | local 020, investigation 007 reconstruction weight 0.20 |
| 021 | `591mt31k` | crashed | 9,100 | local 021, investigation 007 wide/deep decoder |
| 022 | `708jrel8` | crashed | 8,850 | local 022, investigation 007 wide decoder |
| 023 | `a2trqp9c` | crashed | 9,200 | local 023, investigation 007 reconstruction weight 0.10 |
| 024 | `rju7xsh2` | crashed | 9,250 | local 024, investigation 007 reconstruction weight 0.50 |
| 025 | `2xsd5jwr` | crashed | 200 | local 025, investigation 007 256-slot smoke |
| 026 | `7u5zkw6t` | crashed | 200 | local 026, investigation 007 64-slot smoke |
| 027 | `bbrrydax` | crashed | 200 | local 027, investigation 007 64-slot wide-decoder smoke |
| 028 | `ryuh8cpr` | crashed | 200 | local 028, investigation 007 128-slot smoke |
| 029 | `tw685b5g` | crashed | 200 | local 029, investigation 007 reconstruction-weight smoke |
| 030 | `9jxc8i1q` | crashed | 13,050 | local 030, investigation 008 SIGReg 3.0 |
| 031 | `jk8kj7h7` | crashed | 13,100 | local 031, investigation 008 SIGReg 1.0 |
| 032 | `x7z6e0ah` | crashed | 13,150 | local 032, investigation 008 SIGReg 0.3 |
| 033 | `fbqgix1x` | crashed | 13,150 | local 033, investigation 008 SIGReg 10.0 |
| 034 | `xz3nabr9` | crashed | 14,050 | local 034, investigation 009 no-reconstruction control |
| 035 | `jsh6uo7p` | crashed | 14,050 | local 035, investigation 009 residual prediction |
| 036 | `b4lf89if` | killed | 250 | local 036, investigation 010 launch check |
| 037 | `2vbo6pbm` | finished | 14,950 | local 037, investigation 010 full clean run |
| 038 | `1u69hpfm` | finished | 14,950 | local 038, investigation 011 cosine reconstruction |
| 039 | `kttd1fib` | finished | 14,950 | local 039, investigation 011 present-only control |
| 040 | `io74f32b` | finished | 14,950 | local 040, investigation 011 fixed-position decoder |
| 041 | `hcr2qx19` | crashed | 14,400 | local 041, investigation 011 fixed-position present reconstruction |
| 042 | `fq0crddc` | finished | 14,950 | local 042, investigation 011 present geometry |
| 043 | `4f2p1e7b` | finished | 14,950 | local 043, investigation 011 present geometry |
| 044 | `5xockdbh` | finished | 14,950 | local 044, investigation 011 present geometry |
| 045 | `76d6o8d2` | finished | 14,950 | local 045, investigation 011 present geometry |
| 046 | `9ap28tbw` | finished | 14,950 | local 046, investigation 011 present geometry |
| 047 | `az60m6mx` | crashed | 14,150 | local 047, investigation 011 present geometry |
| 048 | `bg7ennr5` | crashed | 14,000 | local 048, investigation 011 present geometry |
| 049 | `bttexglp` | crashed | 14,350 | local 049, investigation 011 present geometry |
| 050 | `h5t89ezx` | crashed | 14,300 | local 050, investigation 011 present geometry |
| 051 | `mtrviiab` | crashed | 14,250 | local 051, investigation 011 present geometry |
| 052 | `662hfy3c` | finished | 14,950 | local 052, investigation 012 sharp-slot reconstruction |
| 053 | `7teohhwc` | crashed | 12,150 | local 053, investigation 013 residual reconstruction |
| 054 | `lx1b6gw2` | finished | 14,950 | local 054, investigation 015 whitened residual reconstruction |
| 055 | `nzz64pl6` | finished | 14,950 | local 055, investigation 015 whitened absolute reconstruction |
| 056 | `tl5dh73c` | finished | 14,950 | local 056, investigation 015 full geometry |
| 057 | `cdvp6hou` | finished | 14,950 | local 057, investigation 015 covariance plus variance |
| 058 | `29a2ora7` | finished | 450 | operational EGO4D 500-step smoke; no local scientific label |
| 059 | `tt4x64hc` | finished | 50 | operational SSv2 100-step regression smoke; no local scientific label |
| 060 | `mvbx96nv` | finished | 14,950 | local 058, investigation 016 EGO4D transfer |
| 061 | `2423b84g` | finished | 14,950 | local 060, investigation 016 reconstruction weight 1.0 |
| 062 | `x03xlpyl` | finished | 14,950 | local 061, investigation 016 32-slot capacity control |
| 063 | `evyokqrm` | finished | 14,950 | local 062, investigation 016 64-slot capacity arm |
| 064 | `7pmvxrxi` | finished | 14,950 | local 063, investigation 016 128-slot capacity arm |
| 065 | `4biwq87o` | finished | 14,950 | local 064, investigation 016 unwhitened memory M=512 |
| 066 | `8gr3je5b` | finished | 14,950 | local 065, investigation 016 unwhitened memory M=1024 |
| 067 | `ufbeokj2` | killed | 11,000 | local 067, investigation 016 SigLIP 2 cov+var center |
| 068 | `guiduvjp` | killed | 10,950 | local 066, investigation 016 V-JEPA2 cov+var center |
| 069 | `j7a3tzj5` | crashed | 14,350 | local 068, investigation 016 SigLIP 2 no-geometry arm |
| 070 | `it7sq8nz` | finished | 14,950 | local 069, investigation 016 DINOv3 no-geometry arm |
| 071 | `kiti1gpc` | killed | 1,000 | investigation 017 duplicate V-JEPA2 center; excluded |
| 072 | `ihiuptdp` | crashed | 7,100 | [investigation 017 V-JEPA2 `16×512`](investigation_017/vjepa2_latent_shape_sweep/ihiuptdp_vjepa2_n16_d512/); invalid partial arm |
| 073 | `lwx0mu34` | finished | 90 | operational DINOv3 whitened cov+var smoke |
| 074 | `qqozribu` | finished | 14,950 | [local 070](investigation_016/run_070_whitened_dinov3_m512_cov_var/), investigation 016 DINOv3 whitened cov+var |
| 075 | `fiactcw6` | finished | 14,950 | [local 071](investigation_016/run_071_unwhitened_dinov3_m512_cov_var/), DINOv3 unwhitened cov+var / investigation 017 center |

## Registered next experiment (2026-07-27)

Local scientific
[run 072](investigation_018/run_072_vjepa2_unwhitened_m512_sigreg10_only/) is registered under
[investigation 018](investigation_018/). It keeps the latest raw/unwhitened `M=512`, `32×256`,
decoder-`512×4`, reconstruction-only recipe, selects V-JEPA2, and replaces covariance plus
variance with `lambda_sigreg=10` alone. No W&B entry exists until launch; the W&B ID remains the
required join key once created.

## Registered sweep (2026-07-28)

[Investigation 019](investigation_019/) supersedes the inactive Investigation-017 queue with a
four-cell-per-encoder, raw reconstruction-only bottleneck design. It holds the settled internal
width at `M=512` and runs the complete `N_c={16,64}` by `D_c={128,512}` factorial independently
for V-JEPA2, DINOv3, and SigLIP 2. All auxiliary geometry weights and feature whitening are off.
The encoder lanes execute in that order; four cells run concurrently on GPUs 0–3 within a lane.
No W&B entry exists until launch.

## Live correction (2026-07-28) — W&B entry 076 and source-diverse diagnostics

This is the current source of truth after the preserved 2026-07-26 reconciliation above.

- Live W&B contains **76 entries**: 35 finished, 29 crashed, 11 killed, 1 failed, and 0 running.
  Entry 076 is local scientific Run 072
  [`utcpfj57`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/utcpfj57); it finished all
  15,000 updates. The 75-entry inventory and “not yet launched” text above are dated history.
- Run 072 was stable and correctly present-only (`prediction_active=0`, zero skipped/NaN updates).
  At the final diagnostic its present/shuffled reconstruction was `0.35721/0.42989` (gap
  `0.07268`), `c_std_mean=0.48411`, `c_effective_rank=57.09`, and the historical-batch
  `c_cross_video_cosine=0.73884`. Pure SIGReg improved substantially over the no-geometry V-JEPA2
  control but did not match the covariance-plus-variance partial reference or pass the registered
  std/rank/cosine gates. Verdict: **collapsed on the recorded within-source batch; decodable,
  cross-source specificity unmeasured**. It is not prediction evidence.
- PR 9 repaired the fixed diagnostic population for future executions. EGO4D now scans validation
  order and keeps the first clip from each distinct source UID; SSv2 retains first-N behavior.
  Shortfall warns without duplicate refill, and fewer than two realized sources fails loudly.
  Therefore the deterministic code roll used by `L_recon_shuffled_c` is also cross-source.
- The same encoder forward now logs `e_cross_video_cosine` on the detailed representation and the
  historical `c_cross_video_cosine` key on the abstract latent, using the same flatten-normalize-
  off-diagonal-mean function. Historical W&B entries 001–076 retain their original metrics:
  EGO4D `c` values are within-source cross-chunk measurements and no `e` key was logged.
- The corrected offline evaluation of unwhitened Run 069 step 15,000 used 16 clips from 16 unique
  EGO4D source UIDs. Two repetitions were identical:
  `e_cross_video_cosine=0.4027678072`, `c_cross_video_cosine=0.0907106772`,
  `c-e=-0.31205713`, or a 77.5% reduction relative to raw DINO. This is checkpoint evidence on the
  live online bottleneck, not a retroactive replacement of Run 069's W&B curve. The current narrow
  evaluator rejects whitened checkpoints.

Current research frontier: the bottleneck can carry strong present representations under some
geometry recipes, and the repaired Run 069 probe shows that one DINO bottleneck removes much of
the raw encoder's shared cross-source direction. No full-prediction run has yet passed both Phase-1
forecasting gates. Investigation 020 is now the first live DINOv3 full-prediction pair, while
Investigation 017's non-center latent-shape sweep remains incomplete.

## Registered follow-up (2026-07-28) — Investigation 020

[Investigation 020](investigation_020/) registers the paired transfer from the selected
Investigation-019 DINOv3 `64×512`, `M=512` bottleneck into the current full-prediction architecture.
Both arms warm-started the exact same DINOv3 online bottleneck and matched decoder checkpoint, kept
the bottleneck trainable, reinitialized the EMA target from the loaded online bottleneck, and
started fresh flow, optimizer, schedule, sampler, RNG, checkpoint, and W&B state. Covariance plus
variance is fixed in both arms; the only scientific difference is residual versus full-latent
temporal prediction. The initialization-only warm start, concurrent temporal-target CLI, and
narrow parity guard are implemented and tested. Both exact-source resource gates passed, and the
pair is running from clean commit `7649f8efde1b104dd81cfbd110af18d499f67304`: residual
[`3y2hxj5t`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3y2hxj5t) and full latent
[`8r6akjsx`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8r6akjsx).

## Update (2026-08-01) — Investigation 020 completed

This is the current W&B reconciliation after the dated launch record above. The project contains
**94 runs**: 49 finished, 33 crashed, 11 killed, 1 failed, and 0 running. The newest two runs are
the completed Investigation-020 full-prediction pair:

| Arm | W&B ID | State | Verdict | Late copy ratio | Late batch ratio |
|---|---|---|---|---:|---:|
| residual | [`3y2hxj5t`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3y2hxj5t) | finished | **Healthy rep, no predictor** | 1.726204 | 1.817973 |
| full latent | [`8r6akjsx`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8r6akjsx) | finished | **Static-`c` trap** | 3.132558 | 0.946963 |

[Investigation 020](investigation_020/) is **CLOSED with no winner**. Both arms completed all
15,000 updates from the same pretrained DINOv3 B/D checkpoint, and W&B provenance confirms that
only `predict_residual` differs scientifically. Neither arm passed the `<=0.70` copy gate or the
`<=0.50` batch-mean gate at any diagnostic point.

Residual prediction preserved the useful substrate: late rank is `376.932/512`, source-diverse
latent cosine is `0.179245` against encoder `0.402768`, and copy loss rose. It still lost to zero
residual and batch mean. Full-latent prediction contracted rank to `296.622/512`, raised latent
cosine to `0.325100`, and reduced copy loss while becoming more than three times worse than copy.
The research frontier is therefore no longer “can a strong present bottleneck be transferred?”;
it is “can the current Fc/flow objective learn temporal dynamics in fixed bottleneck coordinates?”
The registered next probe is residual prediction with B/B_EMA frozen, without simultaneous
encoder, regularizer, horizon, or Fc-capacity changes.
