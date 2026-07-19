# Phase 1 - Research KANBAN

<!-- AUTO-GENERATED-WANDB-KANBAN -->

Phase 1 currently covers coarse dynamics only: frozen V-JEPA 2 encoder, trainable bottleneck `B`, EMA bottleneck `B_EMA`, coarse flow `F_c`, optional feature reconstruction decoder `D`, and diagnostic/regularization knobs used to understand collapse, rank, temporal dynamics, reconstruction honesty, and prediction baselines.

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

## Complete W&B Run Index

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

Newer run index rows (append to the Complete W&B Run Index above):

| # | Run | ID | State | Investigation | Mode | Verdict |
|---:|---|---|---|---|---|---|
| 53 | [`ae_sharp_slots_residual_recon`](investigation_013/run_053_ae_sharp_slots_residual_recon/) | `7teohhwc` | crashed (external) | investigation_013 | present-only | Low-rank decodable |
| 54 | [`ae_latent_stack_whiten_recon_only`](investigation_015/run_054_ae_latent_stack_whiten_recon_only/) | `lx1b6gw2` | finished | investigation_015 | present-only | Low-rank decodable |
| 55 | [`ae_latent_stack_whiten_abs_recon`](investigation_015/run_055_ae_latent_stack_whiten_abs_recon/) | `nzz64pl6` | finished | investigation_015 | present-only | Low-rank decodable |
| 56 | [`ae_latent_stack_whiten_abs_recon_geom`](investigation_015/run_056_ae_latent_stack_whiten_abs_recon_geom/) | `tl5dh73c` | finished | investigation_015 | present-only | Strong present representation |
| 57 | [`ae_latent_stack_whiten_abs_recon_cov_var`](investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/) | `cdvp6hou` | finished | investigation_015 | present-only | Strong present representation |
| 58 | [`ae_latent_stack_whiten_abs_recon_cov_var_ego4d`](investigation_016/run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/) | `mvbx96nv` | finished | investigation_016 | present-only | Collapsed rep / template shortcut |
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
