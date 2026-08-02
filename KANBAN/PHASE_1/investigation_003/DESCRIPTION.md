# investigation_003 - Which Phase 1 knobs keep c_t noncollapsed, and do those knobs make F_c beat the copy baseline?

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Status:** CLOSED  
**Runs covered:** 006, 007, 008, 009, 010, 011, 012, 013, 014  
**Theme:** collapse diagnostics, variance floor, horizon, and slot-loss variants

## Question

Which Phase 1 knobs keep c_t noncollapsed, and do those knobs make F_c beat the copy baseline?

## Why This Investigation Exists

The early runs made it clear that lowering L_flow was not enough. This investigation tested whether variance, horizon, and slot/covariance choices could produce a video-specific abstract latent and a useful predictor.

## W&B-Validated Run Coverage

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 6 | [`exalted-lion-6`](run_006_exalted-lion-6/) | `wv69n7n5` | `finished` | full-prediction | dataset=ssv2_tiny; steps=500; k=4; var=0.1; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Smoke / inconclusive | c_effective_rank=10.3413; c_cross_video_cosine=0.2461; c_std_mean=0.8307; coarse_vs_copy_ratio=2.9663; coarse_vs_batch_mean_ratio=1.9614 |
| 7 | [`sleek-leaf-7`](run_007_sleek-leaf-7/) | `rpxyg9qt` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=4; var=0.1; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Invalid | c_effective_rank=9.4703; c_cross_video_cosine=0.4996; c_std_mean=0.6909; coarse_vs_copy_ratio=2.3479; coarse_vs_batch_mean_ratio=1.3945 |
| 8 | [`serene-cloud-8`](run_008_serene-cloud-8/) | `dhp1i3fk` | `killed` | full-prediction | dataset=ssv2; steps=5000; k=4; var=0.1; cov=0.0027; slot=0.25; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Invalid | c_effective_rank=14.4777; c_cross_video_cosine=0.7435; c_std_mean=0.45; coarse_vs_copy_ratio=11.4562; coarse_vs_batch_mean_ratio=5.9294 |
| 9 | [`confused-butterfly-9`](run_009_confused-butterfly-9/) | `m30jxiye` | `killed` | full-prediction | dataset=ssv2; steps=5000; k=4; var=0.1; cov=0.0027; slot=0.25; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Smoke / inconclusive | n/a |
| 10 | [`skilled-waterfall-10`](run_010_skilled-waterfall-10/) | `27i1r9qi` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.1; cov=0.0027; slot=0.05; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Invalid | c_effective_rank=7.2924; c_cross_video_cosine=0.5756; c_std_mean=0.6035; coarse_vs_copy_ratio=1.8968; coarse_vs_batch_mean_ratio=2.3786 |
| 11 | [`olive-terrain-11`](run_011_olive-terrain-11/) | `q40nq0l3` | `killed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.1; cov=0.0027; slot=0.05; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Invalid | c_effective_rank=9.0648; c_cross_video_cosine=0.6538; c_std_mean=0.5552; coarse_vs_copy_ratio=2.8482; coarse_vs_batch_mean_ratio=2.6544 |
| 12 | [`copper-sky-12`](run_012_copper-sky-12/) | `ejror834` | `killed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.1; cov=0.0027; slot=0.05; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Invalid | c_effective_rank=4.8157; c_cross_video_cosine=0.7495; c_std_mean=0.4585; coarse_vs_copy_ratio=2.0106; coarse_vs_batch_mean_ratio=4.2299 |
| 13 | [`cerulean-snow-13`](run_013_cerulean-snow-13/) | `4lo4j7qb` | `killed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Low-rank rep | c_effective_rank=13.6717; c_cross_video_cosine=0.2356; c_std_mean=1.0378; coarse_vs_copy_ratio=0.9493; coarse_vs_batch_mean_ratio=0.2046 |
| 14 | [`jolly-forest-14`](run_014_jolly-forest-14/) | `8bkeeuio` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Low-rank rep | c_effective_rank=9.392; c_cross_video_cosine=0.2281; c_std_mean=0.9446; coarse_vs_copy_ratio=1.5639; coarse_vs_batch_mean_ratio=0.3607 |

## Current Conclusion

The useful result was not slot loss. The project learned that horizon_k=12 with a stronger variance floor could stabilize basic representation health, but the representation remained low-rank and still did not pass the copy or batch-mean gates.

## Evidence Standard

The run entries above were reconciled against W&B API/export data on 2026-07-02. For run 052, the newest active run, the evidence was additionally refreshed live with `python run_history.py --run 662hfy3c --report`, which showed the run still `running` through step 5600.

## Original Notes Preserved

# Investigation 003 — Why does `c_t` collapse (low rank / video-agnostic codes)?

**Status:** CLOSED  
**Opened:** 2026-06-10 (after `peachy-terrain-5` showed `c_effective_rank ~5`)  
**Closed:** 2026-06 (after `cerulean-snow-13` validated `lambda_var=0.5` + `horizon_k=12`)

## Question

Why does abstract latent `c_t` use only ~2% of its 256 dimensions (`c_effective_rank ~5`),
and why does `c_cross_video_cosine` climb toward video-independent codes — despite
healthy variance-floor metrics on some runs?

## Why it matters

Phase 1 acceptance requires non-collapsed `c_t` and `F_c` beating copy baseline.
Low rank means the bottleneck is not carrying future-relevant structure; Phase 2
(fine flow hierarchy) is pointless on a collapsed coarse state.

## Parent context

- Branched from: [investigation_001](../investigation_001/) (`peachy-terrain-5` rank ~5)
- Plan docs: [`AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/`](../../../AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE)
- Spec: [`AGENT_FILES/PHASES/PHASE_1.md`](../../../AGENT_FILES/PHASES/PHASE_1.md) §12 collapse gates

## Runs in this investigation

Listed in W&B chronological order (the BRIEF "Run N" labels are a coarser narrative
overlay; verified configs are in each run's DESCRIPTION).

| Run | BRIEF | Role (verified vs W&B) |
|---|---|---|
| [`exalted-lion-6`](run_006_exalted-lion-6/) | P1 | **First** — tiny diagnostic; flag-based init ablation → init is a non-lever; pointed at VICReg-C |
| [`sleek-leaf-7`](run_007_sleek-leaf-7/) | **Run 2** | Full SSv2 baseline / VICReg Run A @3500; per-head metric exposes slot collapse (1.62) |
| [`serene-cloud-8`](run_008_serene-cloud-8/) | **Run 3** | k=4, **raw** slot loss 0.25 — early Goodhart |
| [`confused-butterfly-9`](run_009_confused-butterfly-9/) | — | Failed launch (1s, k=4 slot=0.25) |
| [`skilled-waterfall-10`](run_010_skilled-waterfall-10/) | **Run 4** | **k=4** (not 12 — launch drift), raw slot 0.05 **inert** → exposed loss/metric bug |
| [`olive-terrain-11`](run_011_olive-terrain-11/) | Run 5a | **First centered-slot k=12** run (slot=0.05) — Goodhart begins; killed @3900 |
| [`copper-sky-12`](run_012_copper-sky-12/) | **Run 5b** | Re-run of olive — Goodhart confirmed (rank→4.8, cosine 0.84) + grad spikes |
| [`cerulean-snow-13`](run_013_cerulean-snow-13/) | **Run 6** | **Win:** `lambda_var=0.5`, k=12, no slot |
| [`jolly-forest-14`](run_014_jolly-forest-14/) | Run 6b | Winning-config repeat (var=0.5, k=12); reproduced cerulean, crashed @3900 |

## Spawned

- [investigation_004](../investigation_004/) — VICReg-C path (Run A executed; Run B not run)
- [investigation_005](../investigation_005/) — complete 15k with winning config
