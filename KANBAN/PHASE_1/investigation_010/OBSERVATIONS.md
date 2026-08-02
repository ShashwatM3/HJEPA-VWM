# Observations - investigation_010

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Cross-Run Synthesis

Run 037 proved a key negative: c_t can be high-rank, video-specific, and stable while F_c still fails the copy gate. That is the canonical healthy-representation/no-predictor result.

## Run-by-Run Evidence

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 36 | [`upbeat-frog-36`](run_036_upbeat-frog-36/) | `b4lf89if` | `killed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=5; recon=0.05/0.05; residual=true; present_only=false; n_c=32; D=512x4 | Smoke / inconclusive | c_effective_rank=9.4729; c_cross_video_cosine=0.7239; c_std_mean=0.4952; coarse_vs_copy_ratio=22.9366; coarse_vs_batch_mean_ratio=24.3867; L_recon_present=1.0386 |
| 37 | [`soft-universe-37`](run_037_soft-universe-37/) | `2vbo6pbm` | `finished` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=5; recon=0.05/0.05; residual=true; present_only=false; n_c=32; D=512x4 | Healthy rep, no predictor | c_effective_rank=61.0804; c_cross_video_cosine=0.1624; c_std_mean=1.0054; coarse_vs_copy_ratio=1.0631; coarse_vs_batch_mean_ratio=1.1397; L_recon_present=0.5713 |

## Pattern Across The Branch

Best copy ratio in this branch was run 037 at 1.0631; best batch-mean ratio was run 037 at 1.1397. None should be read as a full Phase 1 pass unless both gates pass together.

Verdict distribution: Healthy rep, no predictor=1, Smoke / inconclusive=1.

## What Changed The Research Direction

The research moved to reconstruction geometry and decoder honesty, because prediction failure could no longer be blamed only on rank collapse.

## Original Notes Preserved

Run 037 is the decisive datapoint and the canonical negative result of the project: with the clean
plumbing the residual recipe reached rank 61.08 (>60 gate cleared), cross-video cosine 0.16, std
1.005 — a genuinely healthy, high-rank, video-specific c — yet `coarse_vs_copy_ratio` sat at 1.063
and `coarse_vs_batch_mean_ratio` at 1.140, both above the gates. F_c ties by predicting ~zero
residual even though c now moves a lot (`coarse_copy_loss` rose 0.05 -> 1.56). Representation health
is therefore NOT the limiting factor; the predictor/dynamics is. Run 036 carries no result (aborted).
Full per-run reads: [`run_037_soft-universe-37/OBSERVATIONS.md`](run_037_soft-universe-37/OBSERVATIONS.md).
