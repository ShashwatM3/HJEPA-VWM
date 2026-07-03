# investigation_010 - Can the residual recipe produce a healthy representation and a predictor that beats copy in a clean 15k run?

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Status:** CLOSED  
**Runs covered:** 036, 037  
**Theme:** clean residual run with optimizer/regularization plumbing

## Question

Can the residual recipe produce a healthy representation and a predictor that beats copy in a clean 15k run?

## Why This Investigation Exists

Investigation 009 had promising representation movement but still poor forecasting. This branch removed confounds by using the cleaned residual/reconstruction/SIGReg setup.

## W&B-Validated Run Coverage

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 36 | [`upbeat-frog-36`](run_036_upbeat-frog-36/) | `b4lf89if` | `killed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=5; recon=0.05/0.05; residual=true; present_only=false; n_c=32; D=512x4 | Smoke / inconclusive | c_effective_rank=9.4729; c_cross_video_cosine=0.7239; c_std_mean=0.4952; coarse_vs_copy_ratio=22.9366; coarse_vs_batch_mean_ratio=24.3867; L_recon_present=1.0386 |
| 37 | [`soft-universe-37`](run_037_soft-universe-37/) | `2vbo6pbm` | `finished` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=5; recon=0.05/0.05; residual=true; present_only=false; n_c=32; D=512x4 | Healthy rep, no predictor | c_effective_rank=61.0804; c_cross_video_cosine=0.1624; c_std_mean=1.0054; coarse_vs_copy_ratio=1.0631; coarse_vs_batch_mean_ratio=1.1397; L_recon_present=0.5713 |

## Current Conclusion

Run 037 proved a key negative: c_t can be high-rank, video-specific, and stable while F_c still fails the copy gate. That is the canonical healthy-representation/no-predictor result.

## Evidence Standard

The run entries above were reconciled against W&B API/export data on 2026-07-02. For run 052, the newest active run, the evidence was additionally refreshed live with `python run_history.py --run 662hfy3c --report`, which showed the run still `running` through step 5600.
