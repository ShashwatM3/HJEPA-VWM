# Observations - investigation_012

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Cross-Run Synthesis

Live W&B through step 5600 shows strong reconstruction progress but not healthy representation geometry: c_std_mean is still far below 1, cross-video cosine remains high, and rank has fallen into the low 20s. The run is still active, so the final verdict remains provisional.

## Run-by-Run Evidence

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 52 | [`ae_sharp_slots_recon_only`](run_052_ae_sharp_slots_recon_only/) | `662hfy3c` | `running` | present-only | dataset=ssv2; steps=15000; k=12; var=0; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Collapsed rep | c_effective_rank=22.0355; c_cross_video_cosine=0.8161; c_std_mean=0.4085; L_recon_present=0.3113 |

## Pattern Across The Branch

Present-only evidence peaks at run 052 with rank 22.0355. Best present reconstruction among these runs is run 052 with L_recon_present 0.3113.

Verdict distribution: Collapsed rep=1.

## What Changed The Research Direction

Continue monitoring the late diagnostic window. If the same pattern holds, sharpened slots alone are not enough; geometry regularization or a stronger bottleneck design remains necessary.
