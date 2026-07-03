# investigation_012 - Does sharpened slot attention let pure reconstruction train a healthy bottleneck without variance, SIGReg, or covariance regularizers?

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Status:** RUNNING  
**Runs covered:** 052  
**Theme:** sharp-slot bottleneck reconstruction-only test without geometry regularizers

## Question

Does sharpened slot attention let pure reconstruction train a healthy bottleneck without variance, SIGReg, or covariance regularizers?

## Why This Investigation Exists

Investigation 011 produced good present-only geometry with active regularizers. The sharp-slot run tests whether an architectural attention change alone can make c_t decodable, spread, and video-specific.

## W&B-Validated Run Coverage

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 52 | [`ae_sharp_slots_recon_only`](run_052_ae_sharp_slots_recon_only/) | `662hfy3c` | `running` | present-only | dataset=ssv2; steps=15000; k=12; var=0; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Collapsed rep | c_effective_rank=22.0355; c_cross_video_cosine=0.8161; c_std_mean=0.4085; L_recon_present=0.3113 |

## Current Conclusion

Live W&B through step 5600 shows strong reconstruction progress but not healthy representation geometry: c_std_mean is still far below 1, cross-video cosine remains high, and rank has fallen into the low 20s. The run is still active, so the final verdict remains provisional.

## Evidence Standard

The run entries above were reconciled against W&B API/export data on 2026-07-02. For run 052, the newest active run, the evidence was additionally refreshed live with `python run_history.py --run 662hfy3c --report`, which showed the run still `running` through step 5600.
