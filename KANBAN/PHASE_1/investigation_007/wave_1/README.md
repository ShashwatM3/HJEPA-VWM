# Wave 1 - W&B Run Group

<!-- AUTO-GENERATED-WANDB-KANBAN -->

This wave is a sub-branch of `investigation_007`. It is documented separately because the runs were launched as a coordinated sweep and should be compared against each other before drawing branch-level conclusions.

## Runs

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 20 | [`jolly-glade-20`](run_020_jolly-glade-20/) | `5x7aoxnn` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.2/0; residual=false; present_only=false; n_c=32; D=256x2 | Low-rank rep | c_effective_rank=12.9849; c_cross_video_cosine=0.2506; c_std_mean=1.0427; coarse_vs_copy_ratio=1.5638; coarse_vs_batch_mean_ratio=0.3383; L_recon_present=0.5915 |
| 21 | [`eager-plant-22`](run_021_eager-plant-22/) | `591mt31k` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=32; D=512x4 | Low-rank rep | c_effective_rank=12.9484; c_cross_video_cosine=0.2468; c_std_mean=1.037; coarse_vs_copy_ratio=1.5278; coarse_vs_batch_mean_ratio=0.3246; L_recon_present=0.5845 |
| 22 | [`gallant-dew-22`](run_022_gallant-dew-22/) | `708jrel8` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=32; D=512x2 | Low-rank rep | c_effective_rank=12.5094; c_cross_video_cosine=0.3045; c_std_mean=0.9893; coarse_vs_copy_ratio=1.7402; coarse_vs_batch_mean_ratio=0.3723; L_recon_present=0.5895 |
| 23 | [`toasty-donkey-21`](run_023_toasty-donkey-21/) | `a2trqp9c` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.1/0; residual=false; present_only=false; n_c=32; D=256x2 | Low-rank rep | c_effective_rank=12.9731; c_cross_video_cosine=0.2487; c_std_mean=1.0345; coarse_vs_copy_ratio=1.5857; coarse_vs_batch_mean_ratio=0.3422; L_recon_present=0.5959 |
| 24 | [`light-universe-24`](run_024_light-universe-24/) | `rju7xsh2` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.5/0; residual=false; present_only=false; n_c=32; D=256x2 | Low-rank rep | c_effective_rank=12.6552; c_cross_video_cosine=0.2824; c_std_mean=1.0185; coarse_vs_copy_ratio=1.6644; coarse_vs_batch_mean_ratio=0.3594; L_recon_present=0.5855 |

## Wave-Level Reading

Best copy ratio in this branch was run 021 at 1.5278; best batch-mean ratio was run 021 at 0.3246. None should be read as a full Phase 1 pass unless both gates pass together.

Verdict distribution: Low-rank rep=5.

## Carry-Forward

Read these runs as a coordinated sweep. A single attractive metric is not a winner unless it satisfies the run type's full reading cycle.
