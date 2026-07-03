# Wave 2 - W&B Run Group

<!-- AUTO-GENERATED-WANDB-KANBAN -->

This wave is a sub-branch of `investigation_007`. It is documented separately because the runs were launched as a coordinated sweep and should be compared against each other before drawing branch-level conclusions.

## Runs

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 25 | [`earnest-dragon-25`](run_025_earnest-dragon-25/) | `2xsd5jwr` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=256; D=256x2 | Smoke / inconclusive | c_effective_rank=9.4052; c_cross_video_cosine=0.6885; c_std_mean=0.5243; coarse_vs_copy_ratio=47.1775; coarse_vs_batch_mean_ratio=9.839; L_recon_present=1.0324 |
| 26 | [`pious-mountain-28`](run_026_pious-mountain-28/) | `7u5zkw6t` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=64; D=256x2 | Smoke / inconclusive | c_effective_rank=8.8219; c_cross_video_cosine=0.7359; c_std_mean=0.4799; coarse_vs_copy_ratio=62.9974; coarse_vs_batch_mean_ratio=12.9475; L_recon_present=1.0299 |
| 27 | [`quiet-firebrand-25`](run_027_quiet-firebrand-25/) | `bbrrydax` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.2/0; residual=false; present_only=false; n_c=64; D=512x2 | Smoke / inconclusive | c_effective_rank=8.8218; c_cross_video_cosine=0.7359; c_std_mean=0.4799; coarse_vs_copy_ratio=62.9978; coarse_vs_batch_mean_ratio=12.9475; L_recon_present=1.0447 |
| 28 | [`classic-yogurt-29`](run_028_classic-yogurt-29/) | `ryuh8cpr` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=false; n_c=128; D=256x2 | Smoke / inconclusive | c_effective_rank=9.6446; c_cross_video_cosine=0.7349; c_std_mean=0.4843; coarse_vs_copy_ratio=66.8338; coarse_vs_batch_mean_ratio=14.055; L_recon_present=1.028 |
| 29 | [`helpful-snow-25`](run_029_helpful-snow-25/) | `tw685b5g` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=1/0; residual=false; present_only=false; n_c=32; D=256x2 | Smoke / inconclusive | c_effective_rank=9.4729; c_cross_video_cosine=0.7239; c_std_mean=0.4952; coarse_vs_copy_ratio=62.0671; coarse_vs_batch_mean_ratio=12.6011; L_recon_present=1.0246 |

## Wave-Level Reading

Best copy ratio in this branch was run 025 at 47.1775; best batch-mean ratio was run 025 at 9.839. None should be read as a full Phase 1 pass unless both gates pass together.

Verdict distribution: Smoke / inconclusive=5.

## Carry-Forward

Read these runs as a coordinated sweep. A single attractive metric is not a winner unless it satisfies the run type's full reading cycle.
