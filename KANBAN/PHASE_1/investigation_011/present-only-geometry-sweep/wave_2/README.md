# Wave 2 - W&B Run Group

<!-- AUTO-GENERATED-WANDB-KANBAN -->

This wave is a sub-branch of `investigation_011`. It is documented separately because the runs were launched as a coordinated sweep and should be compared against each other before drawing branch-level conclusions.

## Runs

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 47 | [`po_geom_sig5_cov0p01`](run_047_po_geom_sig5_cov0p01/) | `az60m6mx` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.01; slot=0; sigreg=5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=150.836; c_cross_video_cosine=0.0767; c_std_mean=0.9833; L_recon_present=0.3438 |
| 48 | [`po_geom_sig7p5_cov0p003`](run_048_po_geom_sig7p5_cov0p003/) | `bg7ennr5` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.003; slot=0; sigreg=7.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=89.8678; c_cross_video_cosine=0.0599; c_std_mean=0.9819; L_recon_present=0.3471 |
| 49 | [`po_geom_sig10_cov0p01`](run_049_po_geom_sig10_cov0p01/) | `bttexglp` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.01; slot=0; sigreg=10; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=139.0824; c_cross_video_cosine=0.0742; c_std_mean=0.9635; L_recon_present=0.3485 |
| 50 | [`po_geom_sig7p5_cov0p01`](run_050_po_geom_sig7p5_cov0p01/) | `h5t89ezx` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.01; slot=0; sigreg=7.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=146.0266; c_cross_video_cosine=0.0735; c_std_mean=0.9696; L_recon_present=0.3455 |
| 51 | [`po_geom_sig12p5_cov0p003`](run_051_po_geom_sig12p5_cov0p003/) | `mtrviiab` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.003; slot=0; sigreg=12.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=106.3076; c_cross_video_cosine=0.0868; c_std_mean=0.958; L_recon_present=0.3498 |

## Wave-Level Reading

Present-only evidence peaks at run 047 with rank 150.836. Best present reconstruction among these runs is run 047 with L_recon_present 0.3438.

Verdict distribution: Strong present representation=5.

## Carry-Forward

Read these runs as a coordinated sweep. A single attractive metric is not a winner unless it satisfies the run type's full reading cycle.
