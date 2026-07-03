# Wave 1 - W&B Run Group

<!-- AUTO-GENERATED-WANDB-KANBAN -->

This wave is a sub-branch of `investigation_011`. It is documented separately because the runs were launched as a coordinated sweep and should be compared against each other before drawing branch-level conclusions.

## Runs

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 42 | [`po_geom_sig7p5_cov0`](run_042_po_geom_sig7p5_cov0/) | `fq0crddc` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=7.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Low-rank decodable | c_effective_rank=56.8112; c_cross_video_cosine=0.1044; c_std_mean=0.9678; L_recon_present=0.3459 |
| 43 | [`po_geom_sig5_cov0p003`](run_043_po_geom_sig5_cov0p003/) | `4f2p1e7b` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.003; slot=0; sigreg=5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=91.0551; c_cross_video_cosine=0.0738; c_std_mean=0.9847; L_recon_present=0.3471 |
| 44 | [`po_geom_sig10_cov0p003`](run_044_po_geom_sig10_cov0p003/) | `5xockdbh` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.003; slot=0; sigreg=10; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=105.9453; c_cross_video_cosine=0.0946; c_std_mean=0.9564; L_recon_present=0.3487 |
| 45 | [`po_geom_sig12p5_cov0`](run_045_po_geom_sig12p5_cov0/) | `76d6o8d2` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=12.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=83.4092; c_cross_video_cosine=0.0991; c_std_mean=0.9628; L_recon_present=0.3518 |
| 46 | [`po_geom_sig10_cov0`](run_046_po_geom_sig10_cov0/) | `9ap28tbw` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=10; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=78.5257; c_cross_video_cosine=0.0951; c_std_mean=0.968; L_recon_present=0.349 |

## Wave-Level Reading

Present-only evidence peaks at run 044 with rank 105.9453. Best present reconstruction among these runs is run 042 with L_recon_present 0.3459.

Verdict distribution: Low-rank decodable=1, Strong present representation=4.

## Carry-Forward

Read these runs as a coordinated sweep. A single attractive metric is not a winner unless it satisfies the run type's full reading cycle.
