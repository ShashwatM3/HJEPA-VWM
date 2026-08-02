# Observations - investigation_011

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Cross-Run Synthesis

Present reconstruction can be strong, and SIGReg plus small covariance can raise c_effective_rank above 100 while preserving video specificity. But those are present-side wins; no full-prediction run has inherited them and passed the Phase 1 copy/batch-mean gates.

## Run-by-Run Evidence

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 38 | [`new_recon_loss`](run_038_new_recon_loss/) | `1u69hpfm` | `finished` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=5; recon=0.05/0.05; residual=true; present_only=false; n_c=32; D=512x4 | Healthy rep, no predictor | c_effective_rank=60.3506; c_cross_video_cosine=0.1469; c_std_mean=1.0135; coarse_vs_copy_ratio=1.0575; coarse_vs_batch_mean_ratio=1.1463; L_recon_present=0.3459 |
| 39 | [`original_recon_loss + no-pred`](run_039_original_recon_loss-no-pred/) | `kttd1fib` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0; cov=0; slot=0; sigreg=0; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Collapsed rep | c_effective_rank=10.464; c_cross_video_cosine=0.8638; c_std_mean=0.3472; L_recon_present=0.5149 |
| 40 | [`inv011_fixed_position_decoder`](run_040_inv011_fixed_position_decoder/) | `io74f32b` | `finished` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=5; recon=0.05/0.05; residual=true; present_only=false; n_c=32; D=512x4 | Low-rank rep | c_effective_rank=50.7635; c_cross_video_cosine=0.1652; c_std_mean=1.0053; coarse_vs_copy_ratio=0.9707; coarse_vs_batch_mean_ratio=1.067; L_recon_present=0.3464 |
| 41 | [`inv011_fixed_position_present_recon`](run_041_inv011_fixed_position_present_recon/) | `hcr2qx19` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Low-rank decodable | c_effective_rank=49.5985; c_cross_video_cosine=0.0911; c_std_mean=0.9842; L_recon_present=0.3452 |
| 42 | [`po_geom_sig7p5_cov0`](present-only-geometry-sweep/wave_1/run_042_po_geom_sig7p5_cov0/) | `fq0crddc` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=7.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Low-rank decodable | c_effective_rank=56.8112; c_cross_video_cosine=0.1044; c_std_mean=0.9678; L_recon_present=0.3459 |
| 43 | [`po_geom_sig5_cov0p003`](present-only-geometry-sweep/wave_1/run_043_po_geom_sig5_cov0p003/) | `4f2p1e7b` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.003; slot=0; sigreg=5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=91.0551; c_cross_video_cosine=0.0738; c_std_mean=0.9847; L_recon_present=0.3471 |
| 44 | [`po_geom_sig10_cov0p003`](present-only-geometry-sweep/wave_1/run_044_po_geom_sig10_cov0p003/) | `5xockdbh` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.003; slot=0; sigreg=10; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=105.9453; c_cross_video_cosine=0.0946; c_std_mean=0.9564; L_recon_present=0.3487 |
| 45 | [`po_geom_sig12p5_cov0`](present-only-geometry-sweep/wave_1/run_045_po_geom_sig12p5_cov0/) | `76d6o8d2` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=12.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=83.4092; c_cross_video_cosine=0.0991; c_std_mean=0.9628; L_recon_present=0.3518 |
| 46 | [`po_geom_sig10_cov0`](present-only-geometry-sweep/wave_1/run_046_po_geom_sig10_cov0/) | `9ap28tbw` | `finished` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=10; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=78.5257; c_cross_video_cosine=0.0951; c_std_mean=0.968; L_recon_present=0.349 |
| 47 | [`po_geom_sig5_cov0p01`](present-only-geometry-sweep/wave_2/run_047_po_geom_sig5_cov0p01/) | `az60m6mx` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.01; slot=0; sigreg=5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=150.836; c_cross_video_cosine=0.0767; c_std_mean=0.9833; L_recon_present=0.3438 |
| 48 | [`po_geom_sig7p5_cov0p003`](present-only-geometry-sweep/wave_2/run_048_po_geom_sig7p5_cov0p003/) | `bg7ennr5` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.003; slot=0; sigreg=7.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=89.8678; c_cross_video_cosine=0.0599; c_std_mean=0.9819; L_recon_present=0.3471 |
| 49 | [`po_geom_sig10_cov0p01`](present-only-geometry-sweep/wave_2/run_049_po_geom_sig10_cov0p01/) | `bttexglp` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.01; slot=0; sigreg=10; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=139.0824; c_cross_video_cosine=0.0742; c_std_mean=0.9635; L_recon_present=0.3485 |
| 50 | [`po_geom_sig7p5_cov0p01`](present-only-geometry-sweep/wave_2/run_050_po_geom_sig7p5_cov0p01/) | `h5t89ezx` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.01; slot=0; sigreg=7.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=146.0266; c_cross_video_cosine=0.0735; c_std_mean=0.9696; L_recon_present=0.3455 |
| 51 | [`po_geom_sig12p5_cov0p003`](present-only-geometry-sweep/wave_2/run_051_po_geom_sig12p5_cov0p003/) | `mtrviiab` | `crashed` | present-only | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0.003; slot=0; sigreg=12.5; recon=0.05/0; residual=false; present_only=true; n_c=32; D=512x4 | Strong present representation | c_effective_rank=106.3076; c_cross_video_cosine=0.0868; c_std_mean=0.958; L_recon_present=0.3498 |

## Pattern Across The Branch

Best copy ratio in this branch was run 040 at 0.9707; best batch-mean ratio was run 040 at 1.067. None should be read as a full Phase 1 pass unless both gates pass together.

Present-only evidence peaks at run 047 with rank 150.836. Best present reconstruction among these runs is run 047 with L_recon_present 0.3438.

Verdict distribution: Collapsed rep=1, Healthy rep, no predictor=1, Low-rank decodable=2, Low-rank rep=1, Strong present representation=9.

## What Changed The Research Direction

The next question is whether sharper slot attention or a different bottleneck geometry can retain content without needing external geometry regularizers, then transfer that geometry back into full prediction.

## Original Notes Preserved

This investigation split cleanly into three findings. (1) The cosine reconstruction loss (run 038)
halved the recon floor (~0.585 -> ~0.346) but did not move the copy gate — better readout geometry,
not better forecasting. (2) The fixed-position decoder (runs 040/041) removed the decoder's
unconditional template loophole; run 040 reached the best sustained copy ratio in the project (0.97,
still short of 0.70), and run 041 gave a clean present-only anchor at rank ~50. (3) The present-only
geometry sweep (runs 042-051) proved B+D can carry a very rich present code — rank up to ~150 at
sig5/cov0.01 (run 047), covariance the dominant lever — while keeping videos distinct. The negative
control run 039 (present-only, legacy loss, no geometry) collapsed, confirming those wins come from
the geometry regularizers, not reconstruction alone. The unifying caveat: every strong result here
is PRESENT-side; no full-prediction run has inherited this geometry and passed the Phase 1 gates.
