# investigation_011 - Can reconstruction geometry produce a strong, decodable present bottleneck, and can that evidence be separated from prediction failure?

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Status:** CLOSED  
**Runs covered:** 038, 039, 040, 041, 042, 043, 044, 045, 046, 047, 048, 049, 050, 051  
**Theme:** cosine reconstruction, fixed-position decoder, present-only geometry sweeps

## Question

Can reconstruction geometry produce a strong, decodable present bottleneck, and can that evidence be separated from prediction failure?

## Why This Investigation Exists

The residual branch showed a healthy c_t was still not enough for prediction. This branch isolated the decoder/readout side, changed the reconstruction geometry, and ran present-only sweeps to understand whether B can carry present information at all.

## W&B-Validated Run Coverage

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

## Current Conclusion

Present reconstruction can be strong, and SIGReg plus small covariance can raise c_effective_rank above 100 while preserving video specificity. But those are present-side wins; no full-prediction run has inherited them and passed the Phase 1 copy/batch-mean gates.

## Evidence Standard

The run entries above were reconciled against W&B API/export data on 2026-07-02. For run 052, the newest active run, the evidence was additionally refreshed live with `python run_history.py --run 662hfy3c --report`, which showed the run still `running` through step 5600.

## Original Notes Preserved

# Investigation 011 — Does cosine reconstruction help, and can present-only reconstruction train a rich `c`?

**Status:** OPEN — code switches added; runs not launched.
**Opened:** 2026-06-30
**Closed:** —

## Question

investigation_010 produced the healthiest Phase 1 representation so far, but prediction still tied
the zero-residual/copy baseline. The reconstruction channel may have been undercut by the old
`MSE / Var(e)` objective, which let the decoder game feature norms and made the readout mostly
blind to prediction quality.

This investigation tests two separate reconstruction questions:

- **Run A — cosine reconstruction in the full residual recipe.** Keep the inv010 recipe
  (`λ_sigreg=5`, `λ_var=0.5`, `λ_recon=0.05`, `λ_recon_pred=0.05`, residual prediction,
  decoder `512x4`, `n_c=32`, `k=12`) and switch only the reconstruction formula to
  `--recon-loss-mode cosine`.
- **Run B — present-only reconstruction bottleneck test.** Disable prediction with
  `--present-recon-only` and train only `D(B(e_t)) -> e_t` with the original
  `--recon-loss-mode relative_mse` objective; no `F_c` loss, no residual target, no future
  `c_hat`, no `D(c_hat) -> e_{t+k}` branch.

## Runs

| Run | Config | Role | Main readout |
|---|---|---|---|
| A | full residual + SIGReg + `--recon-loss-mode cosine` | Does the new recon geometry help the current best recipe? | `coarse_vs_copy_ratio`, rank, `L_recon_chat - L_recon_cplus` |
| B | `--present-recon-only`, `--recon-loss-mode relative_mse`, `λ_var=0`, `λ_sigreg=0`, `λ_recon=0.05` | Can original-loss reconstruction alone make `c_t` rich enough to decode `e_t`? | `L_recon_present`, `c_effective_rank`, `c_cross_video_cosine` |

## Interpretation

- **Run A win:** copy ratio drops below inv010 while rank/cosine/stability stay healthy.
- **Run A neutral:** rank and reconstruction change but copy ratio remains near 1, so prediction is
  still the bottleneck.
- **Run B win:** `L_recon_present` falls materially while `c_effective_rank` rises without collapse,
  proving the bottleneck can carry detailed present information under the new objective.
- **Run B fail:** reconstruction stalls and rank stays low, meaning the present decoder path still
  cannot make `c` information-rich by itself.

## Code switches

- `--recon-loss-mode cosine|relative_mse`
- `--present-recon-only`
- `cfg.train.present_recon_only`
- `losses.reconstruction_loss(..., mode=...)`

## Execution

Use [`GUIDE.md`](GUIDE.md).
