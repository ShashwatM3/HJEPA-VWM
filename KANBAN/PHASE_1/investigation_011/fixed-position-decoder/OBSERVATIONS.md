# Observations - fixed-position-decoder

**Status:** FINISHED.
**W&B:** `inv011_fixed_position_decoder` (`io74f32b`), group `inv011_fixed_position_decoder`.
**Commit:** expected `0a9ff46` or later; run used the fixed-position decoder default.
**Command:** see [GUIDE.md](GUIDE.md).
**Full analysis:** [ANALYSIS_inv011_fixed_position_decoder.md](ANALYSIS_inv011_fixed_position_decoder.md).

## Final Metrics

| Metric | Value | Notes |
|---|---:|---|
| `coarse_vs_copy_ratio` | 0.9707 final / 0.9956 plateau | Best recent movement, but gate is <=0.70 |
| `coarse_vs_batch_mean_ratio` | 1.0670 final / 1.0947 plateau | Fails batch-mean gate <=0.50 |
| `coarse_model_loss` | 1.4615 final / 1.4980 plateau | Only slightly below copy at final diag |
| `coarse_copy_loss` | 1.5057 final / 1.5046 plateau | Present and future `c` are separated; not static `c` |
| `c_effective_rank` | 50.76 final / 50.74 plateau | Below target >60 |
| `c_plus_effective_rank` | 50.45 final / 50.40 plateau | EMA target aligned with online rank |
| `c_cross_video_cosine` | 0.1652 final / 0.1655 plateau | Healthy video-specificity |
| `c_std_mean` | 1.0053 final / 1.0051 plateau | Healthy variance |
| `L_recon_present` | 0.3464 final / 0.3465 plateau | Present recon learned under cosine objective |
| `L_recon_cplus` | 0.3441 final / 0.3442 plateau | True-future recon benchmark |
| `L_recon_chat` | 0.3530 final / 0.3541 plateau | Predicted future recon remains close to true future |
| `L_recon_chat - L_recon_cplus` | 0.0088 final / 0.0099 plateau | Gap remains small; recon still blind |
| `grad_skipped` | 0 | Stable; no skip spiral |

## Interpretation Notes

- Compared to `new_recon_loss` (`1u69hpfm`), this run improves copy ratio
  (`1.0652` plateau -> `0.9956`) but loses the rank gate (`60.31` -> `50.74`).
- The run is not a static-`c` trap: `coarse_copy_loss` rises to ~1.5.
- The predictor still fails: copy ratio is near 1 and batch-mean ratio is above 1.
- Reconstruction remains blind: `L_recon_chat - L_recon_cplus` stays around 0.01.
- Verdict: low-rank representation with partial prediction movement; not a passing Phase 1 run.
