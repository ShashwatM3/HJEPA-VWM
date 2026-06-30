# Observations - fixed-position-decoder

**Status:** awaiting run.
**W&B:** TBD.
**Commit:** TBD, expected `0a9ff46` or later.
**Command:** see [GUIDE.md](GUIDE.md).

## Final Metrics

| Metric | Value | Notes |
|---|---:|---|
| `coarse_vs_copy_ratio` | TBD | Main prediction gate; compare to Run A `new_recon_loss` around 1.06 |
| `coarse_vs_batch_mean_ratio` | TBD | Secondary prediction gate |
| `coarse_model_loss` | TBD | Model flow loss |
| `coarse_copy_loss` | TBD | Zero-residual/copy baseline |
| `c_effective_rank` | TBD | Representation health; target remains >60 |
| `c_plus_effective_rank` | TBD | EMA target health |
| `c_cross_video_cosine` | TBD | Collapse watch; should stay well below 0.5 |
| `c_std_mean` | TBD | Variance health |
| `L_recon_present` | TBD | May be higher than learned-query decoder; not alone a failure |
| `L_recon_cplus` | TBD | True future latent reconstruction readout |
| `L_recon_chat` | TBD | Predicted future latent reconstruction readout |
| `L_recon_chat - L_recon_cplus` | TBD | Key diagnostic gap |
| `grad_skipped` | TBD | Stability |

## Interpretation Notes

- Compare first to `new_recon_loss` (`1u69hpfm`): same recipe, learned-query decoder.
- A higher reconstruction loss can be acceptable if it exposes prediction quality more honestly.
- A flat `L_recon_chat - L_recon_cplus` gap would mean another unconditional or shared decoder path
  still masks bad predictions.
- A lower `coarse_vs_copy_ratio` with stable rank/cosine is the strongest positive result.
