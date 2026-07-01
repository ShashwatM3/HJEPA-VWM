# Observations - fixed-position-present-recon

**Status:** NOT LAUNCHED.  
**W&B:** TBD.  
**Command:** see [GUIDE.md](GUIDE.md).  
**Description:** see [DESCRIPTION.md](DESCRIPTION.md).  
**Gradient/readout notes:** see [GRADIENTS_AND_READOUTS.md](GRADIENTS_AND_READOUTS.md).

## Run Metadata

| Field | Value |
|---|---|
| Run name | `inv011_fixed_position_present_recon` |
| Run id | TBD |
| W&B URL | TBD |
| Commit | TBD |
| Dataset | `ssv2` |
| Steps requested | 15000 |
| Last logged training step | TBD |
| Last diagnostic step | TBD |
| State | TBD |

## Config Checklist

Confirm from W&B config:

| Field | Expected | Actual |
|---|---:|---:|
| `present_recon_only` | true | TBD |
| `prediction_active` | false | TBD |
| `recon_loss_mode` | `cosine` | TBD |
| `lambda_recon` | 0.05 | TBD |
| `lambda_recon_pred` | 0.0 | TBD |
| `lambda_sigreg` | 5.0 | TBD |
| `sigreg_warmup_steps` | 2000 | TBD |
| `lambda_var` | 0.5 | TBD |
| `decoder_dim` | 512 | TBD |
| `decoder_blocks` | 4 | TBD |
| `n_c` | 32 | TBD |
| `horizon_k` | 12 | TBD |

## Final Metrics

Final diagnostic step: TBD. Plateau means mean of the last three diagnostic rows.

| Metric | Final | Last-3 plateau | Notes |
|---|---:|---:|---|
| `L_recon_present` | TBD | TBD | Main content readout. |
| `c_effective_rank` | TBD | TBD | Target >60. |
| `c_cross_video_cosine` | TBD | TBD | Target <0.5. |
| `c_std_mean` | TBD | TBD | Target roughly 0.8-1.2. |
| `c_dead_dim_frac` | TBD | TBD | Should stay near 0. |
| `c_slot_diversity_rank` | TBD | TBD | Secondary slot redundancy readout. |
| `c_attn_entropy` | TBD | TBD | Secondary bottleneck attention readout. |
| `c_attn_entropy_min` | TBD | TBD | Watch for saturated heads/slots. |
| `L_sigreg` | TBD | TBD | Geometry regularizer. |
| `L_var` | TBD | TBD | Variance-floor pressure. |
| `grad_norm` | TBD | TBD | Stability. |
| `grad_skipped` | TBD | TBD | Must be 0. |
| `grad_has_nan` | TBD | TBD | Must be 0. |
| `instability_warn` | TBD | TBD | Should be 0. |
| `agc_B_clipped` | TBD | TBD | Bottleneck clipping. |
| `agc_D_clipped` | TBD | TBD | Decoder clipping. |

## Reading Cycle

| Q | Question | Result | Evidence |
|---|---|---|---|
| Q1 | Training alive? | TBD | `grad_skipped`, `grad_has_nan`, `grad_norm`, `instability_warn` |
| Q2 | `c_t` alive/video-specific? | TBD | `c_std_mean`, `c_dead_dim_frac`, `c_cross_video_cosine` |
| Q3 | Rich latent? | TBD | `c_effective_rank`, `c_std_mean` |
| Q5-prime | Present reconstruction bottleneck test? | TBD | `L_recon_present`, rank, cosine, std |
| Q8 | Verdict | TBD | `Invalid`, `Collapsed rep`, `Low-rank decodable`, `Pretty geometry, weak content`, or `Strong present representation` |

## Interpretation Notes

TBD after run.

Use these patterns:

- **Strong present representation:** `L_recon_present` falls materially, rank clears 60, std/cosine
  are healthy, and stability is clean.
- **Low-rank decodable:** reconstruction improves but rank stays below target.
- **Pretty geometry, weak content:** rank/std look healthy but reconstruction stalls.
- **Collapsed rep:** std/cosine/dead-dim metrics fail.
- **Invalid:** skipped-step spiral, NaN, crash, or wrong config.

## Next Step Recommendation

TBD after run.
