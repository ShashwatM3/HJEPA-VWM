# Observations - fixed-position-present-recon

**Status:** READ COMPLETE.  
**W&B:** `inv011_fixed_position_present_recon` (`hcr2qx19`), group
`inv011_fixed_position_present_recon`.  
**W&B state:** `crashed`, but no training-health failure was observed.  
**Commit:** `693c881` (`Add fixed-position present recon run docs`).  
**Command:** see [GUIDE.md](GUIDE.md).  
**Description:** see [DESCRIPTION.md](DESCRIPTION.md).  
**Gradient/readout notes:** see [GRADIENTS_AND_READOUTS.md](GRADIENTS_AND_READOUTS.md).  
**Full analysis:** [ANALYSIS_inv011_fixed_position_present_recon.md](ANALYSIS_inv011_fixed_position_present_recon.md).

## Run Metadata

| Field | Value |
|---|---|
| Run name | `inv011_fixed_position_present_recon` |
| Run id | `hcr2qx19` |
| W&B URL | https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/hcr2qx19 |
| Commit | `693c881` (`Add fixed-position present recon run docs`) |
| Dataset | `ssv2` |
| Steps requested | 15000 |
| Last logged training step | 14400 |
| Last diagnostic step | 14000 |
| State | `crashed` in W&B; treated as externally/manual stopped, not invalid |

## Config Checklist

Confirm from W&B config:

| Field | Expected | Actual |
|---|---:|---:|
| `present_recon_only` | true | true |
| `prediction_active` | false | false |
| `recon_loss_mode` | `cosine` | `cosine` |
| `lambda_recon` | 0.05 | 0.05 |
| `lambda_recon_pred` | 0.0 | 0.0 |
| `lambda_sigreg` | 5.0 | 5.0 |
| `sigreg_warmup_steps` | 2000 | 2000 |
| `lambda_var` | 0.5 | 0.5 |
| `decoder_dim` | 512 | 512 |
| `decoder_blocks` | 4 | 4 |
| `n_c` | 32 | 32 |
| `horizon_k` | 12 | 12 |

## Final Metrics

Final diagnostic step: 14000. Plateau means mean of diagnostic rows 13000, 13500, and 14000.

| Metric | Final | Last-3 plateau | Notes |
|---|---:|---:|---|
| `L_recon_present` | 0.3452 | 0.3454 | Strong present reconstruction. |
| `c_effective_rank` | 49.60 | 49.60 | Below target >60. |
| `c_cross_video_cosine` | 0.0911 | 0.0917 | Excellent video-specificity. |
| `c_std_mean` | 0.9842 | 0.9838 | Healthy spread. |
| `c_dead_dim_frac` | 0.0 | 0.0 | No dead-dim spike. |
| `c_slot_diversity_rank` | 5.08 | 5.07 | Low; slots are redundant. |
| `c_attn_entropy` | 0.7747 | 0.7746 | Attention sharpened. |
| `c_attn_entropy_min` | ~0 | ~0 | At least one head/slot is saturated. |
| `L_sigreg` | 0.0027 at step 14400 | 0.0025 train last-3 | Active and minimized. |
| `L_var` | 0.0151 at step 14400 | 0.0153 train last-3 | Lightly active. |
| `grad_norm` | 0.1658 at step 14400 | 0.1722 train last-3 | Stable. |
| `grad_skipped` | 0 | 0 | No skip spiral. |
| `grad_has_nan` | 0 | 0 | No NaN signal. |
| `instability_warn` | 0 | 0 | No warning. |
| `agc_B_clipped` | 0 | 0 | No bottleneck clipping. |
| `agc_D_clipped` | 0 | 0 | No decoder clipping. |

## Reading Cycle

| Q | Question | Result | Evidence |
|---|---|---|---|
| Q1 | Training alive? | PASS | Correct present-only flags; no skips, NaNs, or instability. |
| Q2 | `c_t` alive/video-specific? | PASS | std 0.984, dead dim 0, cosine 0.091. |
| Q3 | Rich latent? | FAIL | rank 49.60 < 60; slot rank 5.08/32. |
| Q5-prime | Present reconstruction bottleneck test? | PARTIAL | recon strong, geometry not strong enough. |
| Q8 | Verdict | LOW-RANK DECODABLE | Decodable and video-specific, but below rank gate. |

## Interpretation Notes

This is a useful partial win. The old present-only run collapsed (`rank ~10`, `std ~0.35`,
`cross-video cosine ~0.86`). This run fixes that: `c_t` is spread, video-specific, stable, and
strongly decodable.

The remaining failure is not `F_c`; `F_c` was inactive. It is a present-side bottleneck geometry
failure: `B` compresses content into about 50 effective directions and about 5 independent slots.
The fixed-position decoder can reconstruct from that code, but it does not force the bottleneck to
use enough independent dimensions or slots.

## Updated Next Step Recommendation

The cleanest diagnostic-only next step would be shuffled/zero-c reconstruction readouts, but the
operator has chosen to skip that safety diagnostic and move directly to active experiment changes.
Given the current run and prior sweep history, the recommended order is:

1. Run a structured present-only geometry sweep:

```text
lambda_sigreg = {5.0, 7.5, 10.0, 12.5}
lambda_cov    = {0.0, 0.003, 0.01}
lambda_recon  = 0.05
lambda_var    = 0.5
present_recon_only = true
```

Success means `c_effective_rank >= 60`, `c_cross_video_cosine < 0.30` preferred, `c_std_mean` still
near `1.0`, `L_recon_present <= 0.36`, and no further slot-rank regression. This sweep is not a
repeat of the old low-value SIGReg sweep: prior runs showed `0.3` and `1.0` were inert, `3.0`
started moving rank, and `5.0-10.0` is the active band. The new axis is small `lambda_cov`, because
this run logs a nontrivial late `L_cov (~3.3)` but does not optimize it.

2. If global rank clears the gate but `c_slot_diversity_rank` remains near `5`, implement an
anchored-slot bottleneck: fixed/semi-fixed spatiotemporal anchors, local first cross-attention, then
a lightweight slot mixer. This targets the architectural failure directly: global learned slots can
all attend to all detailed tokens, so 32 slots behave like roughly five independent carriers.

3. If dense present reconstruction still admits low-rank solutions, add masked feature
reconstruction: mask/drop detailed tokens before `B`, reconstruct the original detailed tokens from
`c_t`, and start with `bottleneck_token_mask_ratio=0.50`. This makes the reconstruction task harder
and pressures the bottleneck to integrate distributed spatiotemporal content.

Full details and external-source links are in
[ANALYSIS_inv011_fixed_position_present_recon.md](ANALYSIS_inv011_fixed_position_present_recon.md).
