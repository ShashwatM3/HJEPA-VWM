# Observations - run 041 `inv011_fixed_position_present_recon`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `hcr2qx19`  
**State:** `crashed`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Low-rank decodable**  
**Verdict note:** The decoder can reconstruct from c_t, but the bottleneck uses too few effective directions or slots.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=crashed; step=14400; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.1658 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9842; dead_dim=0; cross_video_cosine=0.0911 |
| Q3 | Rich latent? | PARTIAL | c_effective_rank=49.5985; slot_rank=5.0765; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.6406 (0.9858 @ 0 -> 0.3452 @ 14000); last=0.3452; train L_recon=0.3791 @ 14400 |
| Q5 | Geometry and content aligned? | PARTIAL | reconstruction improved through a low-rank or slot-redundant code |
| Q6 | Verdict | Low-rank decodable | The decoder can reconstruct from c_t, but the bottleneck uses too few effective directions or slots. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.2002 @ 0 | 0.0399 @ 14400 | -0.1603 | 0.0253 / 0.0466 / 0.2002 | 0.0365 / 0.0392 / 0.0432 (n=89) |
| `L_flow` | 0 @ 0 | 0 @ 14400 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=89) |
| `L_recon` | 0.9829 @ 0 | 0.3791 @ 14400 | -0.6037 | 0.3646 / 0.4046 / 0.9829 | 0.3646 / 0.3742 / 0.382 (n=89) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14400 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=89) |
| `L_recon_present` | 0.9858 @ 0 | 0.3452 @ 14000 | -0.6406 | 0.3452 / 0.3876 / 0.9858 | 0.3452 / 0.3459 / 0.3471 (n=9) |
| `recon_scale` | 0 @ 0 | 1 @ 14400 | 1 | 0 / 0.9291 / 1 | 1 / 1 / 1 (n=89) |
| `present_recon_only` | 1 @ 0 | 1 @ 14400 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=89) |
| `prediction_active` | 0 @ 0 | 0 @ 14400 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=89) |
| `L_var` | 0.4003 @ 0 | 0.0151 @ 14400 | -0.3852 | 0.0137 / 0.0216 / 0.4003 | 0.0137 / 0.0155 / 0.0177 (n=89) |
| `L_cov` | 2.184 @ 0 | 3.3364 @ 14400 | 1.1524 | 2.184 / 5.9522 / 15.3234 | 3.195 / 3.4013 / 3.7415 (n=89) |
| `L_sigreg` | 0.0417 @ 0 | 0.0027 @ 14400 | -0.0391 | 0.0021 / 0.0042 / 0.0417 | 0.0021 / 0.0025 / 0.0033 (n=89) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14400 | 1 | 0 / 0.9291 / 1 | 1 / 1 / 1 (n=89) |
| `c_std_mean` | 0.4952 @ 0 | 0.9842 @ 14000 | 0.4889 | 0.4952 / 0.9539 / 0.9912 | 0.9787 / 0.9826 / 0.9845 (n=9) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0911 @ 14000 | -0.6328 | 0.0773 / 0.1224 / 0.7239 | 0.0901 / 0.0936 / 0.1002 (n=9) |
| `c_effective_rank` | 9.4728 @ 0 | 49.5985 @ 14000 | 40.1257 | 9.4728 / 32.7867 / 49.6255 | 46.5562 / 48.6746 / 49.6255 (n=9) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 5.0765 @ 14000 | -11.6346 | 1.6055 / 6.4897 / 20.7393 | 4.8428 / 5.007 / 5.0765 (n=9) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.7747 @ 14000 | -0.2253 | 0.7721 / 0.8597 / 1 | 0.7721 / 0.7738 / 0.7757 (n=9) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 4.078e-09 @ 14000 | -0.9998 | 4.078e-09 / 0.2646 / 0.9999 | 4.078e-09 / 4.078e-09 / 4.078e-09 (n=9) |
| `grad_norm` | 1.5798 @ 0 | 0.1658 @ 14400 | -1.414 | 0.1456 / 0.28 / 1.7193 | 0.1456 / 0.1716 / 0.2188 (n=89) |
| `grad_skipped` | 0 @ 0 | 0 @ 14400 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=89) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=9) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0049 @ 14400 | 0.0042 | 6.667e-04 / 0.519 / 1 | 0.0049 / 0.12 / 0.302 (n=89) |

## Last Key Metrics

`c_effective_rank`=49.5985 @ 14000; `c_cross_video_cosine`=0.0911 @ 14000; `c_std_mean`=0.9842 @ 14000; `L_recon_present`=0.3452 @ 14000

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3452, and the key geometry metrics are rank 49.5985, cross-video cosine 0.0911, and std 0.9842. The reading-cycle verdict is **Low-rank decodable** because The decoder can reconstruct from c_t, but the bottleneck uses too few effective directions or slots. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

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
