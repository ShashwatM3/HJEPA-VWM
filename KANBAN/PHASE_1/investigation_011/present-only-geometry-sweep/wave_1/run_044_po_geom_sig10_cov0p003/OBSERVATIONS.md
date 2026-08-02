# Observations - run 044 `po_geom_sig10_cov0p003`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `5xockdbh`  
**State:** `finished`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Strong present representation**  
**Verdict note:** The present branch is decodable, spread, video-specific, and high-rank.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=finished; step=14950; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.1856 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9564; dead_dim=0; cross_video_cosine=0.0946 |
| Q3 | Rich latent? | PASS | c_effective_rank=105.9453; slot_rank=12.2176; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.6372 (0.9858 @ 0 -> 0.3487 @ 14500); last=0.3487; train L_recon=0.3753 @ 14950 |
| Q5 | Geometry and content aligned? | PASS | reconstruction improved while c_t stayed spread, video-specific, and high-rank |
| Q6 | Verdict | Strong present representation | The present branch is decodable, spread, video-specific, and high-rank. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.2067 @ 0 | 0.0443 @ 14950 | -0.1624 | 0.0417 / 0.0543 / 0.2067 | 0.0417 / 0.0438 / 0.0459 (n=100) |
| `L_flow` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon` | 0.9829 @ 0 | 0.3753 @ 14950 | -0.6076 | 0.3688 / 0.4022 / 0.9829 | 0.3688 / 0.3779 / 0.3859 (n=100) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon_present` | 0.9858 @ 0 | 0.3487 @ 14500 | -0.6372 | 0.3487 / 0.3856 / 0.9858 | 0.3487 / 0.3495 / 0.3509 (n=10) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `present_recon_only` | 1 @ 0 | 1 @ 14950 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=100) |
| `prediction_active` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_var` | 0.4003 @ 0 | 0.0239 @ 14950 | -0.3765 | 0.0212 / 0.0293 / 0.4003 | 0.0213 / 0.0233 / 0.0263 (n=100) |
| `L_cov` | 2.184 @ 0 | 0.8687 @ 14950 | -1.3154 | 0.8158 / 2.2058 / 10.8576 | 0.8158 / 0.8857 / 0.9657 (n=100) |
| `L_sigreg` | 0.0417 @ 0 | 0.0011 @ 14950 | -0.0406 | 8.065e-04 / 0.002 / 0.0417 | 8.944e-04 / 0.0011 / 0.0013 (n=100) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 0.9564 @ 14500 | 0.4612 | 0.4952 / 0.9325 / 0.9638 | 0.9534 / 0.9561 / 0.9574 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0946 @ 14500 | -0.6293 | 0.0683 / 0.1233 / 0.7239 | 0.0927 / 0.0952 / 0.1001 (n=10) |
| `c_effective_rank` | 9.4729 @ 0 | 105.9453 @ 14500 | 96.4724 | 9.4729 / 80.8777 / 105.9812 | 103.1647 / 105.3087 / 105.9812 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 12.2176 @ 14500 | -4.4935 | 6.1918 / 11.9586 / 19.5427 | 11.9724 / 12.149 / 12.2206 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.6367 @ 14500 | -0.3632 | 0.617 / 0.7086 / 1 | 0.6341 / 0.6375 / 0.6456 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 8.239e-09 @ 14500 | -0.9998 | 4.081e-09 / 0.1803 / 0.9999 | 4.184e-09 / 9.919e-09 / 2.257e-08 (n=10) |
| `grad_norm` | 1.4368 @ 0 | 0.1856 @ 14950 | -1.2512 | 0.1585 / 0.3114 / 1.5522 | 0.1585 / 0.18 / 0.2074 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=105.9453 @ 14500; `c_cross_video_cosine`=0.0946 @ 14500; `c_std_mean`=0.9564 @ 14500; `L_recon_present`=0.3487 @ 14500

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.3487, and the key geometry metrics are rank 105.9453, cross-video cosine 0.0946, and std 0.9564. The reading-cycle verdict is **Strong present representation** because The present branch is decodable, spread, video-specific, and high-rank. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (grid point sig=10.0, cov=0.003):** reached `c_effective_rank` ~105.9 with
cross-video cosine ~0.06-0.10, std ~0.96-0.98, and `L_recon_present` ~0.345 — a decodable, spread,
video-specific present code. Read against the rest of the sweep, the dominant lever on rank is the
COVARIANCE penalty: the cov=0.01 rows reach 139-151, cov=0.003 rows 90-106, and cov=0 rows only
57-83; SIGReg alone (cov=0) tops out ~83, and raising SIGReg helps only modestly on top of cov.
This is a strong PRESENT representation, but F_c is off, so it is not a prediction result.
