# Observations - run 046 `po_geom_sig10_cov0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `9ap28tbw`  
**State:** `finished`  
**Mode:** present-only run; F_c is inactive and copy/batch-mean gates do not apply  
**Verdict:** **Strong present representation**  
**Verdict note:** The present branch is decodable, spread, video-specific, and high-rank.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Present-only training alive? | PASS | state=finished; step=14950; present_recon_only=1; prediction_active=0; L_flow=0; L_recon_pred=0; grad_skipped max=0; grad_has_nan max=0; grad_norm last=0.1946 |
| Q2 | c_t alive / video-specific? | PASS | std=0.968; dead_dim=0; cross_video_cosine=0.0951 |
| Q3 | Rich latent? | PASS | c_effective_rank=78.5257; slot_rank=8.7357; centered_slot_rank=n/a |
| Q4 | Present reconstruction learning? | PASS | L_recon_present trend=down -0.6368 (0.9858 @ 0 -> 0.349 @ 14500); last=0.349; train L_recon=0.3754 @ 14950 |
| Q5 | Geometry and content aligned? | PASS | reconstruction improved while c_t stayed spread, video-specific, and high-rank |
| Q6 | Verdict | Strong present representation | The present branch is decodable, spread, video-specific, and high-rank. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 0.2002 @ 0 | 0.0437 @ 14950 | -0.1565 | 0.0283 / 0.0538 / 0.2002 | 0.0404 / 0.0446 / 0.0475 (n=100) |
| `L_flow` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon` | 0.9829 @ 0 | 0.3754 @ 14950 | -0.6075 | 0.3683 / 0.4035 / 0.9829 | 0.3683 / 0.3782 / 0.3864 (n=100) |
| `L_recon_pred` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_recon_present` | 0.9858 @ 0 | 0.349 @ 14500 | -0.6368 | 0.349 / 0.3865 / 0.9858 | 0.349 / 0.3496 / 0.3508 (n=10) |
| `recon_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `present_recon_only` | 1 @ 0 | 1 @ 14950 | 0 | 1 / 1 / 1 | 1 / 1 / 1 (n=100) |
| `prediction_active` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `L_var` | 0.4003 @ 0 | 0.0234 @ 14950 | -0.3769 | 0.0207 / 0.0285 / 0.4003 | 0.0207 / 0.0228 / 0.0254 (n=100) |
| `L_cov` | 2.184 @ 0 | 1.6242 @ 14950 | -0.5599 | 1.5833 / 3.7349 / 14.6547 | 1.5833 / 1.6824 / 1.8375 (n=100) |
| `L_sigreg` | 0.0417 @ 0 | 0.0013 @ 14950 | -0.0404 | 0.001 / 0.0027 / 0.0417 | 0.001 / 0.0014 / 0.0017 (n=100) |
| `sigreg_scale` | 0 @ 0 | 1 @ 14950 | 1 | 0 / 0.9317 / 1 | 1 / 1 / 1 (n=100) |
| `c_std_mean` | 0.4952 @ 0 | 0.968 @ 14500 | 0.4728 | 0.4952 / 0.9339 / 0.968 | 0.9591 / 0.9649 / 0.968 (n=10) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.0951 @ 14500 | -0.6287 | 0.0951 / 0.139 / 0.7239 | 0.0951 / 0.1011 / 0.1121 (n=10) |
| `c_effective_rank` | 9.4728 @ 0 | 78.5257 @ 14500 | 69.0529 | 9.4728 / 55.586 / 78.5257 | 75.9412 / 77.8252 / 78.5257 (n=10) |
| `c_slot_diversity_rank` | 16.7111 @ 0 | 8.7357 @ 14500 | -7.9753 | 4.4084 / 9.2103 / 20.4864 | 8.6235 / 8.7261 / 8.7837 (n=10) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.6114 @ 14500 | -0.3885 | 0.6073 / 0.7223 / 1 | 0.6073 / 0.6112 / 0.6134 (n=10) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 5.392e-09 @ 14500 | -0.9998 | 4.374e-09 / 0.2195 / 0.9999 | 4.374e-09 / 6.589e-09 / 9.756e-09 (n=10) |
| `grad_norm` | 1.5798 @ 0 | 0.1946 @ 14950 | -1.3851 | 0.1836 / 0.3529 / 1.7755 | 0.1836 / 0.2072 / 0.236 (n=100) |
| `grad_skipped` | 0 @ 0 | 0 @ 14950 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=100) |
| `grad_has_nan` | 0 @ 0 | 0 @ 14500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=10) |
| `lr_mult` | 6.667e-04 @ 0 | 3.385e-05 @ 14950 | -6.328e-04 | 3.385e-05 / 0.5001 / 1 | 3.385e-05 / 0.1069 / 0.302 (n=100) |

## Last Key Metrics

`c_effective_rank`=78.5257 @ 14500; `c_cross_video_cosine`=0.0951 @ 14500; `c_std_mean`=0.968 @ 14500; `L_recon_present`=0.349 @ 14500

## Interpretation

This is a present-branch experiment. The key content metric is `L_recon_present`, currently 0.349, and the key geometry metrics are rank 78.5257, cross-video cosine 0.0951, and std 0.968. The reading-cycle verdict is **Strong present representation** because The present branch is decodable, spread, video-specific, and high-rank. The run should not be counted as a Phase 1 prediction success, even when reconstruction improves, because the future branch and `F_c` are inactive.

## What This Run Changed

It separated present representation quality from forecasting. A present-only win is useful, but it must later be transferred into a full-prediction run.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (grid point sig=10.0, cov=0.0):** reached `c_effective_rank` ~78.5 with
cross-video cosine ~0.06-0.10, std ~0.96-0.98, and `L_recon_present` ~0.345 — a decodable, spread,
video-specific present code. Read against the rest of the sweep, the dominant lever on rank is the
COVARIANCE penalty: the cov=0.01 rows reach 139-151, cov=0.003 rows 90-106, and cov=0 rows only
57-83; SIGReg alone (cov=0) tops out ~83, and raising SIGReg helps only modestly on top of cov.
This is a strong PRESENT representation, but F_c is off, so it is not a prediction result.
