# Observations - run 033 `lambda_sigreg_10.0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `fbqgix1x`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Healthy rep, no predictor**  
**Verdict note:** Representation health is not the limiting issue; F_c still does not beat copy.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=13150; grad_skipped max=0; grad_has_nan max=0; grad_norm last=2.1387 |
| Q2 | c_t alive / video-specific? | PASS | std=0.9074; dead_dim=0; cross_video_cosine=0.3844 |
| Q3 | Rich latent? | PASS | c_effective_rank=73.7941; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.0526 (0.0533 @ 0 -> 0.1059 @ 13000); ratio=9.0694; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.9604; copy_loss=0.1059; ratio=9.0694; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.103; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | CHECK | present=0.5768; cplus=0.5742; chat=0.6262; chat-cplus=0.0519; reconstruction is active; read with cplus/chat gap and prediction gates |
| Q8 | Verdict | Healthy rep, no predictor | Representation health is not the limiting issue; F_c still does not beat copy. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.5164 @ 0 | 1.0149 @ 13150 | -2.5016 | 0.9047 / 1.1504 / 3.7899 | 0.9047 / 1.0174 / 1.1396 (n=64) |
| `L_flow` | 2.8686 @ 0 | 0.9523 @ 13150 | -1.9164 | 0.843 / 1.0695 / 3.3821 | 0.843 / 0.9534 / 1.0748 (n=64) |
| `L_var` | 0.4081 @ 0 | 0.036 @ 13150 | -0.3721 | 0.0268 / 0.0394 / 0.4081 | 0.0302 / 0.0346 / 0.0425 (n=64) |
| `L_sigreg` | 0.0444 @ 0 | 0.0014 @ 13150 | -0.043 | 0.0013 / 0.0032 / 0.0444 | 0.0013 / 0.0016 / 0.002 (n=64) |
| `L_recon` | 1.0379 @ 0 | 0.6129 @ 13150 | -0.425 | 0.6024 / 0.6393 / 1.0379 | 0.6082 / 0.6164 / 0.6251 (n=64) |
| `L_recon_pred` | 0 @ 0 | 0 @ 13150 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=64) |
| `recon_scale` | 0 @ 0 | 1 @ 13150 | 1 | 0 / 0.9223 / 1 | 1 / 1 / 1 (n=64) |
| `c_std_mean` | 0.4952 @ 0 | 0.9074 @ 13000 | 0.4122 | 0.4952 / 0.8936 / 0.9368 | 0.9066 / 0.9073 / 0.908 (n=7) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 13000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=7) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.3844 @ 13000 | -0.3395 | 0.1093 / 0.3202 / 0.7239 | 0.377 / 0.3816 / 0.3849 (n=7) |
| `c_effective_rank` | 9.4729 @ 0 | 73.7941 @ 13000 | 64.3211 | 9.4729 / 44.7022 / 73.7941 | 69.1149 / 71.7389 / 73.7941 (n=7) |
| `c_slot_diversity_rank` | 16.711 @ 0 | 11.5845 @ 13000 | -5.1266 | 3.8159 / 9.8173 / 16.711 | 10.8633 / 11.2776 / 11.5845 (n=7) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.883 @ 13000 | -0.1169 | 0.8213 / 0.9004 / 0.9999 | 0.8761 / 0.88 / 0.883 (n=7) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.4989 @ 13000 | -0.501 | 0.1764 / 0.5382 / 0.9998 | 0.4473 / 0.4774 / 0.4989 (n=7) |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.1059 @ 13000 | 0.0526 | 0.0533 / 0.2197 / 0.4549 | 0.1059 / 0.1128 / 0.12 (n=7) |
| `coarse_model_loss` | 3.3106 @ 0 | 0.9604 @ 13000 | -2.3502 | 0.7899 / 1.0546 / 3.3106 | 0.7899 / 0.855 / 0.9604 (n=7) |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 0.8707 @ 13000 | 0.608 | 0.2627 / 0.8504 / 0.9467 | 0.8669 / 0.8703 / 0.872 (n=7) |
| `coarse_vs_copy_ratio` | 62.0677 @ 0 | 9.0694 @ 13000 | -52.9984 | 2.4504 / 7.2981 / 62.0677 | 6.5842 / 7.6047 / 9.0694 (n=7) |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 1.103 @ 13000 | -11.4981 | 0.9112 / 1.5382 / 12.6011 | 0.9112 / 0.9823 / 1.103 (n=7) |
| `L_recon_present` | 1.0398 @ 0 | 0.5768 @ 13000 | -0.463 | 0.5761 / 0.6078 / 1.0398 | 0.5761 / 0.5766 / 0.5769 (n=7) |
| `L_recon_cplus` | 1.0407 @ 0 | 0.5742 @ 13000 | -0.4665 | 0.5739 / 0.6086 / 1.0407 | 0.5739 / 0.5742 / 0.5752 (n=7) |
| `L_recon_chat` | 1.04 @ 0 | 0.6262 @ 13000 | -0.4139 | 0.5891 / 0.6275 / 1.04 | 0.5993 / 0.6058 / 0.6262 (n=7) |
| `grad_norm` | 3.9399 @ 0 | 2.1387 @ 13150 | -1.8012 | 1.0869 / 2.1243 / 3.9399 | 2.1387 / 2.4109 / 2.7143 (n=64) |
| `grad_skipped` | 0 @ 0 | 0 @ 13150 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=64) |
| `grad_has_nan` | 0 @ 0 | 0 @ 13000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=7) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0456 @ 13150 | 0.045 | 6.667e-04 / 0.5662 / 1 | 0.0456 / 0.1586 / 0.302 (n=64) |

## Last Key Metrics

`c_effective_rank`=73.7941 @ 13000; `c_cross_video_cosine`=0.3844 @ 13000; `c_std_mean`=0.9074 @ 13000; `coarse_vs_copy_ratio`=9.0694 @ 13000; `coarse_vs_batch_mean_ratio`=1.103 @ 13000; `L_recon_present`=0.5768 @ 13000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 9.0694 and the latest batch-mean ratio is 1.103; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 73.7941, cross-video cosine is 0.3844, and copy loss is 0.1059. The reading-cycle verdict is **Healthy rep, no predictor** because Representation health is not the limiting issue; F_c still does not beat copy. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

It contributed to the SIGReg conclusion: rank can be moved by isotropy pressure, but that movement did not produce a copy-gate win.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

**Mechanism read (W&B `fbqgix1x`):** the decisive point of the sweep. Rank rocketed to ~73.3
(still rising at the cut) — proof the rank-13 ceiling is a regularizer choice, not architectural.
But `coarse_vs_copy_ratio` worsened to ~8.2 (the worst of the sweep) and L_flow rose to ~0.93,
while `coarse_copy_loss` fell to ~0.11 (c froze hardest in time). Verdict Healthy-rep-no-predictor:
a rich, high-rank c that F_c cannot forecast. The variance floor was continuously active here
(L_var ~0.03-0.05, std pinned ~0.90 below the 1.0 target because SIGReg flattens the spectrum) —
which is why investigation_009 kept lambda_var=0.5 as load-bearing, not redundant.
