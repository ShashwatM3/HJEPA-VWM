# Observations - run 031 `lambda_sigreg_1.0`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `jk8kj7h7`  
**State:** `crashed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=crashed; step=13100; grad_skipped max=0; grad_has_nan max=0; grad_norm last=4.0711 |
| Q2 | c_t alive / video-specific? | PASS | std=1.0147; dead_dim=0; cross_video_cosine=0.2892 |
| Q3 | Rich latent? | FAIL | c_effective_rank=13.0161; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1279 (0.0533 @ 0 -> 0.1812 @ 13000); ratio=2.7053; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=0.4903; copy_loss=0.1812; ratio=2.7053; gate <= 0.70 |
| Q6 | F_c uses this video? | PARTIAL | batch_mean_ratio=0.4522; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | BLIND | present=0.5858; cplus=0.583; chat=0.5944; chat-cplus=0.0114; decoder readout barely separates predicted future from true future while prediction is poor |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 3.1171 @ 0 | 0.5519 @ 13100 | -2.5652 | 0.504 / 0.8453 / 3.5603 | 0.504 / 0.5581 / 0.6095 (n=63) |
| `L_flow` | 2.8686 @ 0 | 0.5083 @ 13100 | -2.3604 | 0.4612 / 0.7996 / 3.382 | 0.4612 / 0.5152 / 0.5678 (n=63) |
| `L_var` | 0.4081 @ 0 | 0.0119 @ 13100 | -0.3962 | 0.0054 / 0.0184 / 0.4081 | 0.0054 / 0.0096 / 0.013 (n=63) |
| `L_sigreg` | 0.0444 @ 0 | 0.0064 @ 13100 | -0.038 | 0.0054 / 0.0074 / 0.0444 | 0.0054 / 0.0069 / 0.0089 (n=63) |
| `L_recon` | 1.0379 @ 0 | 0.6258 @ 13100 | -0.4121 | 0.6097 / 0.6447 / 1.0379 | 0.6153 / 0.6237 / 0.6325 (n=63) |
| `L_recon_pred` | 0 @ 0 | 0 @ 13100 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=63) |
| `recon_scale` | 0 @ 0 | 1 @ 13100 | 1 | 0 / 0.9221 / 1 | 1 / 1 / 1 (n=63) |
| `c_std_mean` | 0.4952 @ 0 | 1.0147 @ 13000 | 0.5194 | 0.4952 / 0.968 / 1.0344 | 1.0147 / 1.0263 / 1.0344 (n=7) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 13000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=7) |
| `c_cross_video_cosine` | 0.7239 @ 0 | 0.2892 @ 13000 | -0.4347 | 0.1186 / 0.2497 / 0.7239 | 0.2532 / 0.2672 / 0.2892 (n=7) |
| `c_effective_rank` | 9.4728 @ 0 | 13.0161 @ 13000 | 3.5432 | 9.4728 / 12.6488 / 13.7474 | 13.0161 / 13.2685 / 13.6301 (n=7) |
| `c_slot_diversity_rank` | 16.711 @ 0 | 14.7263 @ 13000 | -1.9848 | 13.3155 / 15.9253 / 20.6161 | 13.5435 / 14.0534 / 14.7263 (n=7) |
| `c_attn_entropy` | 0.9999 @ 0 | 0.8857 @ 13000 | -0.1143 | 0.8857 / 0.9332 / 1 | 0.8857 / 0.8911 / 0.8976 (n=7) |
| `c_attn_entropy_min` | 0.9998 @ 0 | 0.7408 @ 13000 | -0.259 | 0.7395 / 0.8344 / 0.9999 | 0.7395 / 0.7437 / 0.7508 (n=7) |
| `coarse_copy_loss` | 0.0533 @ 0 | 0.1812 @ 13000 | 0.1279 | 0.0533 / 0.2409 / 0.4121 | 0.1812 / 0.2037 / 0.2291 (n=7) |
| `coarse_model_loss` | 3.3106 @ 0 | 0.4903 @ 13000 | -2.8203 | 0.4585 / 0.8272 / 3.3106 | 0.4585 / 0.5085 / 0.5565 (n=7) |
| `coarse_batch_mean_loss` | 0.2627 @ 0 | 1.0842 @ 13000 | 0.8214 | 0.2627 / 0.9977 / 1.1205 | 1.0842 / 1.1046 / 1.1205 (n=7) |
| `coarse_vs_copy_ratio` | 62.0672 @ 0 | 2.7053 @ 13000 | -59.3619 | 2.0929 / 5.0964 / 62.0672 | 2.1854 / 2.5019 / 2.7053 (n=7) |
| `coarse_vs_batch_mean_ratio` | 12.6011 @ 0 | 0.4522 @ 13000 | -12.1489 | 0.4147 / 1.1839 / 12.6011 | 0.4147 / 0.4603 / 0.508 (n=7) |
| `L_recon_present` | 1.0398 @ 0 | 0.5858 @ 13000 | -0.454 | 0.5845 / 0.6134 / 1.0398 | 0.5848 / 0.5855 / 0.5862 (n=7) |
| `L_recon_cplus` | 1.0407 @ 0 | 0.583 @ 13000 | -0.4577 | 0.5823 / 0.614 / 1.0407 | 0.5823 / 0.5832 / 0.5841 (n=7) |
| `L_recon_chat` | 1.04 @ 0 | 0.5944 @ 13000 | -0.4456 | 0.5891 / 0.6216 / 1.04 | 0.5891 / 0.5908 / 0.5944 (n=7) |
| `grad_norm` | 2.0144 @ 0 | 4.0711 @ 13100 | 2.0567 | 0.574 / 2.1745 / 4.3148 | 2.7581 / 3.0379 / 4.3148 (n=63) |
| `grad_skipped` | 0 @ 0 | 0 @ 13100 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=63) |
| `grad_has_nan` | 0 @ 0 | 0 @ 13000 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=7) |
| `lr_mult` | 6.667e-04 @ 0 | 0.0481 @ 13100 | 0.0474 | 6.667e-04 / 0.5682 / 1 | 0.0481 / 0.1604 / 0.302 (n=63) |

## Last Key Metrics

`c_effective_rank`=13.0161 @ 13000; `c_cross_video_cosine`=0.2892 @ 13000; `c_std_mean`=1.0147 @ 13000; `coarse_vs_copy_ratio`=2.7053 @ 13000; `coarse_vs_batch_mean_ratio`=0.4522 @ 13000; `L_recon_present`=0.5858 @ 13000

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 2.7053 and the latest batch-mean ratio is 0.4522; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 13.0161, cross-video cosine is 0.2892, and copy loss is 0.1812. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

It contributed to the SIGReg conclusion: rank can be moved by isotropy pressure, but that movement did not produce a copy-gate win.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.
