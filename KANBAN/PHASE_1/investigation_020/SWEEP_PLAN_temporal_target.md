# Paired plan — residual versus full-latent temporal target

## Experimental unit

This is one controlled two-arm experiment, not a regularizer sweep and not an encoder comparison.
Both arms start from the same selected Investigation-019 DINOv3 checkpoint and execute the same
15,000-update full-prediction schedule.

```text
same pretrained B + matched D
same fresh B_EMA copied from B
same fresh F_c
same fresh optimizer/RNG/sampler
same EGO4D order and seed
same covariance plus variance
same present reconstruction anchor
               |
               +-- residual target
               +-- full-latent target
```

## Registered source

| Field | Required value |
|---|---|
| selected arm label | DINOv3 expanded |
| `N_c` | `64` |
| `D_c` | `512` |
| `M` | `512` |
| source encoder | `dinov3_vitb16` |
| source W&B ID | `60yaqw6d` |
| source checkpoint | `/workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt` |
| source checkpoint SHA-256 | `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1` |
| source encoder fingerprint | `963cf988fb16b1e3971f952ade8afe90b29cef7dfd7103e4f312dada5c0eb415` |
| source dataset fingerprint | `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c` |

## Locked common controls

| Category | Value |
|---|---|
| dataset | full `ego4d` |
| encoder | `dinov3_vitb16`, immutable registry revision |
| seed | `42` |
| batch | `64`, subject only to the pre-launch common-memory gate below |
| steps | `15,000`; fresh schedule from step 0 |
| horizon | `horizon_k=12`, `frame_stride=2` |
| bottleneck | selected `N_c,D_c`; `M=512`; 3 latent blocks |
| decoder | `512×4`, loaded from the source checkpoint and kept trainable |
| reconstruction | present absolute cosine, weight `1.0`, 2,000-step ramp |
| predicted reconstruction | off (`lambda_recon_pred=0`) |
| geometry | `lambda_var=0.5`, `lambda_cov=0.01` |
| other geometry | `lambda_sigreg=0`, `lambda_slot=0` |
| feature transform | raw/unwhitened |
| temporal feature-recon target | absolute (`recon_residual_target=false`) |
| precision | bf16 / SDPA |
| optimizer | fresh AdamW; `lr_B=lr_Fc=lr_D=1e-4` from active YAML |
| clipping | current AGC plus global clip/skip contract |
| EMA | fresh `B_EMA=B` at step 0; current momentum schedule thereafter |
| cadence | log 50, diagnostics 500, checkpoint 2,500 |

If batch 64 fails either arm's resource preflight, do not launch. Register one common lower batch
for both arms in an updated clean commit, explain the effect on variance/covariance sample counts,
and rerun both preflights. Never lower the batch for only one arm.

## Only intended difference

| Arm | `predict_residual` | Flow target | Noise | Endpoint used by diagnostics |
|---|---:|---|---|---|
| residual | `true` | `Delta=c_plus-B_EMA(e_t)` | `std(Delta)*N(0,I)` | `c_t + Delta_hat` |
| full latent | `false` | `c_plus` | `N(0,I)` | predicted full `c_hat` |

The residual arm performs one additional no-grad `B_EMA(e_t)` call. That is part of the target
definition, not a second scientific axis.

## Gradient routing

Common:

- `L_flow` trains `F_c` and online `B` through the undetached `c_t` condition.
- `L_flow` never trains the frozen encoder, `B_EMA`, `e_plus`, or target latents.
- `L_recon_present` trains online `B` and `D`, not `F_c`.
- `L_var` and `L_cov` train online `B` only.
- `B_EMA` updates only after a successful optimizer step.

Residual-only:

- both ends of `Delta` come from `B_EMA` and remain targets;
- `c_t + Delta_hat` is used for prediction-side diagnostic decoding;
- because `lambda_recon_pred=0`, no decoder loss backpropagates through that endpoint.

## Required step-0 parity

Before interpreting training, prove:

- same git SHA;
- same source checkpoint path and SHA-256;
- same source W&B ID;
- same encoder and feature fingerprint;
- same dataset fingerprint and epoch-0 order;
- same loaded online `B`;
- same copied `B_EMA`;
- same loaded `D`;
- same freshly initialized `F_c`;
- same fresh optimizer;
- same trainable initialization hash;
- `next_step=0`;
- different W&B IDs and output roots;
- only `predict_residual` differs in scientific config.

Any other difference invalidates the pair.

## Reading protocol

Apply Reading Cycle A separately to each arm.

### Q1 — stability and mode

Require all 15,000 updates, `grad_skipped=0`, `grad_has_nan=0`, finite gradients, and
`prediction_active=1`. The residual arm must report `predict_residual=true`; the full arm must
report false.

### Q2 — example-specific representation

Use the repaired source-diverse pair:

```text
e_cross_video_cosine
c_cross_video_cosine
```

Require `c_cross_video_cosine < 0.5`, read relative to `e`, and require no material late collapse
from the step-0 warm-start value. Also target `c_std_mean` in the 0.8–1.2 range and
`c_dead_dim_frac` near zero.

### Q3 — shape-aware richness and EMA tracking

The old raw-rank threshold 60 was calibrated at `D_c=256`. Report both:

```text
online_rank_fraction = c_effective_rank / D_c
target_rank_fraction = c_plus_effective_rank / D_c
```

Require at least the historical fraction `60/256 = 0.234375`, plus no material late loss from the
step-0 pretrained baseline. Online and EMA-target fractions should remain close enough that a
moving online coordinate system is not masquerading as a healthy target.

### Q4 — temporal dynamics

Read `coarse_copy_loss`, ratio, and rank together. A falling copy loss with a ratio near one is the
static-`c` trap. A rising copy loss with ratio near one is a dynamic representation with no
predictor.

### Q5/Q6 — prediction gates

Use the median of the final six diagnostic rows, and require the threshold at more than one late
point:

```text
coarse_vs_copy_ratio <= 0.70
coarse_vs_batch_mean_ratio <= 0.50
```

Compare ratios, not absolute `L_flow`, `coarse_model_loss`, or `coarse_copy_loss`, across modes.

### Q7 — reconstruction honesty

Read `L_recon_present`, `L_recon_cplus`, `L_recon_chat`, shuffled-code loss, and the video gap.
The present anchor should retain useful content. If prediction is poor, `L_recon_chat` should
remain worse than `L_recon_cplus`; equality is the known blindness signature.

## Paired decision rule

1. Disqualify any Q1 failure.
2. Label each run independently using Reading Cycle A.
3. If exactly one passes both Q5/Q6 gates and Q2/Q3, it wins.
4. If both pass, choose the lower final-six median copy ratio; use batch-mean ratio, representation
   retention, and stability as ordered tie-breakers.
5. If neither passes, declare no winner. Use the Q4 pattern to distinguish static latent from
   dynamic latent/no predictor.
6. Do not select from `L_flow`, a single best step, or a cross-mode absolute loss.

## Follow-up boundary for SIGReg

The next regularizer probe, if warranted, should keep the winning temporal target fixed and compare
the covariance-plus-variance control against one preregistered SIGReg addition. Pure SIGReg is not
promoted merely because the full prediction path is now active; the current code's objective is not
identical to LeJEPA alignment, and pure SIGReg already lost the matched present-only comparison.
