# Metric readout — residual prediction (`3y2hxj5t`)

## Evidence contract

This readout uses the complete unsampled W&B history for
[`3y2hxj5t`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3y2hxj5t): 300 training rows
at steps 0–14,950 and 30 diagnostic rows at steps 0–14,500. Late values are medians over the final
six diagnostic rows (12,000–14,500), as preregistered in the parent sweep plan.

W&B reports the run `finished`. The last history step is 14,950 because training metrics log every
50 steps; the committed final checkpoint is `phase1_step15000.pt`, SHA-256
`0671516356b149d37b89c3cb3940d7c0eaa870b668eb708a13bab55ae73719dc`.

The W&B run logged two committed artifacts:

- run provenance `3y2hxj5t-provenance:v0`, digest
  `b61ed68ce57bafb4706f76716e446b42`, containing `run_provenance.json`;
- unsampled history `run-3y2hxj5t-history:v0`, digest
  `71b126b20e5d17d7be9c5d43d1d7eb31`, containing `0000.parquet`.

## Resolved identity and config

| Field | Resolved value |
|---|---|
| state / mode | `finished`; full prediction (`present_recon_only=false`) |
| target | EMA future-minus-present residual (`predict_residual=true`) |
| seed / data | `42`; EGO4D; batch 64; horizon 12; frame stride 2 |
| encoder | frozen DINOv3 ViT-B/16, resolved revision `5931719e67bbdb9737e363e781fb0c67687896bc` |
| encoder output | 8 framewise 16×16 grids; `(B,2048,768)` |
| bottleneck / flow | `N_c=64`, `D_c=512`, `M=512`, 3 latent blocks; 6 flow blocks |
| objective | `L_flow + 0.5 L_var + 0.01 L_cov + recon_scale L_recon_present` |
| disabled terms | `lambda_sigreg=0`, `lambda_slot=0`, `lambda_recon_pred=0`, no whitening |
| optimizer schedule | 15,000 steps; 1,500-step warmup; cosine decay; LR `1e-4` for B/Fc/D |
| warm start | B and D from DINO run `60yaqw6d`; fresh exact-copy B_EMA; fresh Fc/optimizer/schedule |
| source checkpoint SHA-256 | `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1` |
| dataset / feature fingerprints | `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c` / `963cf988fb16b1e3971f952ade8afe90b29cef7dfd7103e4f312dada5c0eb415` |
| clean launch commit | `7649f8efde1b104dd81cfbd110af18d499f67304` |

The downloaded provenance artifact records the same dataset, validation-batch, train-order,
feature, source-checkpoint, component-state, and git identities as the full-latent arm. A recursive
W&B config comparison found only the intended target flag plus its derived provenance identity,
display name, and output path.

## Reading Cycle A

| Q | Question | Result | Final-window evidence |
|---|---|---|---|
| Q1 | Did the run train? | **PASS** | 15,000-step final checkpoint; zero skipped/nonfinite updates; max `grad_norm=5.335405` at step 1,400; `prediction_active=1` throughout. |
| Q2 | Is `c_t` alive and video-specific? | **PASS** | `c_std_mean=1.118518`; dead fraction `0`; latent pair cosine `0.179245` versus encoder `0.402768`. |
| Q3 | Is the latent rich and is EMA tracking? | **PASS** | online rank `376.932/512=0.7362`; EMA-future rank `363.957/512=0.7109`; both exceed the shape-aware floor `0.234375`. |
| Q4 | Is the code dynamic? | **YES, but unforecasted** | copy loss rose from `0.962544` at step 0 to a late median `1.300293`; online rank rose from `364.276` to `376.932`. |
| Q5 | Does Fc beat zero residual? | **FAIL** | model/copy ratio `1.726204`; zero of six late points and zero of 30 diagnostic points pass `<=0.70`. |
| Q6 | Does Fc beat the batch mean? | **FAIL** | model/batch-mean ratio `1.817973`; zero late or whole-run points pass `<=0.50`. |
| Q7 | Is reconstruction honest? | **PASS as a diagnostic** | true-future reconstruction `0.100083`; predicted-future reconstruction `0.181156`; correct-vs-shuffled video gap `0.443072`. |
| Q8 | Verdict | **Healthy rep, no predictor** | Representation health and temporal separation pass; both forecasting gates fail. |

## Final six diagnostic rows

| Step | `c_std_mean` | `c_cross_video_cosine` | `c_effective_rank` | `c_plus_effective_rank` | copy loss | copy ratio | batch ratio | `L_recon_present` | `L_recon_cplus` | `L_recon_chat` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12,000 | 1.119276 | 0.177724 | 377.976 | 364.745 | 1.323215 | 1.795698 | 1.887769 | 0.105268 | 0.100384 | 0.185539 |
| 12,500 | 1.119146 | 0.178010 | 376.773 | 364.607 | 1.253055 | 1.705783 | 1.790743 | 0.104992 | 0.100229 | 0.177096 |
| 13,000 | 1.120178 | 0.177200 | 378.844 | 363.930 | 1.296162 | 1.684899 | 1.769730 | 0.104962 | 0.100215 | 0.188197 |
| 13,500 | 1.117192 | 0.181522 | 377.091 | 363.984 | 1.287656 | 1.744104 | 1.841577 | 0.104791 | 0.099951 | 0.180870 |
| 14,000 | 1.117890 | 0.180481 | 376.606 | 363.456 | 1.305107 | 1.721483 | 1.812796 | 0.104701 | 0.099814 | 0.181384 |
| 14,500 | 1.117261 | 0.181365 | 376.335 | 363.049 | 1.304424 | 1.730925 | 1.823150 | 0.104623 | 0.099754 | 0.180927 |
| **median** | **1.118518** | **0.179245** | **376.932** | **363.957** | **1.300293** | **1.726204** | **1.817973** | **0.104876** | **0.100083** | **0.181156** |

`e_cross_video_cosine` is `0.402768` and `c_dead_dim_frac` is zero on every row in this window.
The fixed diagnostic population is source-diverse, so the paired encoder/latent cosine is a valid
cross-video readout rather than the historical single-source EGO4D proxy.

## Temporal trajectory

| Step | copy loss | model loss | copy ratio | batch ratio | online rank | latent pair cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.962544 | 3.021331 | 3.138901 | 3.322235 | 364.276 | 0.109707 |
| 2,000 | 0.951611 | 2.082734 | 2.188639 | 2.311484 | 335.644 | 0.205462 |
| 5,000 | 1.221597 | 1.986245 | 1.625941 | 1.727544 | 364.060 | 0.155839 |
| 7,500 | 1.157985 | 1.989747 | 1.718284 | 1.833979 | 367.539 | 0.170952 |
| 10,000 | 1.276588 | 2.173292 | 1.702423 | 1.807227 | 371.445 | 0.173690 |
| 12,500 | 1.253055 | 2.137440 | 1.705783 | 1.790743 | 376.773 | 0.178010 |
| 14,500 | 1.304424 | 2.257859 | 1.730925 | 1.823150 | 376.335 | 0.181365 |

The best copy ratio in the whole run was `1.574568` at step 6,500, still more than twice the
registered pass threshold. The best batch-mean ratio was `1.658846` at the same step. Copy loss has
a positive post-step-2,000 slope of `0.021374` per 1,000 steps; rank has a positive slope of
`2.351107` per 1,000 steps.

## Reconstruction and optimization facts

- Present reconstruction improved from `0.124882` at step 0 to a final-six median `0.104876`.
- True-future reconstruction improved from `0.118823` to `0.100083`.
- Predicted-future reconstruction remained `0.081073` worse than the true-future readout.
- Shuffling codes across source videos gives late loss `0.547940`, producing a large
  correct-video gap of `0.443072`.
- All 300 training rows have finite `loss`, `L_flow`, `L_var`, `L_cov`, `L_recon`, and
  `grad_norm`. There are no skipped updates, NaNs, or instability warnings.
- Adaptive clipping touched Fc on 25 of 300 logged training rows; B and D were never clipped in
  the logged rows. This did not form a skip or instability pattern.

Absolute residual `L_flow` and coarse losses are intentionally not compared with the full-latent
arm because the target and noise scales differ. The copy and batch-mean ratios are the registered
cross-mode comparison.
