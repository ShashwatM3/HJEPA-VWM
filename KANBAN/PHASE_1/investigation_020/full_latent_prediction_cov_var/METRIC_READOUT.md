# Metric readout — full-latent prediction (`8r6akjsx`)

## Evidence contract

This readout uses the complete unsampled W&B history for
[`8r6akjsx`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8r6akjsx): 300 training rows at
steps 0–14,950 and 30 diagnostic rows at steps 0–14,500. Late values are medians over the final six
diagnostic rows (12,000–14,500), as preregistered in the parent sweep plan.

W&B reports the run `finished`. The last history step is 14,950 because training metrics log every
50 steps; the committed final checkpoint is `phase1_step15000.pt`, SHA-256
`ba47425785ed5d921607bfe114c8f030ab2c2f33f6b55faeadd5c6ec795d7804`.

The W&B run logged two committed artifacts:

- run provenance `8r6akjsx-provenance:v0`, digest
  `db42b2818037ab99749ea8c840807b35`, containing `run_provenance.json`;
- unsampled history `run-8r6akjsx-history:v0`, digest
  `53c93479ed9bf8c54729612fb69386ea`, containing `0000.parquet`.

## Resolved identity and config

| Field | Resolved value |
|---|---|
| state / mode | `finished`; full prediction (`present_recon_only=false`) |
| target | detached EMA full future latent (`predict_residual=false`) |
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
feature, source-checkpoint, component-state, and git identities as the residual arm. A recursive
W&B config comparison found only the intended target flag plus its derived provenance identity,
display name, and output path.

## Reading Cycle A

| Q | Question | Result | Final-window evidence |
|---|---|---|---|
| Q1 | Did the run train? | **PASS** | 15,000-step final checkpoint; zero skipped/nonfinite updates; max `grad_norm=4.449998` at step 11,150; `prediction_active=1` throughout. |
| Q2 | Is `c_t` alive and video-specific? | **ABSOLUTE PASS, MATERIAL DRIFT** | `c_std_mean=1.077082`; dead fraction `0`; latent cosine `0.325100` versus encoder `0.402768`, but up from `0.109707` at step 0. |
| Q3 | Is the latent rich and is EMA tracking? | **FLOOR PASS, PRETRAINED RETENTION FAIL** | online rank `296.622/512=0.5793`; target rank `293.805/512=0.5738`; both clear the floor but fell about 19%/17% from step 0. |
| Q4 | Is the code dynamic? | **STATIC-`c` PATTERN** | copy loss fell from `0.962386` to `0.362214` while copy ratio rose to `3.132558`; rank also contracted. |
| Q5 | Does Fc beat copy-forward? | **FAIL** | model/copy ratio `3.132558`; zero of six late points and zero of 30 diagnostic points pass `<=0.70`. |
| Q6 | Does Fc beat the batch mean? | **FAIL** | model/batch-mean ratio `0.946963`; zero late or whole-run points pass `<=0.50`. |
| Q7 | Is reconstruction honest? | **PASS as a diagnostic** | true-future reconstruction `0.122533`; predicted-future `0.187286`; correct-vs-shuffled video gap `0.406016`. |
| Q8 | Verdict | **Static-`c` trap** | The representation remains non-collapsed in the ordinary sense, but training makes future≈present cheaper while Fc becomes much worse than copy. |

## Final six diagnostic rows

| Step | `c_std_mean` | `c_cross_video_cosine` | `c_effective_rank` | `c_plus_effective_rank` | copy loss | copy ratio | batch ratio | `L_recon_present` | `L_recon_cplus` | `L_recon_chat` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12,000 | 1.078033 | 0.322282 | 299.248 | 299.730 | 0.384550 | 2.949065 | 0.938602 | 0.129309 | 0.121804 | 0.185350 |
| 12,500 | 1.081583 | 0.318905 | 298.876 | 297.221 | 0.380333 | 3.006179 | 0.948908 | 0.129306 | 0.122132 | 0.186887 |
| 13,000 | 1.077725 | 0.324198 | 297.611 | 294.748 | 0.367826 | 3.104114 | 0.954329 | 0.129716 | 0.122395 | 0.187061 |
| 13,500 | 1.076439 | 0.326002 | 295.633 | 292.863 | 0.356601 | 3.161003 | 0.945502 | 0.129940 | 0.122671 | 0.187511 |
| 14,000 | 1.074058 | 0.329169 | 293.726 | 291.466 | 0.351052 | 3.172742 | 0.936746 | 0.130210 | 0.122990 | 0.187817 |
| 14,500 | 1.075348 | 0.327723 | 293.212 | 290.167 | 0.347529 | 3.241400 | 0.948424 | 0.130216 | 0.123137 | 0.188556 |
| **median** | **1.077082** | **0.325100** | **296.622** | **293.805** | **0.362214** | **3.132558** | **0.946963** | **0.129828** | **0.122533** | **0.187286** |

`e_cross_video_cosine` is `0.402768` and `c_dead_dim_frac` is zero on every row in this window.
The fixed diagnostic population is source-diverse, so the paired encoder/latent cosine is a valid
cross-video readout rather than the historical single-source EGO4D proxy.

## Temporal trajectory

| Step | copy loss | model loss | copy ratio | batch ratio | online rank | latent pair cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.962386 | 2.902623 | 3.016070 | 2.333816 | 364.276 | 0.109707 |
| 2,000 | 0.883674 | 1.507960 | 1.706466 | 1.210812 | 344.661 | 0.155461 |
| 5,000 | 0.637411 | 1.209135 | 1.896948 | 1.032706 | 331.877 | 0.250798 |
| 7,500 | 0.498067 | 1.151130 | 2.311194 | 0.984664 | 325.356 | 0.298463 |
| 10,000 | 0.448930 | 1.197496 | 2.667447 | 0.998897 | 305.985 | 0.309157 |
| 12,500 | 0.380333 | 1.143349 | 3.006179 | 0.948908 | 298.876 | 0.318905 |
| 14,500 | 0.347529 | 1.126480 | 3.241400 | 0.948424 | 293.212 | 0.327723 |

The best copy ratio in the whole run was `1.706466` at step 2,000. It then worsened monotonically
in rank order to `3.241400` at step 14,500. The best batch-mean ratio was `0.936746` at step 14,000,
still far above the `0.50` gate. Post-step-2,000 copy loss has slope `-0.025095` per 1,000 steps;
online rank has slope `-3.535468` per 1,000 steps.

## Reconstruction and optimization facts

- Present reconstruction moved from `0.124882` at step 0 to a final-six median `0.129828`.
- True-future reconstruction moved from `0.118823` to `0.122533`.
- Predicted-future reconstruction remained `0.064753` worse than the true-future readout.
- Shuffling codes across source videos gives late loss `0.535873`, producing a large
  correct-video gap of `0.406016`.
- All 300 training rows have finite `loss`, `L_flow`, `L_var`, `L_cov`, `L_recon`, and
  `grad_norm`. There are no skipped updates, NaNs, or instability warnings.
- Adaptive clipping touched Fc on 10 of 300 logged training rows; B and D were never clipped in
  the logged rows. This did not form a skip or instability pattern.

Absolute full-latent `L_flow` and coarse losses are intentionally not compared with the residual
arm because the target and noise scales differ. The copy and batch-mean ratios are the registered
cross-mode comparison.
