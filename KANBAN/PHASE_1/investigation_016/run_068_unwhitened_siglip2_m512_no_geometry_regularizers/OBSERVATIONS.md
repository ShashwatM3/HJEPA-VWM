# Observations — Run 68

## Execution record

Run 68 is running in tmux session `inv016_siglip2_m512_no_geom` with W&B ID [`j7a3tzj5`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/j7a3tzj5),
PID `2087961` at launch, and the isolated checkpoint root documented in `DESCRIPTION.md`. The
first logged row is step `0`; whitening, prediction, and gradient skips are all zero. The process
has entered CUDA training and occupied approximately 20.8 GiB on the A100.

At step `500`, the warmup row remained finite with `grad_skipped=0` and `instability_warn=0`.
`recon_scale=0.25`, fixed-batch `L_recon_present=0.40972`, shuffled-code loss `0.41106`, and
the resulting gap `0.00133`; code cosine was `0.99874`, effective rank `25.51`, centered slot
rank `25.30`, and mean code std `0.03249`. Because reconstruction is only one quarter weighted
at this point, this is an initialization/warmup observation rather than a final geometry verdict.

## Required readout

Record step-500 diagnostics and every 500-step diagnostic thereafter, emphasizing `L_recon_present`,
`L_recon_shuffled_c`, `L_recon_video_gap`, `c_cross_video_cosine`, `c_effective_rank`,
`c_slot_diversity_rank_centered`, `c_std_mean`, `c_dead_dim_frac`, `grad_skipped`, and
`instability_warn`. Confirm `lambda_var=0`, `lambda_cov=0`, and `whiten_active=0` in the resolved
configuration and log.

## 2026-07-26 — terminal W&B reconciliation

W&B state is `crashed`, with the last training row at step 14,350 and the last fixed diagnostic at
step 14,000. The resolved config confirms `lambda_var=0`, `lambda_cov=0`,
`whiten_features=false`, `present_recon_only=true`, and the intended `N_c=32`, `D_c=256`, `M=512`
SigLIP 2 recipe.

Every logged `grad_skipped`, `grad_has_nan`, and `instability_warn` value is zero; late median
gradient norm is `0.08965`. The partial trajectory therefore shows no optimizer cliff before the
external termination, but the W&B record does not establish why the process ended.

Late six-diagnostic medians:

| Metric | Value |
|---|---:|
| fixed correct-code reconstruction | 0.21010 |
| fixed rolled-code reconstruction | 0.28945 |
| exact-chunk gap | 0.07945 |
| `c_std_mean` | 0.25507 |
| within-source pair cosine | 0.91216 |
| `c_effective_rank` | 20.076 |
| centered slot rank | 15.529 |
| dead-dimension fraction | 0.13019 |

The positive gap shows that the decoder used the supplied exact-chunk code. Geometry nevertheless
contracted severely without covariance and variance pressure. Verdict:
**LOW-RANK DECODABLE, PARTIAL ENDPOINT; GLOBAL COLLAPSE INDETERMINATE**. This is not a completed-run
verdict, and the single-source fixed batch cannot support a global cross-source claim.

The pre-run text above is preserved as chronology. [`ANALYSIS.md`](ANALYSIS.md) now contains the
terminal correction.
