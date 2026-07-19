# Observations — Run 64, EGO4D internal memory M=512

## Result

Run 64 completed all 15,000 steps. W&B run
[`4biwq87o`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4biwq87o) finished normally,
the provenance file exists, and the final checkpoint is:

```text
/workspace/ckpt/inv016_unwhitened_memory_m512/phase1_step15000.pt
sha256 1775fe9807886ea820a96cf0a559ce53d81950f6b0658d3c8098780224b62c51
```

The runtime identity stayed correct: full EGO4D, `present_recon_only=1`, prediction inactive,
`whiten_active=0`, `M=512`, `N_c=32`, external `D_c=256`, absolute cosine reconstruction at
weight 1, and zero variance/covariance/SIGReg/slot weights. Existing whitening-stat files were
never passed to or read by this run.

## Present-only reading cycle

| Q | Question | Verdict | Evidence |
|---|---|---|---|
| Q1 | Did it train in the intended mode? | PASS | 15,000 steps; `L_flow=0`; `L_recon_pred=0`; `present_recon_only=1`; prediction and whitening inactive; no NaN, skip, warning, bottleneck AGC clip, or decoder AGC clip. Maximum logged `grad_norm=0.7191`. |
| Q2 | Is the code alive and video-specific? | FAIL ON RECORDED BATCH, GLOBAL STATUS UNRESOLVED | Late median std `0.20490`, far below the 0.8–1.2 health band, and late median cosine `0.95380`. The fixed EGO4D batch is within-source/exact-chunk, so these values prove contraction on that batch but cannot establish global cross-source collapse. Dead-dimension fraction remained zero. |
| Q3 | Is the code rich rather than low-rank? | FAIL ON RECORDED BATCH | Late median effective rank `10.7546` and slot-diversity rank `9.9179`, far below the rank-60 representation target. The final values were `10.5655` and `9.8150`. |
| Q4 | Did reconstruction learn useful content? | PASS | Fixed-batch correct-code loss fell from `0.99081` at initialization to `0.30384` at step 14,500. The active training loss had a late median of `0.26079`; reconstruction was fully warmed up and decoder AGC never clipped. |
| Q5 | Did the decoder use the supplied clip code without preserving healthy geometry? | MIXED | Shuffled-code loss finished at `0.46757`, giving a `0.16373` correct-vs-shuffled gap. Dependence improved strongly, but std/rank/cosine moved in the wrong direction because all geometry forces were intentionally zero. |
| Q6 | Present-only verdict | **LOW-RANK DECODABLE; GLOBAL COLLAPSE INDETERMINATE** | The raw-feature compressor is input-dependent on the recorded exact-chunk batch, but it does not satisfy representation-health gates. No forecasting claim is possible. |

## Preregistered late window

Medians over all six diagnostic points at steps 12,000–14,500:

| Metric | Late median |
|---|---:|
| Active training `L_recon` | 0.260785 |
| Fixed correct-code `L_recon_present` | 0.305063 |
| Fixed shuffled-code `L_recon_shuffled_c` | 0.466626 |
| Correct-vs-shuffled gap | 0.161563 |
| Exact-chunk conditioned share | 23.2486% |
| Mean code std | 0.204902 |
| Cross-example cosine | 0.953795 |
| Effective rank | 10.754587 |
| Slot-diversity rank | 9.917890 |

The conditioned share is computed per diagnostic point as
`gap / (1 - L_recon_present)` and then median-aggregated. It measures how much of the available
cosine improvement disappears when the code is rolled. Because the fixed validation examples are
not source-diverse, it is an exact-chunk/within-source result, not a global video-conditioned
share.

## Interpretation

Removing whitening and preserving a 512-wide stream until the final `512 -> 256` projection made
the raw cosine target much easier than the earlier whitened target, but the raw and whitened loss
values are not directly comparable. The reliable positive result inside this arm is the growth of
the correct-vs-shuffled gap from zero to `0.16373`: the trained decoder no longer behaves like an
input-independent positional template on the recorded batch.

The equally important negative result is geometry. Without variance, covariance, SIGReg, or slot
pressure, reconstruction spent capacity on a narrow, strongly aligned code. This experiment
therefore supports “input-dependent, low-rank decodable compression,” not “healthy information
preservation.” Whether the extra 1,024-wide internal stream improves this tradeoff is the only
causal question answered by the paired Run 65.

Run 65 completed and raised late effective rank from `10.7546` to `15.3759`, but improved the
six-diagnostic training-loss median by only `0.000906` and the correct-versus-shuffled gap by only
`0.002029`. Those changes miss the registered `0.01` and `0.005` thresholds. M=512 is therefore
the selected practical width; the full paired table is in
[`../run_065_unwhitened_internal_memory_m1024/OBSERVATIONS.md`](../run_065_unwhitened_internal_memory_m1024/OBSERVATIONS.md).
