# Observations — Run 65, EGO4D internal memory M=1024

## Result

Run 65 completed all 15,000 steps. W&B run
[`8gr3je5b`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8gr3je5b) finished normally,
the provenance file exists, and the final checkpoint is:

```text
/workspace/ckpt/inv016_unwhitened_memory_m1024/phase1_step15000.pt
sha256 ca35295b4997370cf7d5767bd6e460be0681e9cc235621f4e2cb7fcef52630bd
```

The runtime identity stayed correct: clean commit `9522008`, full EGO4D,
`present_recon_only=1`, prediction inactive, `whiten_active=0`, `M=1024`, `N_c=32`, external
`D_c=256`, absolute cosine reconstruction at weight 1, and zero
variance/covariance/SIGReg/slot weights. Existing whitening-stat files were neither passed nor read.

## Present-only reading cycle

| Q | Question | Verdict | Evidence |
|---|---|---|---|
| Q1 | Did it train in the intended mode? | PASS | 15,000 steps; `L_flow=0`; `L_recon_pred=0`; `present_recon_only=1`; prediction and whitening inactive; no NaN, skip, warning, bottleneck AGC clip, or decoder AGC clip. Maximum logged `grad_norm=1.5579`; global clipping handled larger finite norms normally. |
| Q2 | Is the code alive and video-specific? | FAIL ON RECORDED BATCH, GLOBAL STATUS UNRESOLVED | Late median std `0.21530`, far below the 0.8–1.2 health band, and late median cosine `0.94990`. The fixed EGO4D batch is within-source/exact-chunk, so these values prove contraction on that batch but cannot establish global cross-source collapse. Dead-dimension fraction remained zero. |
| Q3 | Is the code rich rather than low-rank? | FAIL ON RECORDED BATCH | Late median effective rank `15.3759` and slot-diversity rank `13.5693`, far below the rank-60 representation target. Final values were `15.0648` and `13.4338`. |
| Q4 | Did reconstruction learn useful content? | PASS | Fixed-batch correct-code loss fell from `1.01836` at initialization to `0.30229` at step 14,500. Active training loss had a six-diagnostic late median of `0.25988`; reconstruction was fully warmed up and decoder AGC never clipped. |
| Q5 | Did the decoder use the supplied clip code without preserving healthy geometry? | MIXED | Shuffled-code loss finished at `0.46820`, giving a `0.16591` correct-versus-shuffled gap. Dependence improved strongly, but std/rank/cosine still contracted because every geometry force was intentionally zero. |
| Q6 | Present-only verdict | **LOW-RANK DECODABLE; GLOBAL COLLAPSE INDETERMINATE** | The raw-feature compressor is input-dependent on the recorded exact-chunk batch, but it does not satisfy representation-health gates. No forecasting claim is possible. |

## Preregistered late window

Medians over all six diagnostic points at steps 12,000–14,500:

| Metric | Late median |
|---|---:|
| Active training `L_recon` | 0.259879 |
| Fixed correct-code `L_recon_present` | 0.303584 |
| Fixed shuffled-code `L_recon_shuffled_c` | 0.467176 |
| Correct-vs-shuffled gap | 0.163592 |
| Exact-chunk conditioned share | 23.4905% |
| Mean code std | 0.215297 |
| Cross-example cosine | 0.949897 |
| Effective rank | 15.375923 |
| Slot-diversity rank | 13.569283 |

The conditioned share is computed per diagnostic point as
`gap / (1 - L_recon_present)` and then median-aggregated. It is a within-source exact-chunk
measurement because all 16 fixed validation chunks share one EGO4D source UID.

## Paired width decision

The only causal comparison in this bundle is M=512 versus M=1024:

| Metric | M=512 | M=1024 | 1,024 minus 512 |
|---|---:|---:|---:|
| Active training `L_recon` | 0.260785 | 0.259879 | -0.000906 |
| Fixed correct-code loss | 0.305063 | 0.303584 | -0.001478 |
| Fixed shuffled-code loss | 0.466626 | 0.467176 | +0.000550 |
| Correct-vs-shuffled gap | 0.161563 | 0.163592 | +0.002029 |
| Exact-chunk conditioned share | 23.2486% | 23.4905% | +0.2419 pp |
| Mean code std | 0.204902 | 0.215297 | +0.010394 |
| Cross-example cosine | 0.953795 | 0.949897 | -0.003898 |
| Effective rank | 10.754587 | 15.375923 | +4.621336 |
| Slot-diversity rank | 9.917890 | 13.569283 | +3.651394 |
| W&B runtime | 11,473 s | 12,321 s | +848 s (+7.39%) |

The 1,024-wide bottleneck has about 3.86 times as many parameters as the 512-wide bottleneck. The
preregistered rule required at least `0.01` lower late `L_recon`, at least `0.005` more shuffled
gap, and no conditioned-share decrease. M=1024 passes only the share direction: its loss improves
by `0.000906` and its gap by `0.002029`. **Select M=512.** The rank gain is real on the recorded
batch, but it is not a cost-justifying reconstruction/preservation gain.

## Interpretation

Eliminating forced channel reduction before the final `1024 -> 256` projection lets the processor
carry more directions through its latent computation. That is visible in the late rank increase
from `10.75` to `15.38` and the small std/cosine improvement. It also proves that the new width
hyperparameter genuinely changes the complete memory/query/latent stream rather than only a local
preprocessor.

The decoder nevertheless obtains almost the same useful clip-conditioned information from the
fixed `32 x 256` output. This localizes the remaining limit downstream of, or orthogonal to,
internal width: the final external rate, objective/decoder shortcut, or unregularized geometry is
more likely binding than the difference between 512 and 1,024 internal channels. Because both arms
also removed whitening and moved projection timing together, neither intervention can be assigned
an independent causal effect against historical whitened runs.
