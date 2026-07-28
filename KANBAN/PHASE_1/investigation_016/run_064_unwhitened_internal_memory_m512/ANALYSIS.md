# Paired analysis — Runs 64 and 65 · unwhitened internal memory M=512 versus M=1024

> Analyzed 2026-07-19 using the present-reconstruction-only Reading Cycle B in
> `GUIDES/READING_EXPERIMENTS.md`. This is one interleaved analysis of the paired experiment, not
> two independent run summaries. It reads all 300 logged rows and all 30 fixed-batch diagnostic
> points from each W&B run: [M=512 (`4biwq87o`)](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/4biwq87o)
> and [M=1024 (`8gr3je5b`)](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8gr3je5b).
> The canonical report lives in the Run 64 folder and is linked from Run 65's `ANALYSIS.md` so the
> comparison has one source of truth.

## Executive interpretation

Both runs are operationally valid and reconstruct raw V-JEPA features very well, but neither
learns a healthy external representation. The important behavior is not simply the final
reconstruction value near `0.30`; it is the fact that reconstruction continues improving after
the latent reaches its best geometry, while variance, feature rank, slot rank, and cross-example
separation all deteriorate. The decoder is therefore becoming better at reading a progressively
narrower code, not preserving an increasingly broad fraction of the frozen encoder embedding.

Increasing the complete internal width from 512 to 1,024 changes that geometry in a real but
secondary way. M=1024 retains about 4.6 more effective-rank directions and 3.6 more slot-rank
directions in the late window, yet improves active reconstruction by only `0.000906` and the
correct-versus-shuffled gap by only `0.002029`. In other words, the wider processor keeps more
directions alive internally and through the final projection, but the unchanged `32 x 256` output,
raw cosine objective, and decoder extract almost exactly the same useful reconstruction signal.
This misses both preregistered material-effect thresholds and selects M=512 as the practical width.

The high final `c_cross_video_cosine` values (`0.956` and `0.954`) require one crucial qualification:
all 16 fixed validation examples are adjacent chunks from the same EGO4D source UID. The metric is
therefore a cross-example, within-source cosine in these runs, not a genuine cross-source test.
It does prove strong contraction on the recorded batch and agrees with the low std/rank evidence,
but it cannot establish that arbitrary EGO4D videos globally map to one direction. The appropriate
verdict for both arms is consequently **low-rank decodable; global collapse indeterminate**.

## 1. Exact experimental contract

The pair was designed to isolate one variable: the complete width `M` of the bottleneck's memory,
queries, cross-attention, self-attention, and latent MLP stream. Both arms started from scratch on
the same clean commit (`95220080053fbea1a92f9fbcb467c9d0cbd666c8`) with the same seed, data order,
encoder revision, fixed validation batch, objective, schedule, external code, and decoder.

| Contract | Run 64 | Run 65 | Causal status |
|---|---:|---:|---|
| W&B ID | `4biwq87o` | `8gr3je5b` | two finished arms |
| Dataset / seed / steps | EGO4D / 42 / 15,000 | same | held fixed |
| Frozen encoder output | `1024 x 1024` | same | held fixed |
| Complete internal width `M` | **512** | **1,024** | only intended delta |
| External abstract code | `32 x 256` | same | held fixed |
| Final projection | `512 -> 256` | `1024 -> 256` | derived from `M` |
| Latent processor | 3 blocks, 8 heads | same | held fixed |
| Decoder | width 512, 4 blocks | same | held fixed |
| Training mode | present reconstruction only | same | prediction inactive |
| Target / loss | raw absolute V-JEPA / cosine | same | held fixed |
| Reconstruction weight | `1.0`, warmup 2,000 | same | held fixed |
| Whitening / residual target | off / off | same | held fixed |
| `lambda_var/cov/sigreg/slot` | `0/0/0/0` | same | held fixed |

The active path was:

```text
EGO4D clip
  -> frozen V-JEPA2-L
  -> e_t: (B, 1024 tokens, 1024 channels)
  -> trainable bottleneck at M=512 or M=1024
  -> three input-dependent read/compete/refine blocks at the same M
  -> one final M -> 256 projection + LayerNorm
  -> c_t: (B, 32 slots, 256 channels)
  -> fixed-topology trainable decoder
  -> reconstructed raw e_t
  -> mean per-token cosine distance
```

This is a strong width comparison but not a clean ablation of the whole architecture bundle.
Both arms removed whitening and both moved channel down-projection after the latent processor, so
the pair cannot separately attribute the historical loss-floor change to whitening removal versus
projection timing. It also cannot compare its raw cosine values numerically with historical
whitened cosine values, because whitening changes the target distribution and the difficulty of
the loss itself.

Architecturally, M=512 still forces each 1,024-channel V-JEPA token through a `1024 -> 512` map
before global reasoning, whereas M=1024 has no structurally forced channel-rank reduction before
the final abstract projection. The latter is thus a meaningful upper-bound test. Nevertheless,
both arms ultimately compress `1024 x 1024 = 1,048,576` detailed scalars into
`32 x 256 = 8,192` abstract scalars, the same 128:1 external scalar rate, and feed the same decoder.

## 2. Reading Cycle B — interleaved verdict

Late values below are medians over the preregistered six diagnostic points at steps
12,000–14,500 unless explicitly marked final.

| Q | Question | M=512 evidence | M=1024 evidence | Interleaved interpretation |
|---|---|---|---|---|
| Q1 | Did the intended present-only task train stably? | PASS: all 15k steps; max `grad_norm=0.719`; no skips, NaNs, warnings, B clips, or D clips. | PASS: all 15k steps; max `grad_norm=1.558`; no skips, NaNs, warnings, B clips, or D clips. | Both are valid. `present_recon_only=1`, prediction off, `L_flow=0`, and `L_recon_pred=0` throughout. There is no optimizer-failure explanation for the result. |
| Q2 | Is `c_t` spread and example-specific? | FAIL on recorded batch: std `0.20490`, cosine `0.95380`, dead fraction `0`. | FAIL on recorded batch: std `0.21530`, cosine `0.94990`, dead fraction `0`. | M=1024 is marginally better, but both are far from std `0.8–1.2` and cosine `<0.5`. Because the batch is single-source, global cross-source collapse remains unresolved. |
| Q3 | Is the code high-rank with distinct slots? | FAIL: effective rank `10.7546`; raw/centered slot rank `9.9179/9.2271`. | FAIL: effective rank `15.3759`; raw/centered slot rank `13.5693/12.8873`. | Width preserves several additional directions, but both remain far below rank `>60`; neither output is a rich external representation. |
| Q4 | Did present reconstruction learn useful content? | PASS: fixed correct loss `0.99081 -> 0.30384`; late active loss `0.260785`; no decoder clipping. | PASS: fixed correct loss `1.01836 -> 0.30229`; late active loss `0.259879`; no decoder clipping. | Both reconstruct raw features strongly. The `0.00148` fixed-batch and `0.000906` active-loss advantages of M=1024 are negligible. |
| Q5 | Did reconstruction and geometry improve together? | MIXED/FAIL: final gap `0.16373`, but rank/std fall and cosine rises after their mid-run optimum. | MIXED/FAIL: final gap `0.16591`, with the same contraction, although more rank survives. | Correct-code dependence grows while geometry worsens. The objective finds a more decodable low-dimensional subspace rather than preserving broad V-JEPA information. |
| Q6 | Present-only verdict | **LOW-RANK DECODABLE; GLOBAL COLLAPSE INDETERMINATE** | **LOW-RANK DECODABLE; GLOBAL COLLAPSE INDETERMINATE** | M=1024 changes the severity, not the failure class. Neither run supports a forecasting claim. |

The dead-dimension result does not rescue Q2. `c_dead_dim_frac` uses a relative threshold based on
the code's own median std; it can remain zero when every coordinate is weak but similarly weak.
Here the absolute mean std near `0.20`, high cosine, and low covariance rank provide the stronger
geometric diagnosis.

## 3. The paired trajectory, not just the endpoints

Each cell below is `M=512 / M=1024`. Step zero rank is marked mechanical because the 32 orthogonal
learned slot identities create roughly 30 centered directions even though every example receives
the same code (`std=0`, cosine `1`, shuffled gap `0`). It is not evidence of input preservation.

| Step | Correct-code recon | Shuffled gap | Mean std | Cross-example cosine | Effective rank | Centered slot rank |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.9908 / 1.0184` | `0 / 0` | `0 / 0` | `1.000 / 1.000` | `30.00* / 29.72*` | `30.00* / 29.72*` |
| 2,000 | `0.3930 / 0.3885` | `0.0302 / 0.0348` | `0.276 / 0.323` | `0.909 / 0.878` | `18.89 / 21.59` | `14.39 / 15.22` |
| 4,000 | `0.3625 / 0.3579` | `0.0649 / 0.0713` | `0.350 / 0.367` | `0.862 / 0.850` | `19.05 / 22.36` | `13.13 / 14.75` |
| 7,500 | `0.3330 / 0.3328` | `0.1095 / 0.1121` | `0.308 / 0.348` | `0.896 / 0.868` | `15.13 / 21.11` | `11.21 / 14.66` |
| 9,000 | `0.3219 / 0.3217` | `0.1300 / 0.1317` | `0.278 / 0.312` | `0.915 / 0.895` | `13.64 / 19.45` | `10.56 / 14.23` |
| 12,000 | `0.3083 / 0.3072` | `0.1555 / 0.1578` | `0.220 / 0.235` | `0.947 / 0.940` | `11.28 / 16.17` | `9.49 / 13.19` |
| 14,500 | `0.3038 / 0.3023` | `0.1637 / 0.1659` | `0.199 / 0.207` | `0.956 / 0.954` | `10.57 / 15.06` | `9.13 / 12.76` |

### Phase 1 — a template-first opening during warmup

From initialization to step 2,000, fixed-batch reconstruction improves by about `0.598` for M=512
and `0.630` for M=1024, yet the shuffled-code gaps reach only `0.030` and `0.035`. The
reconstruction weight is ramping to full strength over exactly this interval, and the decoder can
rapidly learn shared raw-feature and positional structure before the bottleneck has developed much
example-specific variation. This does not mean the code is unused—the gap is already opening and
std is rising—but most of the early loss reduction is not abolished by rolling codes within the
recorded source.

M=1024 begins separating examples somewhat faster: at step 2,000 its std is `0.323` versus `0.276`,
its cosine is `0.878` versus `0.909`, and its rank is `21.59` versus `18.89`. This is the first
evidence that the complete width hyperparameter is functionally active rather than merely adding
unused parameters.

### Phase 2 — maximum spread around steps 3,500–4,500

Both representations reach their best separation early in the fully warmed objective. M=512 peaks
at std `0.34993` and minimum cosine `0.86215` at step 4,000; M=1024 peaks at std `0.37228` and
minimum cosine `0.84741` at step 4,500. The wider arm is consistently healthier here, but the
absolute geometry is still poor: even at the best point, the examples remain strongly aligned and
effective rank is only around 19–22.

This matters because the late collapse is not an inability to open the input-dependent branches.
The zero-initialized residual outputs do wake up, attention becomes selective, std rises, and the
codes separate. The architecture can route input dependence; the unregularized objective simply
does not preserve the broadest geometry it briefly discovers.

### Phase 3 — reconstruction improves through geometric contraction

After the mid-run optimum, correct reconstruction continues falling and the shuffled gap continues
growing, but every principal geometry metric reverses. From peak to the final diagnostic, std loses
`43.2%` of its maximum in M=512 and `44.4%` in M=1024. Effective rank loses `53.0%` of its
post-initialization maximum in M=512 and `34.2%` in M=1024; centered slot rank loses `58.5%` and
`37.2%`, respectively. Cross-example cosine rebounds by `0.094` and `0.106` from its best value.

The same antagonism appears continuously, not just in selected endpoints. Across post-warmup
diagnostic points, correct reconstruction loss and effective rank have Pearson correlations of
`+0.97` (M=512) and `+0.89` (M=1024): as loss falls, rank falls with it. Correct loss and
cross-example cosine correlate at `-0.83` and `-0.86`: as reconstruction improves, examples become
more directionally aligned. These are trajectory summaries rather than independent causal samples,
but they clearly identify the equilibrium the optimizer is approaching.

The interpretation is not that reconstruction stops using the input. In fact, the correct-minus-
shuffled advantage grows monotonically. Instead, a small set of directions becomes increasingly
useful to the decoder while other directions disappear because the objective assigns them no
benefit and all explicit geometry weights are zero. The result is input-dependent compression,
but not broad information preservation.

## 4. What the reconstruction numbers actually prove

The preregistered late-window comparison is:

| Metric, steps 12,000–14,500 | M=512 | M=1024 | 1,024 minus 512 |
|---|---:|---:|---:|
| Active training `L_recon` | `0.260785` | `0.259879` | `-0.000906` |
| Fixed correct-code loss | `0.305063` | `0.303584` | `-0.001478` |
| Fixed shuffled-code loss | `0.466626` | `0.467176` | `+0.000550` |
| Correct-versus-shuffled gap | `0.161563` | `0.163592` | `+0.002029` |
| Conditioned share | `23.2486%` | `23.4905%` | `+0.2419 pp` |

The conditioned share is computed per diagnostic point as
`(L_shuffled - L_correct) / (1 - L_correct)` and then median-aggregated. It asks how much of the
improvement below a random-cosine loss of 1 disappears when the code is rolled. Roughly 23.3–23.5%
does disappear, so these are not input-independent decoders. However, roughly three quarters of the
gain survives a roll within this validation batch. That surviving part can include decoder/position
priors, feature directions shared across EGO4D, and genuine content shared by adjacent chunks of
the same source; the current probe cannot decompose those possibilities.

M=1024 misses the preregistered requirements of at least `0.01` lower late active loss and at least
`0.005` more shuffled gap, although its conditioned share moves in the required non-decreasing
direction. The observed improvements are only about 9% and 41% of those two thresholds. Because
there is only one seed per width, it would also be unjustified to treat millesimal differences as
high-confidence generalization effects; the predeclared materiality rule correctly prevents that.

The strong raw loss is still scientifically useful. It shows that this architecture bundle can
break the historical numerical `0.6–0.7` region when reconstructing unwhitened V-JEPA features.
It does **not** show that the bundle preserves more total information than the historical whitened
runs, because the target spaces differ. Whitening deliberately equalizes low-variance directions
and makes cosine reconstruction pay attention to content the raw objective can largely ignore;
removing it makes dominant raw directions easier to reconstruct. The honest positive claim is
therefore “much lower raw-feature distortion with a nonzero within-source code-dependence gap,”
not “twice as much V-JEPA information preserved.”

## 5. Geometry: what width helps and what it does not fix

At the final diagnostic, the dormant variance penalty is `0.73494` for M=512 and `0.72996` for
M=1024. Since `L_var` is the mean shortfall below a target std of 1, these large values quantify how
far the code remains from the intended spread; they are logged but contribute no gradient because
`lambda_var=0`. Similarly, the dormant covariance penalty grows to `33.54` for M=512 and `24.94`
for M=1024. The wider processor produces a meaningfully less correlated external code, but both
are far from decorrelated, and `lambda_cov=0` gives the optimizer no reason to reduce those values.

The rank difference is the clearest real benefit of width. M=1024's late effective rank is
`15.38` versus `10.75`, a 43% relative increase, and its late centered slot rank is `12.89` versus
`9.23`, a 40% increase. It also loses less of its peak rank during contraction. Thus, carrying all
1,024 channels through global read/compete/refine computation does let more directions survive the
final `1024 -> 256` map. Width is not irrelevant; it is simply not the binding lever for the
reconstruction or code-conditioning objective at the current external rate.

Attention does not look like a near-uniform averaging failure. Final normalized attention entropy
is `0.5749` for M=512 and `0.5708` for M=1024, where 1.0 would be perfectly uniform, and the
minimum per-head/slot values are `0.4566` and `0.4378`. Both bottlenecks learned selective reads,
with only a tiny width difference. Selective attention is necessary for gathering detail but is
not sufficient for a high-rank code: multiple slots can selectively read correlated evidence, and
the final objective can still retain only a small shared subspace.

The historical evidence also argues against interpreting whitening as the direct cure for high
cosine. In the controlled SSv2 sequence, Run 55 used whitening but zero geometry weights and ended
at cosine `0.824`, std `0.397`, and rank `21.48`; Run 57 kept the whitened base while activating
the variance floor (`0.5`) and covariance penalty (`0.01`, SIGReg still off), and ended at cosine
`0.0585`, std `1.117`, and rank `208.22`. The clean mechanism is that whitening changes the target
substrate and discourages dominant raw-feature shortcuts, while variance and covariance directly
apply gradients to spread and decorrelate `c_t`. Current Runs 64–65 disabled both forces, so their
late contraction is expected rather than evidence that the wider architecture failed to open.

## 6. Where the remaining limit is localized

The M=1024 arm removes the early channel-rank bottleneck: its first `1024 -> 1024` transformation
can be full-rank, and all three latent blocks operate at full V-JEPA channel width. If early
`1024 -> 512` channel destruction were the leading cause of the remaining raw reconstruction or
conditioning limit, M=1024 should have materially beaten M=512. It does not. This localizes the
dominant limit downstream of, or orthogonal to, the 512-versus-1,024 internal-width choice.

The data do not uniquely choose among those downstream explanations, but they constrain them:

- Both arms still exit through the same `32 x 256` code, so external slot/channel rate can erase
  distinctions retained inside the wider processor.
- Both optimize raw per-token cosine distance, which rewards dominant shared directions and has no
  term for preserving the full covariance spectrum of the V-JEPA embedding.
- The decoder can exploit fixed positional structure and within-source commonality, so low loss is
  not equivalent to high exact-chunk dependence.
- All geometry terms are zero, so weight decay and reconstruction are free to remove directions
  that do not immediately improve the decoder's cosine score.

M=1024 contains approximately `70.73M` bottleneck parameters versus `18.32M` for M=512, about
`3.86x` more, while W&B runtime rises from `11,473 s` to `12,321 s` (`+7.39%`). The modest
end-to-end slowdown likely reflects the frozen encoder and data path dominating wall time, but the
larger parameter/state footprint is still real. Paying that cost for a rank improvement that does
not translate into the preregistered content metrics is not justified for the next operating point.

## 7. What this pair establishes—and what it does not

It establishes that both wide, late-projection bottlenecks can learn stable, strongly improved raw
reconstruction and nonzero input dependence. It also establishes causally that M=1024 retains more
external-code geometry than M=512, but does not materially improve reconstruction or within-source
correct-code dependence. That makes M=512 the better engineering choice under the current
objective and external code rate.

It does not establish which bundled change—removing whitening or moving projection late—caused the
historical numerical loss-floor break. It does not establish global EGO4D video specificity,
because the fixed diagnostic batch contains one source UID. It does not measure forecasting,
because the future and coarse-flow paths are inactive. It also does not prove that the retained
directions are semantically or temporally useful, and one seed per width is insufficient to assign
importance to the very small loss differences beyond the preregistered decision rule.

## Conclusions

Runs 64 and 65 reach the same scientific regime: stable and genuinely code-dependent raw-feature
reconstruction, but through a contracted external representation rather than broad information
preservation. M=1024 softens that contraction and retains materially more rank, proving that
internal width affects geometry, yet the nearly identical reconstruction and shuffled-code gap
show that width above 512 is not the binding content bottleneck. The paired verdict is therefore
**low-rank decodable; global collapse indeterminate**, with M=512 selected on effect versus cost.

## Next steps

1. Re-score both saved checkpoints on a source-unique EGO4D diagnostic batch so cosine and shuffled-code dependence become genuinely cross-source measurements.
2. Carry M=512 forward with `lambda_var=0.5` and `lambda_cov=0.01` (SIGReg/slot still off) to test whether healthy geometry can coexist with the newly low raw reconstruction loss.
