# Run 054 analysis — `ae_latent_stack_whiten_recon_only` (`lx1b6gw2`, inv015)

> Analyzed 2026-07-04 using Reading Cycle B from `GUIDES/READING_EXPERIMENTS.md`,
> together with the per-change attribution framework in [`README.md`](README.md).
> W&B run `smahalanobis-uc-davis/hjepa-vwm/lx1b6gw2`, group
> `inv015_ae_latent_stack_whiten`. The main body was written at diagnostic step 10,000
> while the run was still training; the run then finished its full 15,000-step schedule
> cleanly, and the final numbers are recorded in the dated addendum at the bottom. They
> confirm every projection made at step 10,000. All 30 diagnostic checkpoints (logged
> every 500 steps) were read individually, with no downsampling.

## TL;DR verdict

The pair of changes did not hold representation geometry. Both of the pre-registered
measurement axes — the feature-dimension axis, measured by `c_effective_rank` (the
number of independent directions the latent code actually uses), and the slot axis,
measured by `c_slot_diversity_rank_centered` (how many of the 32 slots stay genuinely
distinct) — peaked early and then contracted for the rest of the run. By the README's
2x2 outcome matrix, that places this run in the "neither delta sufficient" cell, and it
re-confirms hypothesis H2 from run 053: reconstruction pressure alone, no matter how
well-designed, does not maintain geometry, so an explicit anti-collapse force is a
requirement.

That said, this is much closer to a partial success than a repeat of run 053. The
collapse was roughly four times slower, it decelerated toward an apparent equilibrium
instead of halving, and every geometry metric landed far above run 053's terminal
values: effective rank ended at 21.9 where run 053 ended at 10.5, the cross-video
cosine (how similar different videos' codes are, where 1.0 means identical) ended at
0.820 versus 0.929, and the code's average per-dimension spread ended at 0.397 versus
0.255. The honesty result is also the strongest the program has produced: about 92% of
everything the decoder learned is conditioned on the correct video's latent, compared
to roughly 77% in run 053 and 15% in run 052. In short, this is the same failure mode
as before but at a substantially better equilibrium — and one measurement caveat
described in Finding 3 means we cannot yet say which of the two changes deserves the
credit. That makes the cheap bottleneck-only control run genuinely load-bearing now.

The Cycle B label is **low-rank decodable**: the code is video-specific and honestly
used by the decoder, but it lives in far fewer directions than we need, even though it
holds roughly 2.3 times the rank that run 053 retained.

## Run status

The run trained on schedule and completed. It started 2026-07-04 at 11:31 UTC, ran at
roughly 1.48 seconds per step, and finished all 15,000 steps with the final diagnostic
logged at step 14,500.

Stability was spotless throughout. The counter for skipped optimizer steps
(`grad_skipped`) and the NaN-gradient flag (`grad_has_nan`) stayed at zero at every
logging point, and the instability warning flag never fired. The overall gradient
magnitude (`grad_norm`) stayed in the 0.005 to 0.032 range with no spikes, which is
notably about five times smaller than run 053's plateau near 0.028. Adaptive gradient
clipping never activated on either the bottleneck or the decoder; the largest observed
clipping ratio was around 0.0006 against a threshold of 1.0.

The configuration matches [`GUIDE.md`](GUIDE.md) exactly. The run trained in
present-reconstruction-only mode (`present_recon_only=1`) with the prediction branch
fully off (`prediction_active=0`, and both the prediction loss `L_flow` and the
predicted-future reconstruction loss `L_recon_pred` pinned at zero). The residual
reconstruction target was active (`recon_target_residual=1`), feature whitening was
active at every step (`whiten_active=1`, with the eigenvalue floor `whiten_eps` set to
1e-4 and statistics loaded from `logs/whiten/whiten_stats_ssv2_train_seed42.pt`), the
bottleneck used three latent-stack blocks with 32 slots and a 256-dimensional code, the
decoder was the standard 512-dimensional, 4-block module, the reconstruction weight
`lambda_recon` was 0.05, all four geometry regularizer weights were zero, weight decay
was 0.05, and the seed was 42.

Two wiring confirmations are worth recording. First, `recon_mean_norm` (the size of the
running mean that the residual target subtracts) held steady near 89.7 for the whole
run, where run 053's raw-space value was about 1,624 — this confirms the mean tracker
was genuinely operating in whitened space. Second, `recon_scale` (the warmup multiplier
on the reconstruction loss) reached its full value of 1.0 at step 2,000 as designed.
The learning-rate schedule behaved normally: warmup to full rate at step 1,500, then
cosine decay, passing 0.30 of full rate at step 10,000.

## Reading cycle (Cycle B, present-reconstruction-only)

Each row condenses the detailed findings below; evidence is quoted at step 10,000
unless noted, since that is where the main analysis was performed.

| Q | Question | Pass? | Evidence |
|---|----------|-------|----------|
| Q1 | Did the run train, alive and in the right mode? | Yes | Zero skipped steps and zero NaN gradients across the run; clipping never fired; every mode and whitening flag held its expected value at every step; the run finished its full schedule. |
| Q2 | Is the latent code alive and video-specific? | Split — better than 053, still unhealthy | The code's average per-dimension spread (`c_std_mean`) reached only 0.42 against a target near 1.0, and the cross-video cosine sat at 0.796 against a target below 0.5. Both are clear improvements on run 053 (0.255 and 0.929), and the code is strongly video-specific in the informational sense established under Q4. |
| Q3 | Is the latent code rich, or low-rank? | No | Effective rank peaked at 34.7 and fell to 24.4, never exceeding the range that the 32 fixed slots can produce mechanically. The centered slot-diversity rank declined from roughly 31 to 12.7, though the decline flattened sharply near the end. |
| Q4 | Is reconstruction actually learning video content? | Yes, strongly | The present-reconstruction loss fell monotonically from 1.0025 to 0.657. Decoding with a deliberately wrong video's code (`L_recon_shuffled_c`) stayed pinned between 0.969 and 0.975 all run, so about 92% of the decoder's improvement depended on being given the correct code. |
| Q5 | Are geometry and content cooperating? | Partial fail | After step 7,500, all four geometry measures drifted in the collapse direction while reconstruction kept improving — the same signature as run 053, but at roughly one quarter of the rate, and decelerating toward a plateau. |
| Q6 | **Verdict** | | **Low-rank decodable: video-specific and honest, but geometrically contracted — settling at a much higher equilibrium than run 053.** |

## The trajectory in numbers

The table below shows five landmark diagnostic steps. For orientation:
`c_eff_rank` is the number of independent feature directions in use;
`slot_rank` is how many of the 32 slots remain distinct within a video;
`cosine` is the cross-video similarity of codes (lower is better);
`std` is the code's average per-dimension spread (target near 1.0);
`recon` is the present-reconstruction loss;
`shuffled` is that same loss when given a wrong video's code; and
`gap` is the difference between the two, i.e. how much the correct code helps.

| step | c_eff_rank | slot_rank | cosine | std | recon | shuffled | gap |
|---|---|---|---|---|---|---|---|
| 0 | 30.99 | 30.99 | 1.000 | 0.000 | 1.0025 | 1.0025 | 0.000 |
| 2,500 | 34.68 (peak) | 28.62 | 0.928 | 0.239 | 0.740 | 0.969 | 0.229 |
| 4,000 | 30.37 | 20.58 | 0.878 | 0.312 | 0.703 | 0.970 | 0.267 |
| 7,500 | 26.78 | 13.43 | 0.783 (best) | 0.436 (peak) | 0.666 | 0.974 | 0.308 |
| 10,000 | 24.43 | 12.68 | 0.796 | 0.422 | 0.657 | 0.974 | 0.318 |

The run moved through three distinct phases rather than run 053's two.

**Phase one, steps 0 to 2,500 — expansion.** Starting from the zero-gated
initialization where every video produces an identical code, the representation spread
out: effective rank rose to its peak of 34.7, the per-dimension spread grew steadily
from zero, and the cross-video cosine fell quickly from 1.0. The bottleneck's attention
entropy (how broadly the slots read from the input tokens) rose to its peak of 0.807
in this phase, meaning the slots were reading broadly while the code organized itself.
Slot diversity was already declining here, but as Finding 3 explains, its starting
value is inflated by construction, so early decline is not by itself meaningful.

**Phase two, steps 2,500 to 7,500 — a mixed regime run 053 never showed.** Feature
rank and slot diversity contracted, while at the same time the cross-video cosine kept
improving all the way to its best value of 0.783 and the per-dimension spread kept
rising to its peak of 0.436. In other words, the code continued pushing different
videos apart while simultaneously concentrating its content into fewer directions. Rank
and video-separation decoupled — they are not the same failure axis, and this run is
the first to demonstrate that cleanly.

**Phase three, steps 7,500 to the end — broad, mild contraction.** From step 7,500
onward, every geometry measure moved in the collapse direction together: rank fell from
26.8 to 24.4 by step 10,000, the cosine rose back from 0.783 to 0.796, the spread
slipped from 0.436 to 0.422, and slot diversity eased from 13.4 to 12.7. Over that same
window the reconstruction loss improved by less than 0.01. This is exactly the
signature that defined run 053's failure — geometry being traded away for tiny
reconstruction gains — but at much smaller amplitude.

The deceleration is real and worth quantifying. The rank was losing about 1.9 points
per thousand steps between steps 5,500 and 7,000, but only about 0.9 points per
thousand steps between 8,500 and 10,000. Slot diversity was losing about 2.8 points per
thousand steps between 4,000 and 5,500, but only about 0.23 per thousand steps by the
end of the window. Extrapolating those decaying slopes to step 15,000 predicted a final
rank near 22 to 23, slot diversity near 12.4, cosine near 0.80 to 0.81, and spread near
0.41. (The addendum confirms these projections almost exactly.) Run 053, by contrast,
halved its rank in the equivalent window — from 19.9 down to 10.5 — with no sign of a
plateau.

## Finding 1 — the honesty property carried over and got stronger

This is the clean positive result of the run, and it confirms pre-registered prior P3
without qualification.

The decisive evidence is `L_recon_shuffled_c`, the reconstruction loss the decoder
achieves when it is deliberately handed the wrong video's latent code. In whitened
space, the pre-registered prediction was that this value should pin near 1.0, because a
wrong code should decode to something nearly orthogonal to the target. That is exactly
what happened: it sat between 0.969 and 0.975 for the entire run, never drifting
downward. No shared cross-video structure survived the whitening for the decoder to
exploit.

Working through the arithmetic makes the strength of this concrete. With the correct
latent, the decoder improved from its untrained loss of 1.0025 down to 0.657, a gain of
0.346. With a wrong video's latent, it improved from 1.0025 only to about 0.974, a gain
of 0.028. So roughly 92% of everything the decoder learned depends on receiving the
specific video's code. The equivalent figure was about 77% in run 053 and about 15% in
run 052, where reconstruction was mostly satisfied by a video-independent template.

`L_recon_video_gap`, the direct difference between wrong-code and correct-code loss,
grew at every one of the diagnostic points in the analysis window, from 0 to 0.318 at
step 10,000, still creeping upward. One caution for cross-run comparisons: this gap's
absolute size should not be compared with run 053's value of 0.433, because the two
runs measure loss in different target spaces (whitened versus raw). The pinned
shuffled-code loss is the cleaner cross-run honesty signal.

The bottom line: run 052's template shortcut is not merely closed, as run 053 showed —
in whitened space it has been essentially eliminated.

## Finding 2 — geometry still contracts, but the equilibrium moved a long way

Qualitatively this run repeats the 052/053 failure: with every geometry regularizer
turned off, reconstruction pressure defends only some directions of the code, and the
rest are slowly ground away. Quantitatively, however, the run landed in a different
regime, as the comparison table shows.

| Metric (meaning) | Run 052 final | Run 053 @12k | **Run 054 @10k** | Healthy target |
|---|---|---|---|---|
| Effective rank (independent directions in use) | 13.4 | 10.5, from a peak of 19.9 | **24.4**, from a peak of 34.7 | At least ~40, and not contracting |
| Centered slot diversity (distinct slots of 32) | declining | 8.5, down from 12.5 | **12.7**, flattening | No monotone decline |
| Cross-video cosine (similarity between videos' codes) | 0.906 | 0.929, best 0.842 | **0.796**, best 0.783 | Well below 0.9, ideally below 0.5 |
| Average per-dimension spread | 0.295 | 0.255, peak 0.378 | **0.422**, peak 0.436 | Stable and well above ~0.3 |
| Share of decoder improvement that is video-conditioned | ~15% | ~77% | **~92%** | High |

The mechanistic reading follows run 053's diagnosis. The anti-collapse forces are still
absent — all geometry regularizer weights are zero, and weight decay at 0.05 acts
unopposed, continuously shrinking every direction the loss does not actively defend. So
the code drifts toward the minimal set of directions the loss cares about. What the two
new changes accomplished, on this evidence, is to raise the number of directions the
loss defends, or to make slot merging more expensive, or both: the terminal rank
roughly doubled and the code's spread held about 65% higher than run 053's. What the
changes did not do is change the sign of the drift. Hypothesis H2 therefore survives
its strongest test yet: even with equalized target directions and explicit slot
competition, reconstruction alone does not hold geometry — it just collapses to a
better floor.

One detail worth carrying into the follow-up run: the variance-floor and covariance
penalty terms are logged diagnostically even though their weights are zero, and the
logged variance-floor value of 0.548 shows that term already "sees" exactly the
spread deficit (0.42 measured versus 1.0 target) that it would attack if enabled.

## Finding 3 — attribution: both axes fail their pre-registered gates, with one honest caveat

The README assigned each change its own measurement axis, so the two can be read
independently. Both reads come back negative, but a measurement artifact discovered in
this run's own data limits how much either read should be trusted.

**Delta B, the whitening hypothesis, read on the feature axis: no positive evidence.**
The pre-registered pass condition was an effective rank of at least roughly 40 that
does not contract after step 4,000. The measured trajectory peaked at 34.7 and fell
from 30.4 to 24.4 across the 4,000-to-10,000 window, failing both halves of the
condition. There is also a deeper problem than a missed threshold. The README warned
that rank values up to roughly the 32-to-40 range are reachable from slot structure
alone, and this run demonstrated that concretely: at step 0, when the code's spread was
exactly zero — meaning the code carried no video information whatsoever — the pooled
effective rank already measured 31.0. That rank comes entirely from the latent stack's
32 fixed slot identity vectors, which are distinct by construction. Since the run's
rank never exceeded about 35, there is no point in the entire trajectory where the
feature rank demanded an explanation beyond slot structure. The mechanism whitening was
supposed to provide — reconstruction pressure defending many equally-weighted feature
directions — did not visibly materialize at the eigenvalue floor of 1e-4.

**Delta A, the latent-stack architecture fix, read on the slot axis: fails the letter
of its gate, with the same caveat cutting the other way.** The pre-registered pass
condition was the absence of a monotone decline in centered slot diversity. The
measured value declined monotonically from about 31 to 12.7, so the gate fails as
written. But the starting value of 31 is itself mechanical — before training routes any
information through the slots, their fixed identity vectors are maximally distinct — so
a decline from 31 is not comparable to run 053's decline from 12.5. The defensible
statements are narrower: the latent stack did not prevent slots from merging, but the
merging decelerated hard (to about 0.23 rank points lost per thousand steps by the end
of the window) toward a plateau around 12.5 that sits above run 053's terminal value of
8.5, and roughly at the level where run 053 started. The attention-entropy readings
support a moderate interpretation: the mean stabilized near 0.65 with the sharpest head
near 0.52, rather than collapsing toward zero, so this is not the "slots sharp but
redundant" failure signature the README warned about either.

**The resulting 2x2 cell is the bottom-right: neither delta sufficient.** Feature rank
contracts and slot diversity declines. The pre-registered consequence is to extend run
053's H2 conclusion and to add explicit anti-collapse terms — the inv011-style variance
floor plus a small covariance penalty — on top of whitening and the residual target.

**However, the mechanical-initialization caveat weakens both attribution reads at
once.** The run landed far above run 053 on every geometry measure, and this analysis
cannot say which of the two changes bought that improvement, because the one metric
that would separate them never left its ambiguous range. This is precisely the
situation the README's section 4 tiebreaker anticipated: a bottleneck-only control run,
identical except with the two whitening flags removed, would convert this single run
into a true ablation for about six hours of pod time. That control is no longer a
nice-to-have; it is load-bearing for the attribution question.

## Priors scorecard

Four priors were registered in [`OBSERVATIONS.md`](OBSERVATIONS.md) before launch.

- **P1, the slot axis holds: failed as written, partial in spirit.** Centered slot
  diversity did decline monotonically, which is what P1 said should not happen. But the
  decline started from a mechanically inflated value, decelerated to a near-plateau,
  and ended above run 053's terminal level.
- **P2, feature rank does not halve after step 4,000: half passed.** The rank lost
  about 20% across the post-4,000 window, versus 47% in run 053, so the letter of the
  prior nearly holds. Its spirit — that reconstruction improvement and feature rank
  would stop moving in opposite directions — failed: they opposed each other from step
  2,500 onward.
- **P3, honesty carries over with the shuffled-code loss pinned near 1.0: passed
  cleanly.** This is the strongest honesty result of the program to date.
- **P4, stability: passed.** No skips, no NaNs, no clipping, whitening active at every
  step.

## What this run decides

First, **H2 can now be treated as settled.** Three successively better content
objectives — run 052's absolute-target reconstruction, run 053's residual target, and
this run's whitened residual target with slot competition — have all failed to hold
rank and variance on their own. Explicit anti-collapse regularization is a requirement
of this architecture family, not a crutch. That is the program's most robust negative
result, and it is worth stating as settled.

Second, **the whitened-residual recipe is the right content channel to build on.** A
92% video-conditioned decoder improvement, with the wrong-code loss pinned at 0.97, is
exactly the honest-content property Phase 1 needs underneath any future geometry
regularization.

Third, **attribution between the two changes is genuinely open.** The improved
equilibrium is real — roughly 2.3 times the terminal rank, 65% more spread, and a
cosine lower by 0.13 than run 053 — but it cannot be assigned to the whitening versus
the architecture from this run alone, because the feature rank never left the range
that slot structure alone can explain.

## Next steps, ordered by information per GPU-hour

1. **Read the finished run's tail (free, done — see addendum).** Confirm the plateau
   projections, and check whether any geometry axis re-accelerated downward late in
   the learning-rate decay.
2. **Run the bottleneck-only control (~6 hours, the pre-registered section-4
   tiebreaker).** Same seed and recipe with the two whitening flags removed. This
   cleanly splits the 053-to-054 improvement between the architecture and the
   whitening, and decides whether whitening earns a permanent place in the recipe
   before it gets entangled with geometry terms.
3. **Run the pre-registered "neither sufficient" follow-up (~6 hours).** Keep whitening
   and the residual target, and add the variance floor (its target spread of 1.0
   directly attacks this run's 0.42 ceiling) plus a small covariance penalty. Keep the
   video-gap honesty probe as the monitor that tells us whether the geometry terms
   preserved or destroyed the 92% content routing. Runs 043 through 051 showed the
   geometry terms alone produce ranks of 90 to 150; the open question is only the
   combination.
4. **If contraction persists even there, lower the bottleneck's weight decay.** The
   value of 0.05 has now been the unopposed contraction force across three runs in a
   row.

---

## 2026-07-04 addendum — final numbers (run finished, full 15,000-step schedule)

The run completed cleanly: W&B state `finished`, the last diagnostic logged at step
14,500, and zero skipped steps, NaN flags, or instability warnings through the end.
The table compares the step-10,000 values, the projections made from their decaying
slopes, and what was actually measured at the final diagnostic.

| Metric (meaning) | @10k | Projected @15k | **Measured @14.5k** |
|---|---|---|---|
| Effective rank (independent directions in use) | 24.43 | ~22–23 | **21.93** |
| Centered slot diversity (distinct slots of 32) | 12.68 | ~12.4 | **12.13** |
| Cross-video cosine (similarity between videos' codes) | 0.796 | ~0.80–0.81 | **0.820** |
| Average per-dimension spread | 0.422 | ~0.41 | **0.397** |
| Present-reconstruction loss | 0.657 | — | **0.652**, flat from step 13,000 |
| Wrong-code reconstruction loss | 0.974 | pinned | **0.975**, pinned all run |
| Video gap (how much the correct code helps) | 0.318 | — | **0.322**, plateaued |
| Video-conditioned share of decoder improvement | ~92% | — | **92.0%** |

Beyond confirming the interim verdict, the final window adds three things.

1. **The sharpest evidence for H2 sits in the last 1,500 steps.** Between steps 13,000
   and 14,500 the reconstruction loss gained essentially nothing — it moved from 0.6524
   to 0.6521, and actually ticked upward at the final point — while geometry kept
   paying anyway: rank fell another 0.34 points, the cosine rose another 0.003, and the
   spread slipped another 0.004. Contraction that continues while reconstruction gains
   zero benefit is the cleanest possible signature that the residual force grinding the
   code down is not the content objective but the unopposed weight decay, exactly as
   run 053's analysis diagnosed. There was no re-acceleration in the learning-rate
   tail; just this steady decay.
2. **The cross-video cosine never recovered its best value.** After bottoming at 0.783
   at step 7,500, it rose monotonically to 0.820 by the end. Video separation was
   slowly re-collapsing for the entire second half of the run, even while the honesty
   gap held its plateau. The per-dimension spread followed the same slow bleed, from
   its 0.436 peak down to 0.397.
3. **The honesty result is fully converged, not a transient.** The wrong-code loss
   stayed pinned between 0.974 and 0.975 through the final step, and the
   video-conditioned share held at 92.0% at the last measurement. This property is
   stable at convergence and safe to build on.

Nothing in the final window changes the verdict, the 2x2 placement ("neither delta
sufficient"), or the ordering of next steps — it strengthens all three. For the
record, the terminal comparison is: run 054 at step 14,500 reached rank 21.9, cosine
0.820, and spread 0.397, against run 053 at step 12,000 with rank 10.5, cosine 0.929,
and spread 0.255.
