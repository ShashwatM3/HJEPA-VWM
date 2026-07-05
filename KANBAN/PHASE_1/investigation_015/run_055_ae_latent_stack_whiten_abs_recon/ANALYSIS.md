# Run 055 analysis — `ae_latent_stack_whiten_abs_recon` (`nzz64pl6`, inv015)

> Analyzed 2026-07-05 using Reading Cycle B from `GUIDES/READING_EXPERIMENTS.md`,
> together with the honesty axis registered in this run's [`DESCRIPTION.md`](DESCRIPTION.md)
> and [`OBSERVATIONS.md`](OBSERVATIONS.md) and the parent framing in
> [`../ANALYSIS_054.md`](../ANALYSIS_054.md). W&B run
> `smahalanobis-uc-davis/hjepa-vwm/nzz64pl6`, group `inv015_ae_latent_stack_whiten`.
> The run finished its full 15,000-step schedule cleanly; all 30 diagnostic checkpoints
> (logged every 500 steps) were read individually, with no downsampling, and every
> reconstruction and geometry curve was overlaid against run 054 (`lx1b6gw2`), whose
> analysis this run was designed to complete.

## TL;DR verdict

This run is the absolute-target arm of the 2x2 honesty design, and it settles the
question the design was built to answer: in whitened feature space, the residual
reconstruction target is no longer doing the heavy lifting, but it is not redundant
either. With the residual target removed and everything else held identical to run 054,
the decoder recovered a small video-independent template that whitening alone leaves
open, and the share of the decoder's improvement that genuinely depends on receiving the
correct video's code fell from run 054's 92% to about 86%. That is a real honesty loss,
but a modest one, and it is nothing like the raw-space collapse of run 052, where only
about 15% of the improvement was video-conditioned. Geometry, by contrast, is
essentially indistinguishable from run 054: the same expansion-then-contraction arc, the
same terminal effective rank near 21.5, the same cross-video cosine near 0.82, and the
same per-dimension spread near 0.40. The Cycle B label is therefore the same as run
054's — **low-rank decodable** — and the practical decision is clear: keep
`--recon-residual-target` in the recipe, because it buys back roughly six points of
honesty for no geometric cost and negligible compute.

## Run status

The run trained on schedule and completed. It started 2026-07-05 at 01:36 UTC and
finished all 15,000 steps, with the last diagnostic logged at step 14,500 and the final
training row at step 14,950. Stability was spotless throughout. The counter for skipped
optimizer steps (`grad_skipped`) and the NaN-gradient flag (`grad_has_nan`) stayed at
zero at every logging point, and the instability warning never fired. The overall
gradient magnitude (`grad_norm`) stayed inside the 0.000 to 0.041 range with no spikes,
in the same small regime run 054 occupied. Adaptive gradient clipping never bit on
either the bottleneck or the decoder; the largest observed clipping ratio was about
0.0006 on the bottleneck and 0.0016 on the decoder, both against a threshold near 1.0.

The configuration matches [`GUIDE.md`](GUIDE.md) exactly, and the run-defining wiring
flags all held their expected values at every diagnostic step. The run trained in
present-reconstruction-only mode (`present_recon_only=1`) with the prediction branch
fully off (`prediction_active=0`, and both `L_flow` and the predicted-future
reconstruction loss `L_recon_pred` pinned at zero all run). Feature whitening was active
at every step (`whiten_active=1`, eigenvalue floor `whiten_eps=1e-4`, statistics loaded
from the same `logs/whiten/whiten_stats_ssv2_train_seed42.pt` file run 054 used). The
two flags that define this run against run 054 read correctly throughout: the residual
reconstruction target was off (`recon_target_residual=0`) and the running mean the
residual target would subtract was inactive, so its logged norm (`recon_mean_norm`) held
at exactly 0.00 for the whole run rather than run 054's whitened-space value of about
89.7. The bottleneck used three latent-stack blocks with 32 slots and a 256-dimensional
code, the decoder was the standard 512-dimensional, 4-block module, the reconstruction
weight `lambda_recon` was 0.05, all four geometry regularizer weights were zero, weight
decay was 0.05, and the seed was 42. The reconstruction warmup multiplier (`recon_scale`)
reached its full value of 1.0 at step 2,000 as designed. In short, this is run 054 with
exactly one mechanism removed, and the removal is confirmed in the logged flags.

## Reading cycle (Cycle B, present-reconstruction-only)

Each row condenses the detailed findings below; evidence is quoted at the final
diagnostic step (14,500) unless noted.

| Q | Question | Pass? | Evidence |
|---|----------|-------|----------|
| Q1 | Did the run train, alive and in the right mode? | Yes | Zero skipped steps and zero NaN gradients across the run; clipping never fired; `whiten_active=1`, `recon_target_residual=0`, `recon_mean_norm=0`, `present_recon_only=1`, `prediction_active=0`, `L_flow=0`, `L_recon_pred=0` all held at every diagnostic step; the run finished its full schedule. |
| Q2 | Is the latent code alive and video-specific? | Split — same regime as 054 | The code's average per-dimension spread (`c_std_mean`) reached only 0.397 against a target near 1.0, and the cross-video cosine (`c_cross_video_cosine`, similarity between different videos' codes) ended at 0.824 against a target below 0.5. Dead dimensions stayed at zero. The code is strongly video-specific in the informational sense established under Q4, but weak on both amplitude and separation, essentially matching run 054. |
| Q3 | Is the latent code rich, or low-rank? | No | Effective rank (`c_effective_rank`, independent feature directions in use) peaked at 33.9 and fell to 21.5, never exceeding the range the 32 fixed slots produce mechanically. The centered slot-diversity rank (`c_slot_diversity_rank_centered`, distinct slots of 32) declined from about 31 to 12.1, with the decline flattening hard near the end. |
| Q4 | Is reconstruction actually learning video content? | Yes, but with a small template leak | The present-reconstruction loss (`L_recon_present`) fell monotonically from 1.004 to 0.642. Decoding with a deliberately wrong video's code (`L_recon_shuffled_c`) dipped to 0.940 early and then settled at 0.953, so about 86% of the decoder's improvement depended on being given the correct code, down from run 054's 92%. |
| Q5 | Are geometry and content cooperating? | Partial fail | After the early expansion, every geometry measure drifted in the collapse direction while reconstruction kept improving — the same signature as run 054, at the same amplitude, decelerating toward the same plateau. |
| Q6 | **Verdict** | | **Low-rank decodable: video-specific and largely honest, but geometrically contracted to the same equilibrium as run 054.** |

## The trajectory in numbers

The table below shows five landmark diagnostic steps. For orientation: `c_eff_rank` is
the number of independent feature directions in use; `slot_rank` is how many of the 32
slots remain distinct within a video; `cosine` is the cross-video similarity of codes
(lower is better); `std` is the code's average per-dimension spread (target near 1.0);
`recon` is the present-reconstruction loss; `shuffled` is that same loss when the
decoder is handed a wrong video's code; and `gap` is the difference between the two,
which is how much the correct code helps.

| step | c_eff_rank | slot_rank | cosine | std | recon | shuffled | gap |
|---|---|---|---|---|---|---|---|
| 0 | 30.99 | 30.99 | 1.000 | 0.000 | 1.004 | 1.004 | 0.000 |
| 2,500 | 33.86 (peak) | 29.27 | 0.950 | 0.210 | 0.724 | 0.940 (min) | 0.217 |
| 8,000 | 26.90 | 13.88 | 0.794 (best) | 0.430 (peak) | 0.653 | 0.951 | 0.298 |
| 10,000 | 24.80 | 13.13 | 0.802 | 0.421 | 0.647 | 0.953 | 0.306 |
| 14,500 | 21.48 | 12.08 | 0.824 | 0.397 | 0.642 | 0.953 | 0.311 |

The run moved through the same three phases run 054 did. In the expansion phase from
step 0 to about step 2,500, the representation spread out from the zero-gated
initialization where every video produces an identical code: the effective rank rose to
its peak of 33.9, the per-dimension spread grew steadily from zero, the cross-video
cosine fell from 1.0, and the bottleneck's attention entropy rose to its peak of 0.814,
meaning the slots were reading broadly while the code organized itself. In the mixed
phase from about step 2,500 to step 8,000, feature rank and slot diversity contracted
while the cross-video cosine kept improving all the way to its best value of 0.794 and
the per-dimension spread kept rising to its peak of 0.430; the code continued pushing
different videos apart while concentrating its content into fewer directions, exactly
the decoupling run 054 was the first to show. From step 8,000 onward, in the broad
contraction phase, every geometry measure moved in the collapse direction together while
reconstruction gained almost nothing, which is the clean signature of unopposed weight
decay grinding down every direction the loss does not actively defend.

## Finding 1 — honesty: the absolute target reopened a small template channel, exactly as predicted

This is the decisive result of the run, and it lands squarely in the intermediate
outcome that prior P1 predicted. The relevant probe is `L_recon_shuffled_c`, the
reconstruction loss the decoder achieves when it is deliberately handed the wrong
video's latent code (a deterministic roll of the batch, so it is RNG-free and decodes
another video's code against each target). In run 054, with the residual target subtracting
the per-position mean, this value pinned between 0.969 and 0.975 for the whole run,
because no shared cross-video structure survived for the decoder to exploit. In this
run, with the absolute target, it behaved differently. It fell faster and deeper early,
reaching a minimum of 0.940 around step 2,000 to 2,500 as the decoder quickly grabbed
the easy video-independent template, and then it recovered and settled at 0.953 for the
rest of the run. That recovery is itself informative: as the code became more
video-specific and the decoder committed more of its machinery to conditioning on the
correct code, feeding it a wrong code produced a slightly worse match again, so the
template contribution partially faded rather than growing.

Working through the arithmetic makes the size of the leak concrete. With the correct
latent, the decoder improved from its untrained loss of 1.004 down to 0.642, a gain of
0.362. With a wrong video's latent, it improved from 1.004 only to 0.953, a gain of
0.051. So roughly 86% of everything the decoder learned depends on receiving the
specific video's code, and the remaining 14% is a video-independent template worth about
0.051 of cosine. That honesty share was not static: it rose over training from about 77%
at step 2,500 to about 85% at step 8,000 and about 86% at the end, because the code kept
getting more video-specific even as the template contribution held roughly constant. The
comparison across the program is the clean read here: the video-conditioned share was
about 15% in run 052 (raw-space absolute target, a near-total template collapse), about
77% in run 053 (raw-space residual target), about 92% in run 054 (whitened residual
target), and about 86% in this run (whitened absolute target).

The mechanistic reading is exactly the one the DESCRIPTION registered. Whitening removes
the single global feature mean, but the per-tubelet-position mean structure survives it
— run 054 measured that surviving structure directly, its whitened-space mean-tracker
norm holding near 89.7 rather than zero. The absolute target lets the decoder learn that
surviving per-position template, because it scores against the raw whitened features
rather than the per-position residual. The residual target's entire job is to subtract
exactly that per-position mean, so removing it hands the decoder back precisely that
channel. The channel is small in whitened space, worth about 0.051 of cosine and 14% of
the decoder's improvement, but it is real and it converges rather than vanishing.

## Finding 2 — geometry is indistinguishable from run 054

Removing the residual target had no material effect on representation geometry. The two
runs' curves lie almost on top of each other for the entire schedule, and their terminal
values differ only at the level of noise.

| Metric (meaning) | **Run 055 @14.5k (abs target)** | Run 054 @14.5k (residual target) |
|---|---|---|
| Effective rank (independent directions in use) | **21.48** (peak 33.86) | 21.93 (peak 34.68) |
| Centered slot diversity (distinct slots of 32) | **12.08** | 12.13 |
| Cross-video cosine (similarity between videos' codes) | **0.824** (best 0.794) | 0.820 (best 0.783) |
| Average per-dimension spread | **0.397** (peak 0.430) | 0.397 (peak 0.436) |
| Dead-dimension fraction | **0.000** | 0.000 |
| Present-reconstruction loss | **0.642** | 0.652 |

The only differences worth naming are second-order. This run contracted marginally
faster through the middle of the run, passing effective rank 28.9 at step 5,000 where
run 054 was still at 30.4, and it bottomed its cross-video cosine slightly higher, at
0.794 against run 054's 0.783, so its best video separation was a touch worse. Both
converge to the same place. The pre-registered geometry read (P2) asked specifically
whether the equilibrium would land materially lower than run 054's terminal rank of
21.9, cosine of 0.820, and spread of 0.397, because a materially lower equilibrium would
have implied the residual target was itself contributing defended feature directions.
The equilibrium landed at 21.5, 0.824, and 0.397, which is not materially different — if
anything the absolute target's rank is a hair lower, the opposite of the residual target
adding directions. The conclusion is therefore that the residual target contributes
essentially zero defended geometry directions. Its entire value is honesty, and the
geometry contraction is driven by the unopposed weight decay in both runs alike. This
also retroactively confirms that the residual target is not a confound in the 053-to-054
geometry improvement: since removing it here changed geometry by nothing, that
improvement is properly attributable to whitening and the latent-stack architecture, not
to the residual target.

One nuance is worth carrying forward. This run's present-reconstruction loss ended
slightly lower than run 054's, at 0.642 against 0.652, which looks like better
reconstruction. It is not better content routing; it is the interpretation guard from
OBSERVATIONS made visible. The absolute target is strictly easier than the residual
target, and part of that extra reconstruction quality comes free from the template
shortcut rather than from information carried through the code. The raw reconstruction
number is better precisely because the run is less honest about how it got there, which
is why the honesty share, not the raw loss, is the metric that decides between the two
targets.

## Finding 3 — what the 2x2 honesty matrix now says, and the recipe decision

This run completes the 2x2 over reconstruction target (absolute or residual) crossed
with feature space (raw or whitened), read on the video-conditioned share of the
decoder's improvement.

| | raw space | whitened space |
|---|---|---|
| **absolute target** | run 052: ~15% honest | **run 055: ~86% honest** |
| **residual target** | run 053: ~77% honest | run 054: ~92% honest |

Reading the matrix along its edges tells the whole story. Moving from raw to whitened
space while holding the absolute target lifts honesty from about 15% to about 86%, so
whitening alone does the overwhelming bulk of the work of closing the template shortcut.
Moving from the absolute to the residual target while holding whitened space lifts
honesty only from about 86% to about 92%, so in whitened space the residual target's
marginal contribution is small. The contrast with raw space is the point: there, moving
from the absolute to the residual target lifted honesty from about 15% to about 77%, so
in raw space the residual target was carrying almost the entire honesty load. In other
words, whitening and the residual target are substitutes for defending honesty rather
than complements. They both attack the same video-independent template, by different
mechanisms — whitening removes the global mean direction and equalizes the target
directions, while the residual target explicitly subtracts the per-position mean — and
once whitening has removed most of the template, the residual target is left to mop up
only the per-position remainder that whitening does not reach.

The recipe decision follows directly and matches the pre-registered verdict rule. A
six-point honesty loss (92% down to 86%) is a material loss, and the residual target
recovers it for no geometric cost, since geometry is identical with and without it, and
for negligible compute, since the mechanism is a single per-position mean tracker with
no parameters. Therefore `--recon-residual-target` stays in the recipe permanently. It
is a cheap honesty top-up that closes the last surviving template channel in whitened
space, and there is no reason to drop it. The alternative reading — that whitening alone
suffices and the residual target is redundant — is rejected, because the shuffled-code
loss did not pin near 1.0 and the honesty share did not hold at run 054's level.

## Priors scorecard

Three priors were registered in [`OBSERVATIONS.md`](OBSERVATIONS.md) before launch.

- **P1, honesty (the decisive axis): confirmed in its main-body form.** The prediction
  was that the absolute target reopens a partial template channel in whitened space, so
  the shuffled-code loss would not pin at 1.0 and the video-conditioned share would land
  clearly below run 054's 92%. Both halves held: the shuffled-code loss dipped to 0.940
  and settled at 0.953 rather than pinning near 1.0, and the share landed at about 86%.
  Neither of the two extreme sub-cases held: honesty did not hold at run 054's level
  (whitening is not fully sufficient on its own), and it did not drop toward run 052's
  15% (whitening is far from useless). The intermediate outcome the main prediction
  described is what happened.
- **P2, geometry: confirmed.** With zero geometry regularizers and weight decay of 0.05
  acting unopposed, expansion then contraction occurred regardless of the target, and
  the equilibrium landed at run 054's values rather than materially lower, so the
  residual target was not contributing defended directions.
- **P3, stability: passed.** No skips, no NaNs, no clipping of consequence, whitening
  active at every step, and the two wiring confirmations `recon_target_residual=0` and
  `recon_mean_norm=0` held throughout.

## What this run decides

First, the residual reconstruction target earns a permanent place in the whitened
recipe. It is not redundant, because whitening leaves a per-position template worth about
six points of honesty that only the residual target removes, and it is not expensive,
because it costs nothing in geometry and almost nothing in compute.

Second, the honesty defense in this architecture family is now understood
mechanistically as two substitutable attacks on one shortcut. Whitening does most of the
work by removing the global mean and equalizing directions; the residual target finishes
the job by removing the per-position mean that survives whitening. This is a cleaner
account than either run 052 or run 053 could give on its own.

Third, this run confirms once more that the geometry contraction is orthogonal to the
content objective. The same expansion-then-contraction arc appeared with and without the
residual target, at the same amplitude, converging to the same low-rank equilibrium,
which re-confirms the settled H2 conclusion that reconstruction pressure of any flavor
does not hold geometry, and an explicit anti-collapse force is required.

## What this run does not decide

This run does not attribute the 053-to-054 geometry improvement between whitening and
the latent-stack architecture. Both run 054 and this run carry the latent stack and the
whitening together, so the clean comparison here isolates the reconstruction target, not
the whitening or the architecture. The pre-registered bottleneck-only control — the same
recipe with the two `--whiten-*` flags dropped and the residual target kept, for about
six hours of pod time — remains open and load-bearing for that attribution question.
This run also says nothing about prediction: it is an autoencoder-only run, so it
neither passes nor fails the Phase 1 copy and batch-mean gates.

## Next steps, ordered by information per GPU-hour

1. **Lock the recipe decision into the record (free, done here).** The whitened residual
   recipe keeps `--recon-residual-target`; this run is the evidence that it is worth its
   place, at about six points of honesty for no geometric cost.
2. **Run the pre-registered bottleneck-only control (~6 hours).** Drop the two whitening
   flags, keep the residual target and the latent stack, same seed. This is the still-open
   tiebreaker that splits the 053-to-054 improvement between whitening and the
   architecture, and it decides whether whitening earns a permanent place before it gets
   entangled with geometry terms.
3. **Run the "neither sufficient" geometry follow-up (~6 hours).** Keep whitening and the
   residual target, and add the variance floor (its target spread of 1.0 directly attacks
   this run's 0.40 ceiling) plus a small covariance penalty, with the video-gap honesty
   probe as the monitor that tells us whether the geometry terms preserved or destroyed
   the content routing. The open question is only the combination, since the geometry
   terms alone are known to produce high ranks.
