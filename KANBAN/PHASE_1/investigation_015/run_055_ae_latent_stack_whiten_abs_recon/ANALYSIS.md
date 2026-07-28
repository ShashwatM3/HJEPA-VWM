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

---

# 2026-07-17 cross-investigation addendum — live W&B audit of the EGO4D bottleneck slot-capacity sweep

> This appendix is intentionally preserved in the path requested by the human. It does **not**
> reinterpret run 055 as part of the slot sweep. Run 055 is the SSv2, 32-slot, zero-geometry,
> absolute-whitened control analyzed above. The actual `N_c=32/64/128` sweep is the later
> investigation 016 folder
> [`bottleneck_slot_capacity_sweep`](../../investigation_016/bottleneck_slot_capacity_sweep/),
> and its canonical analysis remains
> [`ANALYSIS.md`](../../investigation_016/bottleneck_slot_capacity_sweep/ANALYSIS.md).
> This appendix gives the requested independent, deeper audit and records two corrections found
> only after inspecting the live histories, artifacts, and initialization code together.

## Executive verdict

The sweep is operationally valid and the within-seed training-loss ordering is real, but it is
**not a clean, initialization-isolated proof that slot capacity caused the improvement**.

The narrow empirical result is:

- late random-training-batch median reconstruction improves monotonically from `0.67703196` at
  32 slots, to `0.67396140` at 64, to `0.66298261` at 128;
- the 32-to-128 difference is `0.01404935`, or `2.07513837%`, below the registered `0.02` or
  `3.0%` materiality gate;
- after reconstruction warmup, the 128-slot arm is lower than the 32-slot arm on `259` of `260`
  matched logged training batches, so the late scalar ordering is not a dashboard-sampling
  accident;
- fixed-batch example specificity is **not** monotonic: late `std/cosine` is
  `0.80585822/0.47349986`, `0.32573912/0.91421551`, and
  `0.64887702/0.67727712` for 32, 64, and 128 slots;
- all fixed-batch examples are adjacent chunks from one source UID, so those values describe
  within-source exact-chunk specificity, not global cross-source collapse;
- the 64- and 128-slot arms fail the registered representation-health guard, so a 256-slot run is
  not warranted.

The deeper causal qualification is:

1. Changing `N_c` changes the number of abstract activations, but leaves the early
   1,024-to-256 per-token projection untouched. The experiment is a slot-memory test, not a
   general bottleneck-capacity test.
2. The covariance loss pools `B*N_c` slot rows. Its scale and initial geometry therefore change
   mechanically with `N_c`; at step zero `L_cov` is `6.72580338`, `2.93429971`, and
   `0.99159271`.
3. The production constructor uses one RNG stream for the bottleneck, flow, and decoder. The
   `N_c`-dependent orthogonal query initialization consumes a different amount of RNG before
   later modules are built. An exact local replay of the sweep code found `27` same-shaped
   bottleneck tensors and `30` of the decoder's `59` state tensors initialized to different
   values between the 32- and larger-slot arms.
4. There is only one seed per slot count. The 64-slot geometry anomaly can therefore be a
   slot-count/objective interaction, an initialization-basin effect, or both.

The defensible conclusion is consequently narrower than “capacity does not matter”: **under this
one-seed, fully whitened, covariance-regularized recipe, adding query slots gives a small training
reconstruction gain without producing a healthier recorded-batch content code. More slots are
not the leading next lever.** This does not rule out `D_c`, mixer/input-projection width,
whitening strength, or decoder form.

## 1. Prior stated before reading the sweep

The capacity hypothesis made three linked predictions:

1. If the 32-slot tensor is the binding information bottleneck, doubling and quadrupling `N_c`
   should reduce the honest reconstruction residual monotonically and materially.
2. The lower loss should coexist with healthy example-level spread and specificity. A larger
   code that merely stores more slot identity or template structure is not support.
3. If the real floor is the 1,024-to-256 channel squeeze, whitening, decoder form, target
   structure, or optimization, `N_c=32/64/128` should produce only a small or non-monotonic
   change.

The comparison would be considered broken if the runs differed in source code, data order,
encoder features, whitening payload, schedule, or mode. A separate limitation, discovered during
the audit, is that “same seed” did not imply equal shared parameter values across shapes.

## 2. W&B access and evidence surfaces

### Connector status

No callable W&B MCP tool, resource, or resource template is registered in this Codex session.
The connector therefore could not be validated as an MCP endpoint. The authenticated local W&B
Python API **does** work through the credentials in the local netrc, and every live result below
was fetched from `smahalanobis-uc-davis/hjepa-vwm` with a 120-second API timeout.

The required W&B project probe reported:

- step history available;
- `47` sampled scientific metric keys;
- provenance, whitening, history, and events artifacts on all three newest runs;
- no accessible Weave trace surface (`weave_trace_count=null`);
- no W&B sweep-controller object on any arm (`sweep_id=null`).

These are three sequential grouped runs, not a W&B Sweep agent:

| `N_c` | W&B ID | State | Last history step | Display name |
|---:|---|---|---:|---|
| 32 | `x03xlpyl` | finished | 14,950 | `Investigation 16 · Bottleneck capacity · EGO4D 32 slots` |
| 64 | `evyokqrm` | finished | 14,950 | `Investigation 16 · Bottleneck capacity · EGO4D 64 slots` |
| 128 | `7pmvxrxi` | finished | 14,950 | `Investigation 16 · Bottleneck capacity · EGO4D 128 slots` |

All belong to `inv016_bottleneck_capacity_vjepa2_whitened_recon1`.

### Surfaces actually inspected

For every arm, this audit inspected:

- resolved run config and summary;
- all `300` history rows at 50-step cadence;
- all `30` diagnostic checkpoints at 500-step cadence;
- `output.log` in full (`300` metric records per run);
- system telemetry;
- the downloaded provenance JSON;
- the downloaded whitening tensor payload;
- the downloaded history parquet;
- the downloaded system-events parquet;
- logged and consumed artifact inventories;
- final checkpoint path and SHA-256 recorded in the summary.

No fatal error, traceback, OOM, or warning appears in any output log. W&B contains checkpoint
paths and hashes, but **no model-checkpoint artifact**; re-evaluating the final models requires the
remote checkpoint volume or another copy outside W&B.

## 3. What `N_c` changes—and what it cannot change

The active present-only path is:

```text
EGO4D clip
  -> frozen V-JEPA2: e, shape (B, 1024, 1024)
  -> fixed full ZCA whitener
  -> Bottleneck.in_proj, independently per token: 1024 -> 256
  -> shared spatial ConvNeXt mixing + three latent blocks
  -> c, shape (B, N_c, 256)
  -> fixed-position decoder, width 512, four blocks
  -> per-token cosine reconstruction of whitened e
```

`F_c`, future features, flow matching, copy tests, and batch-mean prediction tests are inactive.
This is a feature autoencoder experiment, not a forecasting experiment.

The latent tensor changes as intended:

| `N_c` | Latent scalars/example | Detailed scalars/example | Scalar compression |
|---:|---:|---:|---:|
| 32 | 8,192 | 1,048,576 | 128:1 |
| 64 | 16,384 | 1,048,576 | 64:1 |
| 128 | 32,768 | 1,048,576 | 32:1 |

But the active trainable parameter count barely changes:

| `N_c` | B parameters | D parameters | Active B+D parameters |
|---:|---:|---:|---:|
| 32 | 4,837,123 | 14,318,080 | 19,155,203 |
| 64 | 4,845,315 | 14,318,080 | 19,163,395 |
| 128 | 4,861,699 | 14,318,080 | 19,179,779 |

The 32-to-128 active-parameter increase is only `24,576`, or `0.12829935%`. The meaningful
intervention is more activation memory and more attention positions, not a larger learned channel
map. In particular, the first 1,024-to-256 projection still discards a 768-dimensional per-token
linear complement before slot aggregation. More slots cannot directly restore information lost
there.

This matters because the downloaded whitening artifact is not mild normalization. Its raw
1,024-D channel covariance has effective rank `230.479449608`; after `eps=0.0001` ZCA, the
effective rank is `1022.99999174` and the trace is `1022.88633834`. The top 256 whitened
directions hold only `0.250269680348` of the second-order energy. The channel-only angular oracle
is therefore roughly `0.499730392341` loss before the additional 1,024-token-to-`N_c` compression
and decoder constraints. This is why the earlier architecture audit ranked whitening plus the
early channel squeeze above slot count as the likely floor mechanism.

## 4. Identity and provenance—what is genuinely controlled

The downloaded provenance artifacts differ in only six leaf values:

- `N_c` in common and resolved model config;
- `trainable_init_hash`;
- the derived common-identity hash;
- checkpoint directory;
- tracking display name.

Everything that should be common is common:

| Surface | Shared value |
|---|---|
| Git | clean `820a5b560b8668fa12452e35b1e91b170a2a91a3` |
| Dataset | EGO4D fingerprint `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c` |
| Train order | `7af099758cd119123fda6ac10b63abaafd1027b4c8e0c1a00d3fb3f0f80ada5c` |
| Validation batch | `bb4ee6e7aaf42d2215024c22c72b13d27ac0d43e8a4e89c79ee6b18cf56ed839` |
| Encoder features | `372adef61c7d90a08665faf65991c1fef1f13768cdd9a6850b23393c705e1155` |
| Whitening payload | `fd00b842eebeaa2f5a83a47b82275b39182a58c91131793fd48e3502fe73d0b6` |
| Seed | 42 |
| Mode | present-only, absolute target, whitening on |
| Losses | recon 1.0, variance 0.5, covariance 0.01, SIGReg/slot/predicted recon off |

The three downloaded whitening files are byte-identical: each is `4,207,255` bytes with local
SHA-256 `96ff86baf26354db81419faaa8bef29b22bbf4b9089055be139fc1ba6b5c0492`.
The payload contains 12,800 training clips and 13,107,200 token rows; all tensor values are finite.

Thus there is no config leakage, dataset drift, encoder drift, whitening drift, or schedule drift.
The unresolved control is the parameter initialization induced by changing tensor shape.

## 5. Interpret the metrics before reading their values

### Reconstruction

- Training `L_recon` is per-token `1-cosine` on the current deterministic random training batch.
  Because data order and transform identities are shared, same-step differences are paired-batch
  comparisons.
- `L_recon_present` is the same metric on one fixed 16-example validation batch.
- `L_recon_shuffled_c` rolls codes by one element. Here that swaps adjacent chunks from the same
  long EGO4D source, not unrelated source videos.
- `L_recon_video_gap = shuffled - correct` is therefore an exact-chunk/within-source gap.

### Geometry

- `c_std_mean` and `c_cross_video_cosine` flatten every example's complete `N_c*D_c` code, so
  they are the most direct recorded-batch example-specificity measures.
- `c_effective_rank` pools `B*N_c` slot rows over 256 feature channels. Fixed slot identities and
  more pooled rows can raise it without proving example content.
- centered slot rank measures within-example slot diversity. Its maximum changes with `N_c`:
  31, 63, and 127 centered directions.
- `L_cov` uses the same `B*N_c` pooling as effective rank. A fixed coefficient is not an
  invariant pressure across slot counts.
- `L_var` compares matching flattened coordinates across the current training batch. Tiny train
  `L_var` and weak fixed-batch std can coexist through train/validation and source-composition
  differences.

### Stability

`grad_norm` is after AGC and before the global 0.5 clip. It is a combined-objective norm, not a
per-loss gradient attribution. Neither AGC activity nor clean gradients proves that the code is
informative.

## 6. Reading Cycle B, one run at a time

### 32 slots — `x03xlpyl`

1. **Alive and correctly wired:** pass. `300` training rows, no skipped updates or warnings, no
   diagnostic NaNs, whitening/present-only flags correct, prediction losses zero.
2. **Alive and example-specific:** marginal pass on this one-source batch. Late std is
   `0.80585822`, cosine `0.47349986`, dead fraction zero.
3. **Rich:** conditional pass. Late pooled rank is `85.05537033`; centered slot rank is
   `30.81357294/31`. Rank is interpreted with the slot-identity caveat.
4. **Reconstruction:** pass in isolation. Late train median `0.67703196`; late fixed median
   `0.67116472`; final fixed value `0.67091203`.
5. **Correct-code dependence:** weak. Final rolled-code loss `0.69735569`, gap `0.02644366`, exact
   share `7.65699171%`.
6. **Verdict:** stable, marginally healthy recorded-batch geometry, weak exact-chunk conditioning,
   no prediction evidence.

### 64 slots — `evyokqrm`

1. **Alive and correctly wired:** pass. Same operational validity; late median grad norm
   `0.09111974`.
2. **Alive and example-specific:** fail on the recorded batch. Late std `0.32573912`, cosine
   `0.91421551`, dead fraction zero.
3. **Rich:** pooled metrics do not rescue it. Late rank `83.12630844`; centered slot rank
   `61.97086895/63`.
4. **Reconstruction:** pass in isolation, but the gain is tiny. Late train median `0.67396140`;
   late fixed median `0.67072162`; final fixed `0.67033505`.
5. **Correct-code dependence:** weak. Final gap `0.02765644`, exact share `8.13187861%`.
6. **Verdict:** a valid run in a source-chunk-invariant fixed-batch basin. It is not a healthy
   capacity win, and the one-source manifest prevents a global-collapse label.

The trajectory is the largest surprise. At step 2,000 it temporarily reaches std `0.68589735`
and cosine `0.51970375`; by step 3,500 it is at `0.16407548/0.97720748`, and it finishes near the
same weak-specificity regime. This proves the failure is an evolved equilibrium rather than a
step-zero metric illusion. It does **not** prove initialization was irrelevant; different initial
weights can select a later basin.

### 128 slots — `7pmvxrxi`

1. **Alive and correctly wired:** pass. No skips, warnings, diagnostic NaNs, or AGC activations;
   late median grad norm `0.10019990`.
2. **Alive and example-specific:** fail on the recorded batch. Late std `0.64887702`, cosine
   `0.67727712`, dead fraction zero.
3. **Rich:** pooled rank is high but ambiguous. Late rank `188.84199524`; centered slot rank
   `115.88776183/127`.
4. **Reconstruction:** the best scalar arm. Late train median `0.66298261`; late fixed median
   `0.66570026`; final fixed `0.66532779`.
5. **Correct-code dependence:** still weak, though best of the three. Final gap `0.03431261`, exact
   share `10.00512361%`.
6. **Verdict:** modest scalar improvement with weak recorded-batch specificity; not a healthy
   capacity win and not forecasting evidence.

Normalized centered slot rank adds useful context: the late value is `0.99398622` of its maximum
at 32 slots, `0.98366459` at 64, and only `0.91250206` at 128. The absolute centered rank rises,
but the fraction of available centered slot directions actually used falls in the largest arm.

## 7. Reconstruction analysis

### Full 300-row training histories

The registered late window is steps 12,000 through 14,950, containing 60 logged batches per arm:

| `N_c` | Late median | Late mean | Population SD | Late slope per 1,000 steps |
|---:|---:|---:|---:|---:|
| 32 | `0.67703196` | `0.67702819` | `0.00342801` | `-0.00000786` |
| 64 | `0.67396140` | `0.67391246` | `0.00334774` | `-0.00010244` |
| 128 | `0.66298261` | `0.66297507` | `0.00305811` | `-0.00042873` |

All arms are effectively plateauing. The paired same-batch late differences are:

| Contrast | Median arm-minus-reference | Mean | Arm better fraction |
|---|---:|---:|---:|
| 64 - 32 | `-0.00310391` | `-0.00311573` | `1.00000000` |
| 128 - 32 | `-0.01410100` | `-0.01405312` | `1.00000000` |
| 128 - 64 | `-0.01105312` | `-0.01093739` | `1.00000000` |

Within this deterministic trajectory, the ordering is robust to batch noise. It is still one
realized initialization per shape, and a 60-row time series is not a confidence interval over
seeds or model initializations.

The result lands exactly in the preregistered borderline region: the 32-to-128 late median gain is
`0.01404935` (`2.07513837%`), short of both support thresholds. It is too coherent to call zero,
but too small and geometrically unhealthy to call the bottleneck-floor hypothesis confirmed.

### Fixed-batch baseline adjustment

The endpoint fixed-batch comparison is weaker than it first looks because the untrained arms did
not begin at the same loss:

| `N_c` | Initial | Final | Improvement from own initialization |
|---:|---:|---:|---:|
| 32 | `1.01626515` | `0.67091203` | `0.34535313` |
| 64 | `1.01043403` | `0.67033505` | `0.34009898` |
| 128 | `1.00827813` | `0.66532779` | `0.34295034` |

The 128-slot endpoint is `0.00558424` below the 32-slot endpoint, but it already starts
`0.00798702` lower. Relative to its own untrained scaffold, it learns `0.00240278` **less** fixed-
batch improvement than the 32-slot arm. The full paired training history still supports a real
training-distribution gain, but the fixed-batch endpoint does not independently demonstrate
greater learned capacity.

### Conditioning decomposition

| `N_c` | Learned correct-code improvement | Improvement surviving rolled code | Exact-chunk advantage | Exact share |
|---:|---:|---:|---:|---:|
| 32 | `0.34535313` | `0.31890947` | `0.02644366` | `7.65699171%` |
| 64 | `0.34009898` | `0.31244254` | `0.02765644` | `8.13187861%` |
| 128 | `0.34295034` | `0.30863774` | `0.03431261` | `10.00512361%` |

There is a small, monotonic increase in exact-chunk dependence, but roughly nine tenths of the
128-slot improvement still survives a wrong adjacent-chunk code. Since all chunks share one
source, the remaining channel can contain source, wearer, scene, position-template, or globally
shared information. The current metric cannot decompose those components.

## 8. Why rank and covariance improve so easily with more slots

At initialization, the bottleneck ignores its input and emits normalized learned orthogonal slot
identities for every example. The step-zero metrics expose this directly:

| `N_c` | Example std | Example cosine | Pooled effective rank | Centered slot rank |
|---:|---:|---:|---:|---:|
| 32 | `0.00000203` | `1.00000000` | `30.99746323` | `30.99739504` |
| 64 | `0.00000169` | `0.99999994` | `62.95524597` | `62.95516014` |
| 128 | `0.00000153` | `1.00000000` | `126.80850220` | `126.80836010` |

The examples are identical while pooled rank nearly equals `N_c-1`. That is slot identity, not
content.

The active step-zero objective also changes materially:

| `N_c` | Weighted variance | Raw `L_cov` | Weighted covariance | Total loss |
|---:|---:|---:|---:|---:|
| 32 | `0.50000000` | `6.72580338` | `0.06725803` | `0.56725800` |
| 64 | `0.50000000` | `2.93429971` | `0.02934300` | `0.52934301` |
| 128 | `0.50000000` | `0.99159271` | `0.00991593` | `0.50991595` |

Reconstruction has zero weight at that step. Thus the first update does not place the three arms
under equal geometry pressure even though `lambda_cov=0.01` is numerically fixed. More orthogonal
slots and more pooled rows make covariance easier to satisfy.

The same effect remains late. Median raw `L_cov` after step 12,000 is `0.2817093134`,
`0.1313608885`, and `0.0823624134`; the corresponding weighted contributions are
`0.0028170931`, `0.0013136089`, and `0.0008236241`. Reconstruction contributes more than `99.5%`
of the final scalar objective in every arm, but the earlier geometry trajectory has already shaped
the basin.

This explains why 128 slots can report pooled rank near 189 without passing example specificity:
the objective and metric both reward a distribution of slot rows, while the content question is
whether different examples carry different useful codes.

## 9. Initialization coupling—dated correction to the original sweep read

The canonical sweep analysis calls the arms byte-identical controls apart from `N_c` and says the
64-slot collapse is “not merely an initialization artifact.” The first clause is correct for
code, data, config, and seed but not for shared parameter values; the second is not established.

The production construction order is:

```text
seed one trainable RNG stream
  -> Bottleneck input/mixer/position parameters
  -> orthogonal query tensor with shape (N_c, 256)
  -> Bottleneck latent blocks
  -> EMA deepcopy
  -> CoarseFlow
  -> Decoder
```

Because orthogonal initialization consumes shape-dependent random numbers, later weights shift.
Replaying the exact sweep config and seed on the unchanged production code gives:

| Module | State tensors | Identical | Shape changed | Same shape, different values |
|---|---:|---:|---:|---:|
| B | 93 | 65 | 1 | 27 |
| F_c | 70 | 28 | 2 | 40 |
| D | 59 | 29 | 0 | 30 |

`F_c` is inactive here, but B's latent-block projections and D's attention projections are active.
The W&B provenance independently records different trainable-init hashes:

- 32: `e468284ed15fdc4ea0cb26043997660ebe65e7ff0b6e0a49903d3519ec6d6c82`;
- 64: `6e332ddb78eb7a6ea5f6056f1926abd9d899d7de2a5274d43df533a52922f573`;
- 128: `ac8fc707179ccacc4596289358b3d7a65c82f25dab3d44bf157388879d6b86ba`.

Different hashes are unavoidable when shapes differ; the replay establishes the stronger point
that many shape-matched values differ too. Consequently:

- the observed arm includes both the `N_c` intervention and a deterministic re-draw of many
  shared active weights;
- one seed per shape cannot estimate initialization variance;
- the 64-slot non-monotonic anomaly cannot be confidently assigned to “64” as a structural
  property;
- the decision not to spend on 256 still holds, because the realized larger arms miss the
  preregistered win condition, but the sweep should not be cited as a universal `N_c` law.

## 10. A strong positive control: exact rerun reproducibility

The 32-slot arm is an exact science-metric repeat of investigation 016 run 060 (`2423b84g`). The
two W&B history parquets each contain `300` rows and `48` columns. Excluding only `_timestamp` and
`_runtime`, all `46` scientific columns are exactly equal at every logged step—no differing
column and no differing value.

Their provenance differs in tracking/checkpoint paths, the recorded source commit, the explicit
encoder-revision field, and whitening path, but the resolved feature and whitening payloads are
the same. This exact trajectory reproduction establishes:

1. current same-shape initialization, data order, sample-scoped augmentation, encoder, whitening,
   and training are deterministic across executions;
2. the 32-slot result is not a transient W&B/dashboard artifact;
3. the cross-shape differences arise from the shape-dependent computation/initialization, not
   nondeterministic data loading.

It is a reproducibility check, not an independent statistical seed.

## 11. Operational and system read

All arms are clean:

| `N_c` | Max pre-global-clip grad | Rows above 0.5 | Skips | Warnings | B/D AGC rows |
|---:|---:|---:|---:|---:|---:|
| 32 | `5.02441216` | 33/300 | 0 | 0 | 0/0 |
| 64 | `4.07689619` | 35/300 | 0 | 0 | 0/0 |
| 128 | `5.69471502` | 35/300 | 0 | 0 | 0/0 |

System telemetry shows finite A100 execution and no corrected or uncorrected GPU memory errors.
Maximum allocated GPU memory rises modestly from `12.73220301` GB to `12.82985165` GB to
`13.81688934` GB. Median GPU utilization is `97`, `96`, and `100` percent. The sequential wall
times decrease with larger arms because utilization/data-pipeline conditions differ; that should
not be interpreted as larger attention being intrinsically cheaper.

The resource evidence rules out OOM, thermal failure, numerical failure, AGC saturation, or a
stalled decoder as explanations for the scientific result.

## 12. Reconciliation with the research progression

The sweep makes sense only as the latest link in the prior chain:

1. Early Phase 1 runs established the pipeline and exposed low-rank/static-latent failure.
2. Strong variance and slot losses showed that spread or slot distinction can be Goodharted
   without useful prediction.
3. EMA, AGC, and gradient guards fixed stability but did not make `F_c` beat copy.
4. Reconstruction anchors made the latent decodable; learned output queries and absolute targets
   exposed template shortcuts.
5. Investigation 007's first `N_c` wave died synchronously at step 200 with the entire pod, so it
   supplied no mature capacity evidence.
6. Investigation 011 showed covariance can raise pooled rank dramatically, while prediction still
   fails and rank need not equal content.
7. Runs 052/053 separated low reconstruction from honesty: absolute reconstruction can be
   template-dominated; a residual target restores code dependence without holding geometry.
8. Investigation 014 showed the frozen V-JEPA feature substrate itself has ample rank.
9. Runs 054-057 combined latent-stack processing, full whitening, and covariance/variance. They
   achieved honest or high-rank SSv2 present representations, but at a whitened reconstruction
   cost and with no forecast evidence.
10. EGO4D run 058 appeared globally collapsed, then the source-manifest audit corrected that claim:
    its fixed batch is one long source recording.
11. Run 060 showed a 20x reconstruction weight does not materially clear the floor and is now
    exactly reproduced by the 32-slot control.
12. This sweep is therefore the first mature same-runtime test of the slot-count axis. It finds a
    small scalar response, not a healthy capacity solution.

The current research state is not “the model needs more slots” and not “capacity is irrelevant.”
It is: **the present autoencoder is stable and can hold geometric rank, but content measurement is
source-confounded, full whitening makes the target nearly isotropic before a severe channel
squeeze, and geometry metrics/losses are partly slot-structural.** Prediction remains untested in
this branch and historically has not beaten the copy gate.

## 13. Surprise, mechanism, and hypothesis accounting

### Largest surprise: the 64-slot basin

The expected alternatives were monotonic improvement or saturation. Instead, 64 slots has the
worst example specificity by a large margin while reconstructing almost identically to 32. The
most plausible connected mechanism is not a magical “bad” slot count. It is the interaction of:

- fixed orthogonal slot identities;
- shape-dependent shared initialization;
- `B*N_c`-pooled covariance pressure whose effective scale changes with `N_c`;
- an absolute whitened target that still rewards shared/source/template structure;
- a decoder that can map distinct constant slots plus fixed positions into a position field;
- one single-source diagnostic batch.

That mechanism predicts exactly the observed dissociation: high absolute slot rank, small
reconstruction changes, and poor example-level cosine/std.

### Hypothesis scorecard

| Claim | Status | Reason |
|---|---|---|
| More slots lower training reconstruction monotonically | Supported for this seed | all late paired batches preserve the ordering |
| The effect is materially large | Not supported | `0.01404935` / `2.07513837%` misses both gates |
| More slots produce a healthier content code | Not supported on recorded batch | 64 collapses; 128 does not recover 32-slot specificity |
| Pooled rank proves greater content capacity | Rejected | rank is already `N_c-1` at zero example spread |
| The sweep isolates only activation capacity | Rejected | covariance scaling and shared initialization also change |
| 64 is intrinsically pathological | Unresolved | one seed plus initialization coupling |
| A 256-slot continuation is justified | Rejected | the conditional geometry guard fails |
| `D_c`/mixer/whitening are ruled out | Rejected | those axes were fixed and the first channel squeeze remains |

## 14. Final conclusion and next discriminating probe

Do not launch the 256-slot arm. The result does not meet the preregistered capacity-support rule,
and the lower scalar is paired with weaker recorded-batch content geometry. Carry 32 slots as the
operational control, not because it has been proven globally optimal, but because no larger arm
earned its additional complexity under the joint criterion.

The first falsifiable probe should require **no retraining** if the checkpoints still exist:

> Re-evaluate all three checkpoints on one deterministic validation clip per distinct EGO4D source
> UID, with a verified cross-source code derangement. Change only the validation manifest/pairing.

Prediction and decision rule:

- If the current anomaly is chiefly adjacent-chunk/source confounding, cross-source cosine and
  shuffled-code gaps will become healthy, and the 64-slot arm may no longer be the worst.
- If the codes are globally shared/template-dominated, 64 will remain highly parallel and the
  correct-versus-cross-source gap will remain small.
- If 128 genuinely routes more content, its cross-source gap should exceed 32 while its
  source-diverse std/cosine remain healthy; a lower raw reconstruction number alone does not count.

After that measurement repair, two different questions should be kept separate:

1. **Robustness of the slot result:** rerun the 64-slot arm across additional seeds or isolate
   module-local initialization streams. This decides whether its basin is structural or
   initialization-specific.
2. **Cause of the reconstruction floor:** run the already queued 32-slot no-whitening arm, then a
   separate mixer-width/`D_c` experiment if the cached-feature oracle implicates the early channel
   squeeze. Do not call another `N_c` ladder a general bottleneck-capacity sweep.

No present-only checkpoint should enter prediction until source-diverse reconstruction dependence
is demonstrated. A later prediction run must still beat both copy and batch-mean baselines; none of
the three runs analyzed here supplies that evidence.
