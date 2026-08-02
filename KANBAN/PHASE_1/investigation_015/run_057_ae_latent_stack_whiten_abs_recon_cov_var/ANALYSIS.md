# Run 057 analysis — `ae_latent_stack_whiten_abs_recon_cov_var` (`cdvp6hou`, inv015)

> Analyzed 2026-07-06 using Reading Cycle B from `GUIDES/READING_EXPERIMENTS.md`,
> together with the geometry-plus-honesty axes carried from this run's
> [`GUIDE.md`](GUIDE.md) and the parent framing in
> [`../run_056_ae_latent_stack_whiten_abs_recon_geom/ANALYSIS.md`](../run_056_ae_latent_stack_whiten_abs_recon_geom/ANALYSIS.md)
> and [`../run_055_ae_latent_stack_whiten_abs_recon/ANALYSIS.md`](../run_055_ae_latent_stack_whiten_abs_recon/ANALYSIS.md).
> W&B run `smahalanobis-uc-davis/hjepa-vwm/cdvp6hou`, group
> `inv015_ae_latent_stack_whiten`. The run finished its full 15,000-step schedule
> cleanly; all 30 diagnostic checkpoints (logged every 500 steps) were read
> individually, with no downsampling, and every geometry and reconstruction curve was
> overlaid against run 056 (`tl5dh73c`, the same base with the full geometry bundle) and
> run 055 (`nzz64pl6`, the same base with zero geometry terms).

## What this run is

Run 057 is run 056 with exactly one hyperparameter removed: SIGReg is turned off
(`lambda_sigreg` 5.0 to 0.0), while the covariance penalty (`lambda_cov=0.01`) and the
variance floor (`lambda_var=0.5`) are kept. Everything else is byte-identical to run 056:
present-reconstruction-only, clean/absolute whitened-feature target
(`recon_residual_target=FALSE`), fixed offline whitening reusing the same stats file, the
three-block latent-stack bottleneck, 32 slots, a 256-dimensional code, the 512-wide
4-block decoder, `lambda_recon=0.05`, seed 42, 15,000 steps. It isolates SIGReg's
contribution: overlaying it on run 056 attributes any change to SIGReg alone. The
hypothesis under test was that covariance is the true rank lever while SIGReg's full
`N(0, I)` isotropy constraint is the term most likely to fight reconstruction, so removing
it should retain the rank while relieving the honesty pressure.

## Part A — Interpretation of the metrics (what the curves show)

### Wiring and mode

Every run-defining flag holds its intended value at every diagnostic step. The run
trained in present-reconstruction-only mode (`present_recon_only=1`) with the prediction
branch fully off (`prediction_active=0`, and both `L_flow` and the predicted-future
reconstruction loss `L_recon_pred` pinned at zero all run). Feature whitening was active
throughout (`whiten_active=1`), the clean/absolute target was in force
(`recon_target_residual=0`, and the residual mean tracker inactive so its logged norm
`recon_mean_norm` held at exactly zero). The single defining change is confirmed: the
SIGReg warmup multiplier `sigreg_scale` reads 0 at every step (against run 056's ramp to
1.0), so SIGReg never entered the objective, and the run config carries `lambda_sigreg=0`.
The SIGReg statistic `L_sigreg` still logs a small for-logging-only value near 0.005 to
0.008, which is expected because the term is always computed but only added to the loss
when its weight is positive. The reconstruction warmup `recon_scale` reaches its full
value of 1.0 at step 2,000 as designed.

### Stability

Stability is spotless, and in fact calmer than run 056. The skipped-step counter
(`grad_skipped`), the NaN-gradient flag (`grad_has_nan`), and the instability warning
(`instability_warn`) all stay at zero at every logged point. The gradient magnitude
(`grad_norm`) starts near 0.42, briefly reaches about 1.4 during the reconstruction
warmup, and then settles into a very low 0.04 to 0.10 band for the rest of the run,
ending at 0.036, which is several times smaller than run 056's terminal 0.235. Adaptive
gradient clipping never bites: the bottleneck clipping ratio (`agc_B_max_ratio`) sits
around 0.01 to 0.03 after warmup and the decoder ratio (`agc_D_max_ratio`) stays near
0.0004. The stability prediction is satisfied with margin.

### Geometry — held, and if anything stronger than run 056

Removing SIGReg did not cost rank. Every geometry axis is at least as healthy as run 056,
and none contracts after its early rise.

| step | c_eff_rank | slot_rank_centered | cross_video_cosine | c_std_mean | attn_entropy | attn_min | L_recon_present | shuffled_c | video_gap |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 31.0* | 31.0* | 1.000 | ~0 | 0.677 | 0.227 | 1.004 | 1.004 | 0.000 |
| 500 | 32.8 | 28.6 | 0.617 | 0.592 | 0.674 | 0.211 | 0.975 | 0.992 | 0.017 |
| 1,000 | 76.4 | 28.2 | -0.004 | 0.981 | 0.661 | 0.259 | 0.899 | 0.946 | 0.047 |
| 2,000 | 146.3 | 29.9 | 0.019 | 1.043 | 0.632 | 0.136 | 0.827 | 0.929 | 0.102 |
| 2,500 | 173.8 | 30.0 | 0.023 | 1.078 | 0.636 | 0.098 | 0.802 | 0.930 | 0.128 |
| 5,000 | 193.5 | 30.5 | 0.078 | 1.113 | 0.648 | 0.132 | 0.746 | 0.935 | 0.188 |
| 7,500 | 212.3 (peak) | 30.5 | 0.017 | 1.146 | 0.668 | 0.141 | 0.729 | 0.939 | 0.210 |
| 10,000 | 211.2 | 30.7 | 0.027 | 1.137 | 0.697 | 0.184 | 0.719 | 0.940 | 0.222 |
| 12,500 | 211.3 | 30.7 | 0.055 | 1.120 | 0.719 | 0.271 | 0.714 | 0.940 | 0.226 |
| 14,500 | 208.2 | 30.71 | 0.059 | 1.117 | 0.721 | 0.278 | 0.713 | 0.940 | 0.227 |

(*The step-0 rank and slot rank near 31 are mechanical, from the 32 fixed slot identity
vectors before any information routes through them, the standard run-054 caveat. Slots are
judged on the centered metric.)

The effective rank (`c_effective_rank`, the number of independent feature directions in
use, of a maximum 256) rises from the mechanical 31 to about 146 by step 2,000 and about
174 by step 2,500 — faster than run 056, which reached 122 at step 2,000 — oscillates in
the 160 to 195 band through the middle of the run, and settles at about 208 to 211 over
the final third, peaking at 212.3 at step 7,500 and ending at 208.2. This terminal value
is slightly above run 056's 201.6 and about 9.7 times run 055's 21.5, and it climbs rather
than contracts. Removing the isotropy term did not lower the rank, which identifies
covariance as the operative rank lever, consistent with the covariance loss `L_cov` being
the largest term throughout and falling more completely here (terminal 0.149) than in run
056 (terminal 0.259) — with SIGReg no longer competing for the same code capacity,
decorrelation is more fully achieved.

The centered slot diversity (`c_slot_diversity_rank_centered`, how many of the 32 slots
stay distinct within a video) holds between about 28.6 and 30.7 the whole run, ending at
30.71, slightly higher than run 056's 29.87 and with no monotone decline, against run
055's collapse to 12.1. The dead-dimension fraction (`c_dead_dim_frac`) is zero
throughout. The code's average per-dimension spread (`c_std_mean`, target near 1.0) rises
to 1.0 by step 1,500 and then overshoots to a 1.10 to 1.15 band, ending at 1.117. This
overshoot above 1.0 is the expected fingerprint of removing SIGReg: the variance floor is
a one-sided hinge that only penalizes spread below 1.0, so with SIGReg no longer pinning
the distribution to a unit-variance `N(0, 1)`, the spread is free to float somewhat above
target, which is harmless given the dead-dimension fraction stays at zero.

The cross-video cosine (`c_cross_video_cosine`, similarity between different videos'
codes, lower is better) collapses from 1.0 to essentially zero by step 1,000 — it briefly
touches minus 0.004, meaning videos are on average marginally more than orthogonal — and
then floats in a 0.02 to 0.10 band, ending at 0.059. This is an order of magnitude better
than run 055's 0.82 and near the inv011 sweep's 0.08, though slightly higher than run
056's 0.017. Marginally weaker video separation is the one axis on which SIGReg had been
helping, since its isotropy push spreads videos apart a touch more; the effect is small
and 0.059 is still strong separation.

The bottleneck attention entropy (`c_attn_entropy`, how broadly the query slots read from
the input tokens) is the sharpest qualitative contrast with run 056. Here it starts near
0.68, dips slightly through the early phase, and then rises steadily to 0.72 by the end,
with its per-head minimum (`c_attn_entropy_min`) never falling below about 0.10 and ending
near 0.28. Run 056, under SIGReg, drove the mean down to 0.47 and the minimum to 0.006,
meaning one head had sharpened almost to a delta. Without SIGReg the bottleneck keeps all
heads reading broadly and no head specializes sharply, yet the slot diversity and rank
stay maximal, so the broad reads are not producing redundant slots. The SIGReg isotropy
pressure was what forced the sharp head specialization in run 056; it is not necessary for
a diverse, high-rank code.

### Reconstruction and honesty — modestly better than run 056

The present reconstruction loss (`L_recon_present`) falls monotonically from 1.004 to
0.713 and plateaus after about step 10,000. This terminal value is lower, meaning better
raw reconstruction, than run 056's 0.731, though still higher than run 055's 0.642. The
wrong-video reconstruction loss (`L_recon_shuffled_c`, the loss when the decoder is handed
a deliberately wrong video's code) drops early to about 0.93 and holds near 0.940 all run,
close to run 056's 0.934. The video gap (`L_recon_video_gap`, how much the correct code
helps) grows steadily to about 0.227 and plateaus, which is larger than run 056's 0.204
and closer to run 055's 0.311. Computing the video-conditioned share of the decoder's
improvement the way prior analyses did, as the correct-code gain minus the wrong-code gain
divided by the correct-code gain, gives (0.291 minus 0.064) divided by 0.291, which is
about 78 percent, against run 056's roughly 75 percent, run 055's 86 percent, and run
054's 92 percent. All three honesty measures — raw reconstruction, the video gap, and the
share — moved in the honest direction relative to run 056, by a modest amount, and none
recovered all the way to the zero-geometry run 055.

### Regularizer terms

The variance-floor loss (`L_var`) collapses to essentially zero by step 2,000 and stays
near 0.0001 to 0.001, fully satisfied once the spread reaches and exceeds 1.0. The
covariance loss (`L_cov`) starts as the largest term at 6.7, stays high through the
warmup, and then falls steadily to about 0.15 to 0.30, ending at 0.149. The total training
loss ends near 0.039, lower than run 056's 0.047, with the scaled reconstruction anchor
(0.05 times 0.713, about 0.036) the dominant component and the covariance term
(0.01 times 0.149, about 0.0015) most of the small remainder.

### Cross-run placement

On geometry, run 057 matches or slightly exceeds run 056 on every axis except cross-video
cosine (rank 208 versus 201, slot diversity 30.7 versus 29.9, spread 1.12 versus 1.03,
cosine 0.059 versus 0.017), and it towers over the zero-geometry run 055 (rank 208 versus
21). On honesty it sits modestly above run 056 (share about 78 percent versus 75 percent,
gap 0.227 versus 0.204, reconstruction 0.713 versus 0.731) and still below the
zero-geometry run 055 (86 percent) and the residual-target run 054 (92 percent). Stability
and attention breadth are both cleaner than run 056.

## Part B — What happened in this run (conclusions)

Dropping SIGReg was a strict improvement on essentially every axis, which settles the
attribution the run was built to test. The rank did not fall — it rose slightly to about
208, reached faster — so covariance, not SIGReg, is the operative rank lever, and the
covariance penalty is even more fully satisfied without SIGReg competing for the code's
capacity. Slot diversity, dead dimensions, and stability all held or improved.

SIGReg was a small honesty tax, and removing it recovered part of it. The
video-conditioned share rose from about 75 to about 78 percent, the video gap grew from
0.204 to 0.227, and raw reconstruction improved from 0.731 to 0.713, all at no geometric
cost. The clearest mechanistic fingerprint is attention: SIGReg's full-isotropy pressure
had forced one bottleneck head to sharpen almost to a delta in run 056, and without it the
reads stay broad while the code stays maximally diverse — so the sharp specialization was
a SIGReg artifact, not a requirement for a rich code.

But SIGReg was only a minor contributor to the honesty gap. Only about three of the roughly
eleven points that separated run 056 from the zero-geometry run 055 came back; the
remaining eight points are the cost of the geometry constraint itself. Covariance and the
variance floor still force the code to rank 208 with spread above unit and decorrelated
dimensions, and that broad, isotropic code is intrinsically harder for the decoder to read
than run 055's cramped rank-21 code, which is why reconstruction and the video-conditioned
share remain below the zero-geometry run. That remaining gap is decodability, not the
regularizer mix, and it points to the reconstruction-side levers rather than to any further
change in the geometry terms.

Net, this is the strongest present-only run the program has produced. By Reading Cycle B it
lands in the target cell — reconstruction falling while the code stays high-rank, spread,
and video-specific — which no earlier run reached: it holds genuinely healthy geometry
(rank 208, spread 1.12, cross-video cosine 0.059, slot diversity 30.7) together with
honest, clearly video-specific reconstruction (a stable 0.227 video gap, about 78 percent
conditioned), and it does so more cleanly than run 056, with better reconstruction, a
larger gap, broader attention, and calmer training. The one honest caveat is that its
honesty share, at about 78 percent, still trails the residual-target and zero-geometry
runs, and that residual is a decodability cost of the high-rank code that SIGReg's removal
does not reach.
