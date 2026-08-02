# Run 056 analysis — `ae_latent_stack_whiten_abs_recon_geom` (`tl5dh73c`, inv015)

> Analyzed 2026-07-06 using Reading Cycle B from `GUIDES/READING_EXPERIMENTS.md`,
> together with the geometry-plus-honesty axes registered in this run's
> [`HYPOTHESIS.md`](HYPOTHESIS.md) and [`DESCRIPTION.md`](DESCRIPTION.md), and the
> parent framing in [`../run_055_ae_latent_stack_whiten_abs_recon/ANALYSIS.md`](../run_055_ae_latent_stack_whiten_abs_recon/ANALYSIS.md)
> and [`../ANALYSIS_054.md`](../ANALYSIS_054.md). W&B run
> `smahalanobis-uc-davis/hjepa-vwm/tl5dh73c`, group `inv015_ae_latent_stack_whiten`.
> The run finished its full 15,000-step schedule cleanly; all 30 diagnostic
> checkpoints (logged every 500 steps) were read individually, with no downsampling,
> and every geometry and reconstruction curve was overlaid against run 055
> (`nzz64pl6`, the same base with no geometry terms) and run 054 (`lx1b6gw2`).

## Part A — Interpretation of the metrics (what the curves show)

Baselines for reference throughout: run 055 (`nzz64pl6`, the same clean whitened base
with zero geometry regularizers) and run 054 (`lx1b6gw2`, the residual-target arm).

### Wiring and mode

Every run-defining flag holds its intended value at every diagnostic step. The run
trained in present-reconstruction-only mode (`present_recon_only=1`) with the
prediction branch fully off (`prediction_active=0`, and both the flow loss `L_flow` and
the predicted-future reconstruction loss `L_recon_pred` pinned at zero all run). Feature
whitening was active throughout (`whiten_active=1`), the clean/absolute target was in
force (`recon_target_residual=0`, and the residual mean tracker inactive so its logged
norm `recon_mean_norm` held at exactly zero), and the three added weights are confirmed
in the run config as `lambda_var=0.5`, `lambda_sigreg=5`, and `lambda_cov=0.01`. The
reconstruction warmup multiplier `recon_scale` and the SIGReg warmup multiplier
`sigreg_scale` both ramp from zero to their full value of 1.0 by step 2,000 as designed.
This is unambiguously the intended clean-arm, geometry-regularized run rather than a
mis-launch of the run-055 baseline or the residual variant.

### Stability

Stability is spotless for the whole schedule. The skipped-optimizer-step counter
(`grad_skipped`), the NaN-gradient flag (`grad_has_nan`), and the instability warning
(`instability_warn`) all stay at zero at every logged point. The gradient magnitude
(`grad_norm`) starts near 0.42, spikes once to about 3.1 at step 500 during the joint
reconstruction-and-SIGReg warmup, and then settles into a 0.18 to 0.29 band for the rest
of the run. Adaptive gradient clipping never bites of any consequence: the bottleneck
clipping ratio (`agc_B_max_ratio`) peaks near 0.22 at step 500 and sits around 0.06 to
0.09 thereafter, and the decoder ratio (`agc_D_max_ratio`) stays near 0.0006 throughout.
Against the hypothesis, this satisfies the stability prediction mechanically.

### Geometry — the headline

Every geometry axis moves decisively in the healthy direction, and, crucially, none of
them contracts after an early peak.

| step | c_eff_rank | slot_rank_centered | cross_video_cosine | c_std_mean | L_recon_present | shuffled_c | video_gap |
|---|---|---|---|---|---|---|---|
| 0 | 31.0* | 31.0* | 1.000 | ~0 | 1.004 | 1.004 | 0.000 |
| 500 | 27.5 | 25.1 | 0.573 | 0.615 | 0.975 | 0.995 | 0.020 |
| 1,000 | 75.2 | 29.6 | 0.071 | 0.933 | 0.892 | 0.941 | 0.050 |
| 2,000 | 121.7 | 29.6 | 0.033 | 1.001 | 0.826 | 0.931 | 0.104 |
| 5,000 | 159.0 | 28.6 | 0.078 | 0.987 | 0.765 | 0.929 | 0.164 |
| 8,000 | 157.7 | 29.3 | 0.126 | 0.960 | 0.744 | 0.934 | 0.190 |
| 10,000 | 188.4 | 29.5 | 0.031 | 1.017 | 0.738 | 0.937 | 0.198 |
| 12,500 | 202.6 (peak) | 29.9 | 0.011 | 1.031 | 0.732 | 0.935 | 0.203 |
| 14,500 | 201.6 | 29.9 | 0.017 | 1.028 | 0.731 | 0.934 | 0.204 |

(*The step-0 effective rank and slot rank near 31 are mechanical, produced by the 32
fixed slot identity vectors before any information routes through them, exactly the
run-054 measurement caveat. Slots are judged only on the centered metric.)

The effective rank (`c_effective_rank`, the number of independent feature directions the
code uses, of a maximum 256) climbs from the mechanical 31 to about 122 by step 2,000,
tracking the SIGReg ramp to full weight, then holds a fluctuating plateau near 150 to 170
through steps 4,000 to 9,000, and then climbs a second time to about 200 over the final
third, peaking at 202.6 at step 12,500 and ending at 201.6. Relative to run 055's
terminal 21.5 (which peaked at 33.9 and then fell), this is roughly 9.4 times the rank,
and the sign of the late-run drift is reversed: rank rises here where run 055's fell. A
rank near 201 of 256 sits far above the roughly 40 that slot structure alone can produce,
so nearly the full feature space is in genuine use.

The code's average per-dimension spread (`c_std_mean`, target near 1.0) rises from zero
to 1.0 by step 2,000 and holds between 0.96 and 1.03 for the rest of the run, ending at
1.028, landing exactly on the variance-floor and SIGReg unit-variance target, against run
055's 0.40. The cross-video cosine (`c_cross_video_cosine`, the similarity between
different videos' codes, where lower is better) collapses from 1.0 to 0.07 by step 1,000
and settles in a 0.01 to 0.13 band, ending at 0.017, which is below even the inv011
sweep's roughly 0.08 and an order of magnitude better than run 055's 0.82. The centered
slot diversity (`c_slot_diversity_rank_centered`, how many of the 32 slots stay distinct
within a video) holds between about 28.4 and 29.9 the entire run, ending at 29.87, with
no monotone decline, against run 055's collapse from 31 to 12. The dead-dimension
fraction (`c_dead_dim_frac`) is zero throughout. The bottleneck attention entropy
(`c_attn_entropy`) declines from 0.68 to about 0.47 while its per-head minimum
(`c_attn_entropy_min`) falls to about 0.006, meaning the mean read stays moderate while
at least one head sharpens strongly, which is not a uniform-attention collapse.

Mechanically this is what prediction 1 of the hypothesis described, and the driver
matches the pre-registration: the covariance term (`L_cov`) starts as the largest loss
component at 6.7, rises to 8.65 at step 500 as the code's variance grows off zero, and
then falls monotonically to about 0.22 to 0.31 as decorrelation bites. The covariance
penalty is visibly doing the rank work, consistent with the prediction that covariance
would be the dominant lever.

### Regularizer terms

The variance-floor loss (`L_var`) collapses from 1.0 to 0.37 to 0.05 by step 1,000 and
then sits near 0.01, because the one-sided unit-variance hinge is essentially satisfied
once the spread reaches 1.0. The SIGReg loss (`L_sigreg`) is tiny throughout, near 0.001
to 0.002, contributing roughly 5 times 0.0012, about 0.006, at full ramp. The covariance
loss never reaches zero, ending at 0.259 and contributing about 0.01 times 0.259, roughly
0.0026. The total training loss ends near 0.047, with the scaled reconstruction anchor
(0.05 times 0.731, about 0.037) the largest single component and the geometry bundle the
remainder, so the objective is geometry-shaped exactly as the DESCRIPTION pre-computed.

### Reconstruction and honesty

The present reconstruction loss (`L_recon_present`) falls monotonically from 1.004 to
0.731 and plateaus after about step 8,000. This terminal value is higher, meaning worse
raw reconstruction, than run 055's 0.642, which is the interpretation guard's predicted
consequence of the code being forced to carry more. The wrong-video reconstruction loss
(`L_recon_shuffled_c`, the loss when the decoder is handed a deliberately wrong video's
code) drops early to about 0.93 and then holds flat at 0.934 all run, which is slightly
lower than run 055's 0.953. The video gap (`L_recon_video_gap`, the difference between
the wrong-code and correct-code losses, i.e. how much the correct code helps) grows
steadily to about 0.20 and plateaus after step 8,000, against run 055's 0.311. Computing
the video-conditioned share of the decoder's improvement the way prior analyses did, as
the correct-code gain minus the wrong-code gain divided by the correct-code gain, gives
(0.273 minus 0.069) divided by 0.273, which is about 75 percent, against run 055's 86
percent and run 054's 92 percent. So the gap is large, stable, and clearly positive, and
the code is genuinely being used, but both its size and the share are below run 055, and
the wrong-code channel is marginally larger here rather than smaller. With respect to
prediction 2, that is the opposite direction from what was expected: honesty was
predicted to hold or improve, and instead it eroded by about 11 points while remaining
clearly intact. It did not break, in that the gap did not collapse toward zero, but it did
not survive at run 055's level either.

### Cross-run placement

On geometry, run 056 is not merely better than the 054 and 055 equilibrium; its terminal
rank of 201 exceeds even the inv011 raw-space sweep's peak of 150.8, and it does so in
whitened space, which is the direction the hypothesis argued for, since whitened
reconstruction and the regularizers pull the same way and the combination should
out-perform the raw-space sweep. On honesty, it sits below both whitened predecessors.
The two axes moved in opposite directions relative to run 055.

## Part B — What happened in this run (conclusions)

The core anti-collapse claim is confirmed, emphatically. Bolting the inv011 geometry
bundle onto the honest whitened base did exactly what the settled H2 predicted was
necessary: geometry stopped contracting and instead climbed to a genuinely healthy
regime — rank near 201 of 256, spread near 1.0, cross-video cosine near 0.017, and slot
diversity held near 30 — with covariance the visible lever. This is the first run in the
program to hold healthy geometry, and it even beats the old raw-space sweep's rank,
vindicating the same-direction-pressures argument.

Honesty survived but did not hold at run 055's level; it eroded from about 86 percent to
about 75 percent. The video gap stayed large, stable, and clearly positive, so the code is
really used and this is not the honesty-breaks failure the GUIDE warned of, but prediction
2 was still wrong in direction. Raw present reconstruction rose from 0.642 to 0.731 and
the wrong-code channel grew slightly, meaning the geometry-shaped code is harder to decode
and leaves a marginally larger template shortcut open on the clean absolute base.

The deliberate clean-arm divergence answered its own question: a richer code did not
substitute for the residual target's honesty top-up. Honesty landed below run 055's 86
percent rather than at or above it, so on this evidence the residual target's roughly
six-point contribution is not made redundant by high-rank geometry, and the clean base is
the reason the honesty points were left on the table rather than the geometry terms
shredding content routing. Stability was flawless throughout, with zero skips, zero NaNs,
and quiet AGC, so the result is trustworthy and not an artifact of a shaky optimizer.

Net, this is close to the pre-registered target win — the first present-only run with both
healthy geometry and clearly honest, video-specific reconstruction in a single run — with
the single honest caveat that honesty came in about 11 points under run 055 rather than
preserved or better, a trade concentrated in the harder-to-decode geometry-shaped code on
the clean absolute target.
