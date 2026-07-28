# Analysis — Run 68 · pre-run hypothesis and reading plan

This document is intentionally written before execution. Run 68 is the clean control needed after
Run 67: it keeps SigLIP2, the revised M=512 bottleneck, unwhitened raw features, the decoder, seed,
EGO4D sampling, and the reconstruction schedule fixed, while removing the two geometry losses.
The comparison therefore asks whether the bottleneck changes alone produce input-dependent codes
and usable reconstruction, rather than asking whether covariance/variance pressure can shape the
code. Run 67's regularized trajectory is retained as a reference, not treated as this run's result.

The main hypothesis is that removing `lambda_var` and `lambda_cov` may lower effective rank or
increase cross-example cosine, because those losses were actively pushing code geometry apart.
That would not automatically mean the bottleneck failed: the decisive evidence is whether the
correct-versus-shuffled reconstruction gap remains positive and whether slot diversity, code std,
and dead-dimension diagnostics stay finite and non-degenerate. Conversely, if reconstruction stays
low while the shuffled gap collapses, the decoder may be using a target prior or the bottleneck may
still be information-blind. Raw loss is interpreted only within SigLIP2's feature space.

The pre-run expectation is therefore deliberately asymmetric: Run 68 may have weaker rank and
cosine geometry than Run 67, but it should still show finite, input-dependent reconstruction if
the architectural bottleneck changes are doing the essential work. The final analysis must use
late-window medians and correct/shuffled gaps, not a single diagnostic batch, and must state clearly
that this run has no covariance or variance regularization.

## 2026-07-26 — terminal analysis

The pre-run plan above is retained as registered intent. Live W&B now reports `crashed` at step
14,350; it is not running and did not produce a W&B-recorded final 15,000-step checkpoint. The
resolved run mode and configuration are correct, and all logged skipped-gradient, nonfinite, and
instability-warning fields remain zero.

The final six available diagnostic rows give correct/rolled reconstruction
`0.21010/0.28945`, exact-chunk gap `0.07945`, std `0.25507`, within-source pair cosine `0.91216`,
effective rank `20.076`, centered slot rank `15.529`, and dead-dimension fraction `0.13019`.
Reconstruction is therefore code-dependent, but the external code is weakly spread, highly
aligned, and low-rank.

Reading Cycle B result: **LOW-RANK DECODABLE, PARTIAL ENDPOINT; GLOBAL COLLAPSE INDETERMINATE**.
The partial trajectory supports the architectural claim that SigLIP input information reaches the
decoder. It does not supply a completed-run shape verdict, prediction evidence, or a cross-source
collapse result.
