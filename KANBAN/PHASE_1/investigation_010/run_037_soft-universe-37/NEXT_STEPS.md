# Next Steps - run 037 `soft-universe-37`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Verdict:** **Healthy rep, no predictor** - Representation health is not the limiting issue; F_c still does not beat copy.

## Immediate Consequence

The representation is good enough that the next question is predictor conditioning/dynamics, not basic c_t collapse. Avoid recommendations that only improve rank.

## Linkage To The Research Chain

- This run belongs to `investigation_010`: clean residual run with optimizer/regularization plumbing.
- Parent investigation next direction: The research moved to reconstruction geometry and decoder honesty, because prediction failure could no longer be blamed only on rank collapse.
- Next chronological W&B run: Run 038 [`new_recon_loss`](../../investigation_011/run_038_new_recon_loss/)

## If This Branch Is Revisited

- Re-read the run with `KANBAN/README_for_reading_experiments.md` before copying any config.
- Compare only compatible modes: full-prediction to full-prediction, present-only to present-only.
- Preserve the exact config deltas, checkpoint path, and W&B run id so later investigations can trace the branch without relying on memory.
- Do not use this run as evidence for a later-stage component that was inactive in its config.

## Original Notes Preserved

**Connection:** this run reframed the whole program. Because a fully healthy, rank-61,
video-specific c STILL only ties copy, the next branch (investigation_011) stopped trying to
improve the representation and instead attacked reconstruction GEOMETRY and decoder HONESTY: it
introduced the cosine reconstruction loss (which the very next run 038 applied to this exact
recipe, dropping L_recon_present from ~0.585 to ~0.346) and the fixed-position decoder (removes
the decoder's template loophole). The deeper open question it left — is the copy gate even
reconstruction-fixable, or is it a task/horizon property (c barely moves in the predictable
band)? — is the thread the residual/rho analysis (inv009) and the present-only arc keep circling.
Watch on any follow-up: coarse_vs_copy_ratio decisively < 1, not just c rank.
