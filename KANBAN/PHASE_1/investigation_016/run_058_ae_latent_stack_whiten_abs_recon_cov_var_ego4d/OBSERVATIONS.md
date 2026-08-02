# Observations — run 058

## 2026-07-14 — final W&B read (`mvbx96nv`)

The run finished all 15,000 steps with valid wiring and clean optimization: no skipped steps or
NaN gradients, EGO4D whitening active, present-only mode active, and prediction inactive. The
run-057 recipe did **not** transfer.

At the final diagnostic step (14,500), `c_effective_rank=52.91`, `c_std_mean=0.419`, and
`c_cross_video_cosine=0.863`. Rank had peaked at 80.73 at step 2,000 before contracting, while
cross-video cosine and std moved back into the collapse regime. Centered slot rank remained high
at 30.67, demonstrating that distinct slots did not imply healthy across-video feature geometry.

The decisive failure is reconstruction honesty. `L_recon_present=0.6782` but
`L_recon_shuffled_c=0.6962`, for a video gap of only 0.0180. Relative to the initial 1.0133 loss,
only about **5.4%** of the decoder's improvement depends on the correct video's latent. This is the
pre-registered EGO4D template-shortcut outcome, not a successful substrate transfer.

Verdict: **Collapsed rep / template shortcut**. The run is stable and interpretable, but the
absolute-target EGO4D objective learns mostly shared structure and the train regularizer gains do
not generalize to the fixed validation batch. See [`ANALYSIS.md`](ANALYSIS.md) for the full Cycle-B
read, comparison with SSv2 run 057, and the project-wide `lambda_recon=1` audit.

## 2026-07-16 — correction after validation-batch identity audit

All 16 fixed diagnostic chunks share one EGO4D source UID. The reported cosine is therefore
within-source cross-chunk cosine, and rolling codes swaps adjacent chunks from that same source.
The weak std/rank and small `0.0180` gap remain valid recorded-batch findings, but the 5.4% phrase
does not measure global video conditioning and the batch does not prove a global template.

Current qualified verdict: **WEAK RECORDED-BATCH GEOMETRY AND WEAK EXACT-CHUNK DEPENDENCE; GLOBAL
COLLAPSE/TEMPLATE SHORTCUT INDETERMINATE**. The 2026-07-14 conclusion above is preserved as the
historical interpretation that this audit corrected.
