# Observations - run 055 `ae_latent_stack_whiten_abs_recon`

No run data yet. Launch per [`GUIDE.md`](GUIDE.md); analyze with Reading Cycle B.

Registered priors (2026-07-05, before launch):

- P1 (honesty, the decisive axis): the absolute target re-opens a partial template
  channel in whitened space (per-position mean structure survives whitening). Expected
  outcome is intermediate between 052 and 054: `L_recon_shuffled_c` NOT pinned at ~1.0
  (it declines as the decoder learns the whitened per-position template), and the
  video-conditioned share of decoder improvement lands clearly below run 054's 92%.
  - If shuffled-c stays pinned ~0.97-1.0 and the share stays ~90%+: whitening alone
    blocks the template, and the residual target is redundant in whitened space.
  - If the share drops toward 052's ~15%: whitening does not defend honesty at all;
    the residual target stays mandatory.
- P2 (geometry): H2 is settled — with zero geometry regularizers and weight decay 0.05
  unopposed, expansion then contraction is expected regardless of target. The read is
  WHERE the equilibrium lands vs run 054's terminal rank 21.9 / cosine 0.820 /
  std 0.397. A materially lower equilibrium would suggest the residual target was
  itself contributing defended directions.
- P3 (stability): no skips/NaNs, AGC quiet on B and D, `whiten_active=1` throughout,
  and the wiring confirmations `recon_target_residual=0.0`, `recon_mean_norm=0.0`.

Comparability guards: `L_recon_present` absolute values are comparable to run 054 ONLY
in the loose sense of sharing the whitened space — the absolute target is strictly
easier than the residual target, so compare honesty shares and geometry curves, not raw
loss values. Never compare loss values to 052/053 (raw space).

*(Per PROTOCOL: append dated sections for measured data; do not rewrite the above.)*
