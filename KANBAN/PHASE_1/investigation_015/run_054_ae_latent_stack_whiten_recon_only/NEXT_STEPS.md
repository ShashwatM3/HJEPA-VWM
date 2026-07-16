# Next Steps - run 054 `ae_latent_stack_whiten_recon_only`

**Verdict:** **Low-rank decodable** - the strongest honesty result of the program (92%
video-conditioned) at a much better geometry equilibrium than run 053, but geometry still
contracts, placing this run in the "neither delta sufficient" cell of the attribution matrix.

## Immediate Consequence

Two things are now settled and one is open. Settled: (1) H2 — reconstruction alone never holds
geometry (third confirmation), so the next full attempt MUST include an explicit anti-collapse
term; (2) the whitened-residual recipe is the honest content channel to keep. Open: attribution
between the two deltas cannot be made from this run, because feature rank never left the range
the 32 slots can explain by themselves.

## Linkage To The Research Chain

- This run belongs to `investigation_015`: latent-stack bottleneck + whitened features.
- Previous: Run 053 (raw-space residual target) — this run whitens it and adds slot competition.
- ~~Next planned: Run 055
  [`ae_latent_stack_whiten_abs_recon`](../run_055_ae_latent_stack_whiten_abs_recon/).~~ Completed
  as W&B `nzz64pl6`. Removing the residual target left geometry unchanged and reduced the
  conditioned share from about 92% to 86%, so the residual target was useful but was not the
  geometry lever.
- Runs 056 and 057 then supplied the explicit anti-collapse test. Run 057, covariance plus the
  variance floor without SIGReg, is the settled SSv2 present-only recipe.

## Next steps, ordered by information per GPU-hour (from ANALYSIS_054)

1. **Read the finished tail (free, done).** The final-numbers addendum confirmed the plateau
   projections almost exactly (rank 21.9, cosine 0.820, std 0.397) and found no late
   re-acceleration in the LR decay tail.
2. **Bottleneck-only control (~6h, the load-bearing tiebreaker).** Same seed and recipe with
   the two `--whiten-*` flags removed (keep the residual target). This cleanly splits the
   053->054 improvement between the latent-stack architecture and the whitening, and decides
   whether whitening earns a permanent place before it gets entangled with geometry terms.
3. **The "neither sufficient" follow-up (~6h).** Keep whitening + residual target and add the
   variance floor (its std target 1.0 directly attacks this run's 0.42 spread ceiling; the
   logged `L_var`=0.548 shows the term already "sees" the deficit) plus a small covariance
   penalty. Keep `L_recon_video_gap` as the monitor for whether the geometry terms preserve or
   destroy the 92% content routing. Runs 043-051 showed the geometry terms alone reach rank
   90-150; the open question is only the combination on the honest channel.
4. **If contraction persists even there, lower the bottleneck's weight decay.** 0.05 has been
   the unopposed contraction force across runs 052, 053, and 054.
5. **On a geometry-holding present-only checkpoint, transfer to a full-prediction run** and
   measure against the standing Phase 1 copy / batch-mean gates.

## If This Branch Is Revisited

- Present-recon-only cycle only; never apply the copy/batch-mean gates.
- Never quote `L_recon_present` across whitened (054) and raw (052/053) runs.
- Do not resume across the latent-stack architecture change or the whitening seam; old
  checkpoints are shape-incompatible by design, and a whitened-space checkpoint must be
  evaluated in whitened space (the checkpoint embeds its whitener; `drift_probe.py` rebuilds it).
