# Next Steps - run 053 `ae_sharp_slots_residual_recon`

**Verdict:** **Low-rank decodable** - the code is video-specific and honestly decoded, but it
lives in far too few effective directions; this is NOT template collapse, so the fix to make is
different from run 052's.

## Immediate Consequence

Keep the residual reconstruction target — it earned a permanent place in the recipe by fixing
honesty (77% video-conditioned, video gap +0.433). Do NOT keep trying to make reconstruction
alone hold geometry; run 053 settles that it cannot. The next run must add force on the two
axes reconstruction cannot fix by itself: the feature-space anisotropy (so reconstruction
defends more than ~10 directions) and slot merging.

## Linkage To The Research Chain

- This run belongs to `investigation_013`: the residual reconstruction target as the run-052
  template-collapse fix.
- What it hands forward: H1 (template shortcut) is solved and H2 (recon alone cannot hold
  geometry) is confirmed. The contraction mechanism it exposed is a claim about the FEATURE
  SPACE — cosine reconstruction in raw space is satisfiable by reproducing V-JEPA's few
  dominant directions, so weight decay contracts the rest.
- Next in the chain: [investigation_014](../../investigation_014/) measured that space directly
  (pooled `e` entropy rank ~193/1024 with a long low-energy tail), confirming the mechanism;
  then Run 054 [`ae_latent_stack_whiten_recon_only`](../../investigation_015/) (`lx1b6gw2`)
  reran THIS exact recipe with two changes — fixed offline whitening (equalize the target
  directions so reconstruction must defend many) and the Perceiver latent-stack bottleneck
  (slot self-attention so slots cannot cheaply merge).

## Concrete next run (from ANALYSIS_053)

Rerun this exact recipe with the inv011-style geometry terms restored — variance floor
(`--lambda-var`, std target 1.0, which directly attacks this run's 0.25 amplitude problem)
plus a small covariance penalty — while KEEPING `--recon-residual-target`. That combination is
unconfounded: geometry regularizers alone produced rank 90-150 (runs 043-051) and the residual
target routes real content through `c_t` (this run), so `L_recon_video_gap` becomes the honest
monitor for whether the geometry terms preserve or destroy the content routing. Secondary,
cheaper knob if contraction persists: lower the bottleneck's `weight_decay` from 0.05 (it was
the unopposed contraction force here).

The human's actual ordering inserted the whitening + latent-stack test (run 054) before the
geometry-on combination, to see whether the feature-space and architecture changes alone could
hold geometry. They landed at a much better equilibrium (rank ~22, 92% video-conditioned) but
still contracted, so the geometry-on combination remains the pending confirmatory run.

## If This Branch Is Revisited

- Re-read with the present-recon-only cycle (Cycle B); never apply the copy/batch-mean gates.
- Do not compare `L_recon_present` across the absolute (052), residual (053), and whitened
  (054) runs — different target spaces. Compare trends and the honesty gap.
- Do not resume from this checkpoint into a run with a different bottleneck architecture; the
  latent-stack change makes old checkpoints shape-incompatible by design.
