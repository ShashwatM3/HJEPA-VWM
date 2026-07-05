# Observations - investigation_015

No run data yet. Run 054 (`ae_latent_stack_whiten_recon_only`) launches with
[`GUIDE.md`](GUIDE.md); read it against the attribution framework in
[`README.md`](README.md).

Registered priors (2026-07-04, before launch):

- P1 (bottleneck fix, slot axis): `c_slot_diversity_rank_centered` does NOT repeat the
  052/053 monotone decline (12.5 -> 8.5 in 053); slot self-attention keeps slots
  differentiated while recon improves.
- P2 (whitening, feature axis): `c_effective_rank` does not halve during the post-4k
  contraction window as in 053 (19.9 -> 10.5); recon improvement and feature rank stop
  moving in opposite directions (053's Q5 failure).
- P3 (honesty carried over): `L_recon_video_gap` clearly positive and growing, with
  `L_recon_shuffled_c` pinned near ~1.0 in whitened space.
- P4 (stability): no skips/NaNs; AGC quiet on B and D; `whiten_active=1` throughout.

*(Per PROTOCOL: append dated sections for measured data; do not rewrite the above.)*

## 2026-07-04 — Interim read at step 10,000/15,000 (run `lx1b6gw2`, live)

Full analysis: [`ANALYSIS_054.md`](ANALYSIS_054.md). Measured against the priors:

- **P4 ✅** spotless: 0 skips, 0 NaNs, AGC never clipped, `whiten_active=1` throughout,
  `recon_mean_norm` ~89.7 (whitened space confirmed; 053 raw: ~1624).
- **P3 ✅** strongest honesty result yet: `L_recon_shuffled_c` pinned 0.969–0.975 all
  run (the pre-registered ~1.0 whitened-space prediction), `L_recon_video_gap` 0 →
  0.318 growing at every diag point; **~92% of decoder improvement is
  video-conditioned** (053: ~77%).
- **P2 ⚠️** no halving: `c_effective_rank` peak 34.7 @2.5k → 24.4 @10k (−20% post-4k
  vs 053's −47%), decelerating — but still contracting while recon improves, and rank
  never exceeded the ~32-slot mechanical span (init rank 31.0 with `c_std_mean=0`), so
  no positive evidence for the whitening mechanism at `whiten_eps=1e-4`.
- **P1 ❌ letter / ⚠️ spirit:** `c_slot_diversity_rank_centered` declined monotonically
  31 → 12.7, but the init is mechanical and the curve is plateauing (~−0.23/1k steps)
  above 053's terminal 8.5.
- Geometry equilibrium far better than 053 across the board @10k: cosine 0.796 (best
  0.783 @7.5k) vs 0.929; std 0.422 vs 0.255; rank 24.4 vs 10.5.

README 2x2 cell: **"neither delta sufficient"** — H2 re-confirmed at a third level of
content-objective quality; the §4 bottleneck-only control run is now load-bearing for
attribution. Final-numbers addendum after step 15,000.

## 2026-07-04 — Final read: run finished (state `finished`, last diag step 14,500)

Extrapolations confirmed; verdict and 2x2 cell unchanged. Final @14.5k: rank **21.9**
(predicted 22–23), slot centered **12.1**, cosine **0.820** (kept creeping up from its
0.783 best @7.5k), std **0.397**, `L_recon_present` **0.652** (flat from 13k),
shuffled-c pinned **0.975**, gap **0.322**, video-conditioned share **92.0%**.
Stability clean to the end.

Sharpest new fact: over the final 1.5k steps recon gained *nothing* (0.6524 → 0.6521)
while rank, cosine, and std all kept degrading — geometry paid with zero content
benefit, isolating the residual contraction force (unopposed `weight_decay=0.05`) from
the content objective. Full detail: [`ANALYSIS_054.md`](ANALYSIS_054.md) addendum.

## Original Notes Preserved

Run 054's per-run triad now lives in its own folder:
[`run_054_ae_latent_stack_whiten_recon_only/`](run_054_ae_latent_stack_whiten_recon_only/)
(DESCRIPTION / OBSERVATIONS / NEXT_STEPS). The full narrative analysis remains at the
investigation level in [`ANALYSIS_054.md`](ANALYSIS_054.md).
