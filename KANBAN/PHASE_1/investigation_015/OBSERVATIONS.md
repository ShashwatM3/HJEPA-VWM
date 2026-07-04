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
