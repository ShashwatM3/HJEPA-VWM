# Next Steps - investigation_013

## Current Recommendation

Stop trying to make reconstruction alone hold geometry — that hypothesis (H2) is now
settled negative across three successive content objectives (run 052 absolute target, run 053
residual target). The residual target is a keeper for HONESTY (it belongs in the recipe going
forward), but the next lever must attack the two things reconstruction cannot fix on its own:
the anisotropy of the target feature space (so reconstruction defends more than ~10 directions)
and the ease with which the 32 slots merge.

## Closure / Carry-Forward Status

- Status: **CLOSED**.
- Conclusion to carry forward: the residual reconstruction target fixes the run-052 template
  shortcut (reconstruction becomes provably video-specific, ~77% video-conditioned, video gap
  +0.433 and growing) but does NOT hold representation geometry (rank 10.5, cosine 0.929,
  std 0.255 — marginally worse than run 052). H1 solved, H2 confirmed: an explicit anti-collapse
  force is a requirement of this architecture.
- Runs covered: 053.

## Follow-Up Chain

This investigation feeds into `investigation_014` (the offline V-JEPA rank-budget probe) and
then `investigation_015` (latent-stack bottleneck + whitening). The reason: run 053's
contraction mechanism — cosine reconstruction in raw feature space is satisfiable by
reproducing V-JEPA's few dominant directions, so weight decay contracts the rest of `c` — is
a claim about the FEATURE SPACE. Investigation_014 measured that space directly (pooled `e`
entropy rank ~193/1024 with a long low-energy tail), confirming the mechanism, and
investigation_015 acts on it: whiten the features so reconstruction must defend many equally
weighted directions, and add slot self-attention so slots cannot cheaply merge. Run 054 reruns
run 053's exact recipe with those two changes.

## Original Notes Preserved

Concrete next lever spelled out in [`run_053_ae_sharp_slots_residual_recon/ANALYSIS_053.md`](run_053_ae_sharp_slots_residual_recon/ANALYSIS_053.md):
rerun this exact recipe with the inv011-style geometry terms restored — variance floor
(`lambda_var`, target std 1.0, which directly attacks the 0.25 amplitude problem) plus a small
covariance penalty — while KEEPING `--recon-residual-target`. That combination is now
unconfounded: the geometry regularizers alone produced rank 90-150 (runs 043-051) and the
residual target routes real content through `c_t` (this run), so `L_recon_video_gap` becomes
the honest monitor for whether the geometry terms preserve or destroy the content routing. A
secondary, cheaper knob if the contraction persists: lower the bottleneck's weight decay from
0.05, which was the unopposed contraction force in this run.

Note the ordering that actually happened: rather than jump straight to the geometry-on
combination, the human first inserted investigation_014 (measure the encoder rank budget) and
investigation_015 (whitening + latent stack on the residual-target recipe) to test whether the
feature-space and architecture changes alone could hold geometry before re-adding regularizers.
Run 054 answered "no, but at a much better equilibrium (rank ~22, 92% video-conditioned)," so
the geometry-on combination remains the pending confirmatory run.
