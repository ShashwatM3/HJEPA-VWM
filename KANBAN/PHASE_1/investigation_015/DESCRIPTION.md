# Investigation 015 - Latent-stack bottleneck + whitened features (AE-only)

## Status

OPEN — runs 054–057 are complete. The clean whitening-versus-latent-stack attribution control
remains unrun, but the operational SSv2 recipe and regularizer conclusion are settled.

## Question

With the run-053 recipe held fixed (present-only reconstruction, cosine loss, residual
target, zero geometry regularizers), does the combination of

- the **Perceiver-style latent-stack bottleneck** (repeated read/compete/refine slot
  updates — an architecture fix, not a hypothesis under test), and
- **fixed offline whitening of the frozen V-JEPA features** (the hypothesis under test,
  motivated by investigation_014's rank budget: pooled `e` entropy rank ~193/1024 with a
  long low-energy tail)

hold representation geometry (rank, spread, slot differentiation) that runs 052/053 lost,
while keeping reconstruction honestly video-specific?

## Context

- Run 052 (inv012): absolute cosine recon collapsed via the video-independent template
  (~85% of recon improvement needed no information through `c_t`).
- Run 053 (inv013): residual target fixed the honesty problem (`L_recon_video_gap`
  +0.433, growing) but geometry collapsed anyway (rank 10.5, cosine 0.929, std 0.255)
  — H2 confirmed: recon pressure alone defends only the ~10 directions V-JEPA's energy
  concentrates in; weight decay contracts the rest.
- Whitening equalizes the target directions, so cosine recon pressure must defend many
  more directions of `c` per unit of loss; the latent stack adds slot competition so 32
  slots cannot cheaply merge onto the same readout.
- Implementation: `models.BottleneckLatentBlock` / `models.FeatureWhitener`,
  `whiten_stats.py`, `--whiten-features` wiring (2026-07-04, this branch).

Both changes ship in ONE run (pod time constraint). [`README.md`](README.md) defines how
to attribute the outcome to each change independently from the logged metrics;
[`GUIDE.md`](GUIDE.md) is the pod launch procedure, including the one-time whitening
statistics prerequisite.

## Current conclusion (2026-07-16)

- Run 054 made reconstruction maximally honest but did not hold geometry: rank `21.93`,
  centered slot rank `12.13`, and about `92%` video-conditioned decoder improvement.
- Run 055 removed the residual target. Geometry was unchanged and honesty fell to about `86%`,
  so the residual target remains useful at no geometric cost.
- Run 056 added variance, covariance, and SIGReg. Geometry became genuinely strong (rank
  `201.58`, centered slot rank `29.87`) while reconstruction remained code-conditioned.
- Run 057 removed only SIGReg and improved the joint result: rank `208.22`, reconstruction
  `0.71285`, video gap `0.22681`, broader attention, and calmer gradients. Covariance plus the
  variance floor—not SIGReg—is the settled geometry bundle.

Run 057 therefore became the absolute-target SSv2 present-only control transferred into
[`investigation_016`](../investigation_016/). These runs do not establish forecasting success;
`F_c` was inactive throughout. The residual-target-plus-covariance/variance combination remains
unrun, so run 057 settles the geometry bundle, not the final honesty objective.
