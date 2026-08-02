# Plan — run 058 EGO4D transfer

> **Retrospective record (2026-07-14).** Run 058 had already finished before this KANBAN subtree
> was committed. This file reconstructs the intended plan from the run description, operator
> guide, W&B config, and final analysis; it is not represented as a contemporaneous preregistration.

## Question

Does the settled SSv2 run-057 present-only recipe transfer to full EGO4D when the dataset and its
V-JEPA whitening statistics are the only intended recipe changes?

## Planned configuration

- frozen V-JEPA2 ViT-L/16 detailed features;
- full EGO4D, seed 42, 15,000 steps, physical batch 64;
- present-only absolute-target cosine reconstruction with fixed offline whitening;
- Perceiver bottleneck with 3 latent blocks and `(N_c,D_c)=(32,256)`;
- decoder width/depth 512/4;
- `lambda_recon=0.05`, `lambda_var=0.5`, `lambda_cov=0.01`;
- prediction, residual reconstruction, SIGReg, and slot penalty inactive;
- `lr_bottleneck=1e-4`, `lr_coarse_flow=1e-4`.

No production-code change was planned. The operator procedure lives in [GUIDE.md](GUIDE.md).

## Planned gates

1. Verify full and tiny EGO4D roots and the dataset CLI.
2. Fit and validate EGO4D-specific V-JEPA whitening statistics; never reuse SSv2 statistics.
3. Pass Stage 0 with whitening/present-only wiring active and prediction/residual paths inactive.
4. Launch from scratch into a unique checkpoint/log/W&B identity.
5. Require finite optimization, no skipped/NaN gradient, and correct dataset/stats config.
6. Apply Reading Cycle B: abstract geometry plus reconstruction honesty, with
   `L_recon_video_gap` as the decisive content gate.

## Planned verdict

Transfer would count as successful only if high abstract rank and centered slot diversity held
while shuffled-latent reconstruction became materially worse than correctly conditioned
reconstruction. Stable optimization or low raw reconstruction alone would not count.

The actual result and the limitations of the historical implementation are recorded in
[ANALYSIS.md](ANALYSIS.md) and [OBSERVATIONS.md](OBSERVATIONS.md).
