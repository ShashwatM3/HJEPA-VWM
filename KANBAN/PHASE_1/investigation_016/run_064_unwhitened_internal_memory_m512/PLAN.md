# Plan — Run 64, EGO4D internal memory M=512

## Role

Establish the practical baseline for the new coherent bottleneck: raw V-JEPA features enter a
512-wide memory/query/latent processor, and only the refined slots are projected to external
`D_c=256`.

## Locked config

Full EGO4D; V-JEPA2-L; seed 42; batch 64; 15,000 steps; `N_c=32`; three latent blocks; 512-by-4
decoder; present-only absolute cosine reconstruction; `lambda_recon=1`; every auxiliary geometry
weight zero; no whitening or residual target.

## Readout

Run present-only Reading Cycle B. Record late median `L_recon`, fixed-batch correct/shuffled losses
and gap, within-source conditioned share, stability, geometry context, resource use, and provenance.
This arm alone cannot establish a width trend or global cross-source preservation.
