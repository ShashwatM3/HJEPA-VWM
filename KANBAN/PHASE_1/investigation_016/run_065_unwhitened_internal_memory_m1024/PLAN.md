# Plan — Run 65, EGO4D internal memory M=1024

## Role

Test the no-early-channel-reduction upper bound: the V-JEPA2-L `D_e=1024` channels remain 1,024-D
through memory construction, learned queries, and all three latent blocks before a final projection
to external `D_c=256`.

## Locked config

Byte-for-byte recipe parity with the 512 sibling except `bottleneck_mixer_dim=1024` and derived
parameter/initialization hashes. Full EGO4D; seed 42; batch 64; 15,000 steps; `N_c=32`; 512-by-4
decoder; present-only absolute cosine reconstruction; all auxiliary weights zero; no whitening.

## Acceptance

Both arms must be valid. To call 1,024 improved preservation, require at least 0.01 lower late
median `L_recon`, at least 0.005 larger late correct-versus-shuffled gap, and no decrease in
within-source conditioned share. Otherwise prefer 512 on its roughly 3.86-times smaller bottleneck.
No global cross-source preservation claim is permitted from the current fixed batch.
