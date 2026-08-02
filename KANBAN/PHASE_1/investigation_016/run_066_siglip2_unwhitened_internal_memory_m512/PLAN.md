# Plan — Run 66, SigLIP 2 unwhitened internal-memory M=512

## Role

First same-commit encoder-substrate comparison at the settled unwhitened M=512 operating point:
take the V-JEPA control ([`../run_064_unwhitened_internal_memory_m512`](../run_064_unwhitened_internal_memory_m512/))
and change **only** the frozen encoder to SigLIP 2 ViT-B/16.

## No code change required

This is a pure configuration change. The bottleneck `in_proj` (`D_e -> M`) and the decoder output
projection (`decoder_dim -> D_e`) already resolve their shapes from the selected encoder's
`EncoderSpec`, and `FeatureLayout` drives the ConvNeXt reshape. Swapping `--encoder`/
`--encoder-revision` is sufficient; `models.py`, `losses.py`, and `train.py` are untouched. Confirmed
by the shipped `siglip2_vitb16` adapter and the existing SigLIP run 059.

## Locked config

Full EGO4D; SigLIP 2 ViT-B/16 @ `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`; seed 42; batch 64;
15,000 steps; `N_c=32`; external `D_c=256`; `M=512`; three latent blocks; 512-by-4 decoder;
present-only absolute cosine reconstruction; `lambda_recon=1`; every auxiliary geometry weight zero;
no whitening or residual target. Only intended delta vs run 064: the encoder alias + revision (and
the encoder-only `frame_microbatch` throughput knob, settled by resource preflight).

## Readout

Run present-only Reading Cycle B. Record late median `L_recon`, fixed-batch correct/shuffled losses
and gap, within-source conditioned share, stability, geometry context (std/cosine/effective rank/
slot rank), resource use, and provenance. Compare the *within-run* diagnostics against run 064; do
not compare raw cross-encoder cosine values as if they shared a target space. This single arm cannot
establish a global cross-source preservation claim on the fixed within-source EGO4D batch.
