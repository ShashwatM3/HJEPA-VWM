# Plan — Run 67, DINOv3 unwhitened internal-memory M=512

## Role

Completed DINOv3 encoder-substrate reproduction of the run-066 unwhitened M=512 recipe: change only
the frozen encoder to DINOv3 ViT-B/16 while preserving the full EGO4D scientific configuration.

## No code change required

This was a pure configuration run on branch `codex/task3-dino-run066`, commit
`083cf8a6e87168702efe46ac6bfe485756dcb439`. The bottleneck input projection and decoder output
projection resolved their dimensions from the selected encoder's `EncoderSpec`; no training-code
change was required for the experiment.

## Locked config

Full EGO4D; `dinov3_vitb16` @ `5931719e67bbdb9737e363e781fb0c67687896bc`; seed 42; batch 64;
15,000 steps; frame microbatch 32; `N_c=32`; external `D_c=256`; `M=512`; three latent blocks;
512-by-4 decoder; present-only absolute cosine reconstruction; `lambda_recon=1`; every auxiliary
geometry weight zero; no whitening or residual target; no resume.

## Readout

Record the verified completion identity, final W&B snapshot, checkpoint SHA-256, provenance
artifact, and present-only Reading Cycle B verdict. Preserve the run-066 late-window and Run 064
comparison tables as explicit placeholders until the required W&B window is extracted.
