# Observations — investigation_019

The first V-JEPA2 lane finished with `lambda_var=lambda_cov=0` and is therefore retained only as a
no-geometry ablation. It cannot be compared as though it belonged to the corrected
covariance-plus-variance sweep.

The first DINOv3 launch inherited the same mistaken zero weights. It was stopped before completion;
tmux, all four training processes, and all GPU compute processes were verified absent. The exact
incorrect W&B runs `6ua1sfz7`, `u3s8jxme`, `fjghdjj8`, and `piaiaws0` were deleted. Their pod
checkpoints and logs remain preserved as aborted operational evidence and must never be resumed
under the corrected identities.

The corrected recipe is `lambda_var=0.5`, `lambda_cov=0.01`, `lambda_sigreg=0`, and
`lambda_slot=0`. Corrected runs use fresh `inv019_covvar_*` output identities and the W&B group
`inv019_three_encoder_bottleneck_shape_covvar`.
