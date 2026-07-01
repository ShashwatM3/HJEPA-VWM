# Current Pipeline and Architecture

This note summarizes the current Phase 1 HJEPA-VWM system as implemented in the repo. It is a
working reference for the full coarse-latent training path, including the reconstruction objectives
and their loss modes.

## Goal

HJEPA-VWM learns a video world model in latent space, not pixel space. The core object is a compact
abstract latent `c_t` that should be rich enough to carry useful video state and predictive enough
to support future-latent dynamics.

The model has two representation levels:

- `e_t`: detailed frozen V-JEPA 2 features, shape `(B, 1024, 1024)`.
- `c_t`: trainable abstract bottleneck features, shape `(B, 32, 256)`.

The current implementation is Phase 1: frozen encoder, bottleneck, EMA target bottleneck, coarse
flow predictor, reconstruction decoder, collapse diagnostics, and W&B logging.

## Data Path

Each training sample provides two clips from Something-Something V2:

- `context_clip`: 8 frames at 256x256.
- `target_clip`: 8 frames at 256x256, offset by `horizon_k` original video frames.

The dataloader decodes only the required frames, applies resize/crop/color preprocessing, and uses
the frozen V-JEPA-compatible image normalization.

## Modules

### Frozen Encoder `E`

`E` is `facebook/vjepa2-vitl-fpc64-256`, loaded from Hugging Face and never trained.

```text
context_clip -> E -> e_t
target_clip  -> E -> e_plus
```

Both `e_t` and `e_plus` are detailed latent token grids with 1024 tokens and 1024 feature dimensions.
The encoder has `requires_grad=False`; targets are detached before loss computation.

### Bottleneck `B`

The online bottleneck maps detailed features into the compact abstract state:

```text
e_t -> B -> c_t
```

It uses a 1024-to-256 projection, ConvNeXt-style mixing over the frozen encoder token grid, and
32 learned query slots with cross-attention. Orthogonal query initialization and zero-init output
MLP are built into the current implementation.

### EMA Bottleneck `B_EMA`

The target bottleneck is an EMA copy of `B`:

```text
e_plus -> B_EMA -> c_plus
```

`c_plus` is a stop-gradient target. EMA updates happen only after a successful optimizer step.

### Coarse Flow `F_c`

The coarse flow predicts a rectified-flow velocity in abstract-latent space:

```text
F_c(z_c, tau, c_t) -> u_c_hat
```

The usual full-latent target is `c_plus`. In residual mode, the target is:

```text
Delta = c_plus - c_t_ema
```

where both ends are produced by `B_EMA`. Residual mode keeps the target temporal and detached while
the prediction is still conditioned on online `c_t`.

### Reconstruction Decoder `D`

`D` maps the abstract state back toward frozen detailed features:

```text
c_t -> D -> e_hat
```

This decoder is not the final video/frame generator. It is a Phase 1 reconstruction anchor used to
pressure `c_t` to retain detailed, video-specific information.

## Training Step

The full Phase 1 step is:

1. Encode the context clip with frozen `E` to get `e_t`.
2. Run online `B(e_t)` to get `c_t`.
3. Encode the target clip with frozen `E` to get `e_plus`.
4. Run EMA `B_EMA(e_plus)` to get detached `c_plus`.
5. Build the flow target, either `c_plus` or residual `Delta`.
6. Sample noise `eps` and flow time `tau`.
7. Interpolate `z_c = (1 - tau) * eps + tau * target`.
8. Train `F_c(z_c, tau, c_t)` to predict `target - eps`.
9. Add collapse/geometry regularizers on online `c_t`.
10. Add reconstruction losses when their weights are nonzero.
11. Backprop through online `B`, `F_c`, and `D`.
12. Apply AGC, global grad clipping, skip guard, optimizer step, and EMA update.

## Losses

### Flow Loss

```text
L_flow = mean((u_c_hat - u_c)^2)
```

where:

```text
u_c = flow_target - eps
```

This is the main future-abstract prediction loss.

### Variance Floor

```text
L_var = mean(max(0, 1.0 - std(c_t_dim)))
```

The variance floor prevents constant-code collapse in `c_t`. Current successful experimental
commands use `lambda_var=0.5`, even though the dataclass default remains `0.10`.

### SIGReg

SIGReg regularizes the pooled `c_t` distribution toward an isotropic unit Gaussian. It is used as a
rank/utilization lever and is controlled by:

```text
--lambda-sigreg
--sigreg-warmup-steps
```

When active, it is linearly warmed up so the online bottleneck geometry does not move faster than the
EMA target branch can track.

### Optional Covariance and Slot Terms

`lambda_cov` and `lambda_slot` exist as experimental regularizers. They are off by default and are
not part of the current best-known recipe.

## Reconstruction Objectives

There are two reconstruction anchors that reuse the same decoder `D` and the same reconstruction loss
function.

### Present Reconstruction

```text
e_t -> B -> c_t -> D -> e_hat_t
L_recon = recon_loss(e_hat_t, e_t)
```

This shapes `B` and `D`. The frozen detailed target `e_t` is detached.

### Prediction-Side Reconstruction

```text
c_hat -> D -> e_hat_plus
L_recon_pred = recon_loss(e_hat_plus, e_plus)
```

`c_hat` is the predicted future abstract endpoint from the coarse flow. In residual mode:

```text
c_hat = c_t + predicted_Delta
```

This routes reconstruction pressure through the predicted future abstract state while still comparing
against frozen detailed target features.

## Reconstruction Loss Modes

The reconstruction loss is centralized in `losses.reconstruction_loss` and selected with:

```text
--recon-loss-mode cosine
--recon-loss-mode relative_mse
```

### `cosine`

This is the current new reconstruction objective.

For each tubelet token, both predicted and target detailed vectors are normalized along `D_e`:

```text
e_hat_unit = e_hat / ||e_hat||
e_unit     = e     / ||e||
```

The loss is:

```text
L_recon = mean(1 - cos(e_hat, e))
```

Equivalently, this is one half of the squared distance between unit-normalized tubelet vectors:

```text
0.5 * mean(||e_hat_unit - e_unit||^2)
```

This removes feature magnitude as an escape route. The decoder has to align directionally with the
frozen V-JEPA features instead of lowering loss through norm drift.

### `relative_mse`

This is the legacy reconstruction objective:

```text
L_recon = mean((e_hat - e)^2) / Var(e)
```

The denominator makes the number dimensionless and comparable across runs, but it does not remove
the decoder's ability to change feature magnitudes. It remains available as an explicit ablation
mode for comparison with earlier investigations.

## Total Objective

The full objective is assembled from active weights:

```text
L_total =
    L_flow
  + lambda_var * L_var
  + lambda_sigreg * sigreg_scale * L_sigreg
  + lambda_cov * L_cov
  + lambda_slot * L_slot
  + lambda_recon * recon_scale * L_recon
  + lambda_recon_pred * recon_scale * L_recon_pred
```

Terms with zero weights are logged or skipped according to their implementation path, but they do not
contribute gradients.

## Diagnostics

The important Phase 1 diagnostics are:

- `c_std_mean`, `c_std_median`, `c_dead_dim_frac`: variance health of `c_t`.
- `c_cross_video_cosine`: whether different videos get distinct `c_t`.
- `c_effective_rank`: utilization of the 256-dimensional abstract feature space.
- `c_plus_effective_rank`: EMA target utilization.
- `coarse_vs_copy_ratio`: whether `F_c` beats the copy/no-change baseline.
- `coarse_vs_batch_mean_ratio`: whether `F_c` beats a batch-mean target.
- `L_recon_present`: present reconstruction quality.
- `L_recon_cplus`: reconstruction quality from true future abstract target.
- `L_recon_chat`: reconstruction quality from predicted future abstract state.
- `grad_norm`, `grad_skipped`, `instability_warn`, `agc_*`: training stability.

## Current Experimental Recipe

The current full-recipe investigation uses:

```text
horizon_k = 12
lambda_var = 0.5
lambda_sigreg = 5.0
lambda_recon = 0.05
lambda_recon_pred = 0.05
recon_loss_mode = cosine
predict_residual = true
decoder_dim = 512
decoder_blocks = 4
n_c = 32
lr_bottleneck = 1e-4
lr_coarse_flow = 1e-4
```

The key acceptance bottleneck remains prediction quality: representation health can pass while
`coarse_vs_copy_ratio` stays near 1. A useful run must keep `c_t` healthy and make the predicted
future abstract state beat the copy baseline.
