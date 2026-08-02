# 06 — Decoder, whitening, and reconstruction

This chapter covers three mechanisms often conflated as “reconstruction”: the learned feature
decoder, an optional running feature-mean subtraction, and optional fixed offline whitening.

## Decoder contract

```text
D: latent (B,N_c,D_c) → reconstructed detailed features (B,N_e,D_e)
```

It reconstructs frozen-encoder tokens. It does not emit pixels, RGB frames, optical flow, or VAE
latents.

## Abstract memory

The input slots pass through:

```text
kv_proj: D_c → D_d
```

With shipped values, both widths are 256, but the projection is still learned. The result
`memory ∈ R^(B×N_c×D_d)` supplies keys and values to every cross-attention layer.

## Fixed 3-D output positions

The decoder builds a deterministic table for the encoder's time-major lattice. Width `D_d` is split:

```text
t_dim = floor(D_d/3)
y_dim = floor((D_d-t_dim)/2)
x_dim = D_d-t_dim-y_dim
```

At `D_d=256`, the split is `85/85/86`. Each axis uses sinusoidal functions with exponential
frequencies. The concatenated table has shape `(N_e,256)` and is registered as a buffer, not a
parameter.

This design answers “where should the decoder write?” without learning a per-position content
template.

## Initial cross-attention read

The fixed table goes through non-affine LayerNorm and becomes the initial query. Abstract memory is
both key and value:

```text
hidden, _ = MHA(norm_fixed_pos, memory, memory)
```

`hidden` is therefore a weighted sum of `c`-derived values. Positional codes affect attention
weights but are not injected as residual content.

## Two decoder blocks

Each block:

1. normalizes current hidden content with affine LayerNorm;
2. adds fixed position only to the query;
3. cross-attends to the same abstract memory;
4. adds the attention result;
5. applies affine LayerNorm and `D_d→4D_d→D_d` GELU MLP;
6. adds the MLP result.

Final affine LayerNorm and a `D_d→D_e` linear projection emit `ê`.

An important structural test follows: if abstract memory is zero, the decoder cannot create
position-specific content merely from the fixed code. Biases may create a shared output, but there is
no learned spatial content table to memorize an encoder-mean template.

## Exact parameter formula

For decoder width `D`, `L=2` blocks:

```text
kv projection          = D_c*D + D
initial MHA            = 4*D^2 + 4*D
each decoder block     = 12*D^2 + 13*D
final LayerNorm        = 2*D
output projection      = D*D_e + D_e

P_D = D_c*D+D + 4D^2+4D + L(12D^2+13D) + 2D + D*D_e+D_e
```

The non-affine initial position normalization and fixed position buffer contribute no parameters.

At `D=256,L=2`:

| Encoder | `D_e` | Decoder parameters |
|---|---:|---:|
| V-JEPA2 | 1024 | 2,172,160 |
| SigLIP2/DINOv3 | 768 | 2,106,368 |

## Present reconstruction

The ordinary anchor is:

```text
ê_t = D(c_t)
L_recon = mean_token(1 - cosine(ê_t, stopgrad(e_t)))
```

Both vectors are converted to FP32 and L2-normalized per detailed token with epsilon `1e-6`.
Gradients reach `D` and online `B`. They do not reach `F_c`, `B_EMA`, or `E`.

The legacy `relative_mse` option is:

```text
mean((ê-e)^2) / max(Var_population(e), 1e-8)
```

Cosine is shipped because it removes magnitude as an escape route.

## Prediction-side reconstruction

If `lambda_recon_pred>0`, the same flow output used by `L_flow` constructs `c_hat`, and:

```text
L_recon_pred = reconstruction_loss(D(c_hat), e_future)
```

This gradient reaches:

- `D`, because it decodes;
- `F_c`, because `c_hat` uses `û`;
- online `B`, both through flow conditioning and, in residual mode, the present-latent add-back.

It does not require an extra `F_c` forward. It is the reconstruction objective that can directly
pressure the predictor, unlike present reconstruction.

## Reconstruction blindness

A decoder can obtain similar losses from predicted `c_hat` and true `c⁺` even when prediction
quality is poor. This occurs when the decoder mainly recovers a video-independent mean feature
template. Therefore always compare:

- present `D(c_t)` loss;
- teacher future `D(c⁺)` loss;
- predicted future `D(c_hat)` loss;
- shuffled-latent decoder loss;
- the gap between predicted and teacher readouts.

A low absolute feature-reconstruction loss is not by itself proof of video-specific information.

## Residual feature target

`FeatureMeanTracker` stores one FP32 mean vector at every detailed lattice position:

```text
mean shape = (N_e,D_e)
```

Its first update copies the current training-batch mean. Subsequent updates use:

```text
mean ← 0.99*mean + 0.01*batch_mean
```

The update happens before the same step's reconstruction loss, preventing a zero-mean cold start.
Only training batches update it; fixed validation diagnostics never do.

With `recon_residual_target=True`, the target becomes:

```text
e_residual = e - mean[position]
```

The decoder output is interpreted as residual feature content; a full readout would add the mean
back. The training cosine loss operates on the residual target itself. This removes reward for a
position-specific video-independent template and forces improvements to depend more on video
content.

Storage:

| Encoder | Mean buffer |
|---|---:|
| V-JEPA2, `1024×1024` FP32 | 4 MiB + one boolean |
| frame encoder, `2048×768` FP32 | 6 MiB + one boolean |

The comment that momentum 0.99 is ~99% converged in about 460 updates comes from
`1-0.99^460≈0.9902`.

## Fixed offline ZCA whitening

Whitening acts at the one seam immediately after frozen encoding and before every trainable consumer:

```text
e_w = (e-μ) W
W = U diag((λ+eps)^(-1/2)) Uᵀ
W_inv = U diag((λ+eps)^(1/2)) Uᵀ
e = e_w W_inv + μ
```

This means `B`, `B_EMA`, flow targets, reconstruction targets, tracker means, and diagnostics all
live in the same whitened coordinate system when the flag is active. Whitening only one branch would
make the objective incoherent.

The `FeatureWhitener` stores:

- mean `(D_e,)`, FP32;
- whitening matrix `(D_e,D_e)`, FP32;
- unwhitening matrix `(D_e,D_e)`, FP32;
- initialized boolean.

Storage is approximately:

| `D_e` | Buffers |
|---:|---:|
| 1024 | 8.004 MiB |
| 768 | 4.503 MiB |

`configure` clamps negative eigenvalues to zero, adds positive `eps`, builds both matrices in FP64,
and stores them in FP32. `whiten` and `unwhiten` explicitly disable autocast for the matrix multiply
and return the caller's original dtype.

## Offline statistics contract

Whitening is never fit per batch. `whiten_stats.py`:

1. uses the training split and a pinned encoder;
2. collects an exact deterministic prefix, shipped expectation 12,800 clips;
3. accumulates sums and outer products in FP64;
4. forms the biased maximum-likelihood covariance;
5. uses `torch.linalg.eigh`;
6. saves an artifact whose schema and identity bind encoder fingerprint, dataset identity, seed,
   clip count, eigensolver, and payload hash.

The eigenvalue floor, shipped `1e-4`, is applied when configuring the runtime whitener. This allows a
controlled epsilon sweep from the same raw eigen-statistics.

## Whitening is not normalization or regularization

- RGB normalization changes inputs to `E`.
- Whitening recoordinates frozen feature vectors after `E`.
- Variance/covariance/SIG losses shape learned `c`.
- LayerNorm normalizes individual internal tokens during model computation.

These operations have different axes and purposes.

## Configuration relationships

- `recon_residual_target=True` requires at least one active reconstruction objective.
- Present-only mode requires `lambda_recon>0` and disables prediction-side reconstruction.
- Prediction residual (`predict_residual`) is temporal in abstract space.
- Reconstruction residual (`recon_residual_target`) is a per-position feature-space target.
- Whitening is orthogonal to both residual modes.

Confusing the two “residual” flags is a common oral-exam trap.
