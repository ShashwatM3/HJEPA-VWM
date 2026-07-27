# Sweep plan — external latent shape across three encoders

## Factorial design

Each encoder receives this identical queue:

| Queue | `N_c` | `D_c` | External scalars | Role |
|---:|---:|---:|---:|---|
| 1 | 32 | 256 | 8,192 | center control; reuse exact prior for all three lanes |
| 2 | 16 | 512 | 8,192 | equal-capacity, width-heavy |
| 3 | 64 | 128 | 8,192 | equal-capacity, slot-heavy |
| 4 | 16 | 128 | 2,048 | minimum corner |
| 5 | 16 | 256 | 4,096 | `N_c=16` row |
| 6 | 32 | 128 | 4,096 | `D_c=128` column |
| 7 | 32 | 512 | 16,384 | width expansion at center slots |
| 8 | 64 | 256 | 16,384 | slot expansion at center width |
| 9 | 64 | 512 | 32,768 | maximum corner |

The 8,192-scalar diagonal tests allocation at fixed nominal channel size. Rows and columns estimate
main effects, and the four corners expose interaction. W&B audit found exact scientific center
configurations for V-JEPA2 (`guiduvjp`, step 10,950), SigLIP 2 (`ufbeokj2`, step 11,000), and
DINOv3 (`fiactcw6`, finished). Their resolved model, train, data, encoder, seed, and dataset
identities match the proposed centers. All three cells are reused. DINO Run 69 remains the
no-geometry reference rather than the center.

## Encoder lanes

| Lane | Alias and immutable revision | Native detailed lattice | Frame microbatch | W&B group |
|---|---|---|---:|---|
| V-JEPA2 | `vjepa2_vitl16@b3c1679b7c34d3255ef3547f27c7b226aefab26f` | `4×16×16×1024` tubelets | 8 | `inv017_vjepa2_latent_shape_cov_var` |
| SigLIP 2 | `siglip2_vitb16@3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` | `8×16×16×768` frames | 8 | `inv017_siglip2_latent_shape_cov_var` |
| DINOv3 | `dinov3_vitb16@5931719e67bbdb9737e363e781fb0c67687896bc` | `8×16×16×768` frames | 32 | `inv017_dinov3_latent_shape_cov_var` |

SigLIP 2 and DINOv3 are shape-matched frame encoders but do not share target semantics or
normalization. V-JEPA2 uses four temporally aware tubelet grids and half as many detailed tokens.
No lane pools, drops, or resamples native encoder tokens to manufacture matching input shapes.

## Locked values

```text
data=ego4d, seed=42, steps=15000, batch=64, horizon=12
encoder_precision=bf16, attention_implementation=sdpa
present_recon_only=true, whiten_features=false
bottleneck_mixer_dim=512, bottleneck_latent_blocks=3
decoder_dim=512, decoder_blocks=4
lambda_recon=1.0, recon_loss_mode=cosine, recon_warmup_steps=2000
lambda_var=0.5, lambda_cov=0.01, lambda_sigreg=0, lambda_slot=0
lr_bottleneck=lr_coarse_flow=lr_decoder=1e-4
log_every=50, diag_every=500
```

Within an encoder lane, only `N_c` and `D_c` change. Across lanes, only the immutable encoder
identity and its established frame microbatch change in addition to the intended shape grid.

## What each axis changes in the implementation

- `N_c` changes the learned orthogonal query count, slot position tags, all latent slot streams,
  decoder memory length, nominal channel size, and the `B × N_c` row count used by covariance.
- `D_c` changes the final `M → D_c` abstract projection, all six inactive-present-mode `F_c`
  blocks, decoder memory input width, diagnostics' channel ceiling, and checkpoint shape.
- At `D_c=512`, current code uses an identity for `abstract_proj` because `D_c=M`; the other widths
  use a learned final linear projection. This is a registered part of the width hypothesis, not an
  unnoticed one-variable claim.
- Shape changes consume different amounts of model-initialization RNG after the learned query
  tensor is created. Seed 42 is fixed, but a shape contrast is not a byte-identical initialization
  comparison. A marginal winner must be repeated against the center before being called robust.

`F_c` is built at every width for Phase-1 checkpoint compatibility but receives no gradient in
present-only mode. It still contributes parameters/checkpoint bytes, not the active loss.

## Launch order

After `NEW_POD.md` Step 5, each encoder queue starts only its non-center paid arms directly. Every
lane launches eight arms, beginning with `16×512`; no lane relaunches its `32×256` seed-42 center.
The queue stops on any output collision, nonzero training exit, missing provenance sidecar, or
missing final checkpoint. Lanes run sequentially so their 15,000-step histories do not contend for
one GPU. Do not insert a separate test suite, Stage 0, or resource-preflight process unless the
human asks for one explicitly.

## Reading and normalization

Apply present-only Reading Cycle B to each arm using the final six diagnostic rows and final 50
training rows. Record:

1. operational state, skipped/nonfinite/warned updates, runtime, and peak memory;
2. `L_recon`, fixed `L_recon_present`, `L_recon_shuffled_c`, and `L_recon_video_gap`;
3. conditioned share using that encoder/arm's initial fixed correct loss;
4. `c_std_mean`, `c_cross_video_cosine`, and `c_dead_dim_frac`;
5. effective rank and normalized effective rank with ceiling
   `min(D_c, B_val*N_c - 1)`, where the fixed diagnostic batch has `B_val=16`;
6. centered slot rank and normalized centered slot rank with ceiling `min(N_c - 1, D_c)`;
7. attention entropy and the realized weighted terms `0.5*L_var` and `0.01*L_cov`.

The EGO4D diagnostic batch currently contains adjacent chunks from one source UID. Its rolled-code
gap proves exact-chunk dependence and its pair cosine measures within-source separation; neither is
a global cross-source result.

## Decision rule

Choose a winner separately inside each encoder lane. The center remains selected unless another
shape produces a Pareto improvement that is material and survives the geometry guard:

- reconstruction: at least `0.01` absolute or `3%` relative improvement in late within-lane
  `L_recon`, or at least `0.005` improvement in the fixed rolled-code gap;
- geometry: no material loss of normalized effective rank/slot rank, no `>0.05` increase in pair
  cosine, and no `>0.10` drop in mean std relative to the lane center;
- validity: zero skipped/nonfinite updates and positive correct-versus-shuffled separation.

If metrics trade off, retain the Pareto set rather than forcing a scalar winner. If an apparent
win is near threshold, repeat that arm and the center with a second registered seed before changing
the default. Raw reconstruction magnitudes never select a winner across encoders.
