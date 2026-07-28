# Sweep plan — joint external bottleneck shape

## Why two axes are enough

`M` controls the width of the complete input-memory and latent-processing stream. `D_c` controls
the final external channel width seen by the decoder and future coarse flow. `N_c` controls how
many independently addressable latent slots and decoder memory tokens exist.

The recent same-recipe `M=512/1024` pair established that additional internal width is not the
binding reconstruction lever at `N_c=32,D_c=256`: the 1,024 arm bought a sub-threshold loss/gap
gain at a large parameter cost. The open capacity question is downstream—how the fixed 512-wide
processor allocates its external rate between token multiplicity and feature width. Fixing `M`
also ensures `D_c=512` means “preserve the full internal width,” while `D_c=128` deliberately
tests a final 4:1 channel squeeze.

With only four paid cells, a three-axis design would confound every effect and have no replicated
edge. The selected 2×2 factorial is the smallest joint design that can detect an `N_c × D_c`
interaction.

## Factorial

| GPU | Arm | `N_c` | `D_c` | `M` | External scalars | Relative to `32×256` |
|---:|---|---:|---:|---:|---:|---:|
| 0 | tight | 16 | 128 | 512 | 2,048 | 0.25× |
| 1 | width-heavy | 16 | 512 | 512 | 8,192 | 1.00× |
| 2 | slot-heavy | 64 | 128 | 512 | 8,192 | 1.00× |
| 3 | expanded | 64 | 512 | 512 | 32,768 | 4.00× |

The equal-scalar diagonal is the key allocation comparison. If width-heavy beats slot-heavy, the
external channel squeeze is more binding than decoder memory length. If slot-heavy wins, additional
queries/tokens matter more. If only expanded wins, the axes cooperate; if it does not, the current
center is already beyond the useful rate frontier.

## Encoder order

```text
V-JEPA2 ViT-L/16
  -> wait for all four arms to finish successfully
DINOv3 ViT-B/16
  -> wait for all four arms to finish successfully
SigLIP 2 ViT-B/16
```

The controller stops before the next encoder if any arm exits nonzero or lacks its final
checkpoint/provenance. No scientific recipe is changed during recovery.

## Locked controls

```text
dataset=ego4d
seed=42
global_batch=64
max_steps=stage1_steps=15000
present_recon_only=true
lambda_recon=1
recon_loss_mode=cosine
recon_residual_target=false
recon_warmup_steps=2000
whiten_features=false
lambda_var=lambda_cov=lambda_sigreg=lambda_slot=0
bottleneck_mixer_dim=512
bottleneck_latent_blocks=3
decoder_dim=512
decoder_blocks=4
precision=bf16
log_every=50
diag_every=500
```

The alias registry pins immutable revisions and native detailed lattices. V-JEPA2 uses
`(N_e,D_e)=(1024,1024)`; DINOv3 and SigLIP 2 use `(2048,768)`. Native features are not pooled or
resampled to manufacture matching inputs.

## Required readout

Apply present-only Reading Cycle B to every arm in order:

1. mode and stability: terminal state, `grad_skipped`, `grad_has_nan`, `grad_norm`,
   `present_recon_only`, `prediction_active`, `L_flow`, `L_recon_pred`;
2. example health: paired `e_cross_video_cosine` and `c_cross_video_cosine`, `c_std_mean`,
   `c_dead_dim_frac`;
3. richness: `c_effective_rank`, normalized effective rank, centered slot rank and its normalized
   value, attention entropy;
4. content: `L_recon`, `L_recon_present`, `L_recon_shuffled_c`, and
   `L_recon_video_gap`;
5. cost: runtime, peak GPU memory, bottleneck parameters, and final checkpoint size.

Use the final 50 training rows and final six diagnostic rows. The repaired fixed EGO4D batch is
source-diverse; read `e` and `c` cosine as a paired input-to-bottleneck transformation.

## Decision rule

Choose a winner separately for each encoder. Raw loss endpoints alone do not win.

An arm must:

- finish all 15,000 steps with no skipped/nonfinite updates;
- improve late median `L_recon` by at least `0.01` absolute or 3% relative against the best
  lower-capacity comparator, or improve the late correct-versus-shuffled gap by at least `0.005`;
- keep a positive late reconstruction gap;
- avoid a greater-than-`0.05` increase in `c_cross_video_cosine`, greater-than-`0.10` drop in
  `c_std_mean`, or material normalized-rank loss against the comparator.

If no higher-capacity arm clears the material-gain and geometry guards, select the smallest
non-dominated arm. If reconstruction and geometry trade off, retain a Pareto set and repeat only
the marginal winner before changing the architecture default. Never compare raw reconstruction
losses across encoder lanes.
