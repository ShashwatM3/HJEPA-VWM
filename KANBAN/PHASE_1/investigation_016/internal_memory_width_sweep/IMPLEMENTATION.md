# Implementation — full-width bottleneck with one late projection

## Public interface

The existing `ModelConfig.bottleneck_mixer_dim` is now the single internal-width field. The new CLI
flag is:

```text
--bottleneck-mixer-dim 512
--bottleneck-mixer-dim 1024
```

No second width or projection-timing flag was added. A split set of knobs could create incoherent
states such as 512-D ConvNeXt memory followed by 256-D attention, which is exactly the accidental
behavior this change removes.

## Old and new paths

Before this change, setting a larger mixer width affected only `in_proj`, the two ConvNeXt blocks,
and positional memory. `to_kv` immediately projected that tensor to `D_c=256`; learned queries and
all three latent blocks were still 256-D.

The new path is:

```text
e: (B, N_e, D_e)
-> in_proj: D_e -> M
-> ConvNeXt memory + position: (B, N_e, M)
-> to_kv: M -> M
-> learned queries: (N_c, M)
-> three M-wide cross-read / self-attend / MLP blocks
-> abstract_proj: M -> D_c
-> LayerNorm(D_c)
-> c: (B, N_c, D_c)
```

The decoder, coarse flow, EMA interface, checkpoints outside the bottleneck, losses, and
diagnostics continue to consume the same `(B,N_c,D_c)` external code.

## Initialization and input dependence

Queries remain orthogonally initialized. Cross-attention output, slot-self-attention output, and
the last MLP layer remain zero-initialized residual bridges. Therefore a wide bottleneck still
starts stable and input-independent, returning `LayerNorm(abstract_proj(queries))`; the first
reconstruction update gives nonzero gradient to both the final projection and every cross-attention
output bridge, after which input-dependent evidence reaches the code.

The final `abstract_proj` is orthogonally initialized, not zero-initialized. Zero initialization
would block gradients from reaching the internal slot stream on the first update. Its bias starts
at zero. Its matrix receives normal AdamW decay/AGC treatment, while its bias follows the existing
no-decay rule.

## Default compatibility

When `M == D_c == 256`, `abstract_proj` is `nn.Identity()`. It adds no parameters or state-dict
keys. `to_kv`, queries, latent blocks, and final normalization retain their historical 256-D shapes,
so the shipped default path is numerically unchanged and old 256-wide checkpoints remain strict-load
compatible.

Widths 512 and 1,024 intentionally have different tensor shapes and must start from scratch. Their
EMA bottlenecks are deep copies of the complete online shape and continue to update through the
existing parameter-wise EMA path.

## Validation added

- CLI propagation test proves the width reaches config before Stage 0/module construction.
- Config validation rejects nonpositive widths and widths not divisible by eight attention heads.
- Architecture test proves memory, queries, and every latent block remain at `M` until the final
  `M -> D_c` projection.
- Gradient test proves a wide model is stable/input-independent at initialization and becomes
  input-dependent after one optimizer update.
- Optimizer test proves the late projection matrix is decayed and its bias is not.
- Default-path test proves the identity projection has no state keys.

Whitening code and offline artifacts were deliberately not edited. Disabling whitening for this
run is achieved by omitting every whitening CLI flag, which follows the existing default-false
runtime path.
