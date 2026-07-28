# Investigation 019 — Three-encoder covariance-plus-variance bottleneck shape sweep

## Status

IN PROGRESS — the original V-JEPA2 lane completed with covariance and variance mistakenly disabled,
so it is retained only as no-geometry evidence and is not comparable to the corrected sweep. The
four incorrect DINOv3 runs were stopped and deleted from W&B. Corrected DINOv3 arms run with
covariance plus variance as the next active lane on GPUs 0–3.

## Question

For each frozen encoder, which external bottleneck shape gives the best present-feature
reconstruction while retaining useful source-diverse code dependence and avoiding a materially
worse geometry trajectory?

The tested encoders are:

1. V-JEPA2 ViT-L/16 (`vjepa2_vitl16`);
2. DINOv3 ViT-B/16 (`dinov3_vitb16`);
3. SigLIP 2 ViT-B/16 (`siglip2_vitb16`).

This is a present-only, absolute cosine, raw-feature reconstruction experiment with the established
anti-collapse geometry bundle:

```text
lambda_recon = 1
lambda_var = 0.5
lambda_cov = 0.01
lambda_sigreg = lambda_slot = 0
whiten_features = false
present_recon_only = true
```

Variance and covariance supply gradients; SIGReg and slot loss remain diagnostic-only.

## Axis decision

Sweep two axes, not three:

- vary the number of slots `N_c`;
- vary the external width of each slot `D_c`;
- hold complete internal memory/slot width `M=bottleneck_mixer_dim=512`.

Investigation 016 already isolated `M=512` against `M=1024` on the same raw,
reconstruction-only V-JEPA2 recipe. The wider arm made the bottleneck about 3.86 times larger but
improved late active reconstruction by only `0.000906` and the rolled-code gap by only `0.002029`.
Both missed the preregistered material-effect thresholds, so `M=512` is the established practical
width. Varying `M` again would consume one of only four cells while preventing a complete
`N_c × D_c` interaction design.

## Four combinations

Every encoder receives the same 2×2 factorial:

| Arm | `N_c` | `D_c` | `M` | External scalars | Role |
|---|---:|---:|---:|---:|---|
| tight | 16 | 128 | 512 | 2,048 | low-capacity corner |
| width-heavy | 16 | 512 | 512 | 8,192 | spend equal capacity on slot width |
| slot-heavy | 64 | 128 | 512 | 8,192 | spend equal capacity on slot count |
| expanded | 64 | 512 | 512 | 32,768 | high-capacity corner |

The design spans 16× in external scalar capacity. The two 8,192-scalar arms isolate allocation
between slots and per-slot width; the four cells together estimate both main effects and their
interaction. `D_c=128/512` and `M=512` are divisible by the configured eight attention heads, and
`N_c=16/64` is valid for the 512-wide orthogonal query initialization.

## Prior evidence and interpretation boundary

- The older whitened `N_c=32/64/128` ladder improved late reconstruction by only 2.08% from
  32 to 128 slots and worsened recorded-batch geometry; it argues against spending this sweep on
  another slot-count ladder.
- Raw reconstruction-only `32×256×M512` evidence exists for all three encoders, but those runs
  predate the repaired source-diverse diagnostic sampler. They are historical trajectory anchors,
  not population-matched substitutes for a new arm.
- Raw cosine reconstruction magnitudes are encoder-specific because the encoders expose different
  features and token semantics. Select a shape within each encoder lane; never rank encoders by
  their raw losses.
- This experiment selects the best shape under reconstruction plus the established
  covariance/variance geometry pressure. It does not claim that present-only training establishes
  temporal prediction quality.

## Locked recipe

Full EGO4D, seed 42, batch 64, 15,000 steps, three latent blocks, decoder `512×4`, absolute cosine
target, 2,000-step reconstruction warmup, bf16/SDPA, fixed encoder revisions, no feature whitening,
no residual target, no prediction, `lambda_var=0.5`, and `lambda_cov=0.01`. SIGReg and slot loss
remain off. Only encoder identity, `N_c`, and `D_c` differ; `M=512` is fixed.

Execution and recovery are specified in [`GUIDE.md`](GUIDE.md). The design and decision rule are
specified in [`SWEEP_PLAN_bottleneck_shape.md`](SWEEP_PLAN_bottleneck_shape.md).
