# Plan — unwhitened internal-memory width sweep

## Hypotheses

`H_width`: the early channel squeeze is still binding at 512. If so, `M=1024` should produce a
materially lower late-window reconstruction loss than `M=512` while increasing, or at least
preserving, dependence on the supplied clip code.

`H_rate`: the final `32 x 256` rate, decoder/objective, or shared EGO4D structure is the dominant
limit. If so, 1,024 will be flat versus 512 despite its much larger internal processor.

`H_shortcut`: raw targets make a shared positional/template solution easier. If loss improves but
the shuffled-code gap stays near zero or shrinks, the model found an easier target rather than a
more informative bottleneck.

## Controlled comparison

The two runs share source commit, dataset identity/order, encoder revision and feature fingerprint,
seed, initialization policy, batch, schedule, decoder, losses, optimizer, external latent shape,
and W&B group. The sole between-arm field is `bottleneck_mixer_dim: 512 -> 1024`, which now controls
the complete internal stream and derived parameter shapes.

Both arms intentionally share two changes relative to historical Investigation 016 controls:
whitening is off and the projection to 256 is late. There is no unwhitened `M=256` arm in this
bundle, so do not write a one-variable causal claim about whitening or projection timing.

## Primary readout

For each arm, run present-only Reading Cycle B, then compare:

1. median training `L_recon` over steps 12,000–14,950;
2. median/final fixed-batch `L_recon_present` over the late diagnostic window;
3. `L_recon_shuffled_c`, `L_recon_video_gap`, and conditioned share of learned improvement;
4. stability: `grad_skipped=0`, `grad_has_nan=0`, finite gradients, completed checkpoints;
5. geometry as context: `c_std_mean`, dead dimensions, pair cosine, effective rank, and slot rank;
6. measured resource-preflight peak memory and throughput.

The conditioned share is calculated within each run as:

```text
late gap / (initial correct-code loss - late correct-code loss)
```

Because the current fixed EGO4D diagnostic batch contains adjacent chunks from one source UID,
label this the **exact-chunk/within-source conditioned share**.

## Decision rule

Prefer 1,024 as a preservation result only if all of the following hold:

- both runs are operationally valid;
- its late median `L_recon` improves by at least 0.01 absolute over 512;
- its late correct-code-versus-shuffled-code gap does not shrink, and preferably its conditioned
  share rises by at least two percentage points;
- the gain is not purchased by a clear geometry/stability failure.

If loss improves by less than 0.01, choose 512 as the engineering operating point because 1,024
uses about 3.86 times as many bottleneck parameters. If loss is flat but the gap grows materially,
record a preservation-only result: the code became more input-dependent without solving the final
distortion floor. If loss drops but the gap collapses, classify it as a shortcut, not preservation.

These are project operating thresholds, not universal properties of cosine loss.

## Launch order and resource policy

One A100 is available, so parallel processes would only contend for memory and invalidate timing.
The queue runs:

```text
Stage 0 at M=1024
-> one exact M=1024, batch-64 resource preflight
-> full M=512 run
-> full M=1024 run
```

The largest shape gates the smaller shape. If `M=1024` does not fit at physical batch 64, the queue
stops before W&B training rather than silently changing batch size. Any alternate common-batch
experiment must be documented as a new recipe.

## Stop conditions

- wrong dataset/mode/whitening provenance;
- Stage 0 failure, non-finite loss/gradient, or failed EMA transition;
- largest-arm OOM or invalid resource report;
- W&B authentication/init failure;
- an existing training process, occupied output directory, or overlapping remote tracked change;
- skipped/NaN update spiral during an arm.

A scientific plateau is not an early stop condition. Complete both valid 15,000-step arms so the
late-width comparison remains interpretable.
