# Bottleneck slot-capacity sweep — V-JEPA, whitened reconstruction, EGO4D

**Status:** COMPLETE — all three core arms finished on 2026-07-17

**W&B group:** `inv016_bottleneck_capacity_vjepa2_whitened_recon1`

**Mode:** present-reconstruction-only

**Sweep bundle:** three independent EGO4D W&B runs:

- 32 slots: [`x03xlpyl`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/x03xlpyl)
- 64 slots: [`evyokqrm`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/evyokqrm)
- 128 slots: [`7pmvxrxi`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/7pmvxrxi)

## Question

Is the repeated whitened-feature reconstruction floor caused by insufficient information bandwidth
in the abstract latent `c`?

The scientifically precise quantity is the scalar latent bandwidth

```text
|c| = N_c x D_c
```

This sweep changes that bandwidth through the existing `--n-c` slot-count control while holding
`D_c=256`. It is therefore a **slot-bandwidth sweep**, not a complete sweep of every bottleneck
dimension. A flat result rules out “more slots” as the leading fix; it does not rule out the earlier
`D_e=1024 -> mixer=256` channel squeeze or a future `D_c`/mixer-width experiment.

## Hypothesis

If 32 slots cap the information that `B` can route to `D`, increasing `N_c` should produce a
monotonic decrease in late EGO4D reconstruction loss. If the curve stays flat across a 4x bandwidth
increase, slot count is not the leading cause of the EGO4D floor and the next direct levers are
whitening and the 1,024-to-256 channel projection—not another reconstruction weight ladder.

## Core arm matrix

`D_c` remains 256 in every arm, so the V-JEPA detailed tensor contains 1,048,576 scalars per sample
and the abstract tensor contains:

| Dataset | `N_c` | `N_c x D_c` | Detailed-to-abstract scalar compression | W&B display name |
|---|---:|---:|---:|---|
| EGO4D | 32 | 8,192 | 128:1 | `Investigation 16 · Bottleneck capacity · EGO4D 32 slots` |
| EGO4D | 64 | 16,384 | 64:1 | `Investigation 16 · Bottleneck capacity · EGO4D 64 slots` |
| EGO4D | 128 | 32,768 | 32:1 | `Investigation 16 · Bottleneck capacity · EGO4D 128 slots` |

The 32-slot arms are rerun rather than borrowed from runs 057/060 because the sweep needs one clean
commit, one current encoder identity, one current whitening/provenance contract, and one launch
recipe. Run 057 predates the strict runtime pipeline; run 060 was launched from a dirty commit and
cannot serve as a byte-identical control for new arms.

## Fixed recipe

Every arm uses V-JEPA2 ViT-L/16, full EGO4D, seed 42, batch 64, 15,000 steps, the three-block
latent-stack bottleneck, 512x4 fixed-position decoder, absolute cosine reconstruction, fixed full
ZCA whitening, `lambda_recon=1.0`, `lambda_var=0.5`, `lambda_cov=0.01`, and no SIGReg or slot loss.

`lambda_recon=1.0` is deliberate. Run 060 showed that it is stable and that scalar underweighting
alone does not remove the floor. Holding the strong weight removes “the larger latent was never
asked to carry content” as the obvious interpretation of a flat capacity curve.

Whitening remains on because the colleague proposed removal of whitening as a separate experiment.
Combining no-whitening with larger `c` would prevent attribution. Encoder choice also remains fixed;
SigLIP and the currently unavailable DINO lane are separate orthogonal experiments. No
input-dependent-query architecture change is included in this sweep.

## Why these values

- `32` is the current baseline and is required on the exact sweep commit.
- `64` doubles bandwidth with the smallest meaningful architectural move.
- `128` quadruples bandwidth and matches the slot-wide arm recommended by the reconstruction-floor
  audit while preserving a substantial 32:1 bottleneck.
- `256` is not in the core sweep. It changes the latent to 16:1 compression and makes latent
  self-attention much more expensive. It is a saturation diagnostic only, governed by the
  post-core rule in `PLAN.md` and `GUIDE.md`.

## Relevant prior evidence

- Investigation 007 attempted `N_c=64/128/256`, but all three runs stopped with the same whole-pod
  event at step 200. W&B contains only five rows per run, so that branch never measured the
  capacity hypothesis.
- Run 057 (`cdvp6hou`, SSv2, 32 slots, reconstruction weight 0.05) finished at
  `L_recon_present=0.71285` with strong geometry.
- Run 058 (`mvbx96nv`, EGO4D, 32 slots, reconstruction weight 0.05) finished at
  `L_recon_present=0.67825` with weak recorded-batch geometry.
- Run 060 (`2423b84g`, EGO4D, 32 slots, reconstruction weight 1.0) finished stably at
  `L_recon_present=0.67091`; increasing the weight by 20x moved the floor only about 0.007.

Those values motivate this sweep; they are not substitute controls for its three same-commit EGO4D
arms.

## Interpretation boundary

The EGO4D fixed validation batch in current code contains adjacent chunks from one source UID.
Therefore its shuffled-code gap is an exact-chunk/within-source readout, not proof of cross-source
conditioning. The capacity decision uses the EGO4D late reconstruction curve first and treats the
EGO4D gap as secondary. Prediction is inactive, so no result from this bundle can establish
forecasting or pass the copy/batch-mean gates.

Execution: [`GUIDE.md`](GUIDE.md). Design and decision rules: [`PLAN.md`](PLAN.md).

## Result

All arms completed from clean commit `820a5b560b8668fa12452e35b1e91b170a2a91a3` with the same
dataset, data order, V-JEPA feature, validation-batch, and whitening fingerprints. Late median
training reconstruction improves `0.67703 -> 0.67396 -> 0.66298`; the 32-to-128 change is `0.01405`
(2.08%), below the preregistered `0.02`/3% support threshold. Late fixed-batch reconstruction
improves only `0.67116 -> 0.67072 -> 0.66570`.

The larger codes also fail the representation-health guard. The recorded-batch late
`std/cosine` values are `0.806/0.473` at 32 slots, `0.326/0.914` at 64, and `0.649/0.677` at 128.
Because the fixed EGO4D batch contains one source UID, this is a within-source/source-chunk result,
not proof of global cross-source collapse. It is nevertheless enough to reject the lower loss as a
healthy capacity win. Do not launch 256 slots; move to no-whitening and channel-width/`D_c` work.
Full evidence: [`ANALYSIS.md`](ANALYSIS.md).
