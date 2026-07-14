# Run 041 - `inv011_fixed_position_present_recon`

**Current W&B run name:** `Investigation 11 · Fixed-position decoder · Present only`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Investigation:** [investigation_011](../)  
**W&B:** `hcr2qx19` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/hcr2qx19  
**State:** `crashed`  
**Created:** 2026-07-01T11:39:16Z  
**Last history step:** 14400  
**Runtime in W&B export:** 6.61h  
**Mode:** present-reconstruction-only  
**Reading-cycle verdict:** **Low-rank decodable** - The decoder can reconstruct from c_t, but the bottleneck uses too few effective directions or slots.

## Research Role

Fixed-position present-only run; tests whether present B+D can decode without future branch confounds.

This run sits inside **cosine reconstruction, fixed-position decoder, present-only geometry sweeps**. The parent investigation question is: **Can reconstruction geometry produce a strong, decodable present bottleneck, and can that evidence be separated from prediction failure?**

## Hypothesis Being Tested

The controlled question for this run is whether the configuration below changes the bottleneck or predictor failure mode observed in the preceding run sequence. In this project, a lower `L_flow` is never enough by itself: full-prediction runs must beat the copy and batch-mean baselines, while present-only runs must show a decodable, spread, video-specific, high-rank `c_t`.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2 |
| `steps` | 15000 |
| `stage1_steps` | 15000 |
| `horizon_k` | 12 |
| `frame_stride` | 2 |
| `lambda_var` | 0.5 |
| `lambda_cov` | 0 |
| `lambda_slot` | 0 |
| `lambda_sigreg` | 5 |
| `sigreg_warmup_steps` | 2000 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `recon_loss_mode` | cosine |
| `present_recon_only` | true |
| `predict_residual` | false |
| `n_c` | 32 |
| `decoder_dim` | 512 |
| `decoder_blocks` | 4 |
| `lr_bottleneck` | 1.000e-04 |
| `lr_coarse_flow` | 1.000e-04 |
| `lr_decoder` | 1.000e-04 |
| `grad_clip` | 0.5 |
| `grad_skip_threshold` | 150 |
| `agc_enabled` | true |
| `checkpoint_dir` | /workspace/ckpt/inv011_fixed_position_present_recon |

## Difference From Previous W&B Run

| Config key | Previous run | This run |
|---|---:|---:|
| `lambda_recon_pred` | 0.05 | 0 |
| `present_recon_only` | false | true |
| `predict_residual` | true | false |
| `checkpoint_dir` | /workspace/ckpt/inv011_fixed_position_decoder | /workspace/ckpt/inv011_fixed_position_present_recon |

## Chronological Linkage

- Previous W&B run: Run 040 [`inv011_fixed_position_decoder`](../run_040_inv011_fixed_position_decoder/).
- Next W&B run: Run 042 [`po_geom_sig7p5_cov0`](../present-only-geometry-sweep/wave_1/run_042_po_geom_sig7p5_cov0/).
- Parent investigation conclusion: Present reconstruction can be strong, and SIGReg plus small covariance can raise c_effective_rank above 100 while preserving video specificity. But those are present-side wins; no full-prediction run has inherited them and passed the Phase 1 copy/batch-mean gates.

## How To Read This Run

Use `KANBAN/README_for_reading_experiments.md`. This run uses the **Present Reconstruction Only** cycle. Do not apply copy-ratio gates to this run because the future-prediction branch is off.

## W&B Evidence Sources Checked

| Evidence source | Observed entries |
|---|---|
| W&B run files | `artifact/3084167547/wandb_manifest.json` (0 bytes), `requirements.txt` (3821 bytes), `wandb-metadata.json` (1776 bytes) |
| Logged artifacts | `run-hcr2qx19-history:v0` (wandb-history) |
| API/group fetch caveats | none |

## Original Notes Preserved

# fixed-position-present-recon - Run D: present-only cosine reconstruction with fixed-position D

**Investigation:** [investigation_011](../DESCRIPTION.md)  
**Position:** Run D, after Run C [`fixed-position-decoder`](../run_040_inv011_fixed_position_decoder/DESCRIPTION.md)  
**Status:** READY TO LAUNCH  
**Opened:** 2026-07-01  
**Code requirement:** commit `0a9ff46` or later, because fixed-position `models.Decoder` is the default there  
**W&B group:** `inv011_fixed_position_present_recon`  
**W&B name:** `inv011_fixed_position_present_recon`

## Question

Can the current bottleneck learn a strong present-side representation when the only task is:

```text
x_t -> E -> e_t -> B -> c_t
fixed-position D(c_t) -> e_hat_t
cosine(e_hat_t, e_t)
```

This run intentionally removes the prediction problem. It asks whether `B` can produce a rich,
decodable `c_t` when reconstruction is the main content objective and the decoder cannot use learned
per-output-token content queries.

## Why This Run Exists

Run C, `inv011_fixed_position_decoder` (`io74f32b`), used the fixed-position decoder inside the full
residual/SIGReg/cosine recipe. It was stable and moved the copy ratio in the right direction, but it
did not pass Phase 1:

- `coarse_vs_copy_ratio` plateaued near `1.0`, not the `<=0.70` gate.
- `coarse_vs_batch_mean_ratio` stayed above `1.0`, not the `<=0.50` gate.
- `c_effective_rank` regressed to about `50.8`, below the `>60` gate.
- `L_recon_chat - L_recon_cplus` stayed tiny, so reconstruction remained mostly blind to prediction
  quality.

That result leaves a clean sub-question:

```text
Is the bottleneck plus fixed-position decoder capable of learning a high-quality present
representation at all, before asking F_c to forecast it?
```

If this present-only run fails, the prediction failure is not just an `F_c` issue. It means the
`B -> c_t -> D -> e_t` representational channel is still insufficient. If this run succeeds, then
we have evidence that the bottleneck can carry useful present information, and the next failure to
attack is how `F_c` predicts that representation forward in time.

## Relation To Earlier inv011 Runs

Original Run B was a present-only bottleneck test with the legacy `relative_mse` reconstruction
objective:

```text
--present-recon-only
--recon-loss-mode relative_mse
lambda_var=0
lambda_sigreg=0
lambda_recon=0.05
```

This run is not a strict ablation of original Run B. The goal has changed from "isolate the old
objective" to "make the bottleneck produce the best useful representation space we can." Therefore
this run uses:

```text
--present-recon-only
--recon-loss-mode cosine
--lambda-recon 0.05
--lambda-sigreg 5.0
--lambda-var 0.5
```

## Decoder Architecture

There is no CLI flag for the fixed-position decoder. The fixed-position decoder is the current
default `models.Decoder` implementation.

The intended rule is:

```text
position tells D where to write;
c tells D what to write.
```

The decoder maps:

```text
c_t:     (B, 32, 256)
e_hat_t: (B, 1024, 1024)
```

The fixed 3D tubelet position code identifies the output tubelet location, but it is registered as a
buffer, not a learned `nn.Parameter`. The first hidden state is a weighted sum of values derived
from `c_t`. There is no learned table shaped like `(N_ctx, decoder_dim)` that can become an
unconditional per-position V-JEPA feature template.

Do not resume an old learned-query decoder checkpoint into this run.

## Active Training Path

In `--present-recon-only`, the future branch is off:

```text
context_clip -> E -> detailed = e_t
detailed -> B -> abstract = c_t
abstract -> fixed-position D -> e_hat_t
L_recon = mean(1 - cos(e_hat_t, e_t))
```

Not active:

- no target clip movement to GPU;
- no future encoder call;
- no `B_EMA(E(x_{t+k}))` target;
- no residual target;
- no rectified-flow target;
- no `F_c` forward;
- no `L_flow` gradient;
- no `L_recon_pred`;
- no `c_hat`;
- no copy or batch-mean prediction baselines.

`F_c` is still constructed and included in the optimizer by the current code path, but it should not
receive gradients or update, because no active loss depends on it.

## Objective

The effective loss in this run is:

```text
L_total =
    lambda_var * L_var(c_t)
  + lambda_sigreg * sigreg_scale * L_sigreg(c_t)
  + lambda_recon * recon_scale * L_recon_present(D(c_t), e_t)
```

with:

```text
lambda_var = 0.5
lambda_sigreg = 5.0
lambda_recon = 0.05
sigreg_warmup_steps = 2000
recon_warmup_steps = 2000
```

At step 0, `sigreg_scale=0` and `recon_scale=0`, so early loss is dominated by the variance floor.
By step 2000, both SIGReg and reconstruction are fully active.

## Why Cosine Reconstruction

The goal is not to reproduce the old Run B ablation. The goal is to make `c_t` carry useful
present-side information under the best currently understood reconstruction geometry.

`relative_mse` can reward feature magnitude behavior because it uses raw MSE divided by target
variance. `cosine` normalizes each predicted and target tubelet vector along `D_e`, so the decoder
has to align with the frozen V-JEPA feature direction. That is the cleaner objective for this run:

```text
L_recon_present = mean_p(1 - cos(e_hat_t[p], e_t[p]))
```

## Why SIGReg Plus Reconstruction

The two gradients ask `B` for different, compatible properties:

| Loss | Gradient reaches | What it asks `c_t` to do |
|---|---|---|
| `L_recon_present` | `D` and `B` | Preserve enough per-video content for fixed-position `D` to reconstruct frozen `e_t`. |
| `L_sigreg` | `B` only | Make pooled `c_t` rows high-rank, spread, and approximately isotropic. |
| `L_var` | `B` only | Prevent per-coordinate collapse by pushing std up to the floor. |

This is not risk-free. If SIGReg is too strong, it can make rank/std metrics look good while
reconstruction stalls. If reconstruction dominates, it can learn a decodable but poorly conditioned
code space. This run is deliberately read by both axes:

```text
content success:      L_recon_present falls materially
geometry success:     rank/std/cross-video metrics stay healthy
joint success:        both happen at the same time
```

## Recommended Config

Use the same decoder capacity as Run C:

```text
decoder_dim=512
decoder_blocks=4
n_c=32
```

Run for the same 15k-step budget as the surrounding inv011 runs:

```text
steps=15000
data=ssv2
horizon_k=12
lr_bottleneck=1e-4
lr_coarse_flow=1e-4
```

`horizon_k` is logged for consistency but has no training effect in present-only mode because the
target branch is skipped.

## Main Readouts

Primary:

- `L_recon_present`: must fall materially and not merely flatten at initialization-like values.
- `c_effective_rank`: target is above `60`.
- `c_cross_video_cosine`: target is below `0.5`, ideally in the strong historical range around
  `0.15-0.3`.
- `c_std_mean`: target is roughly `0.8-1.2`.
- `c_dead_dim_frac`: should stay near `0`.

Secondary:

- `L_recon`: train-step present reconstruction loss.
- `L_sigreg` and `sigreg_scale`: geometry regularizer activity.
- `L_var`: variance-floor pressure.
- `c_slot_diversity_rank`: useful warning if the slots become redundant.
- `c_attn_entropy` and `c_attn_entropy_min`: warning if bottleneck attention saturates.
- `grad_norm`, `grad_skipped`, `grad_has_nan`, `instability_warn`: validity/stability.
- `agc_B_*` and `agc_D_*`: whether `B` or `D` gradients are being clipped heavily.

Do not read copy-ratio or batch-mean gates for this run. They are not logged in present-only
diagnostics and are not part of the question.

## Interpretation

**Strong win:**

- training is stable;
- `L_recon_present` falls clearly;
- `c_effective_rank > 60`;
- `c_std_mean` is near `1`;
- `c_cross_video_cosine < 0.5`;
- no large dead-dim fraction;
- no evidence that SIGReg achieved geometry while reconstruction stalled.

**Useful partial win:**

- reconstruction improves strongly, but rank remains around Run C's `~50`;
- or rank clears `60`, but reconstruction only improves modestly.

This still teaches which side is limiting the bottleneck: content capacity versus representation
geometry.

**Fail:**

- `L_recon_present` stalls while rank/std look healthy: SIGReg may be producing pretty geometry
  without decodable content, or fixed-position `D` cannot extract enough information from `c_t`.
- rank stays low and reconstruction stalls: present reconstruction plus SIGReg is not enough to
  make this bottleneck rich.
- cross-video cosine rises toward collapse: `c_t` loses video specificity.
- instability or skipped steps appear: the combined regularizer/reconstruction gradients are too
  aggressive.

## Follow-Up If It Succeeds

If this run produces a genuinely strong present representation, the next experiment should re-enable
prediction while preserving the learned lessons:

```text
full residual/cosine/fixed-position D
same SIGReg and variance geometry
possibly add direct residual diagnostics for F_c
```

The key question would then become whether `F_c` can forecast this healthier `c_t`, not whether
`c_t` can carry content.

## Follow-Up If It Fails

If this run fails, the next design issue is upstream of `F_c`. Possible next checks:

- add a shuffled-`c` reconstruction diagnostic: `D(c_t shuffled across batch) -> e_t` should be
  worse than real `D(c_t) -> e_t`;
- inspect whether fixed-position `D` is too weak or too strong;
- test a different balance of `lambda_sigreg` and `lambda_recon`;
- inspect bottleneck attention/slot redundancy rather than tuning prediction.
