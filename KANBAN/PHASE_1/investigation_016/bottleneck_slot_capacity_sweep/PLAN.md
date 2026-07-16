# Plan — bottleneck slot-capacity sweep

## Decision

Run a six-arm, two-dataset geometric sweep over `N_c in {32, 64, 128}`. The independent variable is
slot-mediated latent bandwidth `N_c x D_c`; `D_c=256` and every other model, objective, schedule,
data, and encoder choice remain fixed.

No code change is required. `train.py --n-c` is already applied before module construction, and
`Bottleneck`, `CoarseFlow`, and `Decoder` derive their slot shapes from `cfg.model.n_c`. Every arm
starts from scratch because checkpoints are shape-incompatible across slot counts.

## Why this is the minimal useful sweep

A two-point comparison cannot distinguish a stable trend from one noisy run. Three geometrically
spaced values give a control, a 2x intervention, and a 4x intervention while keeping every arm a
real abstract bottleneck. Running the same ladder on EGO4D and SSv2 tests whether the response is
architectural or dataset-specific without crossing it with whitening or encoder changes.

The previous investigation-007 slot ladder is not reusable evidence: its `64/128/256` arms all
ended at step 200 in a synchronized pod failure, used an older bottleneck/decoder objective, and
never reached the reconstruction warmup plateau.

## Frozen controls

| Component | Fixed value |
|---|---|
| Encoder | pinned `vjepa2_vitl16`, bf16, SDPA, frame microbatch 8 |
| `D_c`, mixer width | 256, 256 |
| Bottleneck | 2 ConvNeXt mixers, 3 latent read/compete/refine blocks |
| Decoder | width 512, 4 blocks, fixed-position queries |
| Mode | present reconstruction only; absolute target |
| Reconstruction | cosine, weight 1.0, 2,000-step ramp |
| Geometry | variance 0.5, covariance 0.01, SIGReg 0, slot loss 0 |
| Whitening | fixed full ZCA, epsilon `1e-4`, 12,800 clips, dataset-specific artifact |
| Optimization | batch 64, seed 42, 15,000 steps, B/D LR `1e-4`, current schedule/AGC/clipping |
| Data | full EGO4D or full SSv2; only the dataset/stats pair differs between ladders |

The no-whitening EGO4D run and the encoder-substrate runs remain separate experiments. The sweep
must not absorb either delta.

## Execution economy

The guide deliberately avoids a preliminary science campaign. Before the six paid runs it performs
only repository verification plus the largest-arm resource/identity check for each dataset and one
largest-shape Stage 0. These are launch-safety gates, not hypothesis tests. There is no 100-step
W&B smoke per arm.

Default operational layout is two three-GPU waves, one dataset at a time. It reduces aggregate host
pressure and makes a synchronized infrastructure failure obvious; investigation 007 lost a five-wide
capacity wave to a whole-pod event. A six-GPU simultaneous launch is allowed only when host RAM,
CPU, storage throughput, and all GPUs have been checked.

## Primary comparison

Analyze every run with Reading Cycle B, then compare within each dataset:

1. median training `L_recon` over steps 12,000–14,950;
2. median/final fixed-batch `L_recon_present` over the late diagnostic window;
3. stability and mode correctness;
4. `L_recon_video_gap`, std, dead dimensions, and pair cosine as safety/context readouts.

Do not select an arm by raw `c_effective_rank`: learned fixed slot identities mechanically raise
the pooled rank baseline as `N_c` grows. Rank is a health check, read as gain over that arm's own
step-zero value and together with cross-sample spread, not as a capacity score across arms.

Compare each dataset to its own 32-slot arm. Do not rank EGO4D against SSv2 by raw loss; the two
whiteners define different target spaces.

## Pre-registered outcome rules

Treat slot capacity as supported when all runs are valid and:

- late reconstruction improves monotonically from 32 to 64 to 128 within a dataset; and
- the 128-slot arm improves over 32 by at least `0.02` absolute or `3%` relative in the late-window
  primary loss.

Interpret the pair of dataset ladders as follows:

| Result | Conclusion |
|---|---|
| Material monotonic gain on both | Dataset-independent slot-bandwidth limit is supported. |
| Material gain on only one | Capacity is substrate-dependent; choose per dataset rather than declaring a universal bottleneck. |
| Changes below `0.01` and non-monotonic on both | More slots are not the leading floor fix; move to no-whitening and channel-width/mixer hypotheses. |
| `0.01–0.02` or noisy/non-monotonic result | Ambiguous; use the conditional 256-slot rule below on only the ambiguous dataset. |
| Lower loss with unhealthy spread/collapse | Not a capacity win; the larger code found a degenerate solution. |

These are practical effect-size rules, not universal loss thresholds. Report the full curves and
late-window dispersion alongside the label.

## Conditional 256-slot rule

`N_c=256` is not a seventh/eighth automatic arm.

- If 32/64/128 are flat within `0.01` on a dataset, do **not** pay for 256 by default; a 4x slot
  increase already failed, so test whitening or channel width next.
- If the result is borderline (`0.01–0.02`) and directionally monotonic, run 256 on that dataset as
  the one decisive saturation arm.
- If 128 already wins materially and the curve is still steep, capacity is established; run 256
  only if choosing the production operating point requires locating saturation.

## Known limitations accepted for this sweep

- Changing `N_c` changes slot attention compute and the number of rows pooled by covariance, not
  just a byte counter. The conclusion is about the effective slot-capacity architecture.
- Same seed does not make differently shaped modules byte-identical; require a monotonic curve
  rather than over-reading one pairwise delta.
- Current EGO4D honesty diagnostics are within one source UID. Do not write a cross-source claim.
- A flat slot sweep does not rule out `D_c`, mixer width, or whitening as bottlenecks.

## Deliverables after completion

1. Record all six W&B IDs in `DESCRIPTION.md`.
2. Run Reading Cycle B separately for all six arms.
3. Write a comparison `ANALYSIS.md` with within-dataset late-window tables and the rule above.
4. Update `OBSERVATIONS.md`, `NEXT_STEPS.md`, the parent investigation synthesis, and the Phase 1
   run index.
