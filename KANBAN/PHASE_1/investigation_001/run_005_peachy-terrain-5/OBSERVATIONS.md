# Observations - run 005 `peachy-terrain-5`

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**W&B run id:** `1chv2608`  
**State:** `failed`  
**Mode:** full-prediction run; copy and batch-mean gates are decisive  
**Verdict:** **Low-rank rep**  
**Verdict note:** The representation is not rich enough even if some variance/video-specificity metrics are acceptable.

## Reading Cycle

| Q | Question | Result | Evidence and interpretation |
|---|---|---|---|
| Q1 | Training alive? | CHECK | state=failed; step=10950; grad_skipped max=n/a; grad_has_nan max=0; grad_norm last=3.499e+16 |
| Q2 | c_t alive / video-specific? | PASS | std=0.8615; dead_dim=0; cross_video_cosine=0.2681 |
| Q3 | Rich latent? | FAIL | c_effective_rank=4.8975; c_plus_effective_rank=n/a |
| Q4 | Temporal dynamics? | FAIL | copy_loss trend=up 0.1138 (0.0193 @ 0 -> 0.1331 @ 10500); ratio=7.9408; story=dynamics without a learned predictor: present/future separated but F_c did not beat copy |
| Q5 | F_c beats copy? | FAIL | model_loss=1.057; copy_loss=0.1331; ratio=7.9408; gate <= 0.70 |
| Q6 | F_c uses this video? | FAIL | batch_mean_ratio=1.2729; gate <= 0.50 and must be paired with Q5 |
| Q7 | Recon readout honest? | n/a | present=n/a; cplus=n/a; chat=n/a; chat-cplus=n/a; no active reconstruction readout for this run |
| Q8 | Verdict | Low-rank rep | The representation is not rich enough even if some variance/video-specificity metrics are acceptable. |

## Metric Trajectories From W&B History

The table uses the metric history exported from W&B. `First observed` and `Last observed` are metric-specific because training metrics and diagnostic metrics log at different cadences.

| Metric | First observed | Last observed | Delta | Min / mean / max | Late-window step>=10000 |
|---|---:|---:|---:|---:|---:|
| `loss` | 2.9153 @ 0 | 1.1148 @ 10700 | -1.8006 | 1.0457 / 1.2292 / 3.4206 | 1.0579 / 1.0824 / 1.1148 (n=15) |
| `L_flow` | 2.8726 @ 0 | 1.1069 @ 10700 | -1.7658 | 1.0423 / 1.2137 / 3.3788 | 1.0552 / 1.0772 / 1.1071 (n=15) |
| `L_var` | 0.4267 @ 0 | 0.0788 @ 10700 | -0.3479 | 0.0252 / 0.1556 / 0.4381 | 0.0252 / 0.0524 / 0.0788 (n=15) |
| `c_std_mean` | 0.4984 @ 0 | 0.8615 @ 10500 | 0.3631 | 0.4558 / 0.7233 / 0.8744 | 0.8615 / 0.8679 / 0.8744 (n=2) |
| `c_dead_dim_frac` | 0 @ 0 | 0 @ 10500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=2) |
| `c_cross_video_cosine` | 0.7175 @ 0 | 0.2681 @ 10500 | -0.4494 | 0.2331 / 0.4245 / 0.7642 | 0.2331 / 0.2506 / 0.2681 (n=2) |
| `c_effective_rank` | 8.9986 @ 0 | 4.8975 @ 10500 | -4.1011 | 2.7647 / 6.1424 / 10.6924 | 4.6989 / 4.7982 / 4.8975 (n=2) |
| `coarse_copy_loss` | 0.0193 @ 0 | 0.1331 @ 10500 | 0.1138 | 0.0193 / 0.2227 / 0.4834 | 0.1331 / 0.1365 / 0.1399 (n=2) |
| `coarse_model_loss` | 3.3071 @ 0 | 1.057 @ 10500 | -2.2502 | 1.032 / 1.2236 / 3.3071 | 1.057 / 1.057 / 1.057 (n=2) |
| `coarse_batch_mean_loss` | 0.27 @ 0 | 0.8304 @ 10500 | 0.5604 | 0.27 / 0.6351 / 0.8304 | 0.8147 / 0.8225 / 0.8304 (n=2) |
| `coarse_vs_copy_ratio` | 171.5862 @ 0 | 7.9408 @ 10500 | -163.6454 | 2.2927 / 13.4897 / 171.5862 | 7.5559 / 7.7484 / 7.9408 (n=2) |
| `coarse_vs_batch_mean_ratio` | 12.2494 @ 0 | 1.2729 @ 10500 | -10.9765 | 1.2729 / 2.275 / 12.2494 | 1.2729 / 1.2852 / 1.2974 (n=2) |
| `grad_norm` | 0.3553 @ 0 | 3.499e+16 @ 10700 | 3.499e+16 | 0.215 / 1.627e+14 / 3.499e+16 | 0.5134 / 2.333e+15 / 3.499e+16 (n=15) |
| `grad_has_nan` | 0 @ 0 | 0 @ 10500 | 0 | 0 / 0 / 0 | 0 / 0 / 0 (n=2) |
| `lr_mult` | 1.000e-04 @ 0 | 0.9944 @ 10950 | 0.9943 | 1.000e-04 / 0.5431 / 1 | 0.9944 / 0.9981 / 1 (n=20) |

## Last Key Metrics

`c_effective_rank`=4.8975 @ 10500; `c_cross_video_cosine`=0.2681 @ 10500; `c_std_mean`=0.8615 @ 10500; `coarse_vs_copy_ratio`=7.9408 @ 10500; `coarse_vs_batch_mean_ratio`=1.2729 @ 10500

## Interpretation

This is a full-prediction experiment. The latest copy ratio is 7.9408 and the latest batch-mean ratio is 1.2729; the Phase 1 gates are <=0.70 and <=0.50 respectively. The latest c_t rank is 4.8975, cross-video cosine is 0.2681, and copy loss is 0.1331. The reading-cycle verdict is **Low-rank rep** because The representation is not rich enough even if some variance/video-specificity metrics are acceptable. Read this run as part of the connected sequence: it either removes one possible explanation for failure or motivates the next controlled axis, rather than standing alone as a generic training curve.

## What This Run Changed

The project moved from one-off long-run optimism into deliberate smoke, diagnostic, and collapse-control runs.

## Reading Constraints

- `coarse_copy_loss` is diagnostic only; the model does not optimize it directly.
- `L_flow` going down is not Phase 1 success unless the baseline gates pass.
- Present-only runs prove or disprove present bottleneck quality, not forecasting.
- In residual mode, the copy baseline means predicting zero residual.

## Original Notes Preserved

# Observations — peachy-terrain-5

## Key numbers (diagnostic batches, every 500 steps — verified vs W&B `1chv2608`)

| Step | `c_effective_rank` | `coarse_vs_copy_ratio` | `grad_norm` (pre-clip) | Notes |
|---|---|---|---|---|
| 0 | 9.00 | 171.6 | 0.36 | random init |
| 500 | 5.29 | 3.27 | 0.51 | first diag |
| 1000 | 10.48 | 12.93 | 2.44 | rank peak (noise) |
| 1500 | 10.60 | 10.15 | 11.67 | best rank of run; clip already engaging |
| 4500 | 4.45 | 2.76 | **135** | large pre-clip spike, clipped to 1.0 |
| 6000 | 3.95 | 4.24 | **166** | back-half rank settles ~3–5 |
| 7000 | 2.76 | 3.07 | 23.4 | lowest rank logged |
| 8500 | 4.97 | **2.29** | 2.69 | best copy ratio of run |
| 9500 | 5.30 | 7.90 | **47.2** | ratio regressed (prior KANBAN said `0.97` — wrong) |
| 10000 | 4.70 | 7.56 | **176** | warmup ends, `lr_mult=1.0` |
| 10500 | 4.90 | 7.94 | 3.72 | last clean diag |
| 10550 | — | — | **864** | explosion begins (per-step log) |
| 10650 | — | — | **2.6×10⁸** | snowball |
| ~10750 | — | — | NaN | weights corrupted; process died ~10950 on NaN covariance |

Also: `c_std_mean` healthy (~0.74–0.86), `c_dead_dim_frac=0` throughout the healthy
phase. `grad_global_norm` (post-clip) stayed pinned at ≈1.0 the entire run — the
textbook signature of a pre-clip norm that keeps blowing past the clip threshold.

## What broke — the bf16/clip death sequence

The trigger was **not** one freak gradient: large pre-clip spikes (135, 166, 176)
recurred through the whole back half and `grad_clip=1.0` absorbed them. The fatal
interaction came once `lr_mult` reached peak. At step ~10550 a batch produced a
pre-clip norm of **864**; clipping that to norm 1.0 means multiplying every gradient
entry by `1/864 ≈ 0.00116` **in bf16**, which carries only ~2 digits of precision.
The "clipped" gradient was therefore numerical noise pointing in a near-random
direction. The optimizer applied it, the model broke slightly, the next gradient was
bigger (406 → 2.6×10⁸ → 3.5×10¹⁶), and the snowball reached NaN within ~200 steps.
The diagnostics process then crashed running `eigvalsh` on a NaN covariance matrix.
(Full walk-through: chat "death sequence", `cursor_messages` ~4125–4170.)

## What worked

Pipeline, W&B, EMA, frozen encoder, and the variance floor all functioned until the
explosion. Copy ratio showed a real downward trend early (12.93 → 2.29) and rank
briefly reached ~10.6 (step 1500) — proof the stack *can* learn, just not stably to
completion under this schedule.

## Interpretation

Two **independent** problems surfaced: (1) training dynamics unstable at peak LR with
loose clip in bf16 — a numerical/optimizer problem, fixed in code (see
[`NEXT_STEPS.md`](NEXT_STEPS.md)); (2) representational collapse
(`c_effective_rank` ~3–5 against a >60 target) visible the *entire* run, including the
healthy phase — an architectural/regularization problem the stability fixes do **not**
touch. Separating these two is what spawned [investigation_003](../../investigation_003/).
