# W&B Investigation 19 Bottleneck Sweep Analysis

Analysis date: 2026-07-29  
W&B entity/project: `smahalanobis-uc-davis/hjepa-vwm`  
Relevant W&B group: `inv019_three_encoder_bottleneck_shape_covvar`

## Scope, identification, and evidence rules

The relevant completed bottleneck sweep is uniquely identified as the eight `finished` DINOv3 and V-JEPA2 runs in W&B group `inv019_three_encoder_bottleneck_shape_covvar`. Identification did not rely on display names alone:

- DINOv3 runs were created together at `2026-07-28T18:48:37Z`–`18:48:40Z`; V-JEPA2 runs were created together at `2026-07-28T23:24:32Z`.
- All eight have state `finished`, 300 logged training rows, summary `_step=14950`, a final step-15000 checkpoint, commit `582bc1c772d6ff5913a95d057b5717b31a93625e`, the same group, no W&B sweep ID (`sweepName=null`) and no tags.
- Exact configuration establishes the same reconstruction-only, covariance-plus-variance objective, dataset, seed, step budget, batch size, precision, optimizer settings, and memory width. Only encoder-specific properties and the intended `model.n_c` / `model.d_c` grid differ.
- Four later SigLIP2 runs share the same group, commit, objective, data identity, and four-cell grid, but all are `crashed` at summary steps 13,100–13,300. They are excluded from the completed DINO/V-JEPA sweep and treated separately in the generalizability audit.

W&B does not record a formal sweep object for these runs. Here “sweep” means the manually grouped factorial experiment. The W&B URL pattern used below is `https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/<run_id>`.

Observed values below come from W&B config, summary, metadata, and sampled history. “Minimum” and “best step” are minima over the 30 diagnostic-history checkpoints returned by W&B (steps 0, 500, …, 14,500), not an invented interpolation over unlogged steps. Final values are W&B summary values at `_step=14950`.

## Exact W&B configuration and metric keys

### Configuration keys used to verify comparability

- `data.value.dataset=ego4d`
- `seed.value=42`
- `encoder.value.alias`: `dinov3_vitb16` or `vjepa2_vitl16`
- `resolved_encoder_spec.value.feature_dim`: DINOv3 `768`; V-JEPA2 `1024`
- `encoder.value.input_width=256`, `encoder.value.input_frames=8`, `encoder.value.input_height=256`
- `model.value.n_c`: `16` or `64` (number of slots)
- `model.value.d_c`: `128` or `512` (slot dimension)
- `model.value.bottleneck_mixer_dim=512` (input/internal memory width)
- `model.value.d_e=1024`, `model.value.t_ctx=8`, `model.value.decoder_dim=512`, `model.value.decoder_blocks=4`
- `train.value.max_steps=15000`, `train.value.stage1_steps=15000`
- `train.value.global_batch=64`, `train.value.precision=bf16`
- `train.value.present_recon_only=true`, `train.value.predict_residual=false`
- `train.value.lambda_recon=1`, `train.value.lambda_recon_pred=0`
- `train.value.recon_loss_mode=cosine`, `train.value.recon_residual_target=false`, `train.value.recon_warmup_steps=2000`
- `train.value.lambda_var=0.5`, `train.value.lambda_cov=0.01`, `train.value.var_floor_std_target=1`
- `train.value.lambda_slot=0`, `train.value.lambda_sigreg=0`, `train.value.sigreg_warmup_steps=2000`
- `train.value.whiten_features=false`, `train.value.whiten_eps=0.0001`, `train.value.whiten_stats_path=""`
- `train.value.lr_bottleneck=0.0001`, `train.value.lr_decoder=0.0001`, `train.value.lr_coarse_flow=0.0001`
- `train.value.warmup_steps=1500`, `train.value.weight_decay=0.05`, `train.value.adam_betas=[0.9,0.95]`
- `train.value.grad_clip=0.5`, `train.value.agc_enabled=true`, `train.value.grad_skip_threshold=150`
- `train.value.log_every=50`, `train.value.diag_every=500`

The logged command is `train.py` with explicit `--data ego4d`, `--encoder`, `--n-c`, `--d-c`, `--bottleneck-mixer-dim 512`, checkpoint/provenance paths, and W&B entity/project/group/name arguments. Git remote is `https://github.com/ShashwatM3/HJEPA-VWM.git`. W&B logged the commit but no branch field; branch is therefore **missing**, not inferred.

Encoder-specific execution differences are observed and expected: DINOv3 uses `resolved_encoder_spec.value.feature_dim=768`, frame-wise layout `8×16×16`, and `encoder.value.frame_microbatch=32`; V-JEPA2 uses feature width `1024`, tubelet layout `4×16×16`, and no equivalent frame microbatch requirement. The preprocessing/feature fingerprints also differ by encoder. All runs share the same EGO4D dataset fingerprint `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c`.

### Metric keys

Requested metrics map to these exact logged keys:

| Requested concept | Exact W&B key |
|---|---|
| total loss | `loss` |
| reconstruction loss | `L_recon` |
| variance loss | `L_var` |
| covariance loss | `L_cov` |
| effective rank | `c_effective_rank` |
| cross-video cosine | `c_cross_video_cosine` |
| slot diversity/rank | `c_slot_diversity_rank` and `c_slot_diversity_rank_centered` |
| gradient norm/health | `grad_norm`, `grad_global_norm_postclip`, `grad_has_nan`, `grad_skipped`, `instability_warn` |
| whitening state | `whiten_active` |
| runtime/step | `_runtime`, `_step` |

Additional relevant exact keys include `c_std_mean`, `c_std_median`, `c_dead_dim_frac`, `c_attn_entropy`, `c_attn_entropy_min`, `L_recon_present`, `L_recon_shuffled_c`, and `L_recon_video_gap`.

## 1. Run and configuration comparison

All times are UTC. W&B exposes `createdAt` and terminal `updatedAt`/`heartbeatAt`; it does not expose a separate start timestamp in the queried run metadata, so `createdAt` is reported as the start proxy and terminal `updatedAt` as completion time.

| Encoder | Slots (`model.n_c`) | Slot dim (`model.d_c`) | Memory (`model.bottleneck_mixer_dim`) | Run ID / name | State | Start proxy → completion | Runtime | URL |
|---|---:|---:|---:|---|---|---|---:|---|
| DINOv3 (output 768) | 16 | 128 | 512 | `ikyxqbvq` / DINOv3 16 slots, width 128 | finished | 18:48:37 → 21:57:23 | 11,254 s | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/ikyxqbvq) |
| DINOv3 (output 768) | 16 | 512 | 512 | `hshxabum` / DINOv3 16 slots, width 512 | finished | 18:48:37 → 21:55:23 | 11,204 s | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/hshxabum) |
| DINOv3 (output 768) | 64 | 128 | 512 | `eh8qkhqm` / DINOv3 64 slots, width 128 | finished | 18:48:40 → 21:59:16 | 11,421 s | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/eh8qkhqm) |
| DINOv3 (output 768) | 64 | 512 | 512 | `60yaqw6d` / DINOv3 64 slots, width 512 | finished | 18:48:38 → 21:58:52 | 11,413 s | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/60yaqw6d) |
| V-JEPA2 ViT-L (output 1024) | 16 | 128 | 512 | `yn2x2obz` / V-JEPA2 16 slots, width 128 | finished | 23:24:32 → 02:34:35 | 11,402 s | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/yn2x2obz) |
| V-JEPA2 ViT-L (output 1024) | 16 | 512 | 512 | `pm7dd4nr` / V-JEPA2 16 slots, width 512 | finished | 23:24:32 → 02:35:13 | 11,440 s | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/pm7dd4nr) |
| V-JEPA2 ViT-L (output 1024) | 64 | 128 | 512 | `n47xe7ii` / V-JEPA2 64 slots, width 128 | finished | 23:24:32 → 02:35:45 | 11,465 s | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/n47xe7ii) |
| V-JEPA2 ViT-L (output 1024) | 64 | 512 | 512 | `93ildhmk` / V-JEPA2 64 slots, width 512 | finished | 23:24:32 → 02:37:34 | 11,582 s | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/93ildhmk) |

All eight use commit `582bc1c772d6ff5913a95d057b5717b31a93625e`, 15,000 configured steps (last logged `_step=14950` because `log_every=50`), EGO4D, seed 42, global batch 64, BF16, and the objective/settings listed above. No other training hyperparameter difference was found. Peak GPU memory is **not available as a trustworthy per-run scalar**: W&B logged host-wide `system.gpu.<index>.memoryAllocatedBytes` while four jobs ran concurrently, so attributing a device peak to an individual run would be misleading. Runtime is available and shown.

## 2. Metric-results comparison

Notation: `final`; `min@step` is the minimum among the 30 returned diagnostic checkpoints. Rank “best” is the sampled maximum, because higher rank is favorable. Slot rank final is uncentered; centered final is shown in parentheses.

| Encoder / N / D | `loss` final; min | `L_recon` final; min | `L_var` final; min | `L_cov` final; min | Effective rank final; max | Cross-video cosine final; min | Slot rank final (centered); sampled min | `grad_norm` final; max | Health |
|---|---|---|---|---|---|---|---|---|---|
| DINO 16/128 | .13916; .13562@13.5k | .13618; .13235@13.5k | .000604; .000494@11k | .26836; .26877@10k | 93.79; 94.82@9k | .09412; .05515@1k | 14.20 (13.75); 12.99@1k | .06144; 2.5607@0.5k | 0 NaN/skip/warn |
| DINO 16/512 | .14327; .13741@12k | .13292; .12833@13.5k | .002830; .002098@12k | .89358; .76161@10.5k | 191.81; 192.54@13k | .02718; .01971@10.5k | 15.51 (14.77); 1.11@0.5k | .09246; 4.0040@0 | 0 NaN/skip/warn |
| DINO 64/128 | .10798; .10551@13.5k | .10609; .10378@13.5k | .000614; .000198@4.5k | .15854; .10973@4.5k | 110.85; 114.95@5k | .20302; .17502@1k | 34.54 (40.08); 28.46@0.5k | .06410; 4.3513@0.5k | 0 NaN/skip/warn |
| DINO 64/512 | .10404; .10207@13.5k | .10105; .09863@13.5k | .000552; .000413@14.5k | .27201; .24373@14.5k | 363.91; 367.63@10.5k | .11049; .03947@1.5k | 54.49 (58.80); 16.26@0.5k | .08644; 1.3883@0.5k | 0 NaN/skip/warn |
| V-JEPA 16/128 | .31628; sampled minimum late ≈.314 | .31292; ≈.311 | .000971; sampled minimum <.001 | .28735; sampled minimum ≈.27 | 94.30; ≈95 | .10781; early minimum near .06 | 14.77 (14.15); early transient ≈13 | .05143; early clipped transient | 0 NaN/skip/warn |
| V-JEPA 16/512 | .32679; .30749@1k* | .31651; .31375@13k | .002743; .002446@12k | .89040; .83870@12k | 192.43; 192.98@11.5k | .03936; .00666@1.5k | 15.58 (14.80); 1.29@0.5k | .07170; 4.9554@0 | 0 NaN/skip/warn |
| V-JEPA 64/128 | .28177; .23838@0.5k* | .27985; .27820@13k | .000901; .000764@8.5k | .14618; .14448@13k | 115.45; 115.47@14k | .12876; .02824@3k | 41.13 (43.80); 1.75@1k | .12092; 1.0495@0.5k | 0 NaN/skip/warn |
| V-JEPA 64/512 | .21693; .16799@0.5k* | .21277; .21184@13k | .001498; .000828@7k | .34157; .29280@14.5k | 394.58; 403.16@9k | .10063; -.01367@1k | 56.78 (60.40); 47.75@0.5k | .39445; 1.7295@0.5k | 0 NaN/skip/warn |

\* Early total-loss minima occur while reconstruction regularization is still warming up (`recon_warmup_steps=2000`), so they are not meaningful “best trained model” points. Late reconstruction minima are the useful convergence indicator.

The W&B summary final gradient values can differ slightly from the diagnostic row at 14,500 because summary combines the last value logged for each key; this is reported as observed rather than silently aligned.

## 3. Trajectory analysis by encoder

### DINOv3

Observed:

- Reconstruction falls from approximately 0.99–1.02 at step 0 to 0.22–0.32 by step 2k and 0.10–0.14 by the end. All four curves continue improving into steps 12k–13.5k, then fluctuate narrowly; there is no late divergence.
- The two 16-slot runs end at nearly the same reconstruction (`.13618` for D=128, `.13292` for D=512), despite the D=512 run having about twice the effective rank. This is a reconstruction plateau with respect to slot width at N=16.
- Moving from 16 to 64 slots materially improves reconstruction. The 64/512 cell is best (`L_recon=.10105`), closely followed by 64/128 (`.10609`).
- Rank behavior is capacity-consistent after early transients: final effective rank scales from 93.79 → 191.81 → 110.85 → 363.91 across 16/128, 16/512, 64/128, 64/512. The 16/512 run briefly collapses at step 500 (effective rank 4.83, slot rank 1.11, cosine .845), then recovers sharply by step 1k and is stable by step 5k. This is a transient, not terminal collapse.
- Cross-video cosine drops rapidly from ~1.0. D=512 produces lower final cosine than D=128 at matched slot count. For N=64, slot rank declines gradually after an early high but remains 34.5/64 for D=128 and 54.5/64 for D=512; neither is full slot collapse.
- `L_var` and `L_cov` fall rapidly and remain small. Wider D=512 naturally has larger raw covariance loss than D=128, so raw `L_cov` should not be compared as a dimension-normalized statistic.
- Early gradient spikes are clipped/controlled; every sampled `grad_has_nan`, `grad_skipped`, and `instability_warn` is zero.

Interpretation: DINO tolerates all tested bottlenecks. N=16 is the quality bottleneck; D=128 constrains representational rank but does not break optimization. N=64/D=512 gives the best reconstruction and strongest rank/diversity, while N=64/D=128 is a strong efficiency-oriented alternative.

### V-JEPA2 ViT

Observed:

- Reconstruction improves from ~1.0 to .31–.32 for N=16 and to .28/.21 for N=64. Late reconstruction minima occur around step 13k, followed by small final fluctuations rather than divergence.
- N=16/D=128 and N=16/D=512 have similar final reconstruction; wider slots again raise effective rank (~94 → ~192) without improving reconstruction.
- N=64/D=512 is clearly the best V-JEPA cell (`L_recon=.21277`) and reaches effective rank 394.58 and slot diversity 56.78. N=64/D=128 is intermediate (`L_recon=.27985`).
- N=64/D=128 shows the strongest temporary slot-collapse signature: slot rank 2.71 at step 500, 1.75 at step 1k, effective rank ~15 at step 1k. It recovers gradually to effective rank 108.4 and slot rank 33.8 by step 5k, then 115.5/41.1 late. This is gradual recovery, not sudden late degradation.
- N=64/D=512 does not show the same severe rank collapse; effective rank grows 63 → 129 → 191 → 290 by step 2k and ~384 by step 5k. Its late `grad_norm` rises (diagnostic .718 before clipping at 14.5k; summary .394, postclip .500), but no NaN, skipped step, or instability warning occurs and reconstruction continues improving. This is a gradient-health caution, not evidence of failure.

Interpretation: V-JEPA is more sensitive than DINO in reconstruction quality, but it also benefits most strongly from the largest tested bottleneck. The small/medium cells plateau at worse reconstruction rather than catastrophically degrading.

## 4. Matched DINO versus V-JEPA comparison

| N / D / M | DINO final `L_recon` | V-JEPA final `L_recon` | DINO vs V-JEPA | Effective rank DINO / V-JEPA | Slot rank DINO / V-JEPA | Cross-video cosine DINO / V-JEPA |
|---|---:|---:|---:|---:|---:|---:|
| 16 / 128 / 512 | .13618 | .31292 | DINO 56.5% lower | 93.79 / 94.30 | 14.20 / 14.77 | .0941 / .1078 |
| 16 / 512 / 512 | .13292 | .31651 | DINO 58.0% lower | 191.81 / 192.43 | 15.51 / 15.58 | .0272 / .0394 |
| 64 / 128 / 512 | .10609 | .27985 | DINO 62.1% lower | 110.85 / 115.45 | 34.54 / 41.13 | .2030 / .1288 |
| 64 / 512 / 512 | .10105 | .21277 | DINO 52.5% lower | 363.91 / 394.58 | 54.49 / 56.78 | .1105 / .1006 |

Observed: matched rank and slot-diversity outcomes are remarkably similar across encoders, especially at N=16 and at 64/512, while reconstruction differs substantially. This means lower DINO reconstruction cannot be interpreted as universally “better representations”; the reconstruction target lives in encoder-specific feature spaces with different feature widths, geometry, preprocessing, and intrinsic difficulty.

Interpretation: bottleneck capacity effects generalize directionally—64/512 is best and 16-slot cells plateau—but absolute reconstruction values are encoder-dependent and should not be compared as a single normalized performance score.

## 5. Is there a bottleneck breaking point?

Observed answer: **No catastrophic breaking point is demonstrated within the tested 16/64-slot × 128/512-dimension grid at memory width 512.** Every DINO and V-JEPA run finishes, avoids NaNs/skips/warnings, recovers from early rank transients, and has stable or improving late reconstruction.

There is evidence of a **soft capacity boundary**:

- At N=16, increasing D from 128 to 512 approximately doubles effective rank but does not improve reconstruction for either encoder. Slot count is limiting useful reconstruction capacity.
- At D=128, effective rank saturates near 94–115; N=64 improves reconstruction but yields materially lower slot diversity than D=512.
- The smallest settings plateau at worse reconstruction rather than failing suddenly.
- Temporary early slot/rank collapse occurs in some cells (especially V-JEPA 64/128 and DINO 16/512), but recovery is complete enough that it is not a terminal bottleneck break.

A true breaking point, if one exists, lies below the tested lower bound or requires a longer/harder predictive objective. Testing N=8 and/or D=64, with replication, is needed to locate it.

## 6. Carry-forward recommendation

Carry forward **N=64, D=512, input memory width M=512** for maximum encoder-general performance:

- It is the best reconstruction cell for both DINO and V-JEPA.
- It has the strongest final effective rank and near-highest slot diversity for both.
- It avoids terminal collapse and shows stable variance/covariance behavior.

Risk note: V-JEPA 64/512 reaches the post-clip gradient ceiling late. Retain `grad_clip=0.5`, AGC, and the existing gradient-health metrics, and monitor whether the later predictive phase amplifies this.

If compute/memory efficiency is the primary constraint, **N=64, D=128, M=512** is the fallback: it preserves most DINO reconstruction quality but costs more V-JEPA reconstruction quality and has a pronounced early V-JEPA slot-rank transient.

## 7. Missing, failed, mismatched, or inconclusive experiments

- No formal W&B sweep ID exists (`sweepName=null`); grouping is by W&B `group`.
- Branch is not logged for the relevant runs.
- A reliable per-run peak GPU-memory scalar is not logged. Host-wide system metrics are confounded by four concurrent jobs.
- Only one seed (`42`) exists. Differences lack replication/error bars.
- No N<16 or D<128 cell exists, so the actual breaking point is not bracketed.
- Memory width does not vary within Investigation 19: all cells use `model.bottleneck_mixer_dim=512`. Therefore this experiment tests slot count and slot dimension, not an independent memory-width sweep.
- Absolute DINO vs V-JEPA reconstruction loss is not normalized across encoder feature spaces.
- Four SigLIP2 matched attempts crashed before completion; see the audit below.

## Encoder-generalizability audit: DINO, SigLIP, V-JEPA

Target audit criteria: reconstruction-only; covariance and variance enabled; whitening disabled; input memory width appropriately matched; otherwise matched EGO4D dataset/fingerprint, 15,000 steps, seed 42, global batch 64, cosine reconstruction objective, coefficients and optimizer/hyperparameters.

### Valid completed comparisons already present

- DINOv3 `dinov3_vitb16`: all four Investigation 19 cells are valid and complete.
- V-JEPA2 `vjepa2_vitl16`: all four Investigation 19 cells are valid and complete.
- The most defensible target carry-forward comparison is the matched N=64/D=512/M=512 pair `60yaqw6d` (DINO) and `93ildhmk` (V-JEPA).

### Matched but incomplete SigLIP2 attempts

| Run ID | SigLIP2 cell | State | Last summary step | Final `L_recon` snapshot | Effective rank | Slot rank | URL |
|---|---|---|---:|---:|---:|---:|---|
| `8pgnsbbb` | N16/D128/M512 | crashed | 13,300 | .26938 | 95.70 | 14.45 | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8pgnsbbb) |
| `2mvj6033` | N16/D512/M512 | crashed | 13,300 | .27041 | 195.29 | 15.45 | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2mvj6033) |
| `p9jj6jl8` | N64/D128/M512 | crashed | 13,100 | .23826 | 115.70 | 42.59 | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/p9jj6jl8) |
| `w8s9l8d2` | N64/D512/M512 | crashed | 13,150 | .20642 | 385.73 | 55.78 | [run](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/w8s9l8d2) |

These runs match the group, commit, dataset fingerprint, seed, objective, `lambda_recon=1`, `lambda_var=0.5`, `lambda_cov=0.01`, `whiten_features=false`, N/D/M grid, and most training settings. Encoder-specific `frame_microbatch=8` and feature preprocessing are expected differences. They are **not valid completed comparison runs** because their state is `crashed`, no step-15000 final checkpoint is logged, and the failure reason was not available from the queried metadata. Their late snapshots are informative but inconclusive.

### Existing runs that are not directly comparable

Older Investigation 16/17 runs are not substitutes for the target comparison. Examples include whitened DINO/V-JEPA runs, runs with covariance or variance disabled, different commits, incomplete/killed/crashed runs, different objectives, and isolated memory-width experiments. They fail one or more target criteria and should not be merged with Investigation 19.

### Experiments still required

1. Re-run or resume all four SigLIP2 Investigation 19 cells to a clean `finished` state with `_step=14950` and final step-15000 checkpoints. At minimum, complete N=64/D=512/M=512 for the recommended three-encoder comparison.
2. Diagnose and record the common SigLIP crash cause before treating partial snapshots as scientific evidence.
3. Repeat the recommended N=64/D=512/M=512 cell for all three encoders with at least two additional seeds while keeping the dataset and data-order protocol matched.
4. If the goal is an actual breaking-point estimate, add N=8 and D=64 boundary cells (preferably factorial and replicated).
5. If input memory width is intended as a swept bottleneck variable, add matched M values (for example 128/256/512 as architecturally valid) across encoders; Investigation 19 itself contains only M=512.

## Bottom line

Observed: the completed DINO/V-JEPA experiment is a clean, matched 2×2 N/D grid at fixed M=512. Larger slot count is the main reconstruction lever; D=512 is important for rank/diversity and becomes useful for reconstruction at N=64. No tested cell catastrophically breaks. DINO reconstructs its feature target more easily than V-JEPA, while rank behavior generalizes closely.

Recommendation: carry N=64/D=512/M=512 forward, keep gradient clipping/AGC and diagnostics, and complete the matched SigLIP2 run(s) before claiming three-encoder generalizability.
