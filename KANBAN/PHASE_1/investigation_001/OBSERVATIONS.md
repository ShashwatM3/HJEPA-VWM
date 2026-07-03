# Observations - investigation_001

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Cross-Run Synthesis

The long baseline was not a success signal. It exposed late instability and weak prediction, which made optimizer hardening, restart discipline, and clearer acceptance metrics necessary before interpreting longer runs.

## Run-by-Run Evidence

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 5 | [`peachy-terrain-5`](run_005_peachy-terrain-5/) | `1chv2608` | `failed` | full-prediction | dataset=ssv2_tiny; steps=30000; k=4; var=0.1; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Low-rank rep | c_effective_rank=4.8975; c_cross_video_cosine=0.2681; c_std_mean=0.8615; coarse_vs_copy_ratio=7.9408; coarse_vs_batch_mean_ratio=1.2729 |

## Pattern Across The Branch

Best copy ratio in this branch was run 005 at 7.9408; best batch-mean ratio was run 005 at 1.2729. None should be read as a full Phase 1 pass unless both gates pass together.

Verdict distribution: Low-rank rep=1.

## What Changed The Research Direction

The project moved from one-off long-run optimism into deliberate smoke, diagnostic, and collapse-control runs.

## Original Notes Preserved

# Observations — Investigation 001 (training stability)

## Initial belief

The Phase 1 hyperparameters from the original brief (peak `lr_coarse_flow=4e-4`,
10k warmup, 30k steps, `grad_clip=1.0`) should be sufficient for stable
training on `ssv2_tiny`, extrapolating to full SSv2.

## What `peachy-terrain-5` showed

- Training was **healthy for ~10k steps** on collapse metrics: variance alive,
  no NaN, `coarse_vs_copy_ratio` improved toward ~2.29 at step 8500.
- **Early warning ignored:** pre-clip `grad_norm` spiked to 24–27 at steps
  8700–8900 while still below the (then nonexistent) skip guard.
- **Catastrophic failure at ~10550:** LR warmup completed (`lr_mult ≈ 1.0`);
  within ~10 steps `grad_norm` went 864 → 2.6×10⁸ → 3.5×10¹⁶; weights became
  NaN by step 10750; process died at step 11000 diagnostics (`eigvalsh` on NaN
  covariance).
- Root cause chain: peak LR too high + clip too loose + bf16 precision loss on
  extreme clip divisions + no step-skip guard + AdamW `v` absorbing the spike.

## Belief updates

| After | Belief |
|---|---|
| Step 8900 spikes | "Watch" — **should have been abort/tighten** |
| Step 10550 | Peak LR at end of long warmup is the primary trigger |
| Postmortem | Halve peak LRs, tighten clip to 0.5, add `grad_skip_threshold=50`, shorten run to 15k / warmup 1.5k, NaN-safe diagnostics |

## Code changes (not runs)

Landed in commit `611f2cd` (and related): `lr_bottleneck=1e-4`, `lr_coarse_flow=2e-4`,
`warmup_steps=1500`, `stage1_steps=15000`, `grad_clip=0.5`, skip guard in
`train.py`. Documented in
[`AGENT_FILES/KANBAN/02-LAUNCH-FULL-PHASE-1-RUN/POSTMORTEM_RUN1.md`](../../../AGENT_FILES/KANBAN/02-LAUNCH-FULL-PHASE-1-RUN/POSTMORTEM_RUN1.md).

## Conclusion

**Phase 1 was not stable under the original LR/clip schedule.** The recipe is
runnable after the Run 1 retune. Stability is a solved prerequisite, but
requires monitoring `grad_skipped` (should be ~never).

## Cross-investigation notes

- Same run exposed `c_effective_rank ~5` → opened [investigation_003](../investigation_003/).
- Throughput was poor (~1.66 s/step) → [investigation_002](../investigation_002/) ran in parallel.
