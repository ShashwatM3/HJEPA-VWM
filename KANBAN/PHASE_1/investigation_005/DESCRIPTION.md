# investigation_005 - Can the best-known collapse-control recipe finish a 15k Phase 1 run and meet the prediction gates?

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Status:** CLOSED  
**Runs covered:** 015, 016, 017  
**Theme:** 15k acceptance attempts, resume behavior, AGC and skip control

## Question

Can the best-known collapse-control recipe finish a 15k Phase 1 run and meet the prediction gates?

## Why This Investigation Exists

Shorter runs could look stable while longer runs exposed late spikes, rank collapse, or copy-baseline failure. This branch tested the run length and optimizer safety needed for an acceptance attempt.

## W&B-Validated Run Coverage

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 15 | [`elated-snowflake-15`](run_015_elated-snowflake-15/) | `jhodg49x` | `crashed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Invalid | c_effective_rank=13.6879; c_cross_video_cosine=0.1655; c_std_mean=1.1022; coarse_vs_copy_ratio=3.4732; coarse_vs_batch_mean_ratio=1.0147 |
| 16 | [`drawn-elevator-16`](run_016_drawn-elevator-16/) | `0n5mx3qf` | `finished` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Invalid | c_effective_rank=13.621; c_cross_video_cosine=0.167; c_std_mean=1.099; coarse_vs_copy_ratio=3.3319; coarse_vs_batch_mean_ratio=1.3557 |
| 17 | [`royal-cherry-17`](run_017_royal-cherry-17/) | `0xv4upvb` | `killed` | full-prediction | dataset=ssv2; steps=15000; k=12; var=0.5; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Low-rank rep | c_effective_rank=5.837; c_cross_video_cosine=0.3521; c_std_mean=0.9447; coarse_vs_copy_ratio=12.7266; coarse_vs_batch_mean_ratio=1.8137 |

## Current Conclusion

Longer training exposed that nonzero variance was not enough. The runs either became unstable or stayed far from the prediction gates; rank was still low or collapsed, and copy remained competitive.

## Evidence Standard

The run entries above were reconciled against W&B API/export data on 2026-07-02. For run 052, the newest active run, the evidence was additionally refreshed live with `python run_history.py --run 662hfy3c --report`, which showed the run still `running` through step 5600.

## Original Notes Preserved

# Investigation 005 — Can we complete the 15k acceptance run?

**Status:** ACTIVE  
**Opened:** 2026-06 (after `cerulean-snow-13` validated winning config)  
**Closed:** —

## Question

Can Phase 1 Stage 1 run for **15,000 steps** on full SSv2 with the winning collapse config
and pass acceptance gates: stable training, non-collapsed `c_t`, `coarse_vs_copy_ratio < 1`,
rank trajectory acceptable?

## Why it matters

This is the operational definition of "Phase 1 Stage 1 done" before Phase 2 (fine flow).
Partial wins on shorter runs do not unlock the hierarchy work.

## Parent context

- Spec gates: [`AGENT_FILES/PHASES/PHASE_1.md`](../../../AGENT_FILES/PHASES/PHASE_1.md) §12
- Config from: [investigation_003](../investigation_003/) (`cerulean-snow-13`)
- Stability guards from: [investigation_001](../investigation_001/)

## Runs

| Run | Role |
|---|---|
| [`elated-snowflake-15`](run_015_elated-snowflake-15/) | Full 15k attempt — grad-skip death spiral at 8500 |
| [`drawn-elevator-16`](run_016_drawn-elevator-16/) | Resume from ~7500, LR halved — high skip rate (failed) |
| [`royal-cherry-17`](run_017_royal-cherry-17/) | Fresh 15k + AGC — 0% skips; cliff @8600, rank collapse |
