# investigation_001 - Can the first long Phase 1 baseline train far enough to produce a meaningful prediction signal?

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Status:** CLOSED  
**Runs covered:** 005  
**Theme:** first long Phase 1 baseline and late-instability discovery

## Question

Can the first long Phase 1 baseline train far enough to produce a meaningful prediction signal?

## Why This Investigation Exists

This branch converted the initial implementation from smoke-tested code into a longer training attempt. It exists to establish whether the original Phase 1 recipe could run without late numerical failure and whether any early non-collapse signal survived long enough to be trusted.

## W&B-Validated Run Coverage

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 5 | [`peachy-terrain-5`](run_005_peachy-terrain-5/) | `1chv2608` | `failed` | full-prediction | dataset=ssv2_tiny; steps=30000; k=4; var=0.1; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Low-rank rep | c_effective_rank=4.8975; c_cross_video_cosine=0.2681; c_std_mean=0.8615; coarse_vs_copy_ratio=7.9408; coarse_vs_batch_mean_ratio=1.2729 |

## Current Conclusion

The long baseline was not a success signal. It exposed late instability and weak prediction, which made optimizer hardening, restart discipline, and clearer acceptance metrics necessary before interpreting longer runs.

## Evidence Standard

The run entries above were reconciled against W&B API/export data on 2026-07-02. For run 052, the newest active run, the evidence was additionally refreshed live with `python run_history.py --run 662hfy3c --report`, which showed the run still `running` through step 5600.

## Original Notes Preserved

# Investigation 001 — Can Phase 1 train without numerical blow-up?

**Status:** CLOSED  
**Opened:** 2026-06-09 (launch prep on RunPod)  
**Closed:** 2026-06-10 (postmortem + hyperparameter retune landed in code)

Note: [`peachy-terrain-5`](run_005_peachy-terrain-5/) W&B log timestamps show **run start
2026-06-10**; 06-09 reflects decision and launch prep.

## Question

Can the Phase 1 training loop (frozen encoder, AdamW, bf16, gradient clipping)
run for the planned step budget without gradient explosion, NaN weights, or
process crash?

## Why it matters

Every later experiment (collapse fixes, horizon sweeps, acceptance gates) is
wasted if the base recipe is unstable. This investigation gates whether Phase 1
is runnable at all.

## Parent context

- Spec: [`AGENT_FILES/PHASES/PHASE_1.md`](../../../AGENT_FILES/PHASES/PHASE_1.md)
- Branched from: initial Phase 1 implementation (commit `5cb8330` / `e1e1956`)
- Spawned: [investigation_002](../investigation_002/) (throughput), [investigation_003](../investigation_003/) (collapse visible in Run 1 metrics before crash)

## Runs in this investigation

| Run | Role |
|---|---|
| [`peachy-terrain-5`](run_005_peachy-terrain-5/) | First full launch — failed at step ~10750 |
