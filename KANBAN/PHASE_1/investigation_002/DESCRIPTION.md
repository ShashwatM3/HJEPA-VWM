# investigation_002 - Can the training stack launch, log, decode data, and produce interpretable metrics before expensive full-data experiments?

<!-- AUTO-GENERATED-WANDB-KANBAN -->

**Status:** CLOSED  
**Runs covered:** 001, 002, 003, 004  
**Theme:** RunPod, dataloader, W&B, and tiny-data smoke validation

## Question

Can the training stack launch, log, decode data, and produce interpretable metrics before expensive full-data experiments?

## Why This Investigation Exists

Before treating losses or diagnostics as research evidence, the project needed to prove that the pod, W&B sync, SSv2 tiny dataset, and basic training loop worked end to end.

## W&B-Validated Run Coverage

| # | Run | ID | State | Mode | Key config | Verdict | Last key metrics |
|---:|---|---|---|---|---|---|---|
| 1 | [`youthful-pond-1`](run_001_youthful-pond-1/) | `x4pwz33d` | `finished` | full-prediction | dataset=ssv2_tiny; steps=100; k=4; var=0.1; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Smoke / inconclusive | c_effective_rank=8.9986; c_cross_video_cosine=0.7175; c_std_mean=0.4984; coarse_vs_copy_ratio=171.5862; coarse_vs_batch_mean_ratio=12.2494 |
| 2 | [`efficient-aardvark-2`](run_002_efficient-aardvark-2/) | `fz7ztfc8` | `finished` | full-prediction | dataset=ssv2_tiny; steps=200; k=4; var=0.1; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Smoke / inconclusive | c_effective_rank=8.9986; c_cross_video_cosine=0.7175; c_std_mean=0.4984; coarse_vs_copy_ratio=171.5862; coarse_vs_batch_mean_ratio=12.2494 |
| 3 | [`comfy-glade-3`](run_003_comfy-glade-3/) | `0mgmqxxi` | `finished` | full-prediction | dataset=ssv2_tiny; steps=200; k=4; var=0.1; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Smoke / inconclusive | c_effective_rank=8.9986; c_cross_video_cosine=0.7175; c_std_mean=0.4984; coarse_vs_copy_ratio=171.5862; coarse_vs_batch_mean_ratio=12.2494 |
| 4 | [`charmed-haze-4`](run_004_charmed-haze-4/) | `gj8ypv0d` | `finished` | full-prediction | dataset=ssv2_tiny; steps=200; k=4; var=0.1; cov=0; slot=0; sigreg=0; recon=0/0; residual=false; present_only=false; n_c=32; D=n/a | Smoke / inconclusive | c_effective_rank=8.9986; c_cross_video_cosine=0.7175; c_std_mean=0.4984; coarse_vs_copy_ratio=171.5862; coarse_vs_batch_mean_ratio=12.2494 |

## Current Conclusion

The smoke runs validated execution and logging. They also showed that early untrained representations were collapsed or low-rank, so later work had to separate infrastructure success from learning success.

## Evidence Standard

The run entries above were reconciled against W&B API/export data on 2026-07-02. For run 052, the newest active run, the evidence was additionally refreshed live with `python run_history.py --run 662hfy3c --report`, which showed the run still `running` through step 5600.

## Original Notes Preserved

# Investigation 002 — Is dataloader throughput sufficient for full SSv2?

**Status:** CLOSED  
**Opened:** 2026-06 (before / parallel to first long runs)  
**Closed:** 2026-06 (decode-only-needed-frames landed, commit `5e78caa`)

## Question

Is CPU-side video decode the bottleneck for Phase 1 training, and can we fix it
without changing what the model sees?

## Why it matters

At ~1.66 s/step, a 15k–30k run is prohibitively expensive and the A100 sits idle.
Full SSv2 experiments are impractical until throughput improves.

## Parent context

- Operational plan: [`AGENT_FILES/KANBAN/01-OPTIMIZE-DATALOADER/`](../../../AGENT_FILES/KANBAN/01-OPTIMIZE-DATALOADER)
- Related to [investigation_001](../investigation_001/) (same era) but independent question
- Contingency if insufficient: [`AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD/`](../../../AGENT_FILES/KANBAN/03-CPU-TO-GPU-OFFLOAD) — **deferred, not needed yet**

## Runs in this investigation

Smoke runs only — no dedicated training run. Four W&B smokes (all `ssv2_tiny`):
`youthful-pond-1` (connectivity), `efficient-aardvark-2` (1.66 s/step pre-fix baseline),
`comfy-glade-3` (`--log-every 1` dense smoke, pre-fix), `charmed-haze-4` (1.41 s/step
post-fix). Timed throughput checks, not training experiments. See OBSERVATIONS.md.
