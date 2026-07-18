# Investigation 016 — EGO4D transfer of run 057 (whitened AE + cov/var)

## Status

OPEN

## Question

Does the settled inv015 present-only recipe from run 057
(`ae_latent_stack_whiten_abs_recon_cov_var`, W&B `cdvp6hou`) — absolute whitened
reconstruction, Perceiver latent-stack bottleneck, covariance + variance floor, no
SIGReg — transfer cleanly onto the EGO4D sibling corpus?

Single intended delta vs the SSv2 control: `--data ego4d` (plus the matching EGO4D
whitening stats file). Architecture, loss weights, seed, steps, and decoder capacity
stay byte-identical.

## Context

- EGO4D is wired as a sibling dataset (`--data ego4d | ego4d_tiny`). Stages 0–6 of
  [`AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md`](../../../AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md)
  are verified complete on the pod (corpus, `ego4d_tiny`, switchability smoke).
- SSv2 control: investigation_015
  [`run_057_ae_latent_stack_whiten_abs_recon_cov_var`](../investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/)
  — finished cleanly; geometry held (`c_effective_rank` ~208) with cov+var and no
  SIGReg; honesty competitive with the abs-whitened arm.
- Whitening stats are **per-dataset**. Do not reuse
  `logs/whiten/whiten_stats_ssv2_train_seed42.pt` on EGO4D.

## First run

[`run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/`](run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/)
— finished as W&B [`mvbx96nv`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/mvbx96nv).
The transfer failed the geometry and honesty gates despite clean optimization; analysis in that
folder's [`ANALYSIS.md`](run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md).

## Parallel encoder branch

[`run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d/`](run_059_siglip2_latent_stack_whiten_abs_recon_cov_var_ego4d/)
reproduces the run-058 recipe with SigLIP 2 on the current strict pipeline. It does not displace
the preregistered residual-target arm. Because run 058 predates the deterministic
data/encoder/artifact refactor, it is a historical reference; a same-commit V-JEPA companion is
required for causal encoder attribution.

## Weight ablation (absolute target)

[`run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/`](run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/)
finished as W&B [`2423b84g`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2423b84g).
It was intended as the run-058 EGO4D AE recipe with sole config delta
`lambda_recon: 0.05 → 1.0`: same whitening path, absolute target, and cov+var. It finished with
healthier recorded-batch geometry but only a 0.007 reconstruction improvement and 7.66%
exact-chunk conditioned share. Intervening runtime changes prevent a causal one-variable claim,
and the single-source validation batch prevents a global cross-source honesty claim. Full result:
[`run_060.../ANALYSIS.md`](run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/ANALYSIS.md).

## Reconstruction-floor architecture audit

[`reconstruction_floor_architecture_audit/`](reconstruction_floor_architecture_audit/)
traces the full present-only path and corrects the interpretation boundary discovered after run
058: its EGO4D fixed validation batch contains 16 adjacent chunks from one source UID, so the
recorded shuffled-code and “cross-video” metrics are not cross-source measurements. The raw loss
plateau remains real; the global template-collapse claim requires source-diverse remeasurement.

## Same-commit bottleneck slot-capacity sweep

[`bottleneck_slot_capacity_sweep/`](bottleneck_slot_capacity_sweep/) completed the colleague-directed
EGO4D `N_c=32/64/128` ladder on clean commit `820a5b5`, holding `D_c=256`, whitening,
`lambda_recon=1.0`, decoder, schedule, seed, and data identity fixed. W&B runs are `x03xlpyl`,
`evyokqrm`, and `7pmvxrxi`.

Late training reconstruction improves `0.67703 -> 0.67396 -> 0.66298`, but the total gain is only
`0.01405` (2.08%), below the preregistered support threshold. More importantly, fixed-batch
specificity worsens from `std/cosine=0.806/0.473` at 32 slots to `0.326/0.914` at 64 and recovers
only to `0.649/0.677` at 128. Larger `N_c` therefore lowers loss slightly without producing a
healthier recorded-batch content code. The result rejects more query slots as the leading fix; no
256-slot arm is warranted. Full read: [`bottleneck_slot_capacity_sweep/ANALYSIS.md`](bottleneck_slot_capacity_sweep/ANALYSIS.md).

## Unwhitened internal-memory width sweep

[`internal_memory_width_sweep/`](internal_memory_width_sweep/) is the next executable bundle. It
implements one coherent bottleneck width `M` through memory, learned queries, and all three latent
blocks, then projects once to the unchanged external `D_c=256`. Two full-EGO4D arms compare
`M=512` and `M=1024` sequentially on one GPU.

Both arms omit whitening completely and set variance, covariance, SIGReg, and slot-loss weights to
zero. They retain `N_c=32`, `lambda_recon=1`, the 512-by-4 decoder, and the present-only absolute
cosine recipe. Existing whitening files stay on the volume but are ignored. Because whitening
removal and late projection are shared interventions, this bundle isolates only the 512-to-1,024
width delta; it cannot separately attribute a change against historical whitened controls.
