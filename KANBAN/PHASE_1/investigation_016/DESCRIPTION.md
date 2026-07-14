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
