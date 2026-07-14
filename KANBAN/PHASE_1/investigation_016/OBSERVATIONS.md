# Observations — investigation_016

## 2026-07-14 — run 058 finished; transfer failed

Run 058 (`mvbx96nv`) completed cleanly but did not reproduce SSv2 run 057. The SSv2 control ends
with rank 208.2, std 1.117, cross-video cosine 0.059, and video gap 0.227; EGO4D ends with rank
52.9, std 0.419, cosine 0.863, and gap 0.018. Only about 5.4% of the EGO4D decoder improvement is
conditioned on the correct video, so the absolute-target transfer is dominated by a shared
template even though optimization is stable.

The configuration/wiring checks pass, which localizes the failure to substrate-specific target,
whitening/generalization, or objective behavior rather than the dataset flag silently failing.
The next controlled test is the same EGO4D recipe with the residual reconstruction target active.
Full evidence: [`run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md`](run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md).
