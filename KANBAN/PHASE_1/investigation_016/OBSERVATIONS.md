# Observations — investigation_016

## 2026-07-14 — run 058 finished; transfer failed

Run 058 (`mvbx96nv`) completed cleanly but did not reproduce SSv2 run 057. The SSv2 control ends
with rank 208.2, std 1.117, cross-video cosine 0.059, and video gap 0.227; EGO4D ends with rank
52.9, std 0.419, cosine 0.863, and gap 0.018. Only about 5.4% of the EGO4D decoder improvement is
conditioned on the correct video, so the absolute-target transfer is dominated by a shared
template even though optimization is stable.

The configuration/wiring checks pass, which localizes the failure to substrate-specific target,
whitening/generalization, or objective behavior rather than the dataset flag silently failing.
Full evidence: [`run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md`](run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md).

## 2026-07-15 — planned run 060 (`lambda_recon=1.0` absolute-target arm)

Human-requested weight extreme on the run-058 absolute-target EGO4D AE recipe. Folder:
[`run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/`](run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/).
The residual-target control remains in the queue; the run-058 audit still prefers residual
honesty before interpreting weight sweeps.

## 2026-07-16 — correction: run-058 global collapse label is not established

The fixed EGO4D validation batch is the lexically first 16 chunks and all share source UID
`01cab463-9a16-4817-84a4-a00ef5b7bf39`. `torch.roll` therefore substitutes an adjacent chunk
from the same recording, not another source video's code, and `c_cross_video_cosine` is a
within-source measurement. The `0.018` gap still proves weak exact-chunk discrimination on that
batch, but the prior 5.4% “video-conditioned share” cannot distinguish a global template from
shared source/wearer/scene content. Preserve the original conclusion above as history, but do not
use it as a cross-source claim.

The reconstruction plateau itself survives the correction: late random-training-batch means are
`0.70695` for SSv2 run 057 and `0.69610` for EGO4D run 058. The architecture audit identifies
full ZCA target isotropization plus the early 1,024→256 channel projection and 128:1 overall
bottleneck as the leading dataset-independent floor mechanism. Full evidence:
[`reconstruction_floor_architecture_audit/ANALYSIS.md`](reconstruction_floor_architecture_audit/ANALYSIS.md).

## 2026-07-16 — run 060 finished; reconstruction underweighting not supported

Run 060 (`2423b84g`) completed all 15,000 steps with zero skipped/NaN updates. Its final recorded
geometry was healthier than historical run 058: effective rank `84.36`, std `0.800`, pair cosine
`0.479`, centered slot rank `30.81`. But present reconstruction improved only from run 058's
`0.67825` to `0.67091`, while rolled-code reconstruction was `0.69736`; the gap `0.02644` is only
7.66% of the learned improvement.

That gap remains an exact-adjacent-chunk, within-source measurement because the fixed batch still
contains one UID. The comparison is also observational rather than causal: run 060 executed after
the deterministic initialization/data-order, pinned-encoder, strict-whitening, and provenance
refactor. The result is sufficient to reject simple scalar underweighting as the leading account,
but not to assign the geometry difference to `lambda_recon` or to declare global template collapse.
Full read:
[`run_060.../ANALYSIS.md`](run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/ANALYSIS.md).

## 2026-07-17 — slot-capacity sweep complete; more query slots are not the leading fix

The same-commit EGO4D `N_c=32/64/128` bundle finished as W&B `x03xlpyl`, `evyokqrm`, and
`7pmvxrxi`. All runs are operationally valid and share exact dataset/data-order, validation-batch,
encoder-feature, whitening, schedule, seed, and clean-commit identities.

Late median training reconstruction is `0.67703/0.67396/0.66298`. The 32-to-128 improvement is
`0.01405` (2.08%), below the `0.02`/3% capacity-support threshold; late fixed-batch reconstruction
improves only `0.00546`. The representation-health trend is worse: late fixed-batch
`std/cosine` is `0.806/0.473`, `0.326/0.914`, and `0.649/0.677`. Thus 64 collapses on the recorded
within-source batch and 128 only partially recovers, despite pooled rank rising to about 189.

Because the fixed batch contains one source UID, this does not prove global cross-source collapse.
It does show that added slots do not yield a healthy exact-chunk code under the current objective.
The lower loss fails the preregistered geometry guard, so do not run 256. Move to no-whitening and
the channel-width/`D_c` squeeze as separate experiments. Full evidence:
[`bottleneck_slot_capacity_sweep/ANALYSIS.md`](bottleneck_slot_capacity_sweep/ANALYSIS.md).

## 2026-07-18 — full-width late-projection implementation prepared

The next bundle has separate run records for
[`M=512`](run_064_unwhitened_internal_memory_m512/) and
[`M=1024`](run_065_unwhitened_internal_memory_m1024/): unwhitened EGO4D at complete internal widths 512
and 1,024, both returning the same `32 x 256` external code after one final projection. All
auxiliary geometry weights are zero. The paid runs are pending; the code and launch queue are
locally verified before publication.
