# Observations — run 071

## 2026-07-26 — complete W&B reconciliation

W&B `fiactcw6` finished all 15,000 updates and recorded checkpoint
`/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/phase1_step15000.pt` with SHA-256
`fbd4bda4c75780c06b35c837a1291218c4525c1549baf978ef3561eaf81e7cb8`. Every logged
`grad_skipped`, `grad_has_nan`, and `instability_warn` value is zero.

The [PR #8 completion handoff](https://github.com/ShashwatM3/HJEPA-VWM/pull/8) additionally
records upload verification for `7` checkpoint objects (`2.9 GiB`) under
`s3://hjepa-volume/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/` and `1` preflight object
under `s3://hjepa-volume/preflight/inv016_dinov3_unwhitened_memory_m512_cov_var/`.

Late six-diagnostic medians:

| Metric | Value |
|---|---:|
| fixed correct-code reconstruction | 0.13382 |
| fixed rolled-code reconstruction | 0.17961 |
| exact-chunk gap | 0.04584 |
| `c_std_mean` | 0.62217 |
| within-source pair cosine | 0.68489 |
| `c_dead_dim_frac` | 0 |
| `c_effective_rank` | 69.00 |
| centered slot rank | 29.03 |

The code is rank-rich and slot-diverse, and the positive gap proves exact-chunk dependence on the
recorded batch. It nevertheless fails the project spread/alignment thresholds: std stays below
0.8 and pair cosine stays above 0.5. Verdict: **COLLAPSED REP ON THE RECORDED WITHIN-SOURCE BATCH;
GLOBAL COLLAPSE INDETERMINATE**.

This is stronger geometry than the DINO no-regularizer Run 069, but covariance plus variance
without whitening does not reach a healthy recorded-batch equilibrium. It supplies exact tried
center evidence for investigation 017, not a universal DINO default or a prediction result.
