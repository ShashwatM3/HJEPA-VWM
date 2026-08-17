# Erratum for `WANDB_INV019_BOTTLENECK_SWEEP_ANALYSIS.md`

The exact sampled-history extrema for V-JEPA2 N=16, D=128, M=512 (`yn2x2obz`) are:

| Metric | Final | Sampled minimum | Sampled maximum |
|---|---:|---:|---:|
| `loss` | 0.3162788749 | 0.1732129157 at step 500 | 0.5723214746 at step 0 |
| `L_recon` | 0.3129198849 | 0.3100246489 at step 13,000 | 1.0244989395 at step 0 |
| `L_var` | 0.0009710683 | 0.0005631890 at step 14,500 | 1.0 at step 0 |
| `L_cov` | 0.2873461246 | 0.2677092552 at step 13,000 | 7.2321453094 at step 0 |
| `c_effective_rank` | 94.2967147827 | 14.2995290756 at step 0 | 94.8252105713 at step 11,000 |
| `c_cross_video_cosine` | 0.1078081056 | -0.0306040961 at step 500 | 1.0000002384 at step 0 |
| `c_slot_diversity_rank` | 14.7744864821 | 10.1056489944 at step 500 | 15.1486513615 at step 0 |
| `grad_norm` | 0.0514273942 | 0.0491874963 at step 14,500 | 2.6308968067 at step 0 |
| `grad_global_norm_postclip` | 0.0491874977 | 0.0491874977 at step 14,500 | 0.4999998671 at step 0 |

All 30 sampled diagnostic rows have `grad_has_nan=0`, `grad_skipped=0`, and `instability_warn=0`.

This replaces the approximate V-JEPA 16/128 entries in the main report's metric-results table. The WSL filesystem helper failed while attempting the in-place one-line correction, so the exact correction is recorded here without claiming that the original line was changed.
