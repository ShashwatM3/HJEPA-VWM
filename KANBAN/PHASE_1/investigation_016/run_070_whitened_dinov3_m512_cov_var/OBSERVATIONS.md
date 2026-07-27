# Observations — run 070

## 2026-07-26 — complete W&B reconciliation

The 100-step smoke `lwx0mu34` finished with the intended DINO revision, whitening payload,
geometry weights, present-only mode, and zero stability tripwires. It is operational evidence only;
its two diagnostic rows are not a scientific verdict.

The full run `qqozribu` finished all 15,000 updates and recorded checkpoint
`/workspace/ckpt/inv016_dinov3_whitened_memory_m512_cov_var/phase1_step15000.pt` with SHA-256
`ccffffd263897e244ed32260f892d6155231cf63cd812e0618f57e26a40848aa`. Every logged
`grad_skipped`, `grad_has_nan`, and `instability_warn` value is zero.

Late six-diagnostic medians:

| Metric | Value |
|---|---:|
| fixed correct-code reconstruction | 0.34871 |
| fixed rolled-code reconstruction | 0.45704 |
| exact-chunk gap | 0.10831 |
| `c_std_mean` | 0.80948 |
| within-source pair cosine | 0.49264 |
| `c_dead_dim_frac` | 0 |
| `c_effective_rank` | 67.61 |
| centered slot rank | 21.25 |

Reconstruction fell materially from the step-0 fixed loss `0.98019`, while rank exceeded 60, std
entered the healthy 0.8–1.2 band, pair cosine fell just below 0.5, and the rolled-code gap became
positive. Verdict: **STRONG PRESENT REPRESENTATION ON THE RECORDED WITHIN-SOURCE BATCH; GLOBAL
CROSS-SOURCE SPECIFICITY UNMEASURED**.

The qualification is essential: all 16 diagnostic chunks share one EGO4D source UID. The run
does not establish global video specificity or any prediction capability.
