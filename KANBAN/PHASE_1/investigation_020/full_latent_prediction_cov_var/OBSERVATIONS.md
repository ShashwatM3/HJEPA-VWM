# Observations — full-latent prediction

Live W&B run:
[`8r6akjsx`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/8r6akjsx).

Launch provenance:

- clean git commit: `7649f8efde1b104dd81cfbd110af18d499f67304`;
- source SHA-256:
  `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1`;
- trainable-init hash:
  `a50f618198ab12c561cc06a1764b8414edefffac427c0344ff2362d88e07a9db`;
- `predict_residual=false`.

Step 0 was finite: `loss=3.031017`, `L_flow=3.027744`, `prediction_active=1`,
`grad_skipped=0`, and `instability_warn=0`. The run produced its 808 MiB step-2,500 checkpoint
and progressed past step 2,900 at the first audit without a skipped update.

After completion, record the terminal state, Reading-Cycle-A Q1–Q8 evidence, final-six diagnostic
medians, and verdict. Do not infer success from `L_flow`, and do not compare its absolute value to
the residual arm.

## 2026-08-01 — terminal readout

W&B state is `finished`; the run produced the step-15,000 checkpoint with SHA-256
`ba47425785ed5d921607bfe114c8f030ab2c2f33f6b55faeadd5c6ec795d7804`. All 300 logged training
rows are finite, `grad_skipped=0`, `grad_has_nan=0`, and the run stayed in full-prediction
full-latent mode.

Final-six diagnostic medians:

| Metric | Median |
|---|---:|
| `c_std_mean` | 1.077082 |
| `c_dead_dim_frac` | 0 |
| `e_cross_video_cosine` / `c_cross_video_cosine` | 0.402768 / 0.325100 |
| `c_effective_rank` / `c_plus_effective_rank` | 296.622 / 293.805 |
| `coarse_copy_loss` | 0.362214 |
| `coarse_vs_copy_ratio` | 3.132558 |
| `coarse_vs_batch_mean_ratio` | 0.946963 |
| `L_recon_present` / `L_recon_cplus` / `L_recon_chat` | 0.129828 / 0.122533 / 0.187286 |
| `L_recon_video_gap` | 0.406016 |

The code did not become dead or conventionally low-rank, but it lost about 19% of online rank,
latent cross-video cosine nearly tripled, and copy loss fell about 62% from step 0. Fc became more
than three times worse than the increasingly cheap copy baseline. Verdict: **Static-`c` trap**.

Full evidence: [`METRIC_READOUT.md`](METRIC_READOUT.md). Mechanism, paired interpretation, and the
falsifiable frozen-bottleneck probe: [`ANALYSIS.md`](ANALYSIS.md).
