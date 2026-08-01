# Observations — residual prediction

Live W&B run:
[`3y2hxj5t`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/3y2hxj5t).

Launch provenance:

- clean git commit: `7649f8efde1b104dd81cfbd110af18d499f67304`;
- source SHA-256:
  `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1`;
- trainable-init hash:
  `a50f618198ab12c561cc06a1764b8414edefffac427c0344ff2362d88e07a9db`;
- `predict_residual=true`.

Step 0 was finite: `loss=2.205571`, `L_flow=2.202297`, `prediction_active=1`,
`grad_skipped=0`, and `instability_warn=0`. The run produced its 808 MiB step-2,500 checkpoint
and progressed past step 2,850 at the first audit without a skipped update.

After completion, record the terminal state, Reading-Cycle-A Q1–Q8 evidence, final-six diagnostic
medians, and verdict. Do not infer success from `L_flow`, and do not compare its absolute value to
the full-latent arm.

## 2026-08-01 — terminal readout

W&B state is `finished`; the run produced the step-15,000 checkpoint with SHA-256
`0671516356b149d37b89c3cb3940d7c0eaa870b668eb708a13bab55ae73719dc`. All 300 logged training
rows are finite, `grad_skipped=0`, `grad_has_nan=0`, and the run stayed in full-prediction residual
mode.

Final-six diagnostic medians:

| Metric | Median |
|---|---:|
| `c_std_mean` | 1.118518 |
| `c_dead_dim_frac` | 0 |
| `e_cross_video_cosine` / `c_cross_video_cosine` | 0.402768 / 0.179245 |
| `c_effective_rank` / `c_plus_effective_rank` | 376.932 / 363.957 |
| `coarse_copy_loss` | 1.300293 |
| `coarse_vs_copy_ratio` | 1.726204 |
| `coarse_vs_batch_mean_ratio` | 1.817973 |
| `L_recon_present` / `L_recon_cplus` / `L_recon_chat` | 0.104876 / 0.100083 / 0.181156 |
| `L_recon_video_gap` | 0.443072 |

The code stayed dynamic: copy loss rose about 35% from step 0 and online rank ended above its
pretrained start. Fc nevertheless lost to both zero residual and the batch-mean residual at every
diagnostic point. Verdict: **Healthy rep, no predictor**.

Full evidence: [`METRIC_READOUT.md`](METRIC_READOUT.md). Mechanism, paired interpretation, and the
falsifiable frozen-bottleneck probe: [`ANALYSIS.md`](ANALYSIS.md).
