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
