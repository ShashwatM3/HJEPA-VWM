# Observations — fixed residual coordinates

RUNNING — W&B
[`r0s6ouwd`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/r0s6ouwd).

## Launch proof (2026-08-02)

- clean launch commit: `54a207cf9193404401c2c36c3eaf8be09039167c`;
- source checkpoint SHA-256:
  `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1`;
- `F_c` trainable-init hash:
  `ff729701046e64fdcb96b24315790d1c8c8b9514ad3b0602fd21158aa60e529b`;
- frozen hashes: `B=f74f2b4a…b55d`, `B_EMA=cd0566f8…4782`, `D=81ef4e5c…0599a`;
- step 0: `loss=L_flow=2.2022972107`, `prediction_active=1`, `grad_skipped=0`,
  `grad_has_nan=0`, `grad_norm=0.0970680`;
- W&B API state `running`, tmux `inv021_fc_only`, paid PID `355255`, GPU allocation
  approximately `5.38 GiB` at the launch audit.

The first launch attempt failed at required W&B initialization before step 0 because the pod had
no credential. Its evidence was preserved, authentication was repaired without exposing the key,
and the valid run restarted fresh from the same deterministic warm start. The full pod test/smoke
sequence was skipped under explicit human acceleration authorization; the exact full-batch
resource and frozen-state gate passed.

After completion, add only a short verdict and links to `METRIC_READOUT.md` and `ANALYSIS.md`.
Do not write metric values until they have been pulled from unsampled W&B history.
