# Observations — fixed residual coordinates

HUMAN-STOPPED at step 12,450; W&B terminal state `crashed` —
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

## Result (2026-08-02)

The human stopped the run at training step 12,450; the last diagnostic row is step 12,000. W&B
finalized the run as `crashed`, so the registered 15,000-step result is formally incomplete.

The observed trajectory is nevertheless decisive about the current recipe. All fixed-batch
representation and baseline metrics were bitwise stationary. The representation remained healthy
at rank `364.281/512`, latent cross-video cosine `0.109722`, std `1.103687`, and zero dead
dimensions. `F_c` reduced fixed-batch model loss by 53.61%, but its best copy and batch-mean ratios
were still `1.456283` and `1.541340`. It never beat either baseline.

Freezing improved the matched late copy and batch ratios by about 12.5% relative to the joint
Investigation-020 residual run. Coordinate motion was therefore a real obstruction, but not the
dominant one. The remaining question is whether a direct residual objective can extract predictive
signal from this fixed code, or whether the representation/predictor pair lacks that signal.

See [`METRIC_READOUT.md`](METRIC_READOUT.md) for the complete W&B facts and
[`ANALYSIS.md`](ANALYSIS.md) for the causal interpretation and next-investigation recommendation.
