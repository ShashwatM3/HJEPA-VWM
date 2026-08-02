# Observations — investigation_021

## Registered prior (2026-08-02)

1. Investigation-020 residual run `3y2hxj5t` established a healthy, dynamic representation but a
   failed predictor: late copy and batch-mean ratios were `1.726204` and `1.817973`.
2. In that run, `L_flow` updated both online `B` and `F_c`, while the residual target came from
   detached `B_EMA`. Healthy endpoint geometry does not prove that `F_c` learned in a stationary
   input/target coordinate system during the trajectory.
3. The fixed-coordinate run must begin from the same Investigation-019 checkpoint as the prior
   residual arm. Freezing Investigation-020's final `B` would change the starting representation
   and weaken the causal comparison.
4. Under `fc_only`, `B`, `B_EMA`, and `D` are immutable, `B_EMA=B`, EMA updates are disabled, and
   `L_flow` is the sole optimized objective. The configured variance, covariance, and present
   reconstruction coefficients remain part of the recorded recipe but have no training gradient
   because their modules are frozen and those terms are excluded from the optimized total.
5. With a fixed validation batch and fixed representation, `coarse_copy_loss`, rank, spread,
   cross-video cosine, and true-code reconstruction readouts should be stationary. Learning should
   appear only in `F_c`-dependent quantities such as model loss, baseline ratios, and predicted
   future reconstruction.

## 2026-08-02 — launch record

The exact resource gate passed on one A100 before the paid run:

- source checkpoint SHA-256 matched
  `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1`;
- step-0-equivalent `loss=L_flow=2.2022972107`, `grad_skipped=0`, and
  `grad_norm=0.0970680`;
- peak measured memory was `4.1769 GiB` and the exact step took `1.2317 s`;
- provenance bound `F_c` as the only trainable module, disabled EMA, and recorded exact
  `B/B_EMA/D` frozen hashes.

The first paid process stopped before step 0 because the pod did not actually contain the W&B
credential expected from bootstrap step 5. Its log and provenance were preserved under
`failed_wandb_auth_pre_step0`; no optimizer update or W&B run was created. An existing approved
local `api.wandb.ai` credential was transferred through a no-echo PTY, installed as mode-600
`/root/.netrc`, and verified through the remote W&B API before the deterministic relaunch.

The valid run launched from clean commit `54a207cf9193404401c2c36c3eaf8be09039167c`:

- W&B: [`r0s6ouwd`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/r0s6ouwd), state
  `running` at the launch audit;
- trainable-init hash:
  `ff729701046e64fdcb96b24315790d1c8c8b9514ad3b0602fd21158aa60e529b`;
- provenance common identity:
  `601bf967316ad5bf56134baeb7ec944fd48f1e5a282a6cc66ace427a513a4850`;
- step 0: `loss=L_flow=2.2022972107`, `prediction_active=1`, `grad_skipped=0`,
  `grad_has_nan=0`, B/D AGC activity zero;
- step-0 representation/copy diagnostics reproduce the Investigation-020 residual start:
  rank `364.281`, latent cosine `0.109722`, copy loss `0.962544`, copy ratio `3.138901`, and
  batch-mean ratio `3.322235`.

To meet the human's explicit accelerated-launch instruction, the pod skipped the redundant full
test suite and generic model/diagnostic/encoder smokes. The implementation commit had already
passed 254 local tests, formatting, lint, and compilation. The exact source/config/SHA gate and
the materially stronger warm-started full-batch resource/frozen-state gate were retained.
