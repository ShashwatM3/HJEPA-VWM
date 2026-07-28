# Observations — investigation_017

Current state: no queue is active. The accidental duplicate V-JEPA2 center was stopped, and the
first unique arm `16×512` later crashed at step 7,100. No subsequent V-JEPA2 arm launched. SigLIP 2
shape contrasts remain unlaunched; DINO's exact `32×256` center completed separately as
`fiactcw6`, while its eight contrasts remain unlaunched.

## 2026-07-21 — V-JEPA2 direct queue launch

An initial controller that inserted Stage 0 plus a separate full-corpus resource preflight was
stopped at the human's direction before any W&B science run was created. Its partial logs were
preserved and are not science evidence.

The replacement controller follows the requested operator path directly: NEW_POD through Step 5,
then the V-JEPA2 paid queue. It runs on clean training commit `771cbba077d9f846bdf7a7dd48e12dbf29d54b49`
in tmux `inv017_vjepa2_sweep`. The first active process is the fresh `N=32, D=256` center; the
remaining eight arms are queued sequentially. The operational controller is stored outside the Git
worktree at `/workspace/preflight/inv017_vjepa2_direct/RUN_VJEPA2_ARMS_ONLY.sh` with SHA-256
`4bddb2233db60e3fd6f7d5e3377dc23c627747def45dae92f4aac4b4fcdc5989`.

The center registered as W&B run [`kiti1gpc`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/kiti1gpc)
in group `inv017_vjepa2_latent_shape_cov_var`. A full W&B audit then proved it duplicated Run 66
[`guiduvjp`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/guiduvjp): model, training,
data, encoder, seed, dataset fingerprint, and trainable initialization hash are identical. The only
provenance differences are Git/platform identity; the intervening `train.py` change only exposes
the new `--d-c` override and validates shapes, leaving the `32×256` execution unchanged. The
duplicate was stopped at step 1,000, excluded from sweep evidence, and removed from the queue.

The same audit found that SigLIP 2 center `ufbeokj2` is also an exact tried configuration and must
not be relaunched. No other V-JEPA2 or SigLIP 2 grid shape matches the locked recipe. All nine
geometry-active DINOv3 shapes are untried because Run 69 had `lambda_var=lambda_cov=0`.

The replacement operational controller is
`/workspace/preflight/inv017_vjepa2_direct/RUN_VJEPA2_UNTRIED_ARMS_ONLY.sh`, SHA-256
`e34017a1b5d0b6d3f555e6345a5aa2853748b4b95f7f50d09a40ff2eb3fa7c9c`. Its array contains exactly
`16×512`, `64×128`, `16×128`, `16×256`, `32×128`, `32×512`, `64×256`, and `64×512`. It is active in
tmux `inv017_vjepa2_sweep`; the first process is PID 461520 at `16×512`.

## Registered prior — original V-JEPA2 design

Equal scalar capacity need not behave equally. The constant-capacity diagonal `16×512`, `32×256`,
and `64×128` tests whether capacity is better spent on feature width or slot multiplicity. A useful
arm must preserve both healthy geometry and correct-code-dependent reconstruction; raw rank or
reconstruction alone cannot win.

## 2026-07-21 — scope expanded to three encoder lanes before launch

The same factorial is now registered independently for V-JEPA2, SigLIP 2, and DINOv3. The prior is
that the late-projection `M=512` machinery should make capacity effects visible, but historical
`M=512` versus `M=1024` results suggest large reconstruction gains are unlikely. Encoder-specific
target geometry may change which slot/width allocation is best, so no universal winner is assumed.

Covariance plus variance is expected to prevent the severe no-geometry contraction seen in raw
V-JEPA2, SigLIP 2, and DINOv3 baselines. Because covariance itself is shape-dependent, a monotonic
rank trend would describe the combined shape-plus-objective system rather than prove that nominal
capacity alone caused it. Cross-encoder raw losses and unnormalized ranks are preregistered as
non-comparable.

## 2026-07-26 — live W&B reconciliation

The project has 75 W&B entries and none are running.

- Duplicate center `kiti1gpc` remains `killed` at step 1,000 and excluded.
- The first unique V-JEPA2 arm, `ihiuptdp` (`N_c=16`, `D_c=512`), is `crashed` at step 7,100.
  All logged skip/nonfinite/warning fields are zero. Its partial late medians are correct/rolled
  reconstruction `0.38514/0.42271`, gap `0.03759`, std `0.71049`, within-source cosine `0.55934`,
  effective rank `64.81`, and centered slot rank `14.64/15`.
- No later Investigation-17 W&B run exists. The queue therefore did not continue after that
  terminal arm; W&B does not identify the external crash cause.
- DINO `fiactcw6` completed separately under Investigation 016 with the exact registered raw,
  covariance-plus-variance `32×256` scientific configuration. It supplies the DINO center just as
  `guiduvjp` and `ufbeokj2` supply the V-JEPA2 and SigLIP 2 centers. Its full record is
  [Investigation-016 Run 071](../investigation_016/run_071_unwhitened_dinov3_m512_cov_var/).

`ihiuptdp` is **INVALID AS A COMPLETE SWEEP ARM; STABLE, DECODABLE PARTIAL TRAJECTORY**. It cannot
select a shape or be treated as a 15,000-step endpoint. The live design now has three tried centers
and 24 non-center cells, of which only this one has been attempted and remains incomplete.
