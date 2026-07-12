# GUIDE — run 057 `ae_latent_stack_whiten_abs_recon_cov_var` (run 056 minus SIGReg)

This launches investigation_015 run 057 on a 1x A100 pod:

```text
present-only reconstruction
cosine loss
ABSOLUTE (clean) whitened-feature target (residual target OFF, same as runs 055/056)
FIXED offline feature whitening (reuses run 054/055/056 stats file)
Perceiver latent-stack bottleneck (default architecture, 3 blocks)
+ geometry regularizers: lambda_var=0.5, lambda_cov=0.01   (SIGReg OFF: lambda_sigreg=0)
no prediction, no slot penalty
```

Identical to [`../run_056_ae_latent_stack_whiten_abs_recon_geom/GUIDE.md`](../run_056_ae_latent_stack_whiten_abs_recon_geom/GUIDE.md)
**except a single hyperparameter changes: `lambda_sigreg` 5.0 → 0.0** (and its now-moot
`--sigreg-warmup-steps` flag is dropped). Nothing else moves.

## 0. What This Run Is Testing

Run 056 (clean whitened AE + the full inv011 geometry bundle) lifted geometry
spectacularly — `c_effective_rank` ~201, `c_std_mean` ~1.03, `c_cross_video_cosine`
~0.017, centered slot rank ~30 — but the video-conditioned honesty share fell from run
055's ~86% to ~75%. The decomposition of that 11-point drop (relative to run 055) splits
roughly evenly between a larger template channel and a lower correct-code reconstruction
gain, i.e. the isotropic code became harder to decode.

Within the geometry bundle, **covariance is the demonstrated rank lever** (in run 056
`L_cov` was by far the largest term, 6.7 falling to ~0.26, and its decline tracked the
rank climb), the **variance floor** is a cheap one-sided hinge that only forbids constant
dims (satisfied once std hits 1.0), and **SIGReg is the strongest constraint** — it forces
the full `N(0, I)` distribution (Gaussian shape + higher-moment isotropy), not merely high
rank. This run removes SIGReg and keeps covariance + variance floor. The question is
whether covariance alone holds most of the rank (well above the ~40 slot-structure ceiling)
while relieving the over-isotropy pressure so honesty recovers toward run 055's ~86%.

```text
lambda_recon      = 0.05          lambda_var    = 0.5     (ON — unchanged from 056)
lambda_recon_pred = 0.0           lambda_sigreg = 0.0     (OFF — was 5.0 in run 056)
present_recon_only = true         lambda_cov    = 0.01    (ON — unchanged from 056)
recon_residual_target = FALSE     lambda_slot   = 0.0
whiten_features   = true          (no sigreg warmup — SIGReg inactive)
bottleneck_latent_blocks = 3 (default)
```

Keep `recon_residual_target=FALSE` (clean base — the run-055/056 arm, deliberate) and
`lambda_slot=0` (slot loss was rejected in inv003; do not resurrect it). This run isolates
the SIGReg contribution: it is the run-056 recipe with exactly one regularizer removed, so
overlaying it on run 056 attributes the honesty/geometry change to SIGReg alone.

## 1. Verify Code Version

Same commit family as runs 054/055/056 (latent stack + whitening + the three regularizers
must be present). No code change is needed for this run — it is a pure config delta.

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git log --oneline -3
python3 -c "from models import BottleneckLatentBlock, FeatureWhitener; print('modules OK')"
python3 -c "from losses import variance_floor, covariance_floor; print('regularizers OK')"
pytest -q
python3 -c "from models import smoke_test_models; smoke_test_models()"
python3 -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

Stop on any failure.

## 2. Whitening Statistics (REUSE the run 054/055/056 file)

The stats file is one-time per dataset and MUST be the same file the prior inv015 runs
used — never recompute it mid-experiment:

```bash
ls -la /workspace/hierarchal-jepa-flow-world-model/logs/whiten/whiten_stats_ssv2_train_seed42.pt
```

If (and only if) the file is missing — e.g. a fresh pod volume — recompute it exactly as
run 054's guide section 2 did, in tmux:

```bash
tmux new -s inv015_whiten_stats
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
python3 whiten_stats.py --data ssv2 --split train --max-batches 200 --seed 42
```

## 3. Stage 0 With The Exact Recipe

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache

python3 train.py \
  --stage0-only \
  --data ssv2 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 0.0 \
  --lambda-cov 0.01 \
  --lambda-slot 0.0 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ssv2_train_seed42.pt \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32
```

Stop if:

- Stage 0 produces NaN or a traceback (`FileNotFoundError` naming the stats path means
  section 2 was skipped or the path is wrong);
- Stage 0 does not show `whiten_active=1.0`;
- Stage 0 does not show `recon_target_residual=0.0` (this run is the CLEAN arm — if this
  reads 1.0 you launched the residual variant);
- Stage 0 does not show `recon_mean_norm=0.0` (mean tracker inactive without the residual
  target);
- Stage 0 does not show `present_recon_only=1.0` / `prediction_active=0.0`;
- Stage 0 does not show `L_flow=0.0` / `L_recon_pred=0.0`;
- `L_var` and `L_cov` are not both finite and positive (they must be COMPUTED — the weights
  being nonzero is what adds them to the loss).

Note on `L_sigreg`: with `lambda_sigreg=0` it is still COMPUTED and logged (for-logging-only,
per AGENTS.md §8/§9) but is NOT added to the loss. Expect a small finite value; it must not
enter the objective. At step 0, `recon_scale=0.0` (2000-step recon warmup) is expected;
`L_var`/`L_cov` enter the loss at full weight from step 0 (no warmup on those two).

## 4. Launch On 1x A100

Use one tmux session:

```bash
tmux new -s inv015_whiten_abs_recon_cov_var
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv015_ae_latent_stack_whiten
export WANDB_NAME=ae_latent_stack_whiten_abs_recon_cov_var
mkdir -p logs /workspace/ckpt/inv015_whiten_abs_recon_cov_var

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data ssv2 \
  --steps 15000 \
  --seed 42 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 0.0 \
  --lambda-cov 0.01 \
  --lambda-slot 0.0 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ssv2_train_seed42.pt \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv015_whiten_abs_recon_cov_var \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv015_whiten_abs_recon_cov_var.log 2>&1 &

echo "launched inv015 run 057 (cov+var geometry, no SIGReg, clean whitened AE) on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Do NOT pass `--resume` with any checkpoint from runs 052/053 (old bottleneck architecture,
rejected by design) or runs 054/055/056 (this run must train from scratch to be comparable
— same seed, same init path).

## 5. Early Tripwires

```bash
grep -m1 "step=0 " logs/inv015_whiten_abs_recon_cov_var.log
```

Expected at step 0: `whiten_active=1.0`, `recon_target_residual=0.0`, `recon_mean_norm=0.0`,
`present_recon_only=1.0`, `prediction_active=0.0`, `L_flow=0.0`, `L_recon_pred=0.0`, finite
positive `L_var`/`L_cov`, `recon_scale=0.0`, `grad_skipped=0`.

Confirm the weights landed in the W&B run config: `lambda_var=0.5`, `lambda_cov=0.01`, and
critically **`lambda_sigreg=0`**. If `lambda_sigreg` reads 5 you launched run 056 again.

Mechanical, do not read as progress: raw `c_slot_diversity_rank` near 32 and pooled
`c_effective_rank` near 31 at init (fixed slot identities — run 054's measurement caveat).
Judge slots on `c_slot_diversity_rank_centered` only.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv015_whiten_abs_recon_cov_var.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

## 6. Monitor

```bash
watch -n 5 nvidia-smi
tail -f logs/inv015_whiten_abs_recon_cov_var.log
```

In W&B group `inv015_ae_latent_stack_whiten`, overlay THIS run against run 056 (`tl5dh73c`,
same base + full bundle) and run 055 (`nzz64pl6`, same base, zero geometry). Decisive panels:

- Geometry (does covariance alone hold it): `c_effective_rank`, `c_slot_diversity_rank_centered`,
  `c_cross_video_cosine`, `c_std_mean`, `c_dead_dim_frac`. Watch whether rank stays well above
  ~40 without SIGReg.
- Honesty (does it recover toward 86%): `L_recon_shuffled_c`, `L_recon_video_gap`,
  `L_recon_present`, `L_recon`.
- Regularizer terms: `L_var`, `L_cov` (should fall as they bite), `recon_scale`. `L_sigreg`
  still logs but is for-logging-only now.
- Attention: `c_attn_entropy`, `c_attn_entropy_min`.
- Wiring: `whiten_active` (=1), `recon_target_residual` (=0), `recon_mean_norm` (=0).
- Stability: `grad_norm`, `grad_skipped`, `grad_has_nan`, `instability_warn`, `agc_B_*`,
  `agc_D_*`.

Ignore `coarse_*` panels — prediction is intentionally inactive.

## 7. Verdict Rules

Judged jointly on GEOMETRY (does covariance alone hold it) and HONESTY (does it recover):

- **SIGReg was the honesty tax (the hoped-for win):** `c_effective_rank` still holds well
  above ~40 (covariance carries it), AND the video-conditioned honesty share recovers
  clearly above run 056's ~75% toward run 055's ~86%, with `L_recon_shuffled_c` high and the
  video gap large and stable. Confirms SIGReg's full-`N(0,I)` constraint was the term fighting
  reconstruction; cov+var becomes the geometry recipe.
- **Rank collapses without SIGReg:** `c_effective_rank` falls back toward run 055's ~21
  regime. Then covariance alone does not hold the rank at these weights and SIGReg (or a
  larger `lambda_cov`) was load-bearing for rank — a real finding; reconsider the cov weight.
- **Neither axis moves much:** honesty stays near 75% and rank stays near 200. Then SIGReg was
  not the dominant driver of the honesty loss, and the erosion is decodability of the
  full-rank code — points to the recon-side levers (residual target, recon weight, decoder
  capacity), not the regularizer mix.
- **Validity gate:** gradients finite, no sustained skips, AGC quiet, wiring flags correct.

Interpretation guards:

- `L_recon_present` is comparable to runs 055/056 only loosely (all clean whitened space);
  read the honesty SHARE and geometry curves, not the raw recon value. Never compare to
  raw-space runs 052/053 nor to inv011's raw-space `L_recon_present`.
- `c_*` metrics use the same formulas as every prior run and remain cross-run evidence.
