# GUIDE - sharp-slot residual-target reconstruction run

This launches investigation_013 Run 053 on a 1x A100 pod:

```text
present-only reconstruction
cosine loss
residual reconstruction target (e - mean)
no prediction
no SIGReg
no variance floor
no covariance penalty
no slot-diversity penalty
```

This is the direct successor to investigation_012 run_052. The question is narrow:
**does subtracting the per-position feature mean force reconstruction pressure to carry
video-specific information through `c_t` without geometry regularizers?**

Repo path on RunPod:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

## 0. What This Run Is Testing

The primary run must isolate residual-target reconstruction. Keep every geometry/collapse regularizer off:

```text
lambda_recon = 0.05
lambda_recon_pred = 0.0
lambda_var = 0.0
lambda_cov = 0.0
lambda_sigreg = 0.0
lambda_slot = 0.0
present_recon_only = true
prediction_active = false
recon_residual_target = true
```

Do not add covariance or SIGReg on this primary run. Those directly shape representation geometry and
would confound whether the residual target alone fixed run_052's template-collapse failure mode.

## 1. Verify Pod State

```bash
cd /workspace/hierarchal-jepa-flow-world-model

nvidia-smi
git status --short
git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline
```

Expected:

- `git log -1 --oneline` is at or after the residual-target implementation commit;
- full SSv2 exists at `/workspace/data/ssv2`;
- no local pod edits that would make the run unreproducible.

## 2. Runtime Environment

```bash
export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv013_residual_recon_only
export WANDB_NAME=ae_sharp_slots_residual_recon
mkdir -p logs /workspace/ckpt/inv013_residual_recon_only
```

`WANDB_NAME` is set through the W&B environment contract; `train.py` does not expose a run-name CLI flag.

## 3. Smoke Checks

Run the full local test suite:

```bash
pytest -q
```

Run model and diagnostic smoke tests:

```bash
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

Run Stage 0 with the exact training recipe:

```bash
python train.py \
  --stage0-only \
  --data ssv2 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.0 \
  --lambda-sigreg 0.0 \
  --lambda-cov 0.0 \
  --lambda-slot 0.0 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-residual-target \
  --recon-mean-momentum 0.99 \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32
```

Stop if:

- tests fail;
- either smoke test fails;
- Stage 0 produces NaN or a traceback;
- Stage 0 does not show `recon_target_residual=1.0`;
- Stage 0 does not show `present_recon_only=1.0`;
- Stage 0 does not show `prediction_active=0.0`;
- Stage 0 does not show `L_flow=0.0`;
- Stage 0 does not show `L_recon_pred=0.0`.

At step 0, `recon_scale=0.0` because reconstruction warms up over 2000 steps. That is expected.

## 4. Launch On 1x A100

Use one tmux session:

```bash
tmux new -s inv013_residual_recon
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv013_residual_recon_only
export WANDB_NAME=ae_sharp_slots_residual_recon
mkdir -p logs /workspace/ckpt/inv013_residual_recon_only

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 \
  --steps 15000 \
  --seed 42 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.0 \
  --lambda-sigreg 0.0 \
  --lambda-cov 0.0 \
  --lambda-slot 0.0 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-residual-target \
  --recon-mean-momentum 0.99 \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv013_residual_recon_only \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv013_residual_recon_only.log 2>&1 &

echo "launched inv013 residual-target recon-only run on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Detach with `Ctrl-B`, then `D`.

## 5. Early Tripwires

Step 0 should appear within a few minutes:

```bash
grep -m1 "step=0 " logs/inv013_residual_recon_only.log
```

Expected at step 0:

- `recon_target_residual=1.0`;
- finite `recon_mean_norm` (> 0 once initialized);
- `present_recon_only=1.0`;
- `prediction_active=0.0`;
- `L_flow=0.0`;
- `L_recon_pred=0.0`;
- finite `L_recon`;
- finite `loss`;
- `recon_scale=0.0`;
- `sigreg_scale=0.0`;
- `grad_skipped=0`.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv013_residual_recon_only.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

Expected around step 500:

- `present_recon_only=1.0`;
- `prediction_active=0.0`;
- `L_flow=0.0`;
- `L_recon_pred=0.0`;
- `recon_scale` around `0.25`;
- no skipped gradients.

If it stalls:

```bash
tail -n 100 logs/inv013_residual_recon_only.log
nvidia-smi
```

## 6. Monitor

Terminal:

```bash
watch -n 5 nvidia-smi
tail -f logs/inv013_residual_recon_only.log
```

In W&B group `inv013_residual_recon_only`, chart:

- `L_recon_present`
- `L_recon`
- `L_recon_shuffled_c`
- `L_recon_video_gap`
- `recon_target_residual`
- `recon_mean_norm`
- `c_effective_rank`
- `c_cross_video_cosine`
- `c_std_mean`
- `c_std_median`
- `c_dead_dim_frac`
- `c_slot_diversity_rank`
- `c_slot_diversity_rank_centered`
- `c_attn_entropy`
- `c_attn_entropy_min`
- `grad_norm`
- `grad_skipped`
- `grad_has_nan`
- `instability_warn`
- `agc_B_clipped`
- `agc_B_max_ratio`
- `agc_D_clipped`
- `agc_D_max_ratio`
- `present_recon_only`
- `prediction_active`

Do not chart these as success gates for this run:

- `coarse_vs_copy_ratio`
- `coarse_vs_batch_mean_ratio`
- `coarse_copy_loss`
- `coarse_model_loss`

Prediction is intentionally inactive, so these are irrelevant or absent.

## 7. Verdict Rules

This run succeeds as a bottleneck reconstruction test if:

- `L_recon_present` falls materially;
- `L_recon_video_gap` is clearly positive (shuffled latent decodes worse than true latent);
- `c_effective_rank` rises materially from run_052's collapse regime;
- `c_cross_video_cosine` stays low enough that videos do not collapse together;
- gradients remain finite with no sustained skips.

This run fails or is inconclusive if:

- `L_recon_present` improves but `L_recon_video_gap` stays near zero;
- `L_recon_present` improves but geometry remains collapsed (`rank` low, cosine high);
- there are persistent instabilities or skipped gradients.

Important interpretation note:

- Residual-target reconstruction loss is a different objective from run_052's absolute-feature loss, so
  raw `L_recon_present` values are not numerically comparable to run_052's `0.293`.
  Compare trends and honesty diagnostics (`L_recon_video_gap`) instead.
