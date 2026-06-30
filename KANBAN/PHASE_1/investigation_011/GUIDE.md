# GUIDE - running investigation_011 on RunPod

Execution walkthrough for the two reconstruction-objective runs. This assumes the repo is at:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

Run A keeps the full inv010 residual/SIGReg recipe and switches only reconstruction to cosine.
Run B disables prediction and trains only the present reconstruction bottleneck path.

## 0. Verify pod state

```bash
cd /workspace/hierarchal-jepa-flow-world-model

nvidia-smi
git status --short
git log -1 --oneline
```

Expected:

- code includes `--recon-loss-mode` and `--present-recon-only`
- no local pod edits that would make the runs unreproducible
- full SSv2 is available at `/workspace/data/ssv2`

## 1. Runtime environment

```bash
export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv011_recon_objective
mkdir -p logs /workspace/ckpt
```

## 2. Smoke checks

```bash
python -m pytest \
  tests/test_reconstruction_loss.py \
  tests/test_present_recon_only.py \
  tests/test_phase1_contract.py \
  tests/test_optimizer_and_flow.py \
  tests/test_agc.py \
  -q

python -c "from models import smoke_test_models; smoke_test_models()"
```

Stage0 for Run A:

```bash
python train.py \
  --stage0-only \
  --data ssv2 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 5.0 \
  --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.05 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --predict-residual \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32
```

Stage0 for Run B:

```bash
python train.py \
  --stage0-only \
  --data ssv2 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lambda-var 0.0 \
  --lambda-sigreg 0.0 \
  --lambda-recon 0.05 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32
```

Stop if either flag is unrecognized, if present-only launches without `L_flow=0`, or if any smoke
check fails.

## 3. Launch both runs

Use a 2x A100 pod if available. On a 1x A100 pod, run them sequentially.

```bash
tmux new -s inv011
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv011_recon_objective
mkdir -p logs /workspace/ckpt
```

Run A on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 \
  --steps 15000 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 5.0 \
  --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.05 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --predict-residual \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv011_cosine_residual \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv011_cosine_residual.log 2>&1 &
```

Run B on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --data ssv2 \
  --steps 15000 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lambda-var 0.0 \
  --lambda-sigreg 0.0 \
  --lambda-recon 0.05 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv011_present_recon_only \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv011_present_recon_only.log 2>&1 &
```

Detach with `Ctrl-B`, then `D`, and confirm:

```bash
tmux ls
```

## 4. Early tripwires

Check step 0:

```bash
grep -m1 "step=0 " logs/inv011_cosine_residual.log
grep -m1 "step=0 " logs/inv011_present_recon_only.log
```

Run A should show:

- `present_recon_only=0.0`
- `prediction_active=1.0`
- finite `L_flow`, `L_recon`, and `L_recon_pred`
- `recon_loss_mode='cosine'` in W&B config

Run B should show:

- `present_recon_only=1.0`
- `prediction_active=0.0`
- `L_flow=0.0`
- `L_recon_pred=0.0`
- finite `L_recon`

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv011_cosine_residual.log && echo "Run A step 500 OK"
grep -q "step=500 " logs/inv011_present_recon_only.log && echo "Run B step 500 OK"
```

If either stalls:

```bash
tail -n 80 logs/inv011_cosine_residual.log
tail -n 80 logs/inv011_present_recon_only.log
nvidia-smi
```

## 5. Monitor

Run A W&B charts:

- `coarse_vs_copy_ratio`
- `coarse_copy_loss`
- `coarse_model_loss`
- `c_effective_rank`
- `c_cross_video_cosine`
- `L_recon_present`, `L_recon_chat`, `L_recon_cplus`
- `L_recon`, `L_recon_pred`
- `grad_norm`, `grad_skipped`, `agc_*`

Run B W&B charts:

- `L_recon_present`
- `L_recon`
- `c_effective_rank`
- `c_cross_video_cosine`
- `c_std_mean`
- `present_recon_only`
- `prediction_active`
- `grad_norm`, `grad_skipped`, `agc_B_clipped`, `agc_D_clipped`

Do not judge Run B with copy-ratio gates; prediction is intentionally off.

## 6. Result read

Run A answers whether the new reconstruction geometry helps the current best full recipe beat copy.
Run B answers whether present reconstruction alone can make `c_t` rich and decodable without any
future-prediction pressure.
