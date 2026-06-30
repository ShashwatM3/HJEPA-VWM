# GUIDE - fixed-position decoder run

This launches the investigation_011 full residual/cosine recipe with the fixed-position
reconstruction decoder implemented in `models.Decoder`.

Repo path on RunPod:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

There is **no CLI flag** for the fixed-position decoder. The feature is active when the code includes
commit `0a9ff46` or later. Do not resume from old decoder checkpoints, because learned-query decoder
state is intentionally incompatible with this architecture.

## 0. Verify Pod State

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

- `git log -1 --oneline` shows `0a9ff46` or a later commit that includes it;
- full SSv2 exists at `/workspace/data/ssv2`;
- no local pod edits that would make the run unreproducible.

## 1. Runtime Environment

```bash
export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv011_fixed_position_decoder
export WANDB_NAME=inv011_fixed_position_decoder
mkdir -p logs /workspace/ckpt
```

`WANDB_NAME` is set through the W&B environment contract; `train.py` does not expose a run-name CLI
flag.

## 2. Smoke Checks

Run tests that cover the decoder architecture, reconstruction losses, optimizer grouping, and Phase
1 contracts:

```bash
python -m pytest \
  tests/test_decoder.py \
  tests/test_reconstruction_loss.py \
  tests/test_present_recon_only.py \
  tests/test_optimizer_and_flow.py \
  tests/test_phase1_contract.py \
  tests/test_agc.py \
  -q
```

Run the synthetic model smoke test:

```bash
python -c "from models import smoke_test_models; smoke_test_models()"
```

Run Stage 0 with the exact training recipe:

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

Stop if:

- tests fail;
- `smoke_test_models()` fails a gradient contract;
- Stage 0 produces NaN or a traceback;
- the code complains about old learned-query decoder checkpoint state.

## 3. Launch

For a 1x A100 pod:

```bash
tmux new -s inv011_fixed_position_decoder
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv011_fixed_position_decoder
export WANDB_NAME=inv011_fixed_position_decoder
mkdir -p logs /workspace/ckpt

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
  --checkpoint-dir /workspace/ckpt/inv011_fixed_position_decoder \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv011_fixed_position_decoder.log 2>&1 &

echo "launched inv011 fixed-position decoder run on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Detach with `Ctrl-B`, then `D`.

## 4. Early Tripwires

Step 0 should appear within a few minutes:

```bash
grep -m1 "step=0 " logs/inv011_fixed_position_decoder.log
```

Expected:

- `present_recon_only=0.0`;
- `prediction_active=1.0`;
- finite `L_flow`, `L_recon`, `L_recon_pred`, and `loss`;
- `recon_loss_mode='cosine'` in W&B config;
- `predict_residual=True` in W&B config;
- `grad_skipped=0`.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv011_fixed_position_decoder.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

If it stalls:

```bash
tail -n 100 logs/inv011_fixed_position_decoder.log
nvidia-smi
```

## 5. Monitor

```bash
watch -n 5 nvidia-smi
tail -f logs/inv011_fixed_position_decoder.log
```

In W&B group `inv011_fixed_position_decoder`, chart:

- `coarse_vs_copy_ratio`
- `coarse_copy_loss`
- `coarse_model_loss`
- `coarse_vs_batch_mean_ratio`
- `c_effective_rank`
- `c_plus_effective_rank`
- `c_cross_video_cosine`
- `c_std_mean`
- `L_recon_present`
- `L_recon_cplus`
- `L_recon_chat`
- `L_recon`
- `L_recon_pred`
- `L_flow`
- `grad_norm`
- `grad_skipped`
- `agc_D_clipped`
- `agc_D_max_ratio`

## 6. Result Read

The main comparison is Run A from this investigation:

```text
new_recon_loss / 1u69hpfm
same recipe, learned-query decoder, cosine reconstruction
```

A good result does not require `L_recon_present` to be lower. The proposal predicts reconstruction may
be harder at first. The stronger signal is whether readouts become more diagnostic:

- if `c_hat` is poor, `L_recon_chat` should separate from `L_recon_cplus`;
- if the decoder was previously hiding poor prediction behind learned templates, the gap should widen;
- if dynamics improve, `coarse_vs_copy_ratio` should move below Run A's roughly 1.06 plateau.

Fill [OBSERVATIONS.md](OBSERVATIONS.md) after the run finishes or is stopped.
