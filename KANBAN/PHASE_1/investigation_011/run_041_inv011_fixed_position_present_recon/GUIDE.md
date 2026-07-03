# GUIDE - fixed-position present-only reconstruction run

This launches investigation_011 Run D on a 1x A100 pod:

```text
present-only reconstruction
cosine loss
fixed-position decoder
SIGReg + variance floor on c_t
```

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
export WANDB_RUN_GROUP=inv011_fixed_position_present_recon
export WANDB_NAME=inv011_fixed_position_present_recon
mkdir -p logs /workspace/ckpt
```

`WANDB_NAME` is set through the W&B environment contract; `train.py` does not expose a run-name CLI
flag.

## 2. Smoke Checks

Run tests that cover the decoder architecture, reconstruction losses, present-only routing,
optimizer grouping, and Phase 1 contracts:

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
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32
```

Stop if:

- tests fail;
- `smoke_test_models()` fails a gradient contract;
- Stage 0 produces NaN or a traceback;
- Stage 0 does not show `present_recon_only=1.0`;
- Stage 0 does not show `prediction_active=0.0`;
- Stage 0 does not show `L_flow=0.0`;
- the code complains about old learned-query decoder checkpoint state.

At step 0, `recon_scale=0.0` and `sigreg_scale=0.0` because both terms warm up over 2000 steps.
That is expected.

## 3. Launch On 1x A100

Use one tmux session:

```bash
tmux new -s inv011_fixed_position_present_recon
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv011_fixed_position_present_recon
export WANDB_NAME=inv011_fixed_position_present_recon
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
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv011_fixed_position_present_recon \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv011_fixed_position_present_recon.log 2>&1 &

echo "launched inv011 fixed-position present-recon run on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Detach with `Ctrl-B`, then `D`.

## 4. Early Tripwires

Step 0 should appear within a few minutes:

```bash
grep -m1 "step=0 " logs/inv011_fixed_position_present_recon.log
```

Expected at step 0:

- `present_recon_only=1.0`;
- `prediction_active=0.0`;
- `L_flow=0.0`;
- `L_recon_pred=0.0`;
- finite `L_recon`;
- finite `loss`;
- `recon_scale=0.0`;
- `sigreg_scale=0.0`;
- `grad_skipped=0`;
- W&B config has `recon_loss_mode='cosine'`;
- W&B config has `lambda_sigreg=5.0`;
- W&B config has `lambda_var=0.5`.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv011_fixed_position_present_recon.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

Expected around step 500:

- `present_recon_only=1.0`;
- `prediction_active=0.0`;
- `L_flow=0.0`;
- `L_recon_pred=0.0`;
- `recon_scale` around `0.25`;
- `sigreg_scale` around `0.25`;
- no skipped gradients.

If it stalls:

```bash
tail -n 100 logs/inv011_fixed_position_present_recon.log
nvidia-smi
```

## 5. Monitor

Terminal:

```bash
watch -n 5 nvidia-smi
tail -f logs/inv011_fixed_position_present_recon.log
```

In W&B group `inv011_fixed_position_present_recon`, chart:

- `L_recon_present`
- `L_recon`
- `L_sigreg`
- `sigreg_scale`
- `L_var`
- `c_effective_rank`
- `c_cross_video_cosine`
- `c_std_mean`
- `c_std_median`
- `c_dead_dim_frac`
- `c_slot_diversity_rank`
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
- `L_recon_cplus`
- `L_recon_chat`
- `c_plus_effective_rank`

They are prediction/future-branch metrics. Present-only diagnostics intentionally skip them.

## 6. Reading The Result

Use the present-recon-only branch of the experiment reading cycle:

| Q | Question | Main panels |
|---|---|---|
| Q1 | Did training stay alive? | `grad_skipped`, `grad_has_nan`, `grad_norm`, `instability_warn` |
| Q2 | Is `c_t` alive and video-specific? | `c_std_mean`, `c_dead_dim_frac`, `c_cross_video_cosine` |
| Q3 | Is `c_t` high-rank? | `c_effective_rank`, `c_std_mean` |
| Q5-prime | Did present reconstruction train a useful representation? | `L_recon_present`, `c_effective_rank`, `c_cross_video_cosine`, `c_std_mean` |
| Q8 | Verdict | label from `GRADIENTS_AND_READOUTS.md` |

Success should mean both:

```text
L_recon_present falls materially
c_t geometry remains healthy
```

Geometry target:

```text
c_effective_rank > 60
c_std_mean roughly 0.8-1.2
c_cross_video_cosine < 0.5
c_dead_dim_frac near 0
```

Strong practical result:

```text
L_recon_present <= about 0.346
c_effective_rank > 60
c_cross_video_cosine < 0.5
grad_skipped = 0
```

The `0.346` reference is Run C's fixed-position full-recipe final `L_recon_present`. It is not a
hard acceptance threshold, but it is a useful comparator.

## 7. Result Write-Up

Fill [OBSERVATIONS.md](OBSERVATIONS.md) after the run finishes or is stopped.

Include:

- run id and W&B URL;
- final diagnostic step;
- final and last-3 plateau values for `L_recon_present`, rank, std, cross-video cosine, and
  stability;
- whether the result is `Strong present representation`, `Low-rank decodable`, `Pretty geometry,
  weak content`, `Collapsed rep`, or `Invalid`;
- a recommendation for the next experiment.
