# GUIDE - sharp-slot reconstruction-only bottleneck test

This launches the first clean autoencoder-only test after the sharpened bottleneck slot-attention
change:

```text
present-only reconstruction
no prediction
no SIGReg
no variance floor
no covariance penalty
no slot-diversity penalty
```

The question is narrow: **does reconstruction pressure alone improve the bottleneck representation
now that slot identity survives and attention can select memory tokens?**

Repo path on RunPod:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

## 0. What This Run Is Testing

The primary run must isolate reconstruction. Keep every geometry/collapse regularizer off:

```text
lambda_recon = 0.05
lambda_var = 0.0
lambda_cov = 0.0
lambda_sigreg = 0.0
lambda_slot = 0.0
present_recon_only = true
prediction_active = false
```

Do **not** pass `--lambda-cov 0.01` on this primary run. Covariance directly decorrelates `c_t` and
can raise effective rank by itself, so it would confound the answer. If this pure reconstruction run
looks healthy, launch a separate follow-up with `--lambda-cov 0.01`.

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

- `git log -1 --oneline` shows a commit at or after `cc0e318` (`Sharpen bottleneck slot attention`);
- full SSv2 exists at `/workspace/data/ssv2`;
- no local pod edits that would make the run unreproducible.

The commit matters because old code does not include:

- learned memory position embeddings in `B`;
- sharpened cosine cross-attention in `B`;
- residual slot-identity output in `B`;
- centered slot-diversity diagnostic logging.

Do not resume from old checkpoints. The bottleneck state dict changed.

## 2. Runtime Environment

```bash
export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv012_sharp_slot_recon_only
export WANDB_NAME=ae_sharp_slots_recon_only
mkdir -p logs /workspace/ckpt/inv012_sharp_slot_recon_only
```

`WANDB_NAME` is set through the W&B environment contract; `train.py` does not expose a run-name CLI
flag.

## 3. Smoke Checks

Run tests that cover the changed bottleneck, present-only routing, reconstruction losses, decoder,
optimizer grouping, and Phase 1 contracts:

```bash
python -m pytest \
  tests/test_bottleneck_attention.py \
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
  --lambda-var 0.0 \
  --lambda-sigreg 0.0 \
  --lambda-cov 0.0 \
  --lambda-slot 0.0 \
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
- Stage 0 does not show `L_recon_pred=0.0`;
- the Stage 0 command has drifted from the zero-regularizer recipe above.

At step 0, `recon_scale=0.0` because reconstruction warms up over 2000 steps. That is expected.

## 4. Launch On 1x A100

Use one tmux session:

```bash
tmux new -s inv012_recon_only
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv012_sharp_slot_recon_only
export WANDB_NAME=ae_sharp_slots_recon_only
mkdir -p logs /workspace/ckpt/inv012_sharp_slot_recon_only

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
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv012_sharp_slot_recon_only \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv012_sharp_slot_recon_only.log 2>&1 &

echo "launched inv012 sharp-slot reconstruction-only run on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Detach with `Ctrl-B`, then `D`.

## 5. Early Tripwires

Step 0 should appear within a few minutes:

```bash
grep -m1 "step=0 " logs/inv012_sharp_slot_recon_only.log
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
- W&B config has `lambda_var=0.0`;
- W&B config has `lambda_cov=0.0`;
- W&B config has `lambda_sigreg=0.0`;
- W&B config has `lambda_slot=0.0`;
- W&B config has `recon_loss_mode='cosine'`.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv012_sharp_slot_recon_only.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

Expected around step 500:

- `present_recon_only=1.0`;
- `prediction_active=0.0`;
- `L_flow=0.0`;
- `L_recon_pred=0.0`;
- `recon_scale` around `0.25`;
- `sigreg_scale=0.0`;
- no skipped gradients.

If it stalls:

```bash
tail -n 100 logs/inv012_sharp_slot_recon_only.log
nvidia-smi
```

## 6. Monitor

Terminal:

```bash
watch -n 5 nvidia-smi
tail -f logs/inv012_sharp_slot_recon_only.log
```

In W&B group `inv012_sharp_slot_recon_only`, chart:

- `L_recon_present`
- `L_recon`
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

Prediction is intentionally inactive, so those metrics are irrelevant or absent.

## 7. Verdict Rules

This run succeeds as a bottleneck reconstruction test if:

- `L_recon_present` falls materially from its early plateau;
- `c_effective_rank` rises without `lambda_var`, `lambda_cov`, or SIGReg helping it;
- `c_cross_video_cosine` stays low enough that different videos are not collapsing together;
- `c_slot_diversity_rank_centered` improves or stays healthy;
- `c_attn_entropy` trends below `1.0` rather than staying uniform;
- gradients remain finite with no sustained skips.

This run fails or is inconclusive if:

- `L_recon_present` improves but `c_effective_rank` stays low and `c_cross_video_cosine` rises;
- raw `c_slot_diversity_rank` jumps but `c_slot_diversity_rank_centered` does not move;
- `c_attn_entropy` remains near `1.0` for the whole run;
- the run needs `lambda_var`, `lambda_cov`, or SIGReg to look healthy.

The raw slot-diversity number is expected to jump mechanically because fixed slot identities now
survive into `c_t`. Judge slot health primarily by `c_slot_diversity_rank_centered`.

## 8. Optional Follow-up Only After Reading This Run

If reconstruction alone clearly improves the bottleneck, launch a separate covariance follow-up:

```text
same command, but:
WANDB_NAME=ae_sharp_slots_recon_cov0p01
WANDB_RUN_GROUP=inv012_sharp_slot_recon_cov_followup
--lambda-cov 0.01
--checkpoint-dir /workspace/ckpt/inv012_sharp_slot_recon_cov0p01
```

Do not mix the follow-up into the primary verdict. `lambda_cov=0.01` directly attacks feature
correlation, so it answers a different question: whether covariance improves an already useful
reconstruction-shaped bottleneck.
