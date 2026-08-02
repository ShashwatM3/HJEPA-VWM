# GUIDE - running investigation_010 on a 1x A100 pod

Execution walkthrough for the residual-prediction rerun after commit `230096d`
(`Fix Phase 1 optimizer regularization plumbing`). This picks up after you have
already deployed a RunPod pod with **1 A100**, pulled the code, and have the repo at:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

This is a **single-process run pinned to GPU 0**, not DDP. It reruns the
investigation_009 residual arm (`graceful-river-35`) against the new optimizer /
regularization plumbing: decay/no-decay param groups, AGC exclusions for geometry and
zero-init gates, orthogonal bottleneck queries, SIGReg warmup, and clearer gradient
metrics.

---

## 0. What this run is checking

investigation_009 found the first healthy dynamic `c`: residual prediction drove
`coarse_vs_copy_ratio` from ~6 down to ~1.08 and reached rank ~58, but `F_c` mostly
tied copy by predicting `Delta-hat ~= 0`.

This rerun asks whether the latest optimizer/plumbing fixes make that same residual
configuration train more cleanly:

- keep the residual target: `--predict-residual`
- keep the inv009 residual config: `lambda_sigreg=5`, `lambda_var=0.5`, recon
  `0.05 + 0.05`, `k=12`, decoder `512x4`, `n_c=32`
- run from scratch; **do not `--resume` an old checkpoint**, because commit `230096d`
  changes optimizer param-group layout
- compare mainly against inv009 Run 2: `graceful-river-35` (`jsh6uo7p`)

The headline read is unchanged: stable `coarse_vs_copy_ratio < 1` is the win; ratio
near 1 with rising `coarse_copy_loss` still means `F_c` is tying the zero-residual
baseline.

## 1. Verify the pod state

```bash
cd /workspace/hierarchal-jepa-flow-world-model

nvidia-smi
git status --short
git log -1 --oneline
```

Expected:

- `nvidia-smi` shows exactly one visible A100, index `0`
- `git log -1 --oneline` shows commit `230096d` or a later commit that includes it
- no local edits on the pod that would make the run unreproducible

If the latest commit is older than `230096d`, sync before continuing:

```bash
git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline
```

## 2. Set runtime environment

Fresh SSH and tmux shells often lose environment exports, so set these in the shell
that will launch training:

```bash
export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv010_residual_plumbing
mkdir -p logs /workspace/ckpt
```

Assumptions from the existing setup guides:

- full SSv2 is at `/workspace/data/ssv2`
- checkpoints are outside the repo under `/workspace/ckpt` or `/workspace/checkpoints`
- W&B is logged in or `WANDB_API_KEY` is present

## 3. Smoke before burning GPU-hours

Run these once on the pod:

```bash
python -m pytest tests/test_agc.py tests/test_optimizer_and_flow.py tests/test_phase1_contract.py -q

python -c "from models import smoke_test_models; smoke_test_models()"

python train.py \
  --stage0-only \
  --data ssv2 \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lambda-sigreg 5.0 \
  --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.05 \
  --recon-warmup-steps 2000 \
  --predict-residual \
  --lr-coarse-flow 1e-4 \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32
```

Stop here if:

- `--predict-residual` or `--sigreg-warmup-steps` is unrecognized
- tests fail in optimizer / AGC plumbing
- `smoke_test_models()` fails a gradient contract
- stage0 produces NaN / traceback

## 4. Launch in tmux

```bash
tmux new -s inv010
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv010_residual_plumbing
mkdir -p logs /workspace/ckpt

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 \
  --steps 15000 \
  --horizon-k 12 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 5.0 \
  --sigreg-warmup-steps 2000 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.05 \
  --recon-warmup-steps 2000 \
  --predict-residual \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv010_residual_230096d \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv010_residual_230096d.log 2>&1 &

echo "launched inv010 residual rerun on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Detach with `Ctrl-B`, then `D`. Before closing SSH:

```bash
tmux ls
```

`inv010` must still be listed.

## 5. Early tripwires

Step 0 should appear within a couple of minutes:

```bash
grep -m1 "step=0 " logs/inv010_residual_230096d.log
```

Sanity check the first line:

- config includes `predict_residual=True`
- `sigreg_scale` starts near 0 and ramps during the first 2000 steps
- `L_flow`, `loss`, `L_recon`, and `L_recon_pred` are finite
- `agc_active=1`
- `grad_skipped=0`

Step-600 tripwire:

```bash
sleep 900
grep -q "step=500 " logs/inv010_residual_230096d.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

If it stalls:

```bash
tail -n 80 logs/inv010_residual_230096d.log
nvidia-smi
```

Look for traceback, `Killed`, CUDA OOM, pod reclaim, or repeated `grad_skipped=1`.

## 6. Monitor

```bash
watch -n 5 nvidia-smi
tail -f logs/inv010_residual_230096d.log
grep -H "step=15000" logs/inv010_residual_230096d.log
```

In W&B project `hjepa-vwm`, group `inv010_residual_plumbing`, chart:

- `coarse_vs_copy_ratio`
- `coarse_copy_loss`
- `coarse_model_loss`
- `c_effective_rank`
- `c_cross_video_cosine`
- `c_std_mean`
- `L_flow`
- `L_sigreg` and `sigreg_scale`
- `L_recon_present`, `L_recon_chat`, `L_recon_cplus`
- `grad_norm`
- `grad_global_norm_postclip`
- `agc_B_clipped`, `agc_Fc_clipped`, `agc_D_clipped`
- `agc_B_max_ratio`, `agc_Fc_max_ratio`, `agc_D_max_ratio`
- `grad_skipped`, `instability_warn`

Read `grad_norm` as the true pre-clip magnitude. `grad_global_norm_postclip` is expected
to sit near the 0.5 global clip whenever clipping fires.

## 7. How to read the result

Compare primarily to investigation_009 Run 2 (`graceful-river-35`, commit `bc77db6`):

- **Win:** `coarse_vs_copy_ratio` is stably below 1, ideally trending toward 0.7, while
  `c_effective_rank` clears or approaches 60 and `c_cross_video_cosine` stays low.
- **Partial improvement:** ratio stays around 1 but `grad_norm`, AGC, rank, and
  cross-video cosine are cleaner than inv009. The plumbing helped stability/geometry but
  did not solve prediction.
- **Same failure:** ratio returns to ~1.05-1.10 with `coarse_model_loss` tracking
  `coarse_copy_loss`. That is still the `Delta-hat ~= 0` tie.
- **Regression:** rank falls, cosine rises, `grad_skipped` appears, or AGC clips nearly
  everything every step. Treat as an optimizer/plumbing regression and stop early.

The key synthesis to write afterward:

```text
Did commit 230096d make the residual run skillfully beat zero-residual copy, or did it
only harden the training loop around the same tie-by-zero failure?
```

## 8. End-to-end recap

SSH into 1x A100 pod -> verify `230096d` -> export `HF_HOME` and
`WANDB_RUN_GROUP=inv010_residual_plumbing` -> run tests + smoke + residual stage0 ->
start `tmux inv010` -> launch the single GPU-0 residual command -> detach and verify
`tmux ls` -> check step 0 and step 500 -> monitor W&B/logs -> compare to
`graceful-river-35`.

---

### Sources

- Latest code commit: `230096d` (`Fix Phase 1 optimizer regularization plumbing`)
- Parent result: [`../investigation_009/RESULTS_ANALYSIS.md`](../investigation_009/RESULTS_ANALYSIS.md)
- Prior launch format: [`../investigation_009/GUIDE.md`](../investigation_009/GUIDE.md)
