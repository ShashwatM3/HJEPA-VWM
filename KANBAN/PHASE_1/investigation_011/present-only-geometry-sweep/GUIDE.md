# GUIDE - present-only geometry sweep

Execution walkthrough for the investigation_011 present-only geometry sweep. This is the
operational plan for Recommendation 1 from
[`../fixed-position-present-recon/ANALYSIS_inv011_fixed_position_present_recon.md`](../run_041_inv011_fixed_position_present_recon/ANALYSIS_inv011_fixed_position_present_recon.md):

```text
present-only reconstruction
fixed-position decoder
cosine reconstruction loss
SIGReg x VICReg-C geometry sweep on c_t
```

Repo path on RunPod:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

## 0. Logistics Decision

Use **two waves maximum**, **5 GPUs per wave maximum**, and **one experiment per GPU**. This is the
same operational model as investigation 007/008:

- each run is a separate `python train.py` process;
- each process is pinned to one GPU with `CUDA_VISIBLE_DEVICES`;
- each run has its own checkpoint directory;
- each run has its own log file;
- all runs share one W&B group;
- no DDP or distributed training.

The full grid would be 12 runs:

```text
lambda_sigreg = {5.0, 7.5, 10.0, 12.5}
lambda_cov    = {0.0, 0.003, 0.01}
```

But the already-completed run
[`inv011_fixed_position_present_recon`](../run_041_inv011_fixed_position_present_recon/OBSERVATIONS.md) is exactly:

```text
lambda_sigreg = 5.0
lambda_cov = 0.0
lambda_recon = 0.05
present_recon_only = true
```

So this plan runs **10 new experiments** and treats the existing run `hcr2qx19` as the anchor/control.
That gives 11 of the 12 grid points under the two-wave budget. The only omitted point is:

```text
lambda_sigreg = 12.5
lambda_cov = 0.01
```

That point is the highest combined regularization pressure and the least valuable first-pass run. If
`lambda_sigreg=10.0, lambda_cov=0.01` wins cleanly and the high end still looks under-regularized,
run `12.5/0.01` later as a single follow-up. It does not deserve one of the first 10 GPUs.

Use `hcr2qx19` as the anchor only if the pulled code has not changed the training/model/loss path
since that run. Documentation-only changes are fine. If `train.py`, `models.py`, `losses.py`, or
`config.py` changed materially, rerun the anchor as a replacement for the lowest-value Wave 1 point
(`po_geom_sig12p5_cov0`) and mark the historical `hcr2qx19` as context only.

## 1. Wave Plan

| Wave | GPUs | Runs | Purpose |
|---|---:|---:|---|
| Existing anchor | 0 new | 1 existing | `lambda_sigreg=5.0`, `lambda_cov=0.0`; W&B run `hcr2qx19`. |
| Wave 1 | 5 | 5 | Finish the SIGReg-only ladder and sample mild covariance at the current anchor and likely useful mid-high SIGReg. |
| Wave 2 | 5 | 5 | Fill the remaining mild-cov points and test high covariance up to `lambda_sigreg=10.0`. |

Wave 1:

| GPU | W&B name / tag | `lambda_sigreg` | `lambda_cov` | Role |
|---:|---|---:|---:|---|
| 0 | `po_geom_sig7p5_cov0` | 7.5 | 0.0 | SIGReg-only ladder. |
| 1 | `po_geom_sig10_cov0` | 10.0 | 0.0 | SIGReg-only ladder; likely rank-clearing candidate. |
| 2 | `po_geom_sig12p5_cov0` | 12.5 | 0.0 | SIGReg-only high end. |
| 3 | `po_geom_sig5_cov0p003` | 5.0 | 0.003 | Isolate mild covariance against the existing anchor. |
| 4 | `po_geom_sig10_cov0p003` | 10.0 | 0.003 | Test covariance at the likely useful SIGReg band. |

Wave 2:

| GPU | W&B name / tag | `lambda_sigreg` | `lambda_cov` | Role |
|---:|---|---:|---:|---|
| 0 | `po_geom_sig7p5_cov0p003` | 7.5 | 0.003 | Complete the mild-cov mid point. |
| 1 | `po_geom_sig12p5_cov0p003` | 12.5 | 0.003 | Complete the mild-cov high point. |
| 2 | `po_geom_sig5_cov0p01` | 5.0 | 0.01 | High-cov effect at the current SIGReg anchor. |
| 3 | `po_geom_sig7p5_cov0p01` | 7.5 | 0.01 | High-cov mid point. |
| 4 | `po_geom_sig10_cov0p01` | 10.0 | 0.01 | High-cov likely useful upper point. |

All new runs should be grouped under:

```text
WANDB_RUN_GROUP=inv011_present_only_geometry_sweep
```

Constant recipe:

```text
data = ssv2
steps = 15000
horizon_k = 12
present_recon_only = true
prediction_active = false
recon_loss_mode = cosine
sigreg_warmup_steps = 2000
recon_warmup_steps = 2000
decoder_dim = 512
decoder_blocks = 4
n_c = 32
lr_bottleneck = 1e-4
lr_coarse_flow = 1e-4
lambda_recon = 0.05
lambda_recon_pred = 0.0
lambda_var = 0.5
lambda_slot = 0.0
```

Do not add `lambda_slot`. Investigation 003 showed slot-diversity loss can Goodhart the slot-rank
diagnostic. This sweep tests feature geometry through SIGReg and covariance only.

## 2. Pod Setup

This guide picks up where
[`AGENT_FILES/SETUPS/NEW_POD.md`](../../../../AGENT_FILES/SETUPS/NEW_POD.md) leaves off. On a fresh
pod, do that setup first:

1. SSH into the pod from the RunPod Connect tab.
2. Install `git`, `tmux`, `curl`, and certificates if the image is fresh.
3. Set `HF_HOME=/workspace/hf_cache`.
4. Pull the repo into `/workspace/hierarchal-jepa-flow-world-model`.
5. Install `requirements.txt`.
6. Run `wandb login`.
7. Verify SSv2 exists under `/workspace/data/ssv2`.

Quick connect and GPU check:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
nvidia-smi
```

Expected: at least 5 visible GPUs, indexed `0..4`.

## 3. Sync Code And Smoke Checks

Run this once on the pod:

```bash
cd /workspace/hierarchal-jepa-flow-world-model

git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline

export HF_HOME=/workspace/hf_cache
```

Expected code support:

- fixed-position decoder code is present;
- `--present-recon-only` is accepted;
- `--lambda-sigreg` is accepted;
- `--lambda-cov` is accepted;
- `--recon-loss-mode cosine` is accepted.

Run focused tests:

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

Run Stage 0 with a representative covariance config:

```bash
python train.py \
  --stage0-only \
  --data ssv2 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 10.0 \
  --sigreg-warmup-steps 2000 \
  --lambda-cov 0.003 \
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
- Stage 0 throws a traceback or NaN;
- Stage 0 does not show `present_recon_only=1.0`;
- Stage 0 does not show `prediction_active=0.0`;
- Stage 0 does not show `L_flow=0.0`;
- Stage 0 does not show `L_recon_pred=0.0`;
- W&B/config output does not include the intended `lambda_sigreg` and `lambda_cov`.

At step 0, `recon_scale=0.0` and `sigreg_scale=0.0` are expected because both ramp over 2000 steps.

## 4. Launch Wave 1 - 5 GPUs

Launch Wave 1 in tmux:

```bash
tmux new -s inv011_geom_wave1
cd /workspace/hierarchal-jepa-flow-world-model

mkdir -p logs /workspace/ckpt/inv011_present_only_geometry_sweep
export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv011_present_only_geometry_sweep

launch_geom() {
  local TAG=$1
  local LSIG=$2
  local LCOV=$3
  local GPU=$4

  CUDA_VISIBLE_DEVICES=$GPU WANDB_NAME=$TAG python train.py \
    --data ssv2 \
    --steps 15000 \
    --horizon-k 12 \
    --lr-bottleneck 1e-4 \
    --lr-coarse-flow 1e-4 \
    --lambda-var 0.5 \
    --lambda-sigreg $LSIG \
    --sigreg-warmup-steps 2000 \
    --lambda-cov $LCOV \
    --lambda-recon 0.05 \
    --lambda-recon-pred 0.0 \
    --recon-loss-mode cosine \
    --recon-warmup-steps 2000 \
    --present-recon-only \
    --decoder-dim 512 \
    --decoder-blocks 4 \
    --n-c 32 \
    --checkpoint-dir /workspace/ckpt/inv011_present_only_geometry_sweep/$TAG \
    --log-every 50 \
    --diag-every 500 \
    > logs/$TAG.log 2>&1 &

  echo "launched GPU $GPU -> $TAG (sigreg=$LSIG cov=$LCOV pid $!)"
}

wave1=(
  "po_geom_sig7p5_cov0      7.5  0.0"
  "po_geom_sig10_cov0       10.0 0.0"
  "po_geom_sig12p5_cov0     12.5 0.0"
  "po_geom_sig5_cov0p003    5.0  0.003"
  "po_geom_sig10_cov0p003   10.0 0.003"
)

for i in "${!wave1[@]}"; do
  read TAG LSIG LCOV <<< "${wave1[$i]}"
  launch_geom "$TAG" "$LSIG" "$LCOV" "$i"
done

echo "Wave 1 launched. Detach with Ctrl-B then D, then verify with: tmux ls"
```

Detach with `Ctrl-B`, then `D`. Verify before closing SSH:

```bash
tmux ls
```

## 5. Wave 1 Tripwires

Step 0 should appear within a few minutes:

```bash
for f in \
  logs/po_geom_sig7p5_cov0.log \
  logs/po_geom_sig10_cov0.log \
  logs/po_geom_sig12p5_cov0.log \
  logs/po_geom_sig5_cov0p003.log \
  logs/po_geom_sig10_cov0p003.log; do
  echo "== $f"
  grep -m1 "step=0 " "$f" || true
done
```

Expected at step 0:

- `present_recon_only=1.0`;
- `prediction_active=0.0`;
- `L_flow=0.0`;
- `L_recon_pred=0.0`;
- finite `loss`;
- finite `L_recon_present` or `L_recon`;
- `recon_scale=0.0`;
- `sigreg_scale=0.0`;
- `grad_skipped=0`.

Step-600 tripwire:

```bash
sleep 900
for f in \
  logs/po_geom_sig7p5_cov0.log \
  logs/po_geom_sig10_cov0.log \
  logs/po_geom_sig12p5_cov0.log \
  logs/po_geom_sig5_cov0p003.log \
  logs/po_geom_sig10_cov0p003.log; do
  grep -q "step=500 " "$f" && echo "OK   $f" || echo "STALL $f  <-- inspect"
done
```

If any run stalls:

```bash
tail -n 100 <log-file>
nvidia-smi
```

Check the RunPod console events for stop/reclaim if all runs die together. If one run OOMs, lower
`global_batch` or `num_workers` in `config.py` and relaunch that run alone. OOM is not expected;
this sweep keeps the same `n_c`, decoder width, and frozen encoder as the previous present-only run.

## 6. Launch Wave 2 - 5 GPUs

Launch Wave 2 after Wave 1 clears the early health checks. If you are reusing the same 5-GPU pod,
wait until Wave 1 finishes and the GPUs are free. If you deploy a separate 5-GPU pod for Wave 2, it
is fine to start Wave 2 once Wave 1 has reached at least step 500 cleanly on all five runs and no
covariance run shows obvious NaN/skip spirals.

```bash
tmux new -s inv011_geom_wave2
cd /workspace/hierarchal-jepa-flow-world-model

mkdir -p logs /workspace/ckpt/inv011_present_only_geometry_sweep
export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv011_present_only_geometry_sweep

launch_geom() {
  local TAG=$1
  local LSIG=$2
  local LCOV=$3
  local GPU=$4

  CUDA_VISIBLE_DEVICES=$GPU WANDB_NAME=$TAG python train.py \
    --data ssv2 \
    --steps 15000 \
    --horizon-k 12 \
    --lr-bottleneck 1e-4 \
    --lr-coarse-flow 1e-4 \
    --lambda-var 0.5 \
    --lambda-sigreg $LSIG \
    --sigreg-warmup-steps 2000 \
    --lambda-cov $LCOV \
    --lambda-recon 0.05 \
    --lambda-recon-pred 0.0 \
    --recon-loss-mode cosine \
    --recon-warmup-steps 2000 \
    --present-recon-only \
    --decoder-dim 512 \
    --decoder-blocks 4 \
    --n-c 32 \
    --checkpoint-dir /workspace/ckpt/inv011_present_only_geometry_sweep/$TAG \
    --log-every 50 \
    --diag-every 500 \
    > logs/$TAG.log 2>&1 &

  echo "launched GPU $GPU -> $TAG (sigreg=$LSIG cov=$LCOV pid $!)"
}

wave2=(
  "po_geom_sig7p5_cov0p003  7.5  0.003"
  "po_geom_sig12p5_cov0p003 12.5 0.003"
  "po_geom_sig5_cov0p01     5.0  0.01"
  "po_geom_sig7p5_cov0p01   7.5  0.01"
  "po_geom_sig10_cov0p01    10.0 0.01"
)

for i in "${!wave2[@]}"; do
  read TAG LSIG LCOV <<< "${wave2[$i]}"
  launch_geom "$TAG" "$LSIG" "$LCOV" "$i"
done

echo "Wave 2 launched. Detach with Ctrl-B then D, then verify with: tmux ls"
```

Wave 2 tripwire:

```bash
sleep 900
for f in \
  logs/po_geom_sig7p5_cov0p003.log \
  logs/po_geom_sig12p5_cov0p003.log \
  logs/po_geom_sig5_cov0p01.log \
  logs/po_geom_sig7p5_cov0p01.log \
  logs/po_geom_sig10_cov0p01.log; do
  grep -q "step=500 " "$f" && echo "OK   $f" || echo "STALL $f  <-- inspect"
done
```

## 7. Monitor

Terminal:

```bash
watch -n 5 nvidia-smi
tail -f logs/po_geom_sig10_cov0p003.log
grep -H "step=15000" logs/po_geom_sig*.log
```

In W&B group `inv011_present_only_geometry_sweep`, chart:

- `L_recon_present`
- `L_recon`
- `L_sigreg`
- `L_cov`
- `L_var`
- `sigreg_scale`
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
- `agc_D_clipped`
- `present_recon_only`
- `prediction_active`

Do not use prediction/future metrics as success gates for this sweep:

- `coarse_vs_copy_ratio`
- `coarse_vs_batch_mean_ratio`
- `coarse_copy_loss`
- `coarse_model_loss`
- `L_recon_cplus`
- `L_recon_chat`
- `c_plus_effective_rank`

Those are intentionally inactive or irrelevant in present-only mode.

## 8. Read The Results

Use the present-only reading cycle from
[`../../../../GUIDES/READING_EXPERIMENTS.md`](../../../../GUIDES/READING_EXPERIMENTS.md), with this
sweep-specific priority order:

1. Include the existing anchor run `hcr2qx19` (`lambda_sigreg=5.0`, `lambda_cov=0.0`) in every W&B
   comparison chart. It lives in the previous W&B group, so pin/filter it by run id or run name
   alongside the new sweep group.
2. Training health:
   `grad_skipped=0`, `grad_has_nan=0`, `instability_warn=0`, no synchronized pod death.
3. Representation alive and non-static:
   `c_std_mean` near `0.8-1.2`, `c_dead_dim_frac` near `0`, `c_cross_video_cosine < 0.30` preferred
   and `<0.50` hard ceiling.
4. Rank gate:
   `c_effective_rank >= 60` sustained late, not just a transient.
5. Reconstruction preservation:
   `L_recon_present <= 0.36` preferred. Do not accept a high-rank run that destroys present
   reconstruction.
6. Slot redundancy:
   `c_slot_diversity_rank` should not regress below the current run's late value of about `5`.
   Improvement above `8` is a strong sign that the geometry change is not merely global-dim inflation.

Interpretation matrix:

| Pattern | Meaning | Next action |
|---|---|---|
| Rank clears 60, recon stays `<=0.36`, cosine healthy | Present bottleneck geometry is likely sufficient | Use this config as the next substrate. |
| Rank clears 60, slot rank stays near 5 | Global feature geometry improved but slots remain redundant | Build the anchored-slot bottleneck next. |
| Rank improves but recon worsens materially | Regularization is overpowering content | Back down to the nearest lower `lambda_sigreg`/`lambda_cov` pair. |
| Cosine climbs or std drifts outside band | Geometry is becoming static or distorted | Reject that config even if rank looks good. |
| `lambda_cov=0.003` helps but `0.01` hurts | Keep mild covariance; do not push covariance higher | Pick the best mild-cov config. |
| `lambda_sigreg=10.0, lambda_cov=0.01` is best and still below rank gate | We may have dropped a useful high-high point | Run `lambda_sigreg=12.5, lambda_cov=0.01` as a single follow-up. |
| All covariance values fail and SIGReg-only wins | Covariance is not the right active lever here | Use SIGReg-only and move to anchored slots if slot rank remains low. |

## 9. Result Write-Up

After each wave:

1. Record each run's W&B name, run id, URL, final step, and state in the matching wave folder.
2. Update this folder with a short observations file or extend the wave README.
3. After both waves, summarize the best config and failure modes in the parent investigation docs.

Minimum table to fill:

| Run | W&B id | Final step | `lambda_sigreg` | `lambda_cov` | `L_recon_present` | `c_effective_rank` | `c_cross_video_cosine` | `c_std_mean` | `c_slot_diversity_rank` | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `inv011_fixed_position_present_recon` | `hcr2qx19` | 14000 diag | 5.0 | 0.0 | 0.3452 | 49.60 | 0.0911 | 0.9842 | 5.08 | Existing anchor |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 10. End-To-End Recap

Fresh pod setup -> SSH -> `nvidia-smi` -> `git pull` -> tests -> `smoke_test_models()` -> Stage 0
representative config -> tmux Wave 1 -> detach and verify `tmux ls` -> step-0 and step-600
tripwires -> monitor W&B -> launch Wave 2 if Wave 1 is healthy -> compare all new runs against
existing anchor `hcr2qx19` -> read rank, recon, cosine, std, and slot rank together -> write up.

### Sources

- Sibling operational guides:
  [`../fixed-position-present-recon/GUIDE.md`](../run_041_inv011_fixed_position_present_recon/GUIDE.md),
  [`../../investigation_008/GUIDE.md`](../../investigation_008/GUIDE.md),
  [`../../investigation_007/GUIDE.md`](../../investigation_007/GUIDE.md)
- Shared fresh-pod checklist:
  [`../../../../AGENT_FILES/SETUPS/NEW_POD.md`](../../../../AGENT_FILES/SETUPS/NEW_POD.md)
