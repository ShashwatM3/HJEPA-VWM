# GUIDE — run 060 `Investigation 16 · Whitened latent stack EGO4D · Reconstruction weight 1.00`

> **What this is.** Pod operator guide for the run-058 present-only whitened AE recipe on
> EGO4D with a single intended delta: `--lambda-recon 1.0` (was `0.05`).
>
> **Prerequisite.** Stages 0–6 of
> [`AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md`](../../../../AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md)
> are already verified. Run 058 (or this guide §2) already produced
> `logs/whiten/whiten_stats_ego4d_train_seed42.pt` — **reuse that file**.
>
> **How to use.** Sections are ordered. Every label beginning with **Paste** is exactly
> one block to paste at once. Wait for the shell prompt unless the text says a long job
> is still running. Stop on any failed Verify.

This launches investigation_016 run 060 on a 1x A100 pod:

```text
present-only reconstruction
cosine loss
ABSOLUTE (clean) whitened-feature target (residual target OFF)
FIXED offline feature whitening on EGO4D train stats (REUSE run 058 file)
Perceiver latent-stack bottleneck (default architecture, 3 blocks)
+ geometry regularizers: lambda_var=0.5, lambda_cov=0.01   (SIGReg OFF)
no prediction, no slot penalty
dataset = ego4d
lambda_recon = 1.0   ← sole delta vs run 058
```

Identical to
[`../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/GUIDE.md`](../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/GUIDE.md)
**except** `--lambda-recon 0.05` → `--lambda-recon 1.0`, plus new W&B name / group tags /
checkpoint / log paths.

```text
lambda_recon      = 1.0           lambda_var    = 0.5
lambda_recon_pred = 0.0           lambda_sigreg = 0.0
present_recon_only = true         lambda_cov    = 0.01
recon_residual_target = FALSE     lambda_slot   = 0.0
whiten_features   = true
horizon_k = 12                    n_c = 32
decoder_dim = 512                 decoder_blocks = 4
seed = 42                         steps = 15000
--data ego4d
--whiten-stats-path logs/whiten/whiten_stats_ego4d_train_seed42.pt
```

Control reference (do not relaunch — overlay target after the run):

- W&B: `smahalanobis-uc-davis/hjepa-vwm/mvbx96nv`
- display name: `ae_latent_stack_whiten_abs_recon_cov_var_ego4d`
- KANBAN analysis:
  [`../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md`](../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/ANALYSIS.md)

---

## 0. Preflight — EGO4D volume still healthy

Run on the pod SSH session (outside tmux is fine).

**Paste 1 — confirm corpus + tiny subset:**

```bash
test -d /workspace/data/ego4d/train && test -d /workspace/data/ego4d/validation && \
  test -d /workspace/data/ego4d_tiny/train && \
  echo "ego4d roots OK"
echo "ego4d train chunks: $(find /workspace/data/ego4d/train -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
echo "ego4d val chunks:   $(find /workspace/data/ego4d/validation -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
echo "ego4d_tiny train:   $(find /workspace/data/ego4d_tiny/train -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
```

**Verify:** train/val chunk counts are large (order ~10^5 / ~10^4), `ego4d_tiny` train is
4000 if Stage 5 defaults were used. If roots are missing, return to the ego4d GUIDE —
do not continue.

**Paste 2 — confirm CLI accepts ego4d:**

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 train.py --help | grep -E "ego4d"
python3 whiten_stats.py --help | grep -E "ego4d"
```

**Verify:** both help lines list `ego4d` / `ego4d_tiny`.

---

## 1. Verify Code Version

Pull a clean current commit (pluggable encoder pipeline is fine; default encoder remains
V-JEPA2). No code change is required for this run — it is a `lambda_recon` delta on run 058.

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

---

## 2. Whitening Statistics (EGO4D — REUSE run 058 file)

**Do not recompute** if the run-058 artifact is present and provenanced.

Expected artifact:

```text
logs/whiten/whiten_stats_ego4d_train_seed42.pt
```

**Paste 1 — confirm the file exists:**

```bash
cd /workspace/hierarchal-jepa-flow-world-model
ls -la logs/whiten/whiten_stats_ego4d_train_seed42.pt
```

**Paste 2 — verify provenance (must be ego4d / V-JEPA `d_e=1024`):**

```bash
cd /workspace/hierarchal-jepa-flow-world-model
test -f logs/whiten/whiten_stats_ego4d_train_seed42.pt && \
  python3 - <<'EOF'
import torch
from pathlib import Path
p = Path("logs/whiten/whiten_stats_ego4d_train_seed42.pt")
obj = torch.load(p, map_location="cpu", weights_only=False)
print("path:", p)
print("dataset:", obj.get("dataset"))
print("split:", obj.get("split"))
print("seed:", obj.get("seed"))
print("rows:", obj.get("rows"))
print("d_e:", obj.get("d_e"))
print("mean shape:", tuple(obj["mean"].shape))
print("eigvals shape:", tuple(obj["eigvals"].shape))
assert obj.get("dataset") == "ego4d", obj.get("dataset")
assert obj.get("split") == "train", obj.get("split")
assert int(obj.get("seed", -1)) == 42
assert int(obj.get("d_e", -1)) == 1024
print("EGO4D whitening statistics OK — reuse for run 060")
EOF
```

**Verify:** prints `EGO4D whitening statistics OK — reuse for run 060`. Stop if `dataset` is
`ssv2` or `d_e` is not 1024 (SigLIP stats would be 768 — wrong arm).

If the file is missing, compute it exactly as run 058 GUIDE §2 (same CLI), then re-run Paste 2.

---

## 3. Stage 0 With The Exact Recipe

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache

python3 train.py \
  --stage0-only \
  --data ego4d \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 0.0 \
  --lambda-cov 0.01 \
  --lambda-slot 0.0 \
  --lambda-recon 1.0 \
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ego4d_train_seed42.pt \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32
```

Stop if:

- Stage 0 produces NaN or a traceback;
- Stage 0 does not show `whiten_active=1.0`;
- Stage 0 does not show `recon_target_residual=0.0`;
- Stage 0 does not show `recon_mean_norm=0.0`;
- Stage 0 does not show `present_recon_only=1.0` / `prediction_active=0.0`;
- Stage 0 does not show `L_flow=0.0` / `L_recon_pred=0.0`;
- `L_var` and `L_cov` are not both finite and positive;
- W&B / config dump shows `lambda_recon` anything other than `1.0` (or `1`).

At step 0, `recon_scale=0.0` (2000-step recon warmup) is still expected — the higher weight
only bites after the ramp. Confirm `data.dataset` is **`ego4d`**.

---

## 4. Launch On 1x A100

Use one tmux session:

```bash
tmux new -s inv016_whiten_abs_recon_cov_var_ego4d_recon1
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv016_ego4d_transfer
export WANDB_NAME='Investigation 16 · Whitened latent stack EGO4D · Reconstruction weight 1.00'
mkdir -p logs /workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d_recon1

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data ego4d \
  --steps 15000 \
  --seed 42 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 0.0 \
  --lambda-cov 0.01 \
  --lambda-slot 0.0 \
  --lambda-recon 1.0 \
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ego4d_train_seed42.pt \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d_recon1 \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv016_whiten_abs_recon_cov_var_ego4d_recon1.log 2>&1 &

echo "launched inv016 run 060 (run 058 recipe, lambda_recon=1.0) on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Approximate wall time: ~6 h (same class as run 058).

Do NOT pass `--resume` with any run-058 checkpoint — train from scratch.

Do NOT use `--data ego4d_tiny` for this launch.

---

## 5. Early Tripwires

```bash
grep -m1 "step=0 " logs/inv016_whiten_abs_recon_cov_var_ego4d_recon1.log
```

Expected at step 0: `whiten_active=1.0`, `recon_target_residual=0.0`, `recon_mean_norm=0.0`,
`present_recon_only=1.0`, `prediction_active=0.0`, `L_flow=0.0`, `L_recon_pred=0.0`, finite
positive `L_var`/`L_cov`, `recon_scale=0.0`, `grad_skipped=0`.

Confirm in the W&B run config:

- display name = `Investigation 16 · Whitened latent stack EGO4D · Reconstruction weight 1.00`
- `data.dataset` = `ego4d`
- `train.whiten_stats_path` ends with `whiten_stats_ego4d_train_seed42.pt`
- **`lambda_recon=1.0`** (not 0.05)
- `lambda_var=0.5`, `lambda_cov=0.01`, `lambda_sigreg=0`
- `present_recon_only=true`, `recon_residual_target=false`

If `lambda_recon` is still `0.05`, kill and relaunch §4.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv016_whiten_abs_recon_cov_var_ego4d_recon1.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

Watch for sustained `grad_skipped` or pathological `grad_norm` once `recon_scale` rises —
`lambda_recon=1.0` is 20× the control weight.

---

## 6. Monitor

```bash
watch -n 5 nvidia-smi
tail -f logs/inv016_whiten_abs_recon_cov_var_ego4d_recon1.log
```

In W&B group `inv016_ego4d_transfer`, overlay THIS run against run 058 (`mvbx96nv`).
Decisive panels (Reading Cycle B / present-only):

- Geometry: `c_effective_rank`, `c_slot_diversity_rank_centered`, `c_cross_video_cosine`,
  `c_std_mean`, `c_dead_dim_frac`.
- Honesty: `L_recon_shuffled_c`, `L_recon_video_gap`, `L_recon_present`, `L_recon`
  — video gap / conditioned share first; raw recon second.
- Regularizer terms: `L_var`, `L_cov`, `recon_scale`.
- Wiring: `whiten_active` (=1), `recon_target_residual` (=0), `recon_mean_norm` (=0).
- Stability: `grad_norm`, `grad_skipped`, `grad_has_nan`, `instability_warn`, `agc_B_*`,
  `agc_D_*`.

Ignore `coarse_*` panels — prediction is intentionally inactive.

After finish:

```bash
test -f /workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d_recon1/phase1_step15000.pt && \
  echo "run 060 final checkpoint OK"
```

---

## 7. Verdict Rules

Judged against run 058's failure mode (rank collapse / template shortcut), same EGO4D
substrate:

- **Weight helps:** material, growing `L_recon_video_gap` and non-collapsing abstract geometry
  vs run 058 → absolute-target weight was underpowered; residual-target arm still useful as
  a second control.
- **Raw recon improves, honesty does not:** video gap stays near zero / shuffled ≈ present —
  stronger weight did not break the template; residual target remains the higher-information
  next arm.
- **Geometry worsens under weight:** rank/std/cosine degrade vs run 058 while recon drops —
  objective conflict; do not transfer this checkpoint into prediction.
- **Validity gate:** finite grads, correct `lambda_recon=1.0`, `data.dataset=ego4d`, same
  whitening file as run 058, unique checkpoint dir (no accidental resume).

Interpretation guards:

- Overlay geometry against run 058, not SSv2 run 057 raw recon.
- Do not treat ego4d GUIDE smoke runs as science controls.
