# GUIDE — run 058 `ae_latent_stack_whiten_abs_recon_cov_var_ego4d`

> **What this is.** Pod operator guide to launch the EGO4D twin of SSv2 run 057
> (`ae_latent_stack_whiten_abs_recon_cov_var`, W&B `cdvp6hou`). Identical recipe;
> single intended deltas are `--data ego4d` and the matching EGO4D whitening stats file.
>
> **Prerequisite.** Stages 0–6 of
> [`AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md`](../../../../AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md)
> are already verified complete on this volume (`data/ego4d`, `data/ego4d_tiny`,
> switchability smoke). Do not re-run that guide unless a Verify block below fails.
>
> **How to use.** Sections are ordered. Every label beginning with **Paste** is exactly
> one block to paste at once. Wait for the shell prompt unless the text says a long job
> is still running. Stop on any failed Verify.

This launches investigation_016 run 058 on a 1x A100 pod:

```text
present-only reconstruction
cosine loss
ABSOLUTE (clean) whitened-feature target (residual target OFF)
FIXED offline feature whitening on EGO4D train stats (NEW file — not the SSv2 one)
Perceiver latent-stack bottleneck (default architecture, 3 blocks)
+ geometry regularizers: lambda_var=0.5, lambda_cov=0.01   (SIGReg OFF: lambda_sigreg=0)
no prediction, no slot penalty
dataset = ego4d
```

Identical to
[`../../investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/GUIDE.md`](../../investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/GUIDE.md)
**except** `--data ssv2` → `--data ego4d`, the whitening stats path, W&B name/group,
checkpoint/log tags, and the one-time EGO4D whitening computation in §2.

```text
lambda_recon      = 0.05          lambda_var    = 0.5
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

SSv2 control reference (do not launch this — overlay target after the run):

- W&B: `smahalanobis-uc-davis/hjepa-vwm/cdvp6hou`
- display name: `ae_latent_stack_whiten_abs_recon_cov_var`
- KANBAN analysis:
  [`../../investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/ANALYSIS.md`](../../investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/ANALYSIS.md)

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

Same commit family as run 057 (latent stack + whitening + cov/var regularizers). No code
change is required for this run — it is a dataset + stats-path delta.

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

## 2. Whitening Statistics (EGO4D — compute once, then REUSE)

**Do not point `--whiten-stats-path` at the SSv2 file.** Whitening is one-time per
dataset. Expected artifact:

```text
logs/whiten/whiten_stats_ego4d_train_seed42.pt
```

**Paste 1 — check whether the file already exists:**

```bash
cd /workspace/hierarchal-jepa-flow-world-model
ls -la logs/whiten/whiten_stats_ego4d_train_seed42.pt 2>/dev/null \
  && echo "EGO4D whitening stats ALREADY PRESENT — skip Paste 2–3" \
  || echo "EGO4D whitening stats MISSING — run Paste 2–3"
```

If the file is present and was built with `--data ego4d --split train --seed 42
--max-batches 200`, skip to §3. If unsure of provenance, recompute (idempotent overwrite).

**Paste 2 — compute in tmux** (frozen encoder over ~200 train batches; typically
tens of minutes on A100 — leave it running):

```bash
tmux new -s inv016_whiten_stats
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
mkdir -p logs/whiten

python3 whiten_stats.py \
  --data ego4d \
  --split train \
  --max-batches 200 \
  --seed 42 \
  --out logs/whiten/whiten_stats_ego4d_train_seed42.pt \
  --device cuda
```

Detach with `Ctrl-b` then `d` if needed. Re-attach with `tmux attach -t inv016_whiten_stats`.

**Paste 3 — verify the artifact:**

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
print("EGO4D whitening statistics OK")
EOF
```

**Verify:** prints `EGO4D whitening statistics OK` and `dataset: ego4d`. Stop if `dataset`
is `ssv2` — wrong file / wrong CLI `--data`.

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
  --lambda-recon 0.05 \
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

- Stage 0 produces NaN or a traceback (`FileNotFoundError` naming the stats path means
  §2 was skipped or the path is wrong; `No .webm or .mp4 files found` means ego4d roots
  are wrong);
- Stage 0 does not show `whiten_active=1.0`;
- Stage 0 does not show `recon_target_residual=0.0` (this run is the CLEAN arm — if this
  reads 1.0 you launched the residual variant);
- Stage 0 does not show `recon_mean_norm=0.0` (mean tracker inactive without the residual
  target);
- Stage 0 does not show `present_recon_only=1.0` / `prediction_active=0.0`;
- Stage 0 does not show `L_flow=0.0` / `L_recon_pred=0.0`;
- `L_var` and `L_cov` are not both finite and positive (they must be COMPUTED — the weights
  being nonzero is what adds them to the loss).

Note on `L_sigreg`: with `lambda_sigreg=0` it is still COMPUTED and logged (for-logging-only)
but is NOT added to the loss. Expect a small finite value. At step 0, `recon_scale=0.0`
(2000-step recon warmup) is expected; `L_var`/`L_cov` enter the loss at full weight from
step 0.

In the W&B Stage-0 / config dump (if one appears), confirm `data.dataset` is **`ego4d`**,
not `ssv2` or `ego4d_tiny`.

---

## 4. Launch On 1x A100

Use one tmux session:

```bash
tmux new -s inv016_whiten_abs_recon_cov_var_ego4d
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv016_ego4d_transfer
export WANDB_NAME=ae_latent_stack_whiten_abs_recon_cov_var_ego4d
mkdir -p logs /workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d

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
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ego4d_train_seed42.pt \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv016_whiten_abs_recon_cov_var_ego4d.log 2>&1 &

echo "launched inv016 run 058 (run 057 recipe on ego4d) on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Approximate wall time: ~6 h (same class as other 15k-step present-only runs on A100).

Do NOT pass `--resume` with any prior checkpoint — this run must train from scratch
(same seed, same init path) to be the A/B against `cdvp6hou`.

Do NOT use `--data ego4d_tiny` for this launch (tiny is for smoke only).

---

## 5. Early Tripwires

```bash
grep -m1 "step=0 " logs/inv016_whiten_abs_recon_cov_var_ego4d.log
```

Expected at step 0: `whiten_active=1.0`, `recon_target_residual=0.0`, `recon_mean_norm=0.0`,
`present_recon_only=1.0`, `prediction_active=0.0`, `L_flow=0.0`, `L_recon_pred=0.0`, finite
positive `L_var`/`L_cov`, `recon_scale=0.0`, `grad_skipped=0`.

Confirm in the W&B run config:

- `data.dataset` = `ego4d`
- `train.whiten_stats_path` ends with `whiten_stats_ego4d_train_seed42.pt`
- `lambda_var=0.5`, `lambda_cov=0.01`, **`lambda_sigreg=0`**
- `present_recon_only=true`, `recon_residual_target=false`

If `data.dataset` is `ssv2`, you launched the control again. If
`whiten_stats_path` still points at the `ssv2` file, kill the run and restart §4.

Mechanical, do not read as progress: raw `c_slot_diversity_rank` near 32 and pooled
`c_effective_rank` near 31 at init (fixed slot identities). Judge slots on
`c_slot_diversity_rank_centered` only.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv016_whiten_abs_recon_cov_var_ego4d.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

---

## 6. Monitor

```bash
watch -n 5 nvidia-smi
tail -f logs/inv016_whiten_abs_recon_cov_var_ego4d.log
```

In W&B group `inv016_ego4d_transfer`, overlay THIS run against SSv2 run 057 (`cdvp6hou`).
Decisive panels (Reading Cycle B / present-only):

- Geometry: `c_effective_rank`, `c_slot_diversity_rank_centered`, `c_cross_video_cosine`,
  `c_std_mean`, `c_dead_dim_frac`.
- Honesty: `L_recon_shuffled_c`, `L_recon_video_gap`, `L_recon_present`, `L_recon`
  — compare **shape / share**, not absolute level vs SSv2 (different substrate + different
  whitened space).
- Regularizer terms: `L_var`, `L_cov` (should fall as they bite), `recon_scale`. `L_sigreg`
  still logs but is for-logging-only.
- Wiring: `whiten_active` (=1), `recon_target_residual` (=0), `recon_mean_norm` (=0).
- Stability: `grad_norm`, `grad_skipped`, `grad_has_nan`, `instability_warn`, `agc_B_*`,
  `agc_D_*`.

Ignore `coarse_*` panels — prediction is intentionally inactive.

After finish, also confirm:

```bash
test -f /workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d/phase1_step15000.pt && \
  echo "run 058 final checkpoint OK"
```

---

## 7. Verdict Rules

Judged against the SSv2 run-057 pattern (geometry held under cov+var; honesty video-specific),
with cross-dataset raw-recon incomparability kept in mind:

- **Transfer succeeds:** `c_effective_rank` rises well above the ~40 slot-structure ceiling
  and stays there (same ballpark as run 057's ~200 is a strong success; even a clearly
  high-rank plateau with no late collapse counts), centered slot rank stays near the
  full 32, `c_cross_video_cosine` drops off the init-collapse band, `L_recon_video_gap`
  opens and stays positive, gradients clean. Then the inv015 recipe is portable → next
  step is prediction transfer on EGO4D (or a residual-target EGO4D ablation if honesty
  share looks soft).
- **Geometry fails on EGO4D:** rank stays near the low-20s / collapses while wiring flags
  stay correct. Then cov+var weights (or whitening eps / stats sample) may need an EGO4D
  retune — do not blame the dataset flag alone until §2 provenance is re-checked.
- **Honesty fails (template shortcut):** `L_recon_video_gap` near zero / shuffled-c ≈ present
  while recon improves. Then reconsider residual target on EGO4D even though run 057 kept
  absolute target on SSv2.
- **Validity gate:** gradients finite, no sustained skips, AGC quiet, wiring flags correct,
  `data.dataset=ego4d`, correct EGO4D whitening path.

Interpretation guards:

- Never compare raw `L_recon_present` across SSv2 vs EGO4D as if they share a scale.
- `c_*` formulas are shared; geometry curves are the fair overlay.
- Do not treat the ego4d GUIDE smoke runs (`treasured-cherry-58`, `copper-sunset-59`) as
  science controls — they are switchability checks only.
