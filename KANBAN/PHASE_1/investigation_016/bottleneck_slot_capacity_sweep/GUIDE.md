# GUIDE — bottleneck slot-capacity sweep on EGO4D

This guide launches three present-only V-JEPA reconstruction runs:

```text
dataset = ego4d
N_c in {32, 64, 128}
D_c = 256 throughout
```

The exact scientific axis is **slot-mediated latent bandwidth**, `N_c x D_c`. The commands do not
change whitening, encoder, decoder, target type, loss weights, or schedule. Read
[`PLAN.md`](PLAN.md) before launch for the value choice and interpretation rules.

Merely writing or reading this guide is not launch authorization. On a fresh pod, complete
[`AGENT_FILES/SETUPS/NEW_POD.md`](../../../../AGENT_FILES/SETUPS/NEW_POD.md) first. Run every command
below on the RunPod unless a section explicitly says otherwise. Stop at the first failed gate.

## 0. Non-negotiable sweep contract

| Field | Exact value |
|---|---|
| Core arms | EGO4D x `N_c=32,64,128` |
| Encoder | pinned V-JEPA2 ViT-L/16 |
| `D_c` / bottleneck mixer | 256 / 256 |
| Bottleneck latent blocks | 3 |
| Decoder | 512 wide x 4 blocks |
| Mode | present reconstruction only |
| Target | absolute, fixed-whitened encoder features |
| Loss | cosine; `lambda_recon=1.0`; 2,000-step ramp |
| Geometry | `lambda_var=0.5`, `lambda_cov=0.01`, SIGReg/slot loss off |
| Run length | 15,000 steps, physical batch 64, seed 42 |
| W&B group | `inv016_bottleneck_capacity_vjepa2_whitened_recon1` |

Do not add `--recon-residual-target`, remove whitening, change encoder, change `D_c`, lower the
batch, or resume a checkpoint. Those would create different experiments.

## 1. Define the shared identity and output roots

Paste this at the beginning of the SSH shell and again in any manually opened tmux shell:

```bash
cd /workspace/hierarchal-jepa-flow-world-model

export VJEPA_REV=b3c1679b7c34d3255ef3547f27c7b226aefab26f
export FRAME_MB=8
export BATCH=64
export PYTHONHASHSEED=42
export HF_HOME=/workspace/hf_cache
export SWEEP_GROUP=inv016_bottleneck_capacity_vjepa2_whitened_recon1
export STATS_DIR=/workspace/stats/inv016_bottleneck_capacity
export PREFLIGHT_DIR=/workspace/preflight/inv016_bottleneck_capacity
export EGO_STATS="$STATS_DIR/vjepa2_vitl16_ego4d_train_seed42.pt"

mkdir -p /workspace/hf_cache /workspace/ckpt "$STATS_DIR" "$PREFLIGHT_DIR" logs
```

## 2. Pin one clean source commit for all three arms

Do not discard or overwrite unexplained pod changes.

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git status --short
git pull --ff-only
git status --short
git rev-parse HEAD
python -m pip install -r requirements.txt
pytest -q
wandb login --verify
nvidia-smi
```

Require:

- both `git status --short` outputs are empty;
- the intended branch fast-forwards cleanly and the printed commit is recorded for the sweep;
- tests pass, W&B authentication verifies, and every GPU intended for a launch is idle.

All three W&B provenance records must show the same clean Git commit. Run 060 was recorded from a
dirty commit, which is why its 32-slot result is context rather than this sweep's control.

## 3. Prepare exactly one strict EGO4D whitening artifact

All three slot arms must use the same artifact bytes. EGO4D run 060 already used a current strict
12,800-clip payload at
`logs/whiten/whiten_stats_ego4d_train_seed42.pt`. It may be reused only if the exact file is still
present; the resource preflight in Section 4 is the authoritative compatibility check.

### 3.1 EGO4D: paste once; the command decides what to do

You do **not** need to check whether `$EGO_STATS` exists. Paste this entire block. It reuses the
file if it is already there, copies the run-060 file if available, or generates a new file if
neither exists.

```bash
cd /workspace/hierarchal-jepa-flow-world-model

if test -f "$EGO_STATS"; then
  echo "Using existing EGO4D whitening stats: $EGO_STATS"
elif test -f logs/whiten/whiten_stats_ego4d_train_seed42.pt; then
  cp logs/whiten/whiten_stats_ego4d_train_seed42.pt "$EGO_STATS"
  echo "Copied the existing run-060 whitening stats to: $EGO_STATS"
else
  echo "No EGO4D whitening stats found. Generating them now; this is a long GPU job."
  python whiten_stats.py \
    --data ego4d --split train \
    --encoder vjepa2_vitl16 --encoder-revision "$VJEPA_REV" \
    --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
    --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
    --batch-size 8 --max-clips 12800 --seed 42 --device cuda \
    --output "$EGO_STATS"
fi

python whiten_stats.py --inspect "$EGO_STATS"
sha256sum "$EGO_STATS"
```

Require dataset/split `ego4d/train`, exactly 12,800 clips, feature dimension 1,024, the pinned
V-JEPA revision, bf16/SDPA/frame-microbatch 8, finite eigensystem, and the current EGO4D dataset
fingerprint.

## 4. Minimal launch-safety checks at the largest core shape

This is not a preliminary scientific experiment. It is one exact one-step check that validates the
EGO4D artifact and proves that the most expensive core shape fits. If `N_c=128` passes at physical
batch 64, the 32/64 shapes are not separately resource-preflighted.

Paste the function:

```bash
capacity_preflight() {
  local dataset="$1"
  local stats="$2"
  local out="$PREFLIGHT_DIR/${dataset}_nc128_resource.json"
  local ckpt="/workspace/ckpt/inv016_capacity_preflight_${dataset}_nc128"

  python train.py --resource-preflight \
    --data "$dataset" --steps 15000 --seed 42 \
    --encoder vjepa2_vitl16 --encoder-revision "$VJEPA_REV" \
    --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
    --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
    --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
    --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
    --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
    --lambda-sigreg 0 --lambda-slot 0 \
    --whiten-features --whiten-eps 1e-4 --whiten-expected-clips 12800 \
    --whiten-stats-path "$stats" \
    --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 128 \
    --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
    --checkpoint-dir "$ckpt" --provenance-out "$out"
}

capacity_preflight ego4d "$EGO_STATS"
```

Require the report to show:

```text
batch size = 64
whiten_active = 1
present_recon_only = 1
prediction_active = 0
recon_target_residual = 0
L_flow = 0
L_recon_pred = 0
finite loss and gradients
non-null CUDA peak memory and positive throughput
```

If the 128-slot graph does not fit at batch 64, stop and report the measured resource boundary. Do
not quietly lower batch or add gradient accumulation while calling it the same sweep.

Run one Stage 0 on the largest shape after the EGO4D artifact passes:

```bash
python train.py --stage0-only \
  --data ego4d --steps 15000 --seed 42 \
  --encoder vjepa2_vitl16 --encoder-revision "$VJEPA_REV" \
  --encoder-precision bf16 --encoder-frame-microbatch "$FRAME_MB" \
  --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
  --batch-size "$BATCH" --horizon-k 12 --present-recon-only \
  --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
  --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
  --lambda-sigreg 0 --lambda-slot 0 \
  --whiten-features --whiten-eps 1e-4 --whiten-expected-clips 12800 \
  --whiten-stats-path "$EGO_STATS" \
  --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 --n-c 128 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4
```

Require `Stage 0 sanity passed`, finite metrics, a frozen encoder, and a successful EMA transition.

## 5. Three-arm launch matrix

| Dataset | Slots | Suggested GPU | Checkpoint/log tag |
|---|---:|---:|---|
| EGO4D | 32 | 0 | `inv016_capacity_ego4d_nc32` |
| EGO4D | 64 | 1 | `inv016_capacity_ego4d_nc64` |
| EGO4D | 128 | 2 | `inv016_capacity_ego4d_nc128` |

Launch EGO4D on GPUs 0-2 if three GPUs are available. A one-GPU pod can execute the same function
sequentially.

## 6. Define the exact launch function

Paste this function in the shell that contains the variables from Section 1:

```bash
launch_capacity_arm() {
  local dataset="$1"
  local slots="$2"
  local gpu="$3"
  local stats="$4"
  local dataset_label

  case "$dataset" in
    ego4d) dataset_label="EGO4D" ;;
    *) echo "STOP: unsupported dataset $dataset"; return 2 ;;
  esac

  local tag="inv016_capacity_${dataset}_nc${slots}"
  local session="$tag"
  local ckpt="/workspace/ckpt/$tag"
  local log="logs/$tag.log"
  local display_name="Investigation 16 · Bottleneck capacity · ${dataset_label} ${slots} slots"

  mkdir -p "$ckpt" logs

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "STOP: tmux session already exists: $session"
    return 3
  fi

  if find "$ckpt" -maxdepth 1 \
      \( -name 'phase1_step*.pt' -o -name 'run_provenance.json' \) \
      -print -quit | grep -q .; then
    echo "STOP: checkpoint directory is not empty: $ckpt"
    return 4
  fi

  tmux new-session -d -s "$session" \
    "set -o pipefail; \
     cd /workspace/hierarchal-jepa-flow-world-model && \
     export HF_HOME=/workspace/hf_cache PYTHONHASHSEED=42 && \
     CUDA_VISIBLE_DEVICES=$gpu python train.py \
       --data \"$dataset\" --steps 15000 --seed 42 \
       --encoder vjepa2_vitl16 --encoder-revision \"$VJEPA_REV\" \
       --encoder-precision bf16 --encoder-frame-microbatch \"$FRAME_MB\" \
       --encoder-attention-implementation sdpa --hf-cache-dir /workspace/hf_cache \
       --batch-size \"$BATCH\" --horizon-k 12 --present-recon-only \
       --lambda-recon 1.0 --lambda-recon-pred 0 --recon-loss-mode cosine \
       --recon-warmup-steps 2000 --lambda-var 0.5 --lambda-cov 0.01 \
       --lambda-sigreg 0 --lambda-slot 0 \
       --whiten-features --whiten-eps 1e-4 --whiten-expected-clips 12800 \
       --whiten-stats-path \"$stats\" \
       --bottleneck-latent-blocks 3 --decoder-dim 512 --decoder-blocks 4 \
       --n-c \"$slots\" \
       --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 --lr-decoder 1e-4 \
       --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
       --wandb-group \"$SWEEP_GROUP\" --wandb-name \"$display_name\" \
       --checkpoint-dir \"$ckpt\" --provenance-out \"$ckpt/run_provenance.json\" \
       --require-wandb --log-every 50 --diag-every 500 \
       2>&1 | tee \"$log\""

  tmux has-session -t "$session" \
    && echo "launched $display_name on GPU $gpu" \
    || { echo "STOP: tmux session did not persist: $session"; return 5; }
}
```

Every process runs in the foreground of its own tmux session. No arm uses `--resume`.

## 7. Launch the core EGO4D arms

```bash
launch_capacity_arm ego4d 32 0 "$EGO_STATS"
launch_capacity_arm ego4d 64 1 "$EGO_STATS"
launch_capacity_arm ego4d 128 2 "$EGO_STATS"
```

Do not map two processes to one GPU.

## 8. Immediate verification and early tripwire

After each wave launches:

```bash
tmux list-sessions
pgrep -af 'python.*train.py'
nvidia-smi
```

Inspect each session/log, substituting its tag:

```bash
tmux capture-pane -p -t inv016_capacity_ego4d_nc128 -S -120
tail -n 80 logs/inv016_capacity_ego4d_nc128.log
grep -m1 'step=0 ' logs/inv016_capacity_ego4d_nc128.log
```

At step 0 require finite metrics and:

```text
whiten_active = 1
present_recon_only = 1
prediction_active = 0
recon_target_residual = 0
recon_mean_norm = 0
L_flow = 0
L_recon_pred = 0
grad_skipped = 0
```

In W&B confirm the exact dataset, slot count, whitening payload fingerprint, V-JEPA revision,
`lambda_recon=1`, clean Git commit, group, and compliant display name. Confirm that all three arms
share the same EGO4D dataset and whitening fingerprints.

Require every launched arm to reach step 500. If several arms disappear together before then,
preserve logs and inspect the pod event/host-resource layer before assigning an `N_c` failure; that
is how investigation 007's capacity wave became invalid.

## 9. Monitor and complete

```bash
tmux list-sessions
nvidia-smi
tail -f logs/inv016_capacity_ego4d_nc128.log
```

On completion, check each tag has both provenance and a final checkpoint:

```bash
test -f /workspace/ckpt/inv016_capacity_ego4d_nc128/phase1_step15000.pt
test -f /workspace/ckpt/inv016_capacity_ego4d_nc128/run_provenance.json
sha256sum /workspace/ckpt/inv016_capacity_ego4d_nc128/phase1_step15000.pt
```

Repeat for all three tags and confirm W&B reached the final training/diagnostic windows. Record every
W&B ID and URL in [`DESCRIPTION.md`](DESCRIPTION.md).

## 10. Read the sweep without fooling ourselves

Run Reading Cycle B separately on every W&B run before comparing them. The primary capacity table
is:

| Dataset | `N_c` | late median `L_recon` | late/final `L_recon_present` | stability | spread/collapse context |
|---|---:|---:|---:|---|---|
| EGO4D | 32 | | | | |
| EGO4D | 64 | | | | |
| EGO4D | 128 | | | | |

Use the median training `L_recon` over steps 12,000–14,950 and the late fixed-batch diagnostics,
not one lucky point. Compare the 64/128-slot arms against the same-commit 32-slot EGO4D control.

Apply these guards:

- Do not choose the winner by raw `c_effective_rank`. Larger `N_c` mechanically supplies more fixed
  slot-identity directions. Treat rank as an own-arm health trajectory and read it with std, dead
  dimensions, and pair cosine.
- EGO4D `L_recon_video_gap` remains an exact-chunk/within-source metric under the current fixed batch.
  Do not call it cross-source conditioning.
- A lower loss with collapsed spread is not a capacity win.
- Prediction is off. Ignore `coarse_*` and make no forecasting claim.

The core hypothesis is supported when the valid 32→64→128 curve is monotonic and the 128-slot arm
beats 32 by at least 0.02 absolute or 3% relative in the late-window primary loss. Apply the full
decision table in [`PLAN.md`](PLAN.md).

## 11. Conditional 256-slot arm—not part of the core launch

Do not prelaunch 256.

- Core flat within 0.01: stop the slot ladder and move to no-whitening/channel-width work.
- Borderline monotonic 0.01–0.02: run 256 only for EGO4D.
- Clear 128-slot win: capacity is already supported; run 256 only to locate saturation for an
  operating-point decision.

If that rule is triggered, first repeat Section 4 with `--n-c 256`, then launch through the same
function:

```bash
launch_capacity_arm ego4d 256 0 "$EGO_STATS"
```

Its W&B name will remain contract-compliant and its checkpoint/log paths stay isolated.
