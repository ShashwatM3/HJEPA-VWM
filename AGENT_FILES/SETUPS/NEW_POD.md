# NEW_POD.md - Fresh RunPod setup commands

> **Purpose.** Copy-paste checklist for a brand-new RunPod pod attached to the
> existing `/workspace` network volume. This is the short operational guide for
> the repeated "fresh pod" problems: missing pip packages (`transformers`),
> missing `tmux`, missing W&B login, and missing Hugging Face cache env vars.
>
> **Important:** RunPod pods are Linux. Use `apt-get` on the pod, not Homebrew.
> `brew` is only for your Mac laptop if you are installing local tools there.

---

## 0. SSH Into The Pod

Use the exact SSH command shown in the RunPod **Connect** tab. Example shape:

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

If SSH asks for a password, check that your RunPod account has your public key
and that your laptop key permissions are correct:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
```

### Mac-only tools, if missing

Run these on your **Mac laptop**, not on the pod:

```bash
# If Homebrew itself is missing, install it from https://brew.sh first.
brew install git
brew install openssh
```

Most fresh-pod setup happens after SSH, inside Linux, with `apt-get` and `pip`.

---

## 1. Install System Packages

Run this on the pod every time the image is fresh. This fixes the recurring
`tmux: command not found` issue and ensures basic tooling is available.

```bash
apt-get update -qq
apt-get install -y git tmux curl ca-certificates
```

Optional quality-of-life tools:

```bash
apt-get install -y htop nano
```

---

## 2. Set Persistent Cache Paths

Keep Hugging Face downloads on the persistent `/workspace` volume so the V-JEPA
2 checkpoint does not redownload unnecessarily.

```bash
mkdir -p /workspace/hf_cache /workspace/checkpoints
export HF_HOME=/workspace/hf_cache
```

For the current shell, the `export` is enough. If you want it to persist across
new SSH sessions on this pod:

```bash
grep -q "HF_HOME=/workspace/hf_cache" ~/.bashrc || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
```

---

## 3. Get The Code

The expected repo path on the network volume is:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

If the repo already exists:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull origin phase1-v0.2-frozen-encoder
```

If the repo does not exist yet:

```bash
cd /workspace
git clone https://github.com/ShashwatM3/HJEPA-VWM.git hierarchal-jepa-flow-world-model
cd /workspace/hierarchal-jepa-flow-world-model
git checkout phase1-v0.2-frozen-encoder
```

GitHub HTTPS authentication uses a **personal access token** as the password.
Your normal GitHub password will not work if prompted.

Verify the code version:

```bash
git status
git log --oneline -5
```

---

## 4. Install Python Packages

This fixes the recurring:

```text
ModuleNotFoundError: No module named 'transformers'
```

Run:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Quick dependency check:

```bash
python3 - <<'PY'
import torch, transformers, decord, wandb
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("decord OK")
print("wandb OK")
PY
```

---

## 5. Log Into W&B

Run once per fresh pod image:

```bash
wandb login
```

Paste your W&B API key when prompted.

If you are only testing and do not want W&B uploads:

```bash
export WANDB_MODE=offline
```

---

## 6. Verify Data Is Present

The project expects:

```text
/workspace/data/ssv2/train
/workspace/data/ssv2/validation
/workspace/data/ssv2_tiny/train
/workspace/data/ssv2_tiny/validation
```

Count the clips:

```bash
python3 - <<'PY'
from pathlib import Path
from config import DataConfig

d = DataConfig()
print("data_root:", d.data_root)
for name, root in [("ssv2", d.full_root), ("ssv2_tiny", d.tiny_root)]:
    for split in ["train", "validation"]:
        p = Path(root) / split
        n = len(list(p.glob("*.webm"))) if p.exists() else "MISSING DIR"
        print(f"{name}/{split}: {n} ({p})")
PY
```

Expected approximate counts from the current volume:

```text
ssv2/train: 168913
ssv2/validation: 24777
ssv2_tiny/train: 4002
ssv2_tiny/validation: 348
```

Verify the actual dataloader:

```bash
python3 -c "from data import smoke_test_dataloader; smoke_test_dataloader()"
```

For the full dataset dataloader:

```bash
python3 - <<'PY'
import torch
from config import Config
from data import build_dataloader

cfg = Config()
cfg.data.dataset = "ssv2"
loader = build_dataloader(cfg, "train", batch_size=2)
expected = (cfg.model.t_ctx, 3, cfg.model.h, cfg.model.w)
for i, (ctx, tgt) in enumerate(loader):
    print(ctx.shape, tgt.shape, float(ctx.min()), float(ctx.max()))
    assert ctx.shape[1:] == expected and tgt.shape[1:] == expected
    assert torch.isfinite(ctx).all() and torch.isfinite(tgt).all()
    if i == 1:
        break
print("ssv2 full dataloader OK")
PY
```

---

## 7. Stage 0 Preflight

Always run this before a training job:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
python3 train.py --stage0-only
```

If this fails with `No module named 'transformers'`, repeat section 4.

---

## 8. Start A Long Run In tmux

Use `tmux` so the run survives SSH disconnects.

```bash
tmux new -s hjepa
```

Inside tmux:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
python3 train.py --data ssv2 --steps 5000 --log-every 50 --diag-every 250 --lambda-slot 0.25 --lambda-cov 0.0027
```

Detach without killing the run:

```text
Ctrl-b, then d
```

Reattach later:

```bash
tmux attach -t hjepa
```

If you need a full 15k-step run instead of the 5k diagnostic:

```bash
python3 train.py --data ssv2 --steps 15000 --log-every 50 --diag-every 500 --lambda-slot 0.25 --lambda-cov 0.0027
```

---

## 9. Metrics To Watch

For the current slot-collapse fix, watch these together:

```text
L_flow
L_var
L_cov
L_slot
c_slot_diversity_rank
c_attn_entropy_min
c_effective_rank
coarse_vs_copy_ratio
grad_norm
grad_skipped
```

Interpretation:

- `c_slot_diversity_rank` should rise from the bad Run-A value (~1.6/32).
- `c_attn_entropy_min` should fall below near-uniform (~0.96-0.99).
- `c_effective_rank` should rise above the collapsed range (~5-9/256).
- `coarse_vs_copy_ratio` must not regress badly; this is the anti-Goodhart guard.
- `grad_skipped` should stay 0.

---

## 10. Common Fresh-Pod Failures

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'transformers'` | `python3 -m pip install -r requirements.txt` |
| `tmux: command not found` | `apt-get update -qq && apt-get install -y tmux` |
| V-JEPA checkpoint redownloads every pod | `export HF_HOME=/workspace/hf_cache` |
| W&B asks for auth / no metrics online | `wandb login` |
| GitHub asks for password | Use a GitHub personal access token as HTTPS password |
| Data count is 0 / missing dir | Wrong volume attached, or `/workspace/data` layout missing |
| SSH dies and run stops | Launch inside `tmux` |

