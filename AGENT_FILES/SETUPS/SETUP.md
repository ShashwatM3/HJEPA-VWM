# SETUP.md — Operator Guide (Laptop ↔ RunPod)

> **Location:** `AGENT_FILES/SETUPS/SETUP.md` — agent doc index: [`AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`](../AGENT-BEHAVIOUR/PROTOCOL.md) §0.

> **Audience:** You — not the coding agent.
> **Purpose:** Two complete, sequential paths from where you actually are → sitting back watching training metrics stream in.
>
> | Path | When to use |
> |---|---|
> | **[Path A — First time](#path-a--first-time-end-to-end)** | Your situation **right now**: network volume has SSv2 data, **no v0 code on the pod**, **no `ssv2_tiny` yet**, **SSH not set up yet**. You have the project on your laptop and access to the RunPod website. |
> | **[Path B — After local changes](#path-b--after-local-changes-end-to-end)** | Code already deployed on the pod once; you edited files locally, **have not pushed yet** → want to run again and watch metrics. |
>
> **Network volume structure** (current vs target): [`VOLUME_LAYOUT.md`](VOLUME_LAYOUT.md). **RunPod infrastructure** (SSH, troubleshooting): [RunPod docs](https://docs.runpod.io/pods/configuration/use-ssh) and [`GUIDES/MLOPS.md`](../../GUIDES/MLOPS.md).

---

## Document provenance — what is verified (read this)

This guide mixes **three source types**. Nothing below is guesswork without a label.

| Source type | What it covers | How to trust it |
|---|---|---|
| **RunPod official docs** | SSH keys, Connect tab, `/workspace` mount, network volume attach rules, stop vs terminate | Verified against [Use SSH](https://docs.runpod.io/pods/configuration/use-ssh), [Network volumes](https://docs.runpod.io/storage/network-volumes), [Manage Pods](https://docs.runpod.io/pods/manage-pods), [Storage types](https://docs.runpod.io/pods/storage/types). Re-checked when this doc was written. |
| **This project's specs** | Folder layout (`/workspace/data/ssv2`), `make_subset.py`, `train.py` flags, Phase 1 steps, training time **estimates** | From `AGENT_FILES/AGENTS.md`, `config.py`, `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`, and `GUIDES/latest_brief.md` — not from RunPod. |
| **Your volume inspection** | Old paths like `something-something-v2`, symlink counts ~168k/25k, `ssv2_raw` layout | From operator notes at setup time — re-verify on the pod in step A7. |

**Rules used in this doc:**
- RunPod UI labels (Connect tab, Deploy, Network Volume) follow official docs; **always use the exact SSH command your pod shows** — example IPs/ports in this file are illustrations only.
- **`tmux`** and **`git push/pull`** are standard practice, **not** RunPod product features. RunPod docs do not require tmux; it prevents SSH disconnect from killing long jobs.
- **W&B** (`wandb login`) is [Weights & Biases](https://docs.wandb.ai/) — third-party, not RunPod.
- **Training runtime (~4–5 h)** is a **project estimate** (`config.py`: 15k steps on `ssv2_tiny`, A100) — not a RunPod SLA.

**Official RunPod constraints you must know:**
1. Network volumes for Pods are **Secure Cloud only** ([network volumes](https://docs.runpod.io/storage/network-volumes)).
2. A network volume must be **selected when you deploy** the pod; you **cannot attach it later** without deleting the pod ([network volumes](https://docs.runpod.io/storage/network-volumes), [storage types](https://docs.runpod.io/pods/storage/types)).
3. With a network volume, `/workspace` data **persists when you stop or terminate the pod** ([manage pods](https://docs.runpod.io/pods/manage-pods)).
4. **Stopping** releases the GPU but you may still pay for volume storage ([manage pods](https://docs.runpod.io/pods/manage-pods)).
5. Resuming a stopped pod can fail with **“Zero GPU”** if the original machine’s GPU is taken — official workaround: terminate and deploy a **new** pod with the **same network volume** ([zero GPU troubleshooting](https://docs.runpod.io/pods/troubleshooting/zero-gpus)).

If anything in RunPod’s console contradicts this file, **the live console wins**.

---

## Glossary (read once)

| Term | Meaning |
|---|---|
| **Laptop** | Your Mac, where Cursor/local git live. |
| **RunPod website** | [runpod.io](https://www.runpod.io) console — start/stop pods, billing, Connect tab. No terminal required. |
| **Pod** | A rented GPU machine. Ephemeral — can be stopped/started. |
| **Network volume** | Persistent disk, **typically** mounted at `/workspace` when attached to a Pod ([RunPod: network volumes](https://docs.runpod.io/storage/network-volumes)). Data on a network volume persists if you stop or terminate the pod ([RunPod: manage pods](https://docs.runpod.io/pods/manage-pods)). |
| **SSH** | Terminal connection from laptop → pod. You run training commands here. |
| **`ssv2`** | Full Something-Something V2 symlink dataset (~169k train videos). **Already on your volume.** |
| **`ssv2_tiny`** | Small stratified subset (~4k train) for 4–5 hr smoke runs. **Created by `make_subset.py` during first deploy** — not on your volume yet. |
| **Phase 1** | Coarse world model, Stage 0 + Stage 1 training. See [`AGENT_FILES/AGENTS.md`](../AGENTS.md) §13. |

---

## Your starting state (Path A) — confirm before you begin

Check each box — this is exactly where you are today:

- [ ] **Laptop:** Project folder exists (e.g. `HJEPA-VWM/` with `AGENT_FILES/` docs).
- [ ] **Laptop:** Python training code may **not** exist yet (`train.py`, etc.) — that's OK; Path A includes building it.
- [ ] **RunPod website:** Account with credits; **network volume** already created and used before.
- [ ] **Network volume:** Full SSv2 prepared data exists (inside old repo path or already migrated — Path A fixes layout).
- [ ] **Network volume:** **`ssv2_tiny` does NOT exist yet** — expected.
- [ ] **Network volume:** **No v0 codebase** on the pod (or only old implementation you are disregarding).
- [ ] **SSH:** Not configured yet — expected.
- [ ] **Goal:** End up watching **Weights & Biases (W&B)** charts and/or terminal logs while `train.py` runs on the GPU.

**Time budget for Path A (first time):** ~30–60 min setup (SSH, clone, deps, subset) + **~4–5 hours** Phase 1 training (15k steps on `ssv2_tiny`) — **project estimate**, not RunPod official; see `config.py` `stage1_steps`.

---

# Path A — First time (end-to-end)

Follow steps **A1 → A16 in order**. Do not skip unless the step says "skip if."

---

## A1 — Laptop: Install git (skip if already installed)

Open **Terminal** on your Mac:

```bash
git --version
```

If missing, install Xcode Command Line Tools: `xcode-select --install`

---

## A2 — Laptop: Ensure training code exists

You need `train.py`, `config.py`, `data.py`, etc. on your laptop before pushing to the pod.

**If you only have planning docs (`AGENT_FILES/**/*.md`) and no Python yet:**

1. Open Cursor on your laptop project.
2. Tell the agent: **"Read AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md, execute Phase 1."**
3. Wait until these files exist at repo root (or agreed layout):

   ```
   config.py  data.py  models.py  losses.py  diagnostics.py
   train.py   make_subset.py  requirements.txt  README.md
   ```

4. Optionally run on laptop (CPU-only smoke — may be slow or skip GPU tests):

   ```bash
   cd /path/to/HJEPA-VWM
   pip install -r requirements.txt
   python train.py --stage0-only   # only after agent finishes train.py
   ```

**If code already exists:** continue to A3.

---

## A3 — Laptop: Put project on GitHub (skip if remote already exists)

The pod gets code via `git clone` / `git pull`. You need a remote.

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM   # your path

# If not a git repo yet:
git init
git add .
git commit -m "Initial v0: planning docs and Phase 1 implementation"

# Create empty repo on GitHub (website: github.com → New repository)
# Then:
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

**Note:** The repo includes `AGENT_FILES/` for planning docs; Python training code lives at repo root (or your chosen code root). Push the **whole repo** so the pod gets both. Agent docs stay under `AGENT_FILES/` — do not move them unless you update cross-references in `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`.

Verify:

```bash
git remote -v
git log -1 --oneline
```

---

## A4 — RunPod website: Start (or deploy) a pod with your network volume

**Do this in the browser — still no SSH.**

Official procedure: [Network volumes for Pods](https://docs.runpod.io/storage/network-volumes), [Manage Pods](https://docs.runpod.io/pods/manage-pods).

1. Go to [runpod.io](https://www.runpod.io) → log in.
2. **Storage** (or **Storage → Network Volumes**) — confirm your volume exists. Note its **datacenter/region** (GPU options depend on volume location per RunPod docs).
3. **Pods** — choose one path:

   **Path 4a — Resume a stopped pod** (same pod, same volume already attached):
   - Find the **stopped** pod → **Resume** (RunPod API name: `podResume`).
   - Wait until status is **Running**.
   - If you see **“Zero GPU”** when resuming, see [Zero GPU troubleshooting](https://docs.runpod.io/pods/troubleshooting/zero-gpus): often you must **terminate** that pod and use Path 4b instead (your data stays on the network volume).

   **Path 4b — Deploy a new pod** (no running/stoppable pod, or Zero GPU issue):
   - Click **Deploy** (RunPod: “Deploy On-Demand”).
   - **Network Volume:** select your **existing** volume — required at deploy time; cannot attach later without deleting the pod.
   - **GPU:** pick an available type in that volume’s region (e.g. A100 — your choice).
   - **Template:** e.g. **RunPod PyTorch** — official templates include SSH setup ([Use SSH](https://docs.runpod.io/pods/configuration/use-ssh)).
   - **Volume mount path:** leave default **`/workspace`** unless RunPod shows otherwise ([storage types](https://docs.runpod.io/storage/types)).
   - **Port 22/tcp:** expose if the deploy UI/template allows it, for full SSH over public IP ([Use SSH](https://docs.runpod.io/pods/configuration/use-ssh)).
   - Deploy and wait until **Running**.

4. Click the pod → **Connect** tab. **Leave this tab open** — you will copy the **exact** SSH command in A6 (do not use example IPs from this doc).

You have **not** opened a terminal to the pod yet. That is correct.

**Stop vs terminate (official):** **Stop** = GPU released, `/workspace` on network volume kept, you may still pay storage fees. **Terminate** = pod deleted, but **network volume data is still kept** and can be attached to a new pod ([manage pods](https://docs.runpod.io/pods/manage-pods)).

---

## A5 — Laptop: Generate SSH key and add to RunPod (one-time)

Official steps: [Connect to a Pod with SSH — Generate an SSH key and add it to your Runpod account](https://docs.runpod.io/pods/configuration/use-ssh#generate-an-ssh-key-and-add-it-to-your-runpod-account).

Still on your **Mac Terminal**:

```bash
# Official RunPod example (ed25519):
ssh-keygen -t ed25519 -C "YOUR_EMAIL@DOMAIN.COM"

# Show public key — copy the ENTIRE line (must start with ssh-ed25519)
cat ~/.ssh/id_ed25519.pub
```

On **RunPod website** (official: “SSH Public Keys field in your Runpod account settings”):

1. Account **Settings** → **SSH Public Keys**
2. Paste the **full public key** (not the SHA256 fingerprint). One key per line. Save.

If SSH asks for a password, RunPod docs list common causes ([troubleshooting](https://docs.runpod.io/pods/configuration/use-ssh#troubleshooting-ssh-key-authentication)) — often wrong key file or permissions:

```bash
chmod 600 ~/.ssh/id_ed25519
chmod 700 ~/.ssh
```

---

## A6 — Laptop: First SSH connection to the pod

Official steps: [Basic SSH](https://docs.runpod.io/pods/configuration/use-ssh#basic-ssh-with-key-authentication) or [Full SSH via public IP](https://docs.runpod.io/pods/configuration/use-ssh#full-ssh-via-public-ip-with-key-authentication).

On the pod's **Connect** tab, copy the command RunPod shows under **SSH** (basic) or **SSH over exposed TCP** (full). RunPod’s examples look like:

```bash
# Basic (proxied) — no SCP/SFTP per RunPod docs
ssh 8y5rumuyb50m78-6441103b@ssh.runpod.io -i ~/.ssh/id_ed25519

# Full (public IP) — supports SCP/SFTP per RunPod docs
ssh root@213.173.108.12 -p 17445 -i ~/.ssh/id_ed25519
```

**Use your copied command only** — do not use the IP/port above.

Paste into **Mac Terminal**. First connect may ask to trust host — type `yes`.

RunPod states: authentication is by SSH key; **no password** if configured correctly ([troubleshooting](https://docs.runpod.io/pods/configuration/use-ssh#troubleshooting-ssh-key-authentication)).

**Success:** shell prompt on the pod (often `root@...`).

Optional — `~/.ssh/config` shortcut (your values from Connect tab):

```
Host runpod-jepa
    HostName YOUR_POD_IP
    Port YOUR_SSH_PORT
    User root
    IdentityFile ~/.ssh/id_ed25519
```

---

## A7 — Pod: Confirm network volume contents

**Run on the pod** (you are in SSH now):

```bash
df -h /workspace
ls -lah /workspace
```

You should see items like `hierarchal-jepa-flow-world-model/`, `ssv2_raw/` — **your** layout from the chat inspection; names may differ slightly. Re-verify; do not assume.

Inspect SSv2 data location:

```bash
ls /workspace/hierarchal-jepa-flow-world-model/data/something-something-v2/train 2>/dev/null | head
# OR if already migrated:
ls /workspace/data/ssv2/train 2>/dev/null | head
```

If neither shows symlinks, stop and inspect `/workspace` before continuing.

---

## A8 — Pod: One-time data folder layout (project convention — not RunPod)

**Source:** This project's `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md` / [`GUIDES/MLOPS.md`](../../GUIDES/MLOPS.md) — **not** RunPod documentation. RunPod only guarantees a mount at `/workspace`; it does not define `data/ssv2` vs `something-something-v2`.

**What this does:** Moves prepared SSv2 from inside the old repo folder to `/workspace/data/ssv2/`. Takes seconds. **Does not copy video files** — only renames/moves a directory of symlinks on your volume.

```bash
mkdir -p /workspace/data /workspace/checkpoints /workspace/hf_cache

if [ -d /workspace/hierarchal-jepa-flow-world-model/data/something-something-v2 ]; then
  mv /workspace/hierarchal-jepa-flow-world-model/data/something-something-v2 /workspace/data/ssv2
  echo "OK: moved to /workspace/data/ssv2"
elif [ -d /workspace/data/ssv2/train ]; then
  echo "OK: already at /workspace/data/ssv2"
else
  echo "ERROR: SSv2 not found — inspect /workspace manually"
fi

# Archive old checkpoint so new training never loads it by mistake
mkdir -p /workspace/hierarchal-jepa-flow-world-model/archive 2>/dev/null || true
mv /workspace/hierarchal-jepa-flow-world-model/checkpoints/stage1_final.pt \
   /workspace/hierarchal-jepa-flow-world-model/archive/ 2>/dev/null || true

# Verify counts
echo "train symlinks:" $(find /workspace/data/ssv2/train -maxdepth 1 -type l | wc -l)
echo "val symlinks:"   $(find /workspace/data/ssv2/validation -maxdepth 1 -type l | wc -l)
test -f /workspace/data/ssv2/labels.json && echo "labels.json OK"
```

Expected (from **your** chat inspection): ~**168,913** train, ~**24,777** val — re-count on your pod; numbers must match your volume, not this doc.

`ssv2_tiny` **still does not exist** — next steps create it.

---

## A9 — Pod: Clone your v0 codebase (no code on server yet)

Replace `YOUR_GITHUB_REPO_URL` with your actual HTTPS or SSH URL from A3.

```bash
cd /workspace

# Keep old repo as archive (optional but recommended)
if [ -d hierarchal-jepa-flow-world-model ] && [ ! -f hierarchal-jepa-flow-world-model/train.py ]; then
  mv hierarchal-jepa-flow-world-model hierarchal-jepa-flow-world-model-old-$(date +%Y%m%d) 2>/dev/null || true
fi

# Clone fresh (if directory name taken, remove or rename first)
git clone https://github.com/YOUR_USER/YOUR_REPO.git hierarchal-jepa-flow-world-model
cd /workspace/hierarchal-jepa-flow-world-model

git log -1 --oneline
ls -la train.py config.py make_subset.py 2>/dev/null || echo "WARNING: Phase 1 files missing — go back to A2"
```

If `train.py` is missing, stop — complete A2 on laptop, push, re-clone or `git pull` here.

---

## A10 — Pod: Python dependencies + GPU check

```bash
cd /workspace/hierarchal-jepa-flow-world-model
pip install -r requirements.txt

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected: `CUDA: True` and your A100 name.

Install tmux for long runs (optional; **standard Linux**, not RunPod-specific — keeps training alive if SSH disconnects):

```bash
apt-get update && apt-get install -y tmux
```

---

## A11 — Pod: Weights & Biases login (third-party — not RunPod)

Official W&B docs: [wandb login](https://docs.wandb.ai/ref/cli/wandb-login).

```bash
wandb login
```

Paste your API key from [wandb.ai/authorize](https://wandb.ai/authorize).

Optional — persist on volume:

```bash
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
echo 'export WANDB_DIR=/workspace/hierarchal-jepa-flow-world-model/wandb' >> ~/.bashrc
source ~/.bashrc
```

---

## A12 — Pod: Create `ssv2_tiny` (project script — not RunPod)

**Source:** `make_subset.py` and [`AGENT_FILES/AGENTS.md`](../AGENTS.md) §5.

This is the small dataset for ~4–5 hour smoke training. **Your volume does not have it until this runs.**

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python make_subset.py --train-per-class 23 --val-per-class 2 --seed 42
```

Verify:

```bash
ls /workspace/data/ssv2_tiny/manifest.json
echo "tiny train:" $(find /workspace/data/ssv2_tiny/train -maxdepth 1 -type l | wc -l)
echo "tiny val:"   $(find /workspace/data/ssv2_tiny/validation -maxdepth 1 -type l | wc -l)
```

Expected: ~**4,000** train, ~**350** val symlinks, plus `manifest.json`.

---

## A13 — Pod: Stage 0 sanity check (~2 minutes)

Confirms model builds, forward/backward works, no NaN — **before** touching real video data.

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python train.py --stage0-only
```

Expected: exits **0**, prints success, no NaN errors.

If this fails, fix code on laptop (A2), push (A3), `git pull` on pod, retry.

---

## A14 — Pod: Short smoke train (500 steps, ~few minutes)

Confirms dataloader + training loop + W&B logging:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python train.py --data ssv2_tiny --steps 500
```

Watch terminal: loss values print every ~50 steps; no NaN.

---

## A15 — Pod: Full Phase 1 training in tmux (main run)

**Why tmux:** Standard way to keep a process running after SSH disconnect — **not** a RunPod feature. Alternative: RunPod process in foreground while SSH stays open, or `nohup` (see FAQ).

```bash
cd /workspace/hierarchal-jepa-flow-world-model
tmux new -s phase1
```

Inside tmux:

```bash
python train.py --data ssv2_tiny --steps 15000
```

**Detach** (leave training running): press `Ctrl+B`, then `D`.

**Reattach later:**

```bash
ssh runpod-jepa   # or your Connect tab command
tmux attach -t phase1
```

Expected runtime: **~4–5 hours** on A100 80GB with `ssv2_tiny` at 15k steps — **project estimate** from `config.py`, not RunPod.

To resume after interruption:

```bash
python train.py --data ssv2_tiny --steps 15000 --resume /workspace/checkpoints/phase1_stepXXXX.pt
```

---

## A16 — Watch metrics (sit back)

You now have **two places** to watch training:

### Option 1 — Weights & Biases (recommended)

1. Open [wandb.ai](https://wandb.ai) in your browser on laptop.
2. Open your project (name set in `train.py` / config).
3. Watch live charts (v0.2): `L_flow`, `L_var`, the three `c_t` monitors (variance, cross-video cosine, effective rank), learning rate, grad norm, diagnostics.

The terminal prints a W&B run URL when training starts — click it.

### Option 2 — Terminal log

```bash
ssh runpod-jepa
tmux attach -t phase1
```

Scroll through step logs. Detach again with `Ctrl+B`, `D`.

### Option 3 — Pod log file (if you used nohup instead of tmux)

```bash
tail -f /workspace/hierarchal-jepa-flow-world-model/train.log
```

### What "done" looks like for Phase 1

- 15,000 steps complete, no OOM/NaN
- Checkpoint at `/workspace/checkpoints/phase1_step15000.pt`
- W&B diagnostics: see [`GUIDES/READING_EXPERIMENTS.md`](../../GUIDES/READING_EXPERIMENTS.md) for acceptance gates (`coarse_vs_copy_ratio`, etc.)

**Path A complete.** Future code changes → use [Path B](#path-b--after-local-changes-end-to-end).

---

# Path B — After local changes (end-to-end)

**Starting state:**

- [ ] Pod was set up before (Path A done at least once).
- [ ] `/workspace/data/ssv2` and `/workspace/data/ssv2_tiny` exist on volume.
- [ ] Repo exists on pod at `/workspace/hierarchal-jepa-flow-world-model`.
- [ ] You edited code on **laptop** and **have NOT pushed yet**.

Follow **B1 → B9 in order**.

---

## B1 — Laptop: Finish editing, review changes

```bash
cd /path/to/HJEPA-VWM
git status
git diff
```

Confirm only intended files changed.

---

## B2 — Laptop: Commit and push

```bash
git add -A
git commit -m "Describe your change clearly"
git push origin main
```

Verify push succeeded:

```bash
git log -1 --oneline
# GitHub website: confirm commit visible on remote
```

**You still have not touched RunPod yet.** Code is on GitHub only.

---

## B3 — RunPod website: Ensure pod is running

1. [runpod.io](https://www.runpod.io) → **Pods**
2. If pod is **Stopped** → **Start/Resume** → wait for **Running**
3. If pod was terminated → **Deploy** new pod with **same network volume** (see A4)

Billing note: you pay while the pod is running, even if not training.

---

## B4 — Laptop: SSH into pod

```bash
ssh runpod-jepa
# OR paste full command from pod Connect tab (see A6)
```

---

## B5 — Pod: Pull latest code

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git pull origin main
git log -1 --oneline   # should match laptop commit from B2
```

If pull conflicts, resolve on laptop and push again — do not hand-edit conflict files on pod unless emergency.

---

## B6 — Pod: Reinstall deps (only if requirements.txt changed)

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git diff HEAD~1 --name-only | grep -q requirements.txt && pip install -r requirements.txt || echo "requirements unchanged, skip pip"
```

Or always run `pip install -r requirements.txt` — safe, slightly slower.

---

## B7 — Pod: Quick sanity (recommended after non-trivial changes)

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python train.py --stage0-only
```

Skip only for doc-only changes.

If you changed data pipeline, optionally:

```bash
python -c "from data import smoke_test_dataloader; smoke_test_dataloader()"
```

---

## B8 — Pod: Start training in tmux

**Fresh Phase 1 run:**

```bash
tmux new -s train
cd /workspace/hierarchal-jepa-flow-world-model
python train.py --data ssv2_tiny --steps 15000
```

**Resume from checkpoint:**

```bash
tmux new -s train
cd /workspace/hierarchal-jepa-flow-world-model
python train.py --data ssv2_tiny --steps 15000 --resume /workspace/checkpoints/phase1_step15000.pt
```

**Later stages** (when implemented): adjust `--steps` and `--resume` per human direction and `GUIDES/latest_brief.md`.

Detach: `Ctrl+B`, then `D`.

---

## B9 — Watch metrics

Same as **A16**:

- **W&B:** [wandb.ai](https://wandb.ai) → your project → live run
- **Terminal:** `tmux attach -t train`

**Path B complete.**

---

# Quick reference cards

## First time checklist (print mentally)

```
[ ] A2  Code on laptop (agent Phase 1)
[ ] A3  Pushed to GitHub
[ ] A4  Pod running on RunPod website
[ ] A5  SSH key in RunPod settings
[ ] A6  SSH connected
[ ] A8  Data at /workspace/data/ssv2
[ ] A9  git clone on pod
[ ] A10 pip install + CUDA OK
[ ] A11 wandb login
[ ] A12 make_subset.py → ssv2_tiny
[ ] A13 stage0-only
[ ] A14 500-step smoke
[ ] A15 tmux → 15k steps
[ ] A16 watch wandb
```

## Local change checklist

```
[ ] B2  git push
[ ] B3  pod running (website)
[ ] B4  ssh in
[ ] B5  git pull
[ ] B7  stage0-only (if code changed)
[ ] B8  tmux train
[ ] B9  watch wandb
```

---

# Related docs

| Doc | Role |
|---|---|
| [`VOLUME_LAYOUT.md`](VOLUME_LAYOUT.md) | Current vs target volume tree, path contract |
| [`GUIDES/MLOPS.md`](../../GUIDES/MLOPS.md) | RunPod SSH, W&B, checkpoints |
| [`../AGENTS.md`](../AGENTS.md) | Architecture, defaults, acceptance context |
| [`../AGENT-BEHAVIOUR/PROTOCOL.md`](../AGENT-BEHAVIOUR/PROTOCOL.md) | Agent operating rules |
| [`GUIDES/latest_brief.md`](../../GUIDES/latest_brief.md) | Architecture narrative and empirical notes (not ground truth) |

---

# FAQ

**Do I use Jupyter or SSH?**  
You prefer SSH. RunPod also offers web terminal / Jupyter via the Connect tab; this guide follows SSH per your choice. SSH steps are from [RunPod Use SSH](https://docs.runpod.io/pods/configuration/use-ssh).

**Can I run training from the RunPod web terminal?**  
RunPod provides a browser terminal on the Connect tab. Closing the tab typically ends that session — for multi-hour jobs, SSH + tmux (or similar) is standard practice outside RunPod’s docs.

**Pod stopped overnight — is data lost?**  
**On a network volume:** No — RunPod states `/workspace` data is preserved when you stop or terminate the pod ([manage pods](https://docs.runpod.io/pods/manage-pods)). `ssv2`, `ssv2_tiny`, and `checkpoints/` should remain. Resume pod → SSH → continue training with `--resume`.

**Resume failed with “Zero GPU”?**  
Official RunPod issue when the original machine’s GPU is busy ([zero GPU](https://docs.runpod.io/pods/troubleshooting/zero-gpus)). Fix: terminate pod, deploy **new** pod with the **same network volume** (step A4 Path 4b).

**Do I re-run make_subset every time?**  
No. Once `ssv2_tiny` exists on the volume, skip A12.

**Do I re-run data migration (A8) every time?**  
No. Once `/workspace/data/ssv2` exists, skip A8.

**Laptop has no GPU — is that OK?**  
Yes. Implement on laptop, train on RunPod only.
