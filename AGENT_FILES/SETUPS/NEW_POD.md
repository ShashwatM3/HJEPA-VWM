# NEW_POD — Canonical five-step RunPod bootstrap

> **Purpose:** Prepare a newly deployed RunPod pod so the exact experiment `GUIDE.md` can take over.
>
> **Boundary:** New-pod setup is only the five steps in this file. Dataset checks, repository tests,
> whitening/statistics work, Stage 0, `tmux` launch commands, training, and monitoring belong to the
> specific run's `GUIDE.md`.

Run these commands on the pod as `root`, after the `hjepa-runpod` SSH alias works. A human may run
them in one SSH shell. An authorized coding agent may run the same commands through the alias.

Use these fixed values:

| Item | Value |
|---|---|
| SSH alias | `hjepa-runpod` |
| Repository URL | `https://github.com/ShashwatM3/HJEPA-VWM.git` |
| Repository path | `/workspace/hierarchal-jepa-flow-world-model` |
| Working branch | `phase1-v0.2-frozen-encoder` |
| Hugging Face cache | `/workspace/hf_cache` |
| Checkpoint root | `/workspace/checkpoints` |

---

## 1. Install system packages

**Pod shell — run this one command:**

```bash
apt-get update -qq
```

**Pod shell — run this one command:**

```bash
apt-get install -y git tmux curl ca-certificates ffmpeg
```

This is the step that makes `command -v tmux` return a path such as `/usr/bin/tmux`.

---

## 2. Create cache and checkpoint paths

**Pod shell — run this one command:**

```bash
mkdir -p /workspace/hf_cache /workspace/checkpoints
```

**Pod shell — run this one command:**

```bash
export HF_HOME=/workspace/hf_cache
```

The `export` applies to the current shell. Each run-specific `GUIDE.md` must export `HF_HOME` again
before model downloads, statistics jobs, Stage 0, or training.

---

## 3. Clone the repository and select the working branch

This is the canonical path when the repository does not exist on the new pod.

If `/workspace/hierarchal-jepa-flow-world-model` already exists, do not clone over it. An agent must
inspect its branch, commit, and worktree first; a human should stop and ask the agent to reconcile it.

**Pod shell — run this one command:**

```bash
cd /workspace
```

**Pod shell — run this one command:**

```bash
git clone https://github.com/ShashwatM3/HJEPA-VWM.git hierarchal-jepa-flow-world-model
```

**Pod shell — run this one command:**

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

**Pod shell — run this one command:**

```bash
git checkout phase1-v0.2-frozen-encoder
```

The run-specific guide or agent must still verify the exact intended commit before training. Never
use `git reset --hard`, `git clean`, or force-push as a bootstrap shortcut.

---

## 4. Install the repository's Python requirements

The dependency import check is not part of the canonical bootstrap. The run-specific guide owns any
imports, tests, smoke checks, and scientific preflights it requires.

**Pod shell — run this one command:**

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

**Pod shell — run this one command:**

```bash
python3 -m pip install --upgrade pip
```

**Pod shell — run this one command:**

```bash
python3 -m pip install -r requirements.txt
```

---

## 5. Authenticate W&B

### Human-interactive path

Run this inside the pod shell:

```bash
wandb login
```

When W&B waits for input, paste the API key directly into that terminal and press Return. Do not
paste the key into an agent chat, committed file, command argument, or command transcript.

If you are not already inside the pod shell, run this one command in Mac Terminal instead:

```bash
ssh -t hjepa-runpod 'wandb login'
```

Then paste the API key directly into the visible W&B prompt.

### Coding-agent path

A coding agent may invoke `wandb login`, but it must not request or paste an API key received through
chat. If no W&B credential is already configured, the agent must pause and give the human the exact
Mac Terminal command above. Resume the agent after the human confirms that login succeeded.

For fully unattended bootstrap, configure a RunPod Secret before deployment and map it to the pod's
`WANDB_API_KEY` environment variable. W&B checks `WANDB_API_KEY` before falling back to its settings,
`.netrc`, or an interactive prompt. The agent may verify this authentication without printing the
key.

Official references:

- [W&B `wandb login` authentication order](https://docs.wandb.ai/models/ref/cli/wandb-login)
- [W&B `WANDB_API_KEY` environment variable](https://docs.wandb.ai/models/track/environment-variables)
- [RunPod Secrets](https://docs.runpod.io/pods/templates/secrets)

---

## Bootstrap complete — switch to the exact run guide

After W&B authentication succeeds, stop using this file. Read and execute the exact `GUIDE.md` in
the current KANBAN investigation/run folder.

That run guide owns all subsequent repository verification, data checks, required artifacts,
statistics, tests, Stage 0, `tmux` naming, W&B naming, training launch, early tripwires, and monitoring.

Saying “bootstrap/setup the new pod” authorizes only steps 1–5 above. It does not authorize starting
a paid training process. Launch authority must name the run guide and explicitly say to start it.

---

## Minimal troubleshooting

| Symptom | Correct response |
|---|---|
| `tmux` is not found | Repeat step 1. |
| Repository destination already exists | Do not clone over it; inspect and reconcile it. |
| GitHub asks for credentials | Stop for approved GitHub authentication; never paste a token into agent chat. |
| Python reports a missing package | Repeat step 4 using the checked-out branch's `requirements.txt`. |
| W&B waits for a key | Complete step 5 directly in a human-controlled terminal. |
| Dataset, artifact, Stage 0, or training check fails | Follow the exact run `GUIDE.md`; it owns those checks. |
