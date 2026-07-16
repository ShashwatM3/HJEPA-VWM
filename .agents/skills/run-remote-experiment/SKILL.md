---
name: run-remote-experiment
description: Access and operate this HJEPA-VWM project's GPU training host over SSH, including RunPod connection setup/checks, canonical fresh-pod bootstrap, exact Git commit synchronization, tmux session management, training launch/resume/stop, and log/GPU/W&B/checkpoint monitoring. Use whenever a task mentions SSH, RunPod, a new/fresh pod or remote server, NEW_POD.md, tmux, remote commands, executing a KANBAN GUIDE.md, starting or monitoring a training run, inspecting remote logs/checkpoints/GPU state, or configuring agent SSH access for this repository. Do not use for W&B-only analysis that requires no remote shell.
---

# Run remote experiments

## Load the operating contract

Read [`AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md`](../../../AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md)
before the first SSH command in a task. Treat it as the credential, authorization, synchronization,
execution, and reporting contract.

Also read the task-specific sources in this order:

1. `AGENT_FILES/AGENTS.md` and `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`.
2. `AGENT_FILES/SETUPS/NEW_POD.md` when the pod is fresh or dependencies may be missing.
3. `KANBAN/PROTOCOL.md`, the active investigation triad, and the exact run/investigation
   `GUIDE.md` when launching an experiment.
4. `GUIDES/MLOPS.md` for repository, W&B, log, checkpoint, and volume conventions.

Reconcile old guides with current code, tests, and `AGENT_FILES/AGENTS.md`; code wins. Preserve the
scientific recipe unless the user authorizes a change.

## Guide a human through SSH setup

When SSH is not configured, use Part A of `AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md` and start from the
human's stated progress. Never tell them to regenerate a key they already created.

For beginner-facing instructions:

- put exactly one shell command in each fenced block;
- name the place where it runs: Mac Terminal, RunPod web terminal, or remote SSH shell;
- distinguish shell commands, file contents, examples, and agent-chat prompt text;
- explain every interactive prompt and the exact save/exit keystrokes for an editor;
- distinguish a local private-key passphrase prompt from a remote-account password prompt, treating
  the latter as failed public-key authentication rather than asking the human to guess a password;
- show where each copied value is used in the immediately following step;
- never ask for a private key, password, API token, or `.env` value.

Do not run a remote command merely because the human is still configuring access. Wait until they
request a connection test or remote task and the alias passes the guide's non-interactive check.

Distinguish successful SSH authentication from remote dependency readiness. If `SSH_OK` succeeds but
`command -v <tool>` is empty, report the missing tool and follow the authorized fresh-pod bootstrap;
do not send the human back through key generation or alias setup.

## Bootstrap a fresh pod

Treat “bootstrap/setup this fresh pod” as authorization for only the five stages in
`AGENT_FILES/SETUPS/NEW_POD.md`:

1. Install the specified system packages.
2. Create the cache/checkpoint paths and export `HF_HOME` for the working shell.
3. Clone the absent repository and check out `phase1-v0.2-frozen-encoder`; if it already exists,
   inspect and preserve it instead of cloning or resetting over it.
4. Install the checked-out `requirements.txt` after upgrading `pip`.
5. Establish W&B authentication.

Stop bootstrap after stage 5. Leave dataset checks, imports, tests, statistics, Stage 0, `tmux`,
training, tripwires, and monitoring to the exact run `GUIDE.md`.

Never request, receive, echo, or paste a W&B API key through chat or tool logs. If an approved
`WANDB_API_KEY`, W&B settings file, or `.netrc` credential already works, verify authentication
without printing the credential. Otherwise pause and instruct the human to run this in Mac Terminal:

```bash
ssh -t hjepa-runpod 'wandb login'
```

Resume only after the human confirms login. For unattended future pods, recommend a RunPod Secret
mapped to `WANDB_API_KEY`; never create or populate that secret without explicit user authorization.

## Classify the authorized operation

- Treat SSH setup, connection testing, and read-only inspection as authorized only when requested or
  necessary for the requested remote task.
- Treat `launch`, `start`, `run`, `resume`, or equivalent explicit wording as authorization to start
  the paid process described by the task. Merely writing or reviewing a `GUIDE.md` is not launch
  authorization.
- Treat “complete control for this run” as authority to manage only that run's process, tmux session,
  logs, and checkpoints. It does not authorize unrelated process termination or pod lifecycle changes.
- Commit/push only when the user authorizes publication or when their launch instruction explicitly
  includes the repository's local-dev -> push -> remote-pull workflow. Never force-push.
- Never deploy, terminate, resize, or restart the pod; delete datasets/checkpoints; kill unrelated
  sessions; weaken SSH security; or rotate/reveal credentials without explicit authorization.

## Connect without exposing credentials

Use the concrete alias `hjepa-runpod` unless the user names another alias.

Never search the repository for private keys, read key contents, print tokens, copy a private key into
the repository, or paste credentials into commands/logs. If the alias or authentication is missing,
stop credential probing and ask the human to complete the guide or provide the public connection
details. An exact provider-generated `ssh ...` command is connection metadata; a private key,
passphrase, W&B key, GitHub token, or `.env` value is secret.

Use a non-interactive preflight before mutation:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 hjepa-runpod '
  printf "SSH_OK\n"
  hostname
  id
  command -v git
  command -v tmux
  nvidia-smi -L
  git -C /workspace/hierarchal-jepa-flow-world-model status --short --branch
'
```

If it fails, report the exact failing layer: DNS/route, host key, authentication, missing tool, GPU,
or repository. Do not fall back to passwords or disable host-key checking.

## Synchronize reproducibly

Use the repository contract: local implementation -> tests -> intentional commit/push -> remote
fetch/pull. Do not hand-copy changed training files to the pod.

Before launch:

1. Record the local branch and commit SHA.
2. Preserve any unrelated local changes.
3. Inspect the remote worktree before changing it. Stop if it is dirty in an overlapping way.
4. Fetch and fast-forward the intended remote branch without force/reset.
5. Verify the remote SHA exactly matches the intended local SHA.
6. Run the guide's tests, Stage 0, provenance, and resource preflights before spending the full run.

Do not launch from an uncommitted or unidentifiable remote state unless the user explicitly approves
a non-reproducible smoke test and the run is labeled honestly.

## Execute through tmux

Use a deterministic, investigation-specific tmux session name. Inspect an existing session before
reusing it; never overwrite or kill it blindly. Prefer a detached session with training in the
foreground inside tmux so tmux session state reflects process state.

Follow the exact task `GUIDE.md` command. Require:

- the current W&B display-name contract from `AGENT_FILES/AGENTS.md`;
- an experiment-specific checkpoint directory;
- a persistent log path;
- `--require-wandb` for paid runs;
- no accidental resume from an incompatible checkpoint.

Immediately verify the tmux session, training PID, first log output, GPU utilization, W&B identity,
and checkpoint destination. A successful `tmux new-session` alone does not prove training started.

## Monitor and recover within scope

Use bounded checks rather than leaving a foreground SSH tail as the only monitor:

```bash
ssh hjepa-runpod 'tmux list-sessions; pgrep -af "python.*train.py"; nvidia-smi'
ssh hjepa-runpod 'tail -n 100 /workspace/hierarchal-jepa-flow-world-model/logs/<run>.log'
ssh hjepa-runpod 'tmux capture-pane -p -t <session> -S -120'
```

Compare early output against the guide's explicit tripwires. If a tripwire fails, preserve evidence,
stop only the authorized run when continued GPU spending is clearly wasteful, diagnose, and relaunch
only when the task grants recovery authority and the scientific recipe remains unchanged. Otherwise
report the blocker.

## Report the durable handoff

After launch or inspection, report:

- SSH alias and remote hostname;
- local and remote commit SHA plus branch;
- tmux session name and training PID/state;
- exact W&B display name, group, and run ID/URL when available;
- log and checkpoint paths;
- GPU status and latest verified step;
- tests/preflights performed and every deviation from the guide;
- the exact reconnect/monitor command.

Never claim success until the process exists and the first expected training evidence is present.
