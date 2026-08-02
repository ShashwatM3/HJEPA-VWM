# GUIDE_AUTONOMOUS_REMOTE_RUN — unattended execution from an SSH command

> **Audience:** coding agents operating HJEPA-VWM experiments.
> **Purpose:** turn a provider SSH command plus a KANBAN `GUIDE.md` path into a reproducible,
> monitored, end-to-end remote run without making the human supervise routine shell work.

This is the autonomous execution layer. Credential setup and authorization boundaries remain in
[`GUIDE_AGENT_SSH_ACCESS.md`](GUIDE_AGENT_SSH_ACCESS.md); the exact scientific recipe remains in the
requested KANBAN `GUIDE.md`; code and tests remain authoritative for implemented behavior.

## 1. The minimal handoff an agent should accept

A human should only need to provide:

1. an exact provider-generated SSH command, or the working `hjepa-runpod` alias;
2. the repository-relative path to the experiment `GUIDE.md`;
3. an explicit outcome such as “execute this end to end while I am away.”

An explicit end-to-end launch request authorizes the requested experiment's inspections, gates,
tmux processes, logs, checkpoints, W&B runs, monitoring, and recovery. It does not authorize pod
termination/redeployment, deletion of data or unrelated checkpoints, force-pushes, weakened SSH
security, or killing unrelated processes.

Do not ask the human to translate the guide into commands that are already written in the repo.
Resolve routine details from the repository and remote state. Ask only when a genuinely scientific
choice, missing credential, destructive action, or out-of-scope infrastructure change is required.

## 2. Auto-discovery and mandatory read order

For every remote experiment, read these before the first SSH command:

1. `AGENT_FILES/AGENTS.md` and `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`;
2. `.agents/skills/run-remote-experiment/SKILL.md`;
3. `AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md` and this guide;
4. `KANBAN/PROTOCOL.md`, `KANBAN/PHASE_1/README.md`, the active investigation triad, and the
   requested run/investigation `GUIDE.md`;
5. `GUIDES/MLOPS.md` and `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`;
6. `AGENT_FILES/SETUPS/NEW_POD.md` only when the pod is fresh or dependencies are missing.

The root `AGENTS.md` and the repository skill point here so agents discover this workflow from an
SSH/run task without the human naming this file.

## 3. Treat the SSH command as connection metadata

- A command such as `ssh root@HOST -p PORT -i ~/.ssh/id_ed25519` is safe connection metadata.
- Never read or print the private key, its passphrase, `.env` contents, W&B tokens, or GitHub tokens.
- Preserve the user, host, port, and identity path exactly. Do not use `eval` on pasted text.
- Add `BatchMode=yes` and a bounded connection timeout for agent checks.
- Never add `StrictHostKeyChecking=no`. A host-key mismatch is a stop-and-verify event.
- When both RunPod proxy SSH and exposed TCP are provided, prefer exposed TCP after a successful
  host-key/authentication check; retain the proxy route as a fallback.
- A stable alias is convenient but not required. Do not spend time editing SSH config when the
  provider command already works non-interactively.

First remote action: one read-only preflight that prints only non-secret state:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 root@HOST -p PORT -i IDENTITY '
  printf "SSH_OK\n"
  hostname
  id
  command -v git
  command -v tmux
  command -v python
  nvidia-smi -L
  git -C /workspace/hierarchal-jepa-flow-world-model status --short --branch
  git -C /workspace/hierarchal-jepa-flow-world-model rev-parse HEAD
  tmux list-sessions 2>/dev/null || true
  pgrep -af "python.*train.py|whiten_stats.py" || true
'
```

Classify failures precisely: route/DNS, host key, public-key authentication, missing repository,
missing dependency, missing GPU, dirty worktree, or an existing experiment process.

## 4. Reconcile state before changing anything

Record locally and remotely:

- branch, commit SHA, tracked diff, and untracked files;
- GPU count/model/free memory;
- running training/statistics PIDs and tmux sessions;
- required data, stats, checkpoint, and log paths;
- W&B authentication status without printing credentials.

Preserve unrelated work. Never reset, clean, overwrite, or broadly kill to obtain a convenient
starting state. If an existing PID belongs to the requested gate/run, inspect it and continue it
when healthy. If it is demonstrably stuck or invalid, preserve evidence and stop only that PID.

Generated output must not make source provenance dirty. Repository-generated roots such as
`/logs/`, `/wandb/`, and checkpoint directories belong in `.gitignore`; do not hide arbitrary
untracked source files with a global `status.showUntrackedFiles=no` setting.

## 5. Pin a reproducible source state

Use the normal chain:

```text
local relevant changes -> tests -> intentional commit -> push
-> remote fetch -> remote pull --ff-only -> exact SHA comparison
```

Do not hand-copy Python files to the pod. Do not launch paid work from an unidentified source tree.
If no implementation change is required, use the already-published clean commit and record it.
Commit/push only relevant files; leave unrelated local edits untouched.

After synchronization, require:

- no tracked remote changes;
- the exact intended branch and SHA;
- a clean `git status --porcelain` after generated outputs are properly ignored;
- the guide's W&B display names and group;
- separate log/checkpoint paths for every arm.

## 6. Execute gates as observable jobs

Run the guide's tests, stats checks, provenance/resource preflights, and Stage 0 in its stated order.
Every command that may take more than a minute should be observable from another SSH connection:

- use `PYTHONUNBUFFERED=1`;
- write stdout/stderr through `tee` to a persistent, gate-specific log;
- use a deterministic tmux session for a long gate;
- record the PID, start time, command, and output path before waiting.

Full-dataset provenance can be silent while it opens every video header. No GPU utilization during
that phase is expected. Before calling it hung, take at least three bounded observations and check:

```bash
ps -o pid,ppid,stat,etime,pcpu,pmem,args -p PID
cat /proc/PID/io
lsof -p PID | grep '/workspace/data/' | tail
nvidia-smi
```

Changing open video paths, increasing I/O counters, or sustained CPU proves forward progress. Do
not kill a healthy inventory scan merely because it has not printed yet. If all progress signals
are unchanged across repeated checks, inspect the exact wait channel, file/socket, and traceback
before recovery.

## 7. Launch paid work safely

Copy the scientific flags from the requested `GUIDE.md`; do not re-derive them from memory. Before
launch, mechanically compare the final command with the guide for dataset, seed, steps, batch,
encoder revision, architecture, losses, schedule, stats file, W&B identity, checkpoint root, and
resume policy.

Each paid process must have:

- `--require-wandb`;
- a compliant explicit W&B name and the intended group;
- one unique checkpoint directory and provenance output;
- one persistent unbuffered log;
- one deterministic tmux session or one arm inside a deterministic queue session;
- training in the foreground inside tmux so session death is meaningful.

For multiple GPUs, run at most one arm per GPU using explicit `CUDA_VISIBLE_DEVICES`. For one GPU,
run sweep arms sequentially. A useful unattended one-GPU controller has this behavior:

```text
QUEUE_START
  ARM_START variant_1 -> exact GUIDE command -> ARM_DONE only on exit 0
  ARM_START variant_2 -> exact GUIDE command -> ARM_DONE only on exit 0
  ...
QUEUE_DONE
```

Use `set -Eeuo pipefail`; stop the queue on the first nonzero arm; never allow a failed command
hidden behind `tee` to start the next arm. The controller changes only scheduling, not the recipe.
Inspect existing session and checkpoint names before launch; never overwrite or accidentally resume.

## 8. Verify launch with evidence

Immediately verify all of the following, not just tmux command acceptance:

1. the expected session and PID exist;
2. only the intended GPU is occupied;
3. the log contains resolved config/provenance and the first expected step;
4. the checkpoint/provenance destinations are correct;
5. W&B reports the exact display name, group, run ID, config, and active streaming;
6. every early tripwire in the requested guide passes.

Keep early monitoring bounded and frequent. Once an arm is stable past its guide threshold, reduce
poll frequency while continuing to verify that step, log size, checkpoint time, and GPU activity
advance.

## 9. Recover autonomously without changing the experiment

- **SSH disconnect:** reconnect through the other verified route; tmux should preserve work.
- **Silent but progressing gate:** wait and keep evidence; do not restart.
- **Command/config error before training:** preserve the log, correct the operational transcription,
  rerun the gate, and keep the scientific recipe identical.
- **W&B authentication/init failure:** use an already-approved remote credential. Never print or
  shuttle a token through logs. If no approved credential exists, this is a genuine human blocker.
- **OOM/resource boundary:** preserve memory evidence and obey the guide. Do not lower batch or
  change architecture while retaining the same experiment label.
- **Transient pod/process failure:** resume only the same arm from a compatible checkpoint when the
  guide/code permits exact resume; otherwise restart that arm from an empty unique directory and
  record why.
- **Scientific tripwire failure:** stop only when the guide declares it a stop condition. Do not
  tune the recipe ad hoc.

Never use broad commands such as `pkill python`. Target the recorded PID/session for this run.

## 10. Prove completion

An arm is complete only when all applicable evidence agrees:

- the process exited successfully and its session/controller logged `ARM_DONE`;
- the final expected step exists in the log and W&B history;
- W&B state is `finished`, not merely absent from `pgrep`;
- the final checkpoint and `run_provenance.json` exist at the intended paths;
- checkpoint checksum is recorded;
- Git SHA/dirty state, dataset fingerprint, encoder revision, whitening payload fingerprint, seed,
  architecture value, and W&B identity match the intended arm.

For a sweep, repeat this proof for every arm and verify the controlled fields match across arms.
Then update the run/investigation KANBAN records with real W&B IDs and evidence; never invent a
metric or mark a run complete from process disappearance alone.

## 11. Durable handoff

Report or record:

- SSH route and hostname;
- local/remote branch and SHA;
- gate results and any guide deviation;
- session/controller name and PID state;
- each W&B name, ID, URL, and final state;
- log, provenance, checkpoint, and checksum paths;
- latest/final verified step and GPU state;
- recovery actions taken;
- exact reconnect/monitor command if anything remains active.

The default agent posture is: execute routine mechanics, preserve scientific controls, keep long
work observable, recover only within scope, and stay with the requested outcome until every arm has
durable completion evidence or a genuine human-only blocker exists.
