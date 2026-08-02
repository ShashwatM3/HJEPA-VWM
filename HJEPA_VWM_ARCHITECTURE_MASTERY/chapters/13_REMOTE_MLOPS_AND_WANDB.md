# 13 — Remote MLOps and W&B

This chapter is the operational architecture for RunPod work. It summarizes the repository's
mandatory remote-run playbook; the authoritative command-level guides remain
[`GUIDE_AGENT_SSH_ACCESS.md`](../../AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md) and
[`GUIDE_AUTONOMOUS_REMOTE_RUN.md`](../../AGENT_FILES/GUIDE_AUTONOMOUS_REMOTE_RUN.md).

## Authorization boundary

Read-only inspection, local preparation, and non-mutating verification can proceed within scope.
Starting paid training, provisioning/deleting pods, deleting data, or stopping another run requires
clear authorization. An instruction to “monitor” does not authorize “restart with changed config.”

No remote command was run to create this corpus.

## Fresh-pod bootstrap: five stages

1. **System tools:** install Git, tmux, curl, CA certificates, and ffmpeg.
2. **Persistent directories/environment:** create data/cache/checkpoint layout and set `HF_HOME` to
   `/workspace/hf_cache`.
3. **Repository:** clone or inspect the existing checkout, then reach the exact required branch/SHA
   without discarding unknown work.
4. **Python:** upgrade pip and install the repository-pinned `requirements.txt`.
5. **Tracking:** authenticate W&B through a secure mechanism and verify the intended entity/project.

Bootstrap only establishes the machine. The experiment's KANBAN `GUIDE.md` owns tests, data/stats
preflights, exact command, tmux tag, output paths, monitoring tripwires, and acceptance.

## Reproducible code deployment

The approved chain is:

```text
local tests
→ intentional commit
→ push
→ remote fetch
→ fast-forward/exact checkout
→ verify `git rev-parse HEAD`
```

Do not hand-copy modified Python files to a pod and then call the run reproducible. Do not use hard
reset/clean against a checkout whose state has not been resolved. Record both expected and actual SHA.

## Storage preflight

Before paid work:

- confirm `/workspace` is the persistent network volume;
- verify free capacity;
- verify dataset roots and all manifests;
- resolve symlinks;
- verify HF cache writability;
- choose a unique `/workspace/ckpt/<run_tag>`;
- choose unique log/provenance paths;
- ensure no output destination aliases another arm.

Pod-local ephemeral storage is not an acceptable sole checkpoint destination.

## Data and stats preflight

For the selected recipe:

- materialize exact dataset identity;
- require full EGO source completeness when appropriate;
- validate train/validation counts and source leakage;
- if whitening is active, verify the exact encoder/dataset/12,800-clip artifact;
- if a feature cache is used, verify its fingerprint and covered offsets;
- run repository tests and recipe-specific preflight/smoke commands.

Any identity mismatch is a stop condition, not a warning to ignore.

## Tmux launch contract

Training runs in the foreground of a named detached tmux session. One GPU hosts one training process
unless a guide explicitly defines another topology. Set `CUDA_VISIBLE_DEVICES` deliberately.

For sequential variants on one GPU, run one queue script/process that waits for each child to finish
and stops on failure. Do not background multiple Python commands and hope they serialize.

Each run has:

- unique tmux session;
- exact logged command;
- unique W&B name;
- unique checkpoint directory;
- unique stdout/stderr log;
- unique provenance JSON.

Compliant W&B naming follows:

```text
Investigation NN · axis · variant
```

## W&B required versus optional

Research/paid runs use `--require-wandb`. In required mode:

- initialization failure is fatal;
- log failure is fatal;
- final summary/checksum failure is fatal;
- provenance is uploaded as a run artifact;
- a fresh whitening run uploads its whitening artifact.

Without required mode, W&B failure degrades to local logging. That is useful for local smoke tests
but not sufficient for an unattended paid run whose guide requires tracking.

## What W&B stores here

W&B receives:

- resolved config and run identity;
- train/diagnostic histories;
- run provenance artifact;
- optional whitening-stats artifact;
- final checkpoint path and SHA-256 in summary.

W&B does **not** receive the Phase 1 checkpoint file by default. Checkpoint durability is the
operator's volume responsibility.

## Launch verification

Immediately after launch, establish all of:

1. tmux session exists;
2. expected Python PID/command exists;
3. GPU utilization/memory correspond to the run;
4. local log advances;
5. W&B run appears under correct name/config;
6. provenance file exists and identities match;
7. checkpoint/log paths are correct;
8. no tripwire has fired.

A tmux session alone does not prove training is alive. A W&B run alone can exist before the first
optimizer update.

## Monitoring tripwires

Watch:

- process/tmux disappearance;
- CUDA OOM;
- exceptions or nonzero queue child status;
- no log progress;
- W&B logging failure in required mode;
- non-finite loss/gradients;
- repeated `grad_skipped`;
- late `grad_norm` spikes;
- checkpoint cadence failure;
- disk pressure;
- copy/batch-mean gates at diagnostic cadence;
- cross-video/rank collapse.

Unchanged metrics between expected cadences are normal. Unchanged step/log timestamps while the GPU
is idle are not.

## Completion proof

“The process disappeared” is not completion. Require:

- zero exit status;
- expected final step;
- W&B state finished, not crashed;
- final checkpoint exists at intended path;
- SHA-256 computed and recorded;
- provenance JSON exists;
- encoder/dataset/whitening identities match the recipe;
- queue produced every promised arm and no extras;
- late-window acceptance metrics evaluated using the correct reading cycle.

Only then can a paid run be handed off as complete.

## Resume on remote

Before resume:

- hash and inspect the checkpoint;
- verify exact code SHA/config compatibility;
- verify dataset/encoder/whitening fingerprints;
- use the original W&B run ID and `resume=must`;
- preserve checkpointed sampler state;
- use explicit reset/transfer flags only when the scientific design calls for them;
- retain the source checkpoint and record lineage.

A dataset transfer or optimizer reset is a new experimental phase, even if it starts from weights.

## SSH transport

RunPod may expose:

- direct SSH, which supports ordinary SSH/SCP/SFTP when configured;
- proxy SSH, where interactive commands may work while SCP/SFTP do not.

Use host-key verification. Never disable it broadly to make automation easier. Do not print private
keys or W&B API keys into logs/provenance.

For small marker-delimited results over a command-only proxy,
[`fetch_drift_outputs.sh`](../../fetch_drift_outputs.sh) provides a base64 transfer workaround. Large
artifacts should use a supported durable transport rather than terminal copy/paste.

## Stop semantics

Resolve exact target PID/session first. Prefer a graceful signal and verify termination. Never use
broad pattern kills that can stop other users' runs. Stopping training does not authorize deleting
its checkpoints, data, pod, or volume.

## Network-volume recovery

The repository's [`NETWORK_VOLUME_RECOVERY`](../../NETWORK_VOLUME_RECOVERY/README.md) corpus records
a prior recovery/forensic process. General lessons:

- inventory before mutation;
- distinguish live volume, recovered snapshot, and copied artifact;
- hash checkpoints and manifests;
- map old paths to current paths explicitly;
- validate recovered artifacts through current schema/fingerprint checks;
- do not assume W&B can restore checkpoints, because this project logs only path/checksum.

Recovery evidence is historical infrastructure documentation, not a promise that a particular old
volume or bucket is currently available.

## Operator handoff template

```text
code SHA:
working-tree status:
host/pod:
GPU:
dataset fingerprint:
encoder fingerprint:
whitening fingerprint:
exact command:
tmux session:
PID:
W&B URL/ID/name:
checkpoint directory:
log path:
provenance path:
latest step:
tripwire state:
final checkpoint SHA-256:
acceptance verdict:
```

This makes the run reviewable without relying on oral memory.

## Forbidden shortcuts

- secrets in commands, docs, logs, or provenance;
- `StrictHostKeyChecking=no`;
- hand-copied code presented as a Git-identified run;
- shared output directory for parallel arms;
- broad `pkill`/destructive cleanup;
- deleting raw data before verified chunk completion;
- declaring success from GPU utilization alone;
- treating a W&B link as checkpoint backup;
- changing the recipe while “resuming” without explicit provenance.
