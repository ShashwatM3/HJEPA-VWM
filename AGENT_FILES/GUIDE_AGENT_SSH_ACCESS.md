# GUIDE_AGENT_SSH_ACCESS — Give coding agents SSH access safely

> **Purpose:** Set up SSH once, then let Codex, Claude Code, Cursor, or another coding agent run
> HJEPA-VWM commands on a GPU server, use `tmux`, launch training, and monitor it.
>
> **Scope:** The standard target is a RunPod GPU pod. The same key-and-alias pattern works for a
> normal SSH server. This file contains no real host, private key, password, or API token.

## Read this before pasting anything

The label immediately above every block tells you what the block is.

- **Mac Terminal — run this one command:** paste that block, press Return, and wait for it to finish.
- **RunPod web terminal — run this one command:** paste it into RunPod's browser terminal.
- **Remote SSH shell — run this one command:** paste it only after an `ssh` command has connected.
- **File contents — do not run:** open the named file first, then paste the block inside that file.
- **Example only — do not paste:** use it only to understand the shape of your real value.
- **Prompt text — paste into an agent chat:** this is an instruction for an AI, not shell code.

In the human setup below, each shell block contains exactly one command. Never paste the next block
until the current command has finished or returned you to a prompt.

### Your resume point if you followed the older version

You said you completed 2.1 and added the key in RunPod Settings, but did not run the old 2.2 command
block. Do **not** repeat 2.1 and do **not** generate another key.

Use this decision:

1. If the current pod was **deployed after** you saved the key in RunPod Settings, skip to 2.3.
2. If the current pod was already running when you saved the key, complete 2.2B, then go to 2.3.
3. If you are unsure, try 2.3. If it says `Permission denied (publickey)`, return to 2.2B.

The short mental model is:

```text
agent runs on your laptop
  -> laptop SSH reads ~/.ssh/hjepa_codex
  -> the alias hjepa-runpod reaches the pod
  -> the agent runs commands in /workspace
  -> tmux keeps training alive after SSH disconnects
```

There is no special SSH plugin. Once the non-interactive test in 2.6 works from your Mac, an agent
with terminal and network permission can use the same alias.

---

## 0. Standard names for this repository

Use these names throughout the guide:

| Item | Standard value |
|---|---|
| Mac SSH alias | `hjepa-runpod` |
| Mac private key | `~/.ssh/hjepa_codex` |
| Mac public key | `~/.ssh/hjepa_codex.pub` |
| Remote repository | `/workspace/hierarchal-jepa-flow-world-model` |
| Persistent data root | `/workspace/data` |
| Persistent Hugging Face cache | `/workspace/hf_cache` |
| Persistent checkpoints | `/workspace/checkpoints` or the run guide's explicit directory |
| Current development branch | Verify locally and in the run guide; never guess |

`hjepa-runpod` stays the same even when RunPod changes the real IP or port. When that happens, edit
one alias entry instead of rewriting every experiment guide.

---

## 1. Know what each permission means

These permissions are separate:

1. **Technical SSH access:** the agent's host can authenticate to the server.
2. **Remote mutation:** the user asked the agent to install, edit, pull, or run something remotely.
3. **Paid execution:** the user explicitly asked to start or resume GPU training.
4. **Git publication:** the user authorized any required commit and push.
5. **Destructive infrastructure:** pod lifecycle changes, data deletion, or unrelated process kills.

Owning the key does not grant every permission. “Write a `GUIDE.md`” does not authorize a paid run.

“Implement this guide and start the training run; control this run” authorizes that run. It does not
authorize deleting other data, killing other sessions, or terminating the pod.

---

## Part A — Human setup on your Mac and RunPod

Do section 2 on the Mac that runs your coding agent. Only steps labeled **RunPod web terminal** go
inside RunPod.

## 2. One-time SSH setup

### 2.1 Create the dedicated key on your Mac

If you already completed 2.1, skip this subsection. Do not regenerate the key.

Open the macOS **Terminal** app.

**Mac Terminal — run this one command:**

```bash
ls -l ~/.ssh/hjepa_codex ~/.ssh/hjepa_codex.pub
```

If both files are listed, the key already exists. Skip to the `ssh-add` command near the end of this
subsection.

If Terminal says both files do not exist, continue below.

**Mac Terminal — run this one command:**

```bash
mkdir -p ~/.ssh
```

**Mac Terminal — run this one command:**

```bash
chmod 700 ~/.ssh
```

**Mac Terminal — run this one command:**

```bash
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/hjepa_codex -C "hjepa-codex"
```

The command pauses at `Enter passphrase`. Choose one option:

- Recommended: type a passphrase, press Return, then type it again.
- Simplest unattended setup: press Return twice for no passphrase. This is less secure.

Nothing appears while you type a passphrase. That is normal.

If it asks whether to overwrite an existing key, type `n` and press Return. Never overwrite a key
whose public half may already be installed on a server.

After `ssh-keygen` finishes, run the next commands one at a time.

**Mac Terminal — run this one command:**

```bash
chmod 600 ~/.ssh/hjepa_codex
```

**Mac Terminal — run this one command:**

```bash
chmod 644 ~/.ssh/hjepa_codex.pub
```

If you chose a passphrase, load the key into macOS Keychain.

**Mac Terminal — run this one command:**

```bash
ssh-add --apple-use-keychain ~/.ssh/hjepa_codex
```

Enter the key passphrase if asked. Again, typed characters will not appear.

Now copy the **public** key to your Mac clipboard.

**Mac Terminal — run this one command:**

```bash
pbcopy < ~/.ssh/hjepa_codex.pub
```

No output is expected. Your clipboard now contains one line beginning with `ssh-ed25519`.

Never copy, print, upload, or commit `~/.ssh/hjepa_codex`. That file is the private key.

### 2.2 Put the public key on RunPod

There are two parts. Part A makes the key available to future pods. Part B is needed only when the
current pod was already running before you saved the key.

RunPod documents that account keys saved before deployment are injected at startup.

An already-running pod needs a manual `authorized_keys` update or redeployment. See the official
RunPod references in section 10.

#### 2.2A Save the key in RunPod account settings

If your clipboard no longer contains the public key, copy it again.

**Mac Terminal — run this one command:**

```bash
pbcopy < ~/.ssh/hjepa_codex.pub
```

Now use your browser:

1. Sign in to RunPod.
2. Open your account **Settings**.
3. Find **SSH Public Keys**.
4. Click the control for adding a new public key.
5. Click inside the public-key field.
6. Press **Command-V** once.
7. Confirm the pasted text starts with `ssh-ed25519`.
8. Save the key.

Do not paste a `SHA256:...` fingerprint. Do not paste the private-key file.

If the pod was deployed after this save, RunPod should inject the key automatically. Skip 2.2B and
continue to 2.3.

#### 2.2B Add the key to a pod that was already running

Do this only if the current pod was running before you saved the account key, or if 2.3 later fails
with `Permission denied (publickey)`.

First, put the public key on your Mac clipboard again.

**Mac Terminal — run this one command:**

```bash
pbcopy < ~/.ssh/hjepa_codex.pub
```

In your browser, open the pod's **Connect** panel and launch its **Web Terminal**. The next commands
go into that browser terminal, not your Mac Terminal.

Keep this same web-terminal tab open until the `unset HJEPA_PUBLIC_KEY` command is complete. The
temporary variable exists only in that shell session.

**RunPod web terminal — run this one command:**

```bash
install -d -m 700 ~/.ssh
```

Wait for the RunPod prompt to return.

**RunPod web terminal — run this one command:**

```bash
touch ~/.ssh/authorized_keys
```

**RunPod web terminal — run this one command:**

```bash
chmod 600 ~/.ssh/authorized_keys
```

The next command tells the shell to wait for one line of text.

**RunPod web terminal — run this one command:**

```bash
read -r HJEPA_PUBLIC_KEY
```

The cursor may sit on a blank line with no new prompt. That is expected.

Press **Command-V** once. If the normal RunPod prompt does not return automatically, press Return
once. The copied public-key line may already include its own final newline, so do not paste twice.

Now append that key only if it is not already present.

**RunPod web terminal — run this one command:**

```bash
grep -qxF "$HJEPA_PUBLIC_KEY" ~/.ssh/authorized_keys || printf '%s\n' "$HJEPA_PUBLIC_KEY" >> ~/.ssh/authorized_keys
```

Verify that the key is present.

**RunPod web terminal — run this one command:**

```bash
grep -nF "$HJEPA_PUBLIC_KEY" ~/.ssh/authorized_keys
```

Expected result: one numbered line beginning with `ssh-ed25519`.

Clear the temporary shell variable.

**RunPod web terminal — run this one command:**

```bash
unset HJEPA_PUBLIC_KEY
```

You may now close the RunPod web terminal.

For a permanent non-RunPod server, ask its administrator to install the public key for a dedicated,
least-privilege account.

### 2.3 Copy **and use** RunPod's SSH command once

This step tests the real provider command before you create an alias from it.

1. In RunPod, open the current pod.
2. Open **Connect**.
3. Prefer **SSH over exposed TCP** when it is available.
4. Click the copy button beside RunPod's complete `ssh ...` command.

**Example only — do not paste:**

```text
ssh root@203.0.113.10 -p 17445 -i ~/.ssh/id_ed25519
```

Your real user, host, and port will be different.

Open macOS **TextEdit**. Choose **Format → Make Plain Text**, then press **Command-V** to paste the
copied command.

In TextEdit, change only the identity-file path:

- Replace `~/.ssh/id_ed25519` with `~/.ssh/hjepa_codex`.
- If the command has no `-i` option, add ` -i ~/.ssh/hjepa_codex` at the end.

**Example only — do not paste:**

```text
ssh root@203.0.113.10 -p 17445 -i ~/.ssh/hjepa_codex
```

Write down these three values from your real command:

| Name used in 2.4 | Where to find it | Example only |
|---|---|---|
| `REMOTE_USER` | Text between `ssh ` and `@` | `root` |
| `REMOTE_HOST` | Text after `@` and before the next space | `203.0.113.10` |
| `REMOTE_PORT` | Number after `-p`; use `22` when there is no `-p` | `17445` |

Select the full edited command in TextEdit and press **Command-C**.

Return to your Mac Terminal. Press **Command-V** once, visually confirm the command contains
`-i ~/.ssh/hjepa_codex`, then press Return.

On the first connection, SSH may ask `Are you sure you want to continue connecting?` Check that the
host and port match RunPod's Connect panel. Then type `yes` and press Return.

SSH can show two very different prompts:

- `Enter passphrase for key '/Users/.../.ssh/hjepa_codex':` asks for the optional local key
  passphrase. Because you left it empty, you should not see this prompt.
- `root@... password:` or `<user>@... password:` asks for a remote account password. This means
  public-key authentication failed. Do not enter a password or press Return repeatedly.

If the second prompt appears, press **Control-C**. Return to 2.2B and install the public key in the
already-running pod. Do not continue to the remote-shell commands below.

When the remote shell opens, run these commands separately.

**Remote SSH shell — run this one command:**

```bash
hostname
```

**Remote SSH shell — run this one command:**

```bash
nvidia-smi -L
```

You should see a hostname and at least one GPU. Disconnect cleanly.

**Remote SSH shell — run this one command:**

```bash
exit
```

If the connection says `Permission denied (publickey)`, return to 2.2B. Do not continue to 2.4 until
the edited provider command can open the remote shell.

RunPod's proxied SSH can still run shell commands and `tmux`, but it may not support SCP/SFTP. The
alias in 2.4 works for either connection style.

### 2.4 Create the stable `hjepa-runpod` alias

This is where the user, host, and port you wrote down in 2.3 are used.

You will edit `~/.ssh/config`. The template below is **file content**, not a sequence of Terminal
commands.

First, make sure the config file exists.

**Mac Terminal — run this one command:**

```bash
touch ~/.ssh/config
```

Protect it with the required permissions.

**Mac Terminal — run this one command:**

```bash
chmod 600 ~/.ssh/config
```

Make a backup before editing.

**Mac Terminal — run this one command:**

```bash
cp -p ~/.ssh/config ~/.ssh/config.backup-before-hjepa
```

Check whether an older `hjepa-runpod` entry already exists.

**Mac Terminal — run this one command:**

```bash
grep -n '^Host hjepa-runpod$' ~/.ssh/config
```

No output means there is no existing entry. If it prints a line number, edit that existing block in
the next step instead of adding a duplicate.

Open the file in the beginner-friendly terminal editor `nano`.

**Mac Terminal — run this one command:**

```bash
nano ~/.ssh/config
```

You are now **inside nano**, not at a shell prompt.

If `grep` printed an existing line number, press **Control-_** in nano. On many Mac keyboards this is
**Control-Shift-hyphen**. Type the line number and press Return. Edit the existing block.

If `grep` printed nothing, use **Control-V** repeatedly to move to the bottom. Put the cursor on a
new blank line.

Copy the following template, then press **Command-V** to paste it inside nano. Do **not** paste this
template at the normal Terminal prompt.

**File contents for `~/.ssh/config` — do not run:**

```sshconfig
Host hjepa-runpod
    HostName REPLACE_WITH_REMOTE_HOST
    User REPLACE_WITH_REMOTE_USER
    Port REPLACE_WITH_REMOTE_PORT
    IdentityFile ~/.ssh/hjepa_codex
    IdentitiesOnly yes
    AddKeysToAgent yes
    UseKeychain yes
    ForwardAgent no
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

Inside nano, replace exactly three placeholders with the real values from 2.3:

1. Replace `REPLACE_WITH_REMOTE_HOST` with `REMOTE_HOST`.
2. Replace `REPLACE_WITH_REMOTE_USER` with `REMOTE_USER`.
3. Replace `REPLACE_WITH_REMOTE_PORT` with `REMOTE_PORT`.

Do not type the words `REMOTE_HOST`, `REMOTE_USER`, or `REMOTE_PORT`. Type their real values.

Save and close nano:

1. Press **Control-O**. This is the letter O, not zero.
2. Nano shows the filename at the bottom. Press Return once.
3. Press **Control-X** to exit.

You should now be back at the normal Mac Terminal prompt.

Check that SSH can parse the file.

**Mac Terminal — run this one command:**

```bash
ssh -G hjepa-runpod >/dev/null
```

No output means the file parsed successfully.

Print only the four resolved connection fields.

**Mac Terminal — run this one command:**

```bash
ssh -G hjepa-runpod | grep -E '^(hostname|user|port|identityfile) '
```

Confirm that the output matches the real host, user, port, and `~/.ssh/hjepa_codex`. If it does not,
run `nano ~/.ssh/config` again and correct the entry.

### 2.5 Test the alias manually

The long provider command worked in 2.3. Now prove that the short alias reaches the same server.

**Mac Terminal — run this one command:**

```bash
ssh hjepa-runpod
```

If SSH asks about host trust, compare the destination with RunPod's current Connect panel. Type `yes`
only when it matches.

After the remote prompt appears, check the host again.

**Remote SSH shell — run this one command:**

```bash
hostname
```

Check the GPU.

**Remote SSH shell — run this one command:**

```bash
nvidia-smi -L
```

Disconnect.

**Remote SSH shell — run this one command:**

```bash
exit
```

Do not add `StrictHostKeyChecking no` to the alias.

If a redeployed pod changes its host key, verify the new host and port in RunPod before removing any
stale host-key entry.

### 2.6 Prove an agent can connect without questions

This is the decisive test. It must print `SSH_OK` without asking for a password, key passphrase, or
host confirmation.

**Mac Terminal — run this one command:**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 hjepa-runpod 'printf "SSH_OK\n"'
```

If it prints `SSH_OK`, run the remaining checks separately.

**Mac Terminal — run this one command:**

```bash
ssh -o BatchMode=yes hjepa-runpod 'hostname'
```

**Mac Terminal — run this one command:**

```bash
ssh -o BatchMode=yes hjepa-runpod 'id'
```

**Mac Terminal — run this one command:**

```bash
ssh -o BatchMode=yes hjepa-runpod 'command -v tmux'
```

Expected output is a path such as `/usr/bin/tmux`.

If this command prints nothing, SSH still worked: `tmux` is simply not installed on the pod. Do not
repeat the key or alias setup. Continue to Part B section 4 to bootstrap the fresh pod, then repeat
this one `command -v tmux` check.

**Mac Terminal — run this one command:**

```bash
ssh -o BatchMode=yes hjepa-runpod 'nvidia-smi -L'
```

If a passphrased key fails, reload it into macOS Keychain.

**Mac Terminal — run this one command:**

```bash
ssh-add --apple-use-keychain ~/.ssh/hjepa_codex
```

Then repeat only the first `SSH_OK` test.

When `SSH_OK`, `hostname`, `id`, and `nvidia-smi -L` work, SSH access is complete. A missing command
such as `tmux` means pod bootstrapping remains; it does not invalidate SSH access.

You never need to send an agent your private key.

---

## Part B — Agent access and operating contract

Part A was the human copy/paste walkthrough. Part B tells coding agents how to use the working
alias safely.

Humans do not need to paste every Part B block. Multi-line blocks in Part B are technical examples
for agents that can inspect a complete shell command before running it.

## 3. Give the coding agent terminal permission

The application running the agent must be allowed to:

- execute the local `ssh` binary;
- make outbound network connections to the SSH host and port;
- read the selected identity through OpenSSH/Keychain;
- run the requested remote commands.

Keep the private key in `~/.ssh`, never inside this repository. If an agent sandbox blocks SSH, grant
network/key access through that agent's normal permission UI or configuration. Prefer a scoped
network-enabled workspace profile. Full un-sandboxed access is appropriate only for a trusted
repository and dedicated/revocable credentials.

For Codex, network access and approval policy are separate controls. A common interactive local
profile is workspace-write plus network access with on-request approvals. A fully unattended profile
has substantially more risk because any prompt-injected repository content may reach network and
credentials.

---

## 4. One-time setup on each fresh pod

[`AGENT_FILES/SETUPS/NEW_POD.md`](SETUPS/NEW_POD.md) is the canonical source. Fresh-pod bootstrap
contains exactly five stages:

1. Install Git, `tmux`, `curl`, CA certificates, and `ffmpeg` with `apt-get`.
2. Create `/workspace/hf_cache` and `/workspace/checkpoints`, then export `HF_HOME`.
3. Clone the repository and check out `phase1-v0.2-frozen-encoder`.
4. Upgrade `pip` and install the checked-out `requirements.txt`.
5. Authenticate W&B with `wandb login`.

After step 5, bootstrap ends and the exact KANBAN run `GUIDE.md` takes over. Dataset checks, import
checks, tests, statistics, Stage 0, `tmux`, training, and monitoring are not generic bootstrap steps.

An authorized coding agent can perform steps 1–4 without human typing. It must inspect first and
must not clone over an existing repository or discard an existing worktree.

For step 5, an agent can execute `wandb login`, but it cannot safely paste an API key received in
chat or expose the key in a command/log. Use one of these paths:

- If `WANDB_API_KEY`, W&B settings, or `.netrc` already provides authentication, the agent verifies
  it without printing the key and continues.
- Otherwise, the agent pauses. The human runs `ssh -t hjepa-runpod 'wandb login'` in Mac Terminal,
  pastes the API key directly into the visible prompt, and then tells the agent to resume.
- For hands-free future pods, map an approved RunPod Secret to `WANDB_API_KEY` before deployment.

Never put W&B, Hugging Face, GitHub, or SSH secrets in an agent chat, KANBAN guide, committed file,
or command transcript.

---

## 5. Copy-paste authorization prompt

Use this bootstrap-only prompt for a fresh pod. It authorizes setup but not training:

**Prompt text — paste into an agent chat, not Terminal:**

```text
Use the run-remote-experiment skill and SSH alias hjepa-runpod.
The remote repository is /workspace/hierarchal-jepa-flow-world-model.

Bootstrap the current fresh pod using AGENT_FILES/SETUPS/NEW_POD.md. You are explicitly authorized
to inspect the pod and perform its exact five-step bootstrap: install the listed system packages;
create the cache/checkpoint paths; clone the repository and check out
phase1-v0.2-frozen-encoder; install requirements.txt; and establish W&B authentication.

Preserve any existing repository/worktree. Do not run dataset checks, tests, statistics, Stage 0,
tmux, training, or monitoring; the exact run GUIDE.md owns those. Do not change the pod lifecycle,
delete data/checkpoints, force/reset Git state, expose secrets, or alter unrelated sessions.

For W&B, use an already-configured credential without printing it. If interactive login is needed,
stop and tell me to run `ssh -t hjepa-runpod 'wandb login'` in my own terminal. Never request the API
key in chat. Continue after I confirm login succeeded.

Report completion of each of the five steps plus the repository branch/SHA/state. Stop after
bootstrap; do not launch a paid process.
```

Use this when you want the agent to implement and launch an experiment end to end:

**Prompt text — paste into an agent chat, not Terminal:**

```text
Use the run-remote-experiment skill and SSH alias hjepa-runpod.
The remote repository is /workspace/hierarchal-jepa-flow-world-model.

Implement <PATH/TO/GUIDE.md> and launch the experiment. You are explicitly authorized to:
- complete the exact five-step AGENT_FILES/SETUPS/NEW_POD.md bootstrap first if this pod is fresh,
  pausing for human W&B login only when no approved credential is already configured;
- edit and test the local repository;
- commit and push the current branch without force-pushing;
- fetch and fast-forward that exact commit on the pod;
- install missing experiment dependencies;
- create, inspect, stop, and restart tmux sessions belonging to this investigation;
- start the paid training process;
- monitor logs, GPU state, checkpoints, and W&B until the guide's early tripwires pass.

Do not terminate/redeploy/resize the pod, delete datasets or checkpoints, kill unrelated
sessions, force-push, change unrelated branches, or expose secrets.
Continue until the requested run is verified or a genuine stop condition in the guide occurs.
```

If you want implementation but no remote launch, say so directly:

**Prompt text — paste into an agent chat, not Terminal:**

```text
Implement the guide and run local tests, but do not SSH, push, or start paid training.
```

If local changes are already pushed, remove commit/push authorization and give the exact branch/SHA
the pod should pull.

---

## 6. Required agent workflow for a launch

Every coding agent must follow this sequence.

### 6.1 Load the repository contract

Read:

1. `AGENT_FILES/AGENTS.md`.
2. `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`.
3. `KANBAN/PROTOCOL.md` and the active investigation/run triad.
4. The exact investigation/run `GUIDE.md`.
5. `AGENT_FILES/SETUPS/NEW_POD.md` if the pod is fresh.
6. `GUIDES/MLOPS.md` for W&B, logs, checkpoints, and Git synchronization.

Old KANBAN guides are experiment history, not implementation truth. Reconcile them with current code,
CLI flags, tests, and `AGENT_FILES/AGENTS.md`. Preserve the intended scientific recipe; report any
necessary operational correction.

### 6.2 Verify authorization

Before mutation, identify whether the task authorizes:

- read-only remote inspection;
- local edits;
- commit/push;
- remote dependency installation;
- paid launch;
- stop/restart of this run;
- destructive infrastructure changes.

Do not silently broaden one category into another.

### 6.3 Implement and verify locally

Follow the plan/guide, preserve unrelated worktree changes, and run the tests required by the touched
code. Keep `PLAN.md`, `DESCRIPTION.md`, and `GUIDE.md` synchronized with the final command.

Any W&B-producing command must use the current display-name contract:

```text
Investigation NN · experiment axis · defining variant
```

### 6.4 Verify SSH and remote state read-only

The following fence is one SSH command. Its quoted remote script intentionally spans several lines.

**Agent local shell — paste this whole block as one command:**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 hjepa-runpod '
  printf "SSH_OK\n"
  hostname
  id
  command -v git
  command -v tmux
  nvidia-smi -L
  test -d /workspace/hierarchal-jepa-flow-world-model/.git
  git -C /workspace/hierarchal-jepa-flow-world-model status --short --branch
'
```

Stop if the wrong server/repository is reached, the GPU is absent, or the remote worktree has
overlapping dirty changes. Never “fix” a dirty remote with `git reset --hard` or `git clean`.

### 6.5 Synchronize an exact commit

Normal path:

```text
local verified changes
  -> intentional commit
  -> push intended branch
  -> remote git fetch
  -> remote git pull --ff-only
  -> compare local and remote git rev-parse HEAD
```

Do not hand-copy Python files to the pod. A paid run should be reproducible from a named commit. Do not
launch if local and remote SHAs differ unless the user explicitly authorized a labeled,
non-reproducible smoke test.

### 6.6 Run gates before the paid process

Run the exact guide gates. These commonly include the following separate commands.

**Agent remote shell — run this one command:**

```bash
pytest -q
```

**Agent remote shell — run this one command:**

```bash
python3 -c "from models import smoke_test_models; smoke_test_models()"
```

**Agent remote shell — run this one command:**

```bash
python3 -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

Then run the guide's Stage 0, provenance preflight, resource preflight, stats prerequisite, or encoder
smoke as applicable. Stop on its declared failure conditions.

### 6.7 Launch in a detached tmux session

Use an investigation/run-specific session name such as:

```text
inv015_whitened_latent_stack
```

Before creating it:

```bash
ssh hjepa-runpod 'tmux list-sessions 2>/dev/null || true'
```

If the intended name already exists, inspect it. Do not kill or overwrite it blindly.

Run training in the foreground *inside* the detached tmux session when practical. That makes a dead
tmux session meaningful evidence that the process exited. Redirect output to the guide's persistent
log path and use a unique checkpoint directory. Paid runs should pass `--require-wandb`.

The exact launch command belongs in the run `GUIDE.md`; do not maintain a second divergent command in
this access guide.

### 6.8 Verify evidence, not merely command acceptance

Immediately check each signal separately.

**Agent local shell — run this one command:**

```bash
ssh hjepa-runpod 'tmux list-sessions'
```

**Agent local shell — run this one command:**

```bash
ssh hjepa-runpod 'pgrep -af "python.*train.py" || true'
```

**Agent local shell — run this one command:**

```bash
ssh hjepa-runpod 'nvidia-smi'
```

**Agent local shell — run this one command:**

```bash
ssh hjepa-runpod 'tail -n 100 /workspace/hierarchal-jepa-flow-world-model/logs/<run>.log'
```

**Agent local shell — run this one command:**

```bash
ssh hjepa-runpod 'tmux capture-pane -p -t <session> -S -120'
```

Verify all of the following:

- tmux session exists;
- expected `train.py` PID exists;
- the intended GPU is active;
- the log has the expected config and first step/progress output;
- W&B has the compliant display name/group and is streaming;
- checkpoint/log destinations match the guide;
- early metrics satisfy the guide's tripwires.

Creating a tmux session is not proof that training launched successfully.

### 6.9 Recover only within scope

If an early tripwire fails:

1. Preserve the log, traceback, `nvidia-smi`, tmux pane, command, and commit SHA.
2. Stop only the authorized run if continuing would waste GPU time or corrupt evidence.
3. Diagnose without changing the experiment's scientific recipe.
4. Relaunch only when recovery authority is explicit and the fix is verified.
5. Otherwise report the blocker and exact next human action.

Never kill unrelated Python processes or tmux sessions by broad patterns such as `pkill python`.

### 6.10 Report a durable handoff

Report:

```text
SSH alias + hostname
local and remote branch/SHA
tmux session
training PID/state
W&B display name, group, run ID/URL
log path
checkpoint path
GPU status
latest verified step
tests/preflights run
deviations from GUIDE.md
exact reconnect/monitor command
```

Do not say “launched successfully” until the process and first expected evidence exist.

---

## 7. Common read-only command cookbook

Connection and host:

```bash
ssh -o BatchMode=yes hjepa-runpod 'hostname; id; uptime'
```

Repository:

```bash
ssh hjepa-runpod '
  git -C /workspace/hierarchal-jepa-flow-world-model status --short --branch
  git -C /workspace/hierarchal-jepa-flow-world-model log -1 --oneline
'
```

GPU and processes:

```bash
ssh hjepa-runpod 'nvidia-smi; pgrep -af "python.*train.py" || true'
```

tmux sessions:

```bash
ssh hjepa-runpod 'tmux list-sessions 2>/dev/null || true'
```

```bash
ssh hjepa-runpod 'tmux capture-pane -p -t <session> -S -120'
```

Logs and checkpoints:

```bash
ssh hjepa-runpod 'tail -n 100 /workspace/hierarchal-jepa-flow-world-model/logs/<run>.log'
```

```bash
ssh hjepa-runpod 'find /workspace/checkpoints/<run> -maxdepth 1 -type f -print'
```

Use narrow process/session names. Treat even read-only commands as potentially sensitive: terminal
output can contain repository URLs, usernames, W&B links, or configuration values. Do not print
environment variables or credential files.

---

## 8. Troubleshooting

### `Permission denied (publickey)`

If SSH asks for `<user>@<host>'s password`, press **Control-C**. That is also a public-key failure;
it is not the passphrase you chose while creating the local key.

Check, in order:

1. The public key—not the `SHA256:` fingerprint—was added.
2. The pod started after account-key addition, or `authorized_keys` was updated manually.
3. `IdentityFile` points to `~/.ssh/hjepa_codex`.
4. `IdentitiesOnly yes` is present.
5. `~/.ssh` is mode `700`; private key and config are mode `600`.
6. A passphrased key is loaded into Keychain/ssh-agent.

Do not switch to password authentication as an agent workaround.

### Host key changed

This can be legitimate after RunPod redeployment, or it can indicate the wrong destination. Compare
the current IP/port against the RunPod Connect tab before removing the stale `[host]:port` entry.
Never globally disable host-key checking.

### `tmux: command not found`

On a fresh pod:

```bash
apt-get update -qq
```

```bash
apt-get install -y tmux
```

Then continue with `AGENT_FILES/SETUPS/NEW_POD.md` because other dependencies may also be missing.

### SSH disconnects during training

Reconnect and inspect the named tmux session. A correctly launched tmux job survives the SSH
connection. If no session exists, inspect the persistent log before relaunching.

### Agent can SSH manually but not from its tool

The agent process may lack outbound network permission, access to the user's ssh-agent/Keychain, or
permission to read `~/.ssh/config`. Adjust the agent's own sandbox/approval settings; do not move the
private key into the repository.

### A new skill is not visible

Start a new agent chat or restart/reload the coding tool. Existing long-lived chats may retain the
skill catalog captured when they started.

---

## 9. Revoke agent SSH access

To revoke access:

1. Remove the public key from RunPod account settings.
2. Remove the matching public line from the server's `authorized_keys` if it was added manually.
3. Remove or disable the `Host hjepa-runpod` block in `~/.ssh/config`.
4. Remove the identity from ssh-agent/Keychain if applicable.
5. Delete the dedicated local key pair only after confirming it is no longer needed.

Revoking this dedicated key does not disturb unrelated personal SSH identities.

---

## 10. Repository-local agent skill discovery

The canonical skill is:

```text
.agents/skills/run-remote-experiment/SKILL.md
```

Discovery paths:

| Agent | Repository path | Invocation |
|---|---|---|
| Codex app/CLI/IDE | `.agents/skills/run-remote-experiment/` | `$run-remote-experiment` or implicit |
| Cursor | `.agents/skills/run-remote-experiment/` | `/run-remote-experiment` or implicit |
| Claude Code | `.claude/skills/run-remote-experiment` symlink | `/run-remote-experiment` or implicit |
| Other Agent Skills implementations | `.agents/skills/run-remote-experiment/` | Tool-specific or implicit |
| Agents without skill discovery | Root `AGENTS.md` pointer | Read skill manually |

Keep `.agents/skills/run-remote-experiment` as the single source of truth. Do not create divergent
copies under tool-specific directories.

Official references:

- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)
- [OpenAI: Connect to an SSH host](https://learn.chatgpt.com/docs/remote-connections#connect-to-an-ssh-host)
- [Claude Code: Extend Claude with skills](https://code.claude.com/docs/en/slash-commands)
- [Cursor: Agent Skills](https://cursor.com/docs/skills)
- [RunPod: Connect to a Pod with SSH](https://docs.runpod.io/pods/configuration/use-ssh)
- [RunPod: Connect with VS Code or Cursor](https://docs.runpod.io/pods/configuration/connect-to-ide)
