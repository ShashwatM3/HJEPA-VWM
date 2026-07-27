# Rebuild a fresh current-project volume from the ground up

Use this only for survivor-audit Case C: the old RunPod volume and all byte backups are gone, but the
Mac repository, GitHub, dataset entitlements, W&B cloud state, and model access are still available.

This procedure creates a **new derivation**, not the deleted filesystem. It can restore current
experiment capability; it cannot recreate missing checkpoint tensors, optimizer/RNG state, Pod-only
dirty code, unsynced W&B queues, old logs, or old preflight evidence.

## Stage 0 — prerequisites and expected result

Do not start until all of these are true:

- [`01_SURVIVOR_AUDIT_AND_RECOVERY_CLASSIFICATION.md`](01_SURVIVOR_AUDIT_AND_RECOVERY_CLASSIFICATION.md)
  is complete and the Mac evidence directory has two copies;
- the RunPod account can again pay for a network volume and a temporary Pod;
- Qualcomm SSv2 research-use access is available, or SSv2 is explicitly marked blocked;
- EGO4D approval can be obtained/renewed, or EGO4D is explicitly marked blocked;
- Hugging Face access is available for the selected encoder lanes, including gated DINOv3;
- the user controls the GitHub and W&B accounts used by this project.

Target topology:

```text
/workspace/
├── hierarchal-jepa-flow-world-model/  current Mac tree over verified Git base
├── data/
│   ├── ssv2/{train,validation}/       exact official split membership, absolute symlinks
│   ├── ssv2_tiny/{train,validation}/  4002/348 deterministic symlinks
│   ├── ego4d/{train,validation}/      newly generated project chunks
│   └── ego4d_tiny/{train,validation}/ 4000/350 deterministic symlinks
├── ssv2_raw/20bn-something-something-v2/
├── ego4d_raw/
├── ckpt/
├── checkpoints/
├── stats/
├── preflight/
├── hf_cache/
├── logs/
├── archive/
└── recovery-input/
```

## Stage 1 — freeze the exact Mac inputs

Return to the Mac survivor-audit directory created in the previous guide:

```bash
printf '%s\n' "$HJEPA_SURVIVOR_AUDIT"
test -s "$HJEPA_SURVIVOR_AUDIT/HJEPA-VWM-all-refs.bundle"
test -f "$HJEPA_SURVIVOR_AUDIT/mac-index.patch"
test -f "$HJEPA_SURVIVOR_AUDIT/mac-worktree.patch"
test -s "$HJEPA_SURVIVOR_AUDIT/mac-untracked-files.tar.gz"
shasum -a 256 -c "$HJEPA_SURVIVOR_AUDIT/SHA256SUMS"
```

If the shell was restarted and the variable is empty, set it to the exact existing audit directory;
do not create a second incomplete one:

```bash
export HJEPA_SURVIVOR_AUDIT="$HOME/Desktop/REPLACE_WITH_EXACT_HJEPA_VOLUME_SURVIVOR_AUDIT_DIRECTORY"
```

Confirm that the base commit still exists remotely:

```bash
git ls-remote https://github.com/ShashwatM3/HJEPA-VWM.git refs/heads/phase1-v0.2-frozen-encoder
```

The audited output SHA is `771cbba077d9f846bdf7a7dd48e12dbf29d54b49`. If the branch moved, that is
fine only if the exact commit can still be fetched; never silently replace the base with the new tip.

## Stage 2 — calculate capacity and cash before creating the volume

For a pure ground-up current-project rebuild, use **300 GB**. This is the established project budget
for one ~72 GB transient EGO4D batch plus accumulated ~50–90 GB chunks, SSv2, code, caches, and
headroom. RunPod volumes can grow but cannot shrink.

At the published first-tier rate:

```text
300 GB × $0.07/GB-month ≈ $21/month before tax
```

This is not the right size for restoring a historic backup; the full-backup guide calculates that
from its inventory. In the RunPod console, also record:

- exact monthly storage quote;
- available datacenters and their current GPU types/prices;
- account balance after creating the volume;
- enough balance for at least setup, EGO4D preprocessing, encoder smokes, and validation.

## Stage 3 — create the new RunPod network volume

In the RunPod website:

1. Sign in and open **Storage**.
2. Choose **New Network Volume**.
3. Name it `hjepa-vwm-rebuilt-2026-07`.
4. Choose `US-MO-1` if it offers a suitable GPU and you want the old datacenter/endpoint convention.
   Otherwise choose another datacenter that both offers the required GPU and appears in RunPod's
   current S3-compatible API list.
5. Enter `300` GB.
6. Review the live recurring price.
7. Choose **Create Network Volume**.
8. Wait until status is ready.
9. Record the **new** volume ID and datacenter. The new ID will not be `4hzrwzk8ja`.

Set the following on the Mac, replacing every placeholder with the new volume's real values. The
region value is the lower-case datacenter ID used for signing:

```bash
export NEW_RUNPOD_VOLUME_ID="REPLACE_WITH_NEW_VOLUME_ID"
export NEW_RUNPOD_DATACENTER="US-MO-1"
export NEW_RUNPOD_REGION="us-mo-1"
export NEW_RUNPOD_ENDPOINT="https://s3api-us-mo-1.runpod.io"
```

If another datacenter was chosen, use its exact documented endpoint. Do not point at the old
US-MO-1 endpoint with a volume created elsewhere.

## Stage 4 — create a temporary S3 key and configure a Mac upload remote

In RunPod **Settings → S3 API Keys**, create a key named
`temporary-ground-up-volume-load-2026-07`. Save its one-time secret in a password manager. This is a
separate S3 key, not the normal RunPod API key.

Install rclone on the Mac if it is absent:

```bash
command -v rclone || brew install rclone
rclone version
```

Use in-memory configuration only:

```bash
export RCLONE_CONFIG=/dev/null
read -r -p "RunPod S3 access key (user_...): " NEW_RUNPOD_S3_ACCESS_KEY
read -r -s -p "RunPod S3 secret (rps_...): " NEW_RUNPOD_S3_SECRET_KEY; printf '\n'
```

Configure the remote without writing its secret to a file:

```bash
export RCLONE_CONFIG_NEWRUNPOD_TYPE=s3
export RCLONE_CONFIG_NEWRUNPOD_PROVIDER=Other
export RCLONE_CONFIG_NEWRUNPOD_ENV_AUTH=false
export RCLONE_CONFIG_NEWRUNPOD_ACCESS_KEY_ID="$NEW_RUNPOD_S3_ACCESS_KEY"
export RCLONE_CONFIG_NEWRUNPOD_SECRET_ACCESS_KEY="$NEW_RUNPOD_S3_SECRET_KEY"
export RCLONE_CONFIG_NEWRUNPOD_REGION="$NEW_RUNPOD_REGION"
export RCLONE_CONFIG_NEWRUNPOD_ENDPOINT="$NEW_RUNPOD_ENDPOINT"
export RCLONE_CONFIG_NEWRUNPOD_FORCE_PATH_STYLE=true
export RCLONE_CONFIG_NEWRUNPOD_NO_CHECK_BUCKET=true
```

Prove the new empty volume is reachable:

```bash
rclone lsf "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --max-depth 1
```

An empty successful result is correct. `AccessDenied` is a key problem; `NoSuchBucket` means the new
ID/endpoint/datacenter combination is wrong or not ready.

## Stage 5 — upload the frozen Mac recovery inputs without a Pod

Copy the evidence directory into a fixed path on the new volume:

```bash
rclone copy "$HJEPA_SURVIVOR_AUDIT" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/recovery-input/mac-audit" --size-only --transfers 4 --checkers 8 --retries 20 --low-level-retries 50 --stats 30s -P
```

Verify every uploaded evidence byte against the Mac:

```bash
export HJEPA_UPLOAD_CHECK_REPORT="${HJEPA_SURVIVOR_AUDIT}-check-new-volume-upload.txt"
rclone check "$HJEPA_SURVIVOR_AUDIT" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/recovery-input/mac-audit" --download --checkers 4 --combined "$HJEPA_UPLOAD_CHECK_REPORT" -P
```

Require exit status zero and only `=` lines in the combined report. The report is deliberately a
sibling of the checked source directory so it cannot become a new, falsely missing source file.
Copy it to a separate evidence prefix:

```bash
rclone copyto "$HJEPA_UPLOAD_CHECK_REPORT" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/recovery-input/transfer-evidence/mac-audit-check.txt"
```

This step uses no RunPod Pod.

Next, perform the SSv2 acquisition/upload in
[`04_REBUILD_SSV2.md`](04_REBUILD_SSV2.md) through its pre-Pod upload gate. Doing this now avoids
paying for an idle Pod during a ~19.4 GB internet upload.

## Stage 6 — deploy a Pod with the new volume

In the RunPod website:

1. Open **Pods → Deploy**.
2. Select **Network Volume**, then choose the new 300 GB volume. Network volumes must be attached at
   Pod creation and cannot be attached later without deleting the Pod.
3. Choose an available Secure Cloud GPU in the same datacenter. Use the GPU family required by the
   next documented experiment; A100 80 GB is the historical project baseline, not a mandatory
   provider promise.
4. Choose a current official RunPod PyTorch template whose Python is at least 3.10.
5. Keep the network-volume mount at `/workspace`.
6. Expose `22/tcp` and choose a public-IP capable machine if full SSH/SCP is desired. Basic proxied
   SSH works for terminal access but not SCP/SFTP.
7. Review the live hourly price and deploy On-Demand.
8. Wait for Running, then open **Connect** and copy the exact SSH command shown by RunPod.

On the Mac, paste that exact SSH command. Do not use an example IP from documentation.

## Stage 7 — verify the new mount before writing project state

Run inside the Pod:

```bash
df -hT /workspace
find /workspace -mindepth 1 -maxdepth 3 -printf '%y\t%p\n' | sort | sed -n '1,120p'
```

Require approximately 300 GB total capacity and the uploaded
`/workspace/recovery-input/mac-audit` files. Verify their checksums:

```bash
cd /workspace/recovery-input/mac-audit
sha256sum -c SHA256SUMS
git bundle list-heads HJEPA-VWM-all-refs.bundle
```

Create only the documented top-level roots:

```bash
mkdir -p /workspace/data /workspace/ssv2_raw /workspace/ego4d_raw
mkdir -p /workspace/ckpt /workspace/checkpoints /workspace/stats
mkdir -p /workspace/preflight /workspace/hf_cache /workspace/logs /workspace/archive
```

## Stage 8 — install system tools on the fresh Pod

Run inside the Pod:

```bash
apt-get update -qq
apt-get install -y git tmux curl ca-certificates ffmpeg unzip rsync
```

Verify the tools individually:

```bash
git --version
tmux -V
ffmpeg -version | head -1
unzip -v | head -1
```

## Stage 9 — recreate the Mac code tree exactly enough to continue development

Clone the audited base repository into the required path:

```bash
cd /workspace
git clone https://github.com/ShashwatM3/HJEPA-VWM.git hierarchal-jepa-flow-world-model
cd /workspace/hierarchal-jepa-flow-world-model
git checkout --detach 771cbba077d9f846bdf7a7dd48e12dbf29d54b49
```

If and only if GitHub cannot supply that exact commit, remove no existing tree. From a still-empty
`/workspace/hierarchal-jepa-flow-world-model` target, clone the verified frozen bundle instead:

```bash
cd /workspace
git clone /workspace/recovery-input/mac-audit/HJEPA-VWM-all-refs.bundle hierarchal-jepa-flow-world-model
cd /workspace/hierarchal-jepa-flow-world-model
git checkout --detach 771cbba077d9f846bdf7a7dd48e12dbf29d54b49
git remote set-url origin https://github.com/ShashwatM3/HJEPA-VWM.git
```

Do not run both clone paths into the same directory. If a failed clone left a partial directory,
preserve its error/log, choose a different empty staging path, and only rename after verifying the
exact commit.

Verify the commit before applying anything:

```bash
test "$(git rev-parse HEAD)" = "771cbba077d9f846bdf7a7dd48e12dbf29d54b49"
git bundle verify /workspace/recovery-input/mac-audit/HJEPA-VWM-all-refs.bundle
git status --short
```

Apply the frozen staged changes to both the index and working tree first:

```bash
git apply --check --index --allow-empty /workspace/recovery-input/mac-audit/mac-index.patch
git apply --index --allow-empty /workspace/recovery-input/mac-audit/mac-index.patch
```

Then apply the frozen unstaged changes to the working tree only:

```bash
git apply --check --allow-empty /workspace/recovery-input/mac-audit/mac-worktree.patch
git apply --allow-empty /workspace/recovery-input/mac-audit/mac-worktree.patch
```

Extract the frozen untracked files over that tree:

```bash
tar -xzf /workspace/recovery-input/mac-audit/mac-untracked-files.tar.gz -C /workspace/hierarchal-jepa-flow-world-model
```

Compare the resulting status to the captured Mac status:

```bash
git status --porcelain=v2 --untracked-files=all > /workspace/recovery-input/pod-reconstructed-status.txt
diff -u /workspace/recovery-input/mac-audit/mac-status-porcelain-v2.txt /workspace/recovery-input/pod-reconstructed-status.txt
```

Require no diff. If paths differ, stop and diagnose; do not `git pull`, reset, or force checkout over
the reconstruction. The Git bundle remains the fallback if the remote commit later disappears.

Because this exact state is detached from the remote branch and dirty by design, create a reviewed
commit/branch later before launching paid scientific runs. This handbook does not silently commit or
push the user's current work.

## Stage 10 — install the current Python environment and record its new identity

The deleted environment cannot be exact without an old image digest and `pip freeze`. Install the
current declared dependencies and label the result as a new runtime:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 --version
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Require the repository's hard pin:

```bash
python3 - <<'PY'
import torch
import transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert transformers.__version__ == "4.57.6"
assert torch.cuda.is_available()
PY
```

Create a dated environment record:

```bash
export REBUILD_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "/workspace/preflight/ground-up-${REBUILD_STAMP}"
python3 -m pip freeze --all > "/workspace/preflight/ground-up-${REBUILD_STAMP}/pip-freeze.txt"
ffmpeg -version > "/workspace/preflight/ground-up-${REBUILD_STAMP}/ffmpeg-version.txt" 2>&1
nvidia-smi -q > "/workspace/preflight/ground-up-${REBUILD_STAMP}/nvidia-smi-q.txt"
git status --short > "/workspace/preflight/ground-up-${REBUILD_STAMP}/git-status.txt"
git rev-parse HEAD > "/workspace/preflight/ground-up-${REBUILD_STAMP}/git-head.txt"
```

## Stage 11 — configure caches and credentials outside the volume

Set the persistent model cache path:

```bash
export HF_HOME=/workspace/hf_cache
grep -q 'HF_HOME=/workspace/hf_cache' ~/.bashrc || printf '%s\n' 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
```

Authenticate interactively as needed:

```bash
hf auth login
```

```bash
wandb login
```

Use a Hugging Face read token. Request gated DINOv3 access in the browser before its smoke. W&B and
HF credential stores belong under `/root`, not `/workspace`; do not copy them into recovery
manifests. EGO4D credentials are configured only in the EGO4D component guide and expire after 14
days.

## Stage 12 — rebuild both dataset families

Finish SSv2 first:

1. Execute every remaining stage in [`04_REBUILD_SSV2.md`](04_REBUILD_SSV2.md).
2. Require exactly 220,847 raw videos, 168,913 train links, 24,777 validation links, 4002 tiny train
   links, and 348 tiny validation links.
3. Keep the official archives/labels until an independent backup is verified.

Then execute [`05_REBUILD_EGO4D.md`](05_REBUILD_EGO4D.md):

1. obtain fresh licensed credentials;
2. create a new 210-hour, seed-42 selection because the old manifest was deleted;
3. perform all four download→chunk→verify→delete-raw batches in order;
4. require split isolation, final count ranges, decode geometry, and 4000/350 tiny links.

If either owner denies access, mark that lane blocked in the survivor ledger. Do not fill it from an
unverified mirror.

## Stage 13 — repopulate the three pinned encoder lanes

Run one at a time so a failure is attributable:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
python3 encoders.py --smoke --encoder vjepa2_vitl16 --batch-size 1 --hf-cache-dir /workspace/hf_cache | tee "/workspace/preflight/ground-up-${REBUILD_STAMP}/encoder-vjepa2.txt"
```

```bash
python3 encoders.py --smoke --encoder siglip2_vitb16 --batch-size 1 --hf-cache-dir /workspace/hf_cache | tee "/workspace/preflight/ground-up-${REBUILD_STAMP}/encoder-siglip2.txt"
```

```bash
python3 encoders.py --smoke --encoder dinov3_vitb16 --batch-size 1 --hf-cache-dir /workspace/hf_cache | tee "/workspace/preflight/ground-up-${REBUILD_STAMP}/encoder-dinov3.txt"
```

Each smoke must print the exact requested/resolved revision from the root README and zero trainable
encoder parameters. Never change a failing immutable revision to `main`.

## Stage 14 — handle permanently lost experiment state honestly

With no backup, leave old checkpoint paths absent. Do not create placeholder files. Record:

- `/workspace/ckpt`: fresh output root only;
- `/workspace/checkpoints`: fresh default output root only;
- old `stats`, `preflight`, logs, and probe caches: lost;
- W&B cloud runs: retained as metrics/config/evidence only unless explicit artifacts downloaded;
- old exact resume: unavailable.

Use [`06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md`](06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md) to
download any real W&B artifact survivors and to generate new whitening/preflight/probe artifacts for
specific experiment recipes. New artifacts need new paths, timestamps, hashes, and provenance. Do
not recreate an old filename merely because a KANBAN guide references it.

## Stage 15 — run the full science-readiness gates

Execute [`07_VALIDATE_EXPERIMENT_READINESS.md`](07_VALIDATE_EXPERIMENT_READINESS.md) from Gate 0
through final sign-off. For a no-backup rebuild:

- old/new inventory comparison is marked “not available—old volume terminated without inventory,”
  not falsely passed;
- historic checkpoint gates are “not applicable—bytes lost” unless an independent checkpoint was
  found;
- current code/dataset/encoder/provenance/resource/W&B/disposable-training gates remain mandatory.

Do not launch a real 15k-step experiment until all applicable gates pass.

## Stage 16 — create an independent backup before calling the rebuild complete

The replacement RunPod volume is not a backup of itself. Choose AWS S3 or another approved private
destination, then adapt Perspective 1's non-deleting whole-volume copy and inventory checks to the
**new** volume ID/endpoint. At minimum preserve immediately:

- the Mac audit inputs and reconstructed Git tree;
- official SSv2 archives/labels and extracted raw videos;
- EGO4D metadata, tier manifest, selection/UID lists, generated chunks, and tiny manifest;
- all new checkpoints, whitening artifacts, preflight evidence, and offline W&B queues;
- source/destination inventories and a full content check.

Revoke the temporary volume-load S3 key only after the independent copy is verified:

```bash
unset NEW_RUNPOD_S3_ACCESS_KEY NEW_RUNPOD_S3_SECRET_KEY RCLONE_CONFIG_NEWRUNPOD_ACCESS_KEY_ID RCLONE_CONFIG_NEWRUNPOD_SECRET_ACCESS_KEY
```

Then revoke it in the RunPod console. Keep a separate long-term backup schedule; RunPod explicitly
says it is not designed as long-term storage.
