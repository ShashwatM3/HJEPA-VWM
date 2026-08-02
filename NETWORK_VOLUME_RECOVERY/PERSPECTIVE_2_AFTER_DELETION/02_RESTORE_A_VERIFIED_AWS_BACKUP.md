# Restore a verified AWS backup after the old RunPod volume is deleted

Use this after the emergency AWS copy is complete and RunPod funds are available again. It restores
the intended `/workspace` topology, not merely an unexamined pile of S3 objects.

There are two restore modes:

- **POSIX-aware restore (recommended):** restore every object except the six documented symlink-farm
  directories, then recreate those links from the preserved source inventory. This avoids duplicating
  video bytes and recreates the intended filesystem structure.
- **Literal object restore:** restore every AWS object. This is simpler but normally creates regular
  files, not Unix symlinks, and may require far more RunPod capacity.

The exact source-object snapshot remains in AWS in both cases.

## Stage 0 — Do not start until these facts are recorded

From `recovery-manifests/<SNAPSHOT_ID>/`, record:

- AWS bucket name;
- snapshot ID;
- `source-size.json` and `destination-size.json` object counts/bytes;
- whether `check-size` and `check-download` exited zero;
- any missing/different/error paths;
- source object count for each excluded link-tree prefix;
- AWS plan/credit expiry date;
- intended new RunPod datacenter and volume capacity.

Do not delete the AWS copy during restore. A new RunPod volume is not trusted until all gates in the
next two chapters pass.

## Stage 1 — Calculate the new capacity

### Literal object restore capacity

Use destination `bytes` from `destination-size.json`, convert to GB, and add at least 20% headroom:

```text
minimum practical GB = ceil(destination_bytes / 1,000,000,000 × 1.20)
```

This mode can be much larger than the old physical volume because S3 may have materialized each
SSv2/tiny symlink as another video object.

### POSIX-aware restore capacity

Subtract the logical bytes beneath these known link views from the AWS total before adding headroom:

```text
data/ssv2/train/**
data/ssv2/validation/**
data/ssv2_tiny/train/**
data/ssv2_tiny/validation/**
data/ego4d_tiny/train/**
data/ego4d_tiny/validation/**
```

Do **not** subtract `/data/ego4d/train` or `/data/ego4d/validation`; those are unique generated MP4
files. Do not subtract `/ssv2_raw`; those are unique raw videos.

Add at least 20% for checkpoints, new runs, package/cache updates, temporary preprocessing, and
filesystem overhead. If EGO4D will be regenerated, budget temporary raw-video space separately.

RunPod lets a volume grow but not shrink and documents direct support up to 4 TB. If the calculated
literal restore exceeds 4 TB, use the POSIX-aware mode or contact RunPod before creating anything.

## Stage 2 — Request the AWS migration-off credit before downloading

If the backup exceeds the unused monthly 100 GB outbound allowance, follow the AWS support process in
[`../PERSPECTIVE_1_BEFORE_DELETION/02_AWS_ACCOUNT_SECURITY_AND_COSTS.md`](../PERSPECTIVE_1_BEFORE_DELETION/02_AWS_ACCOUNT_SECURITY_AND_COSTS.md).
Request approval
**before** copying. Do not assume approval; record the case ID and written terms.

If AWS denies it, ensure enough AWS credit/cash is available for the estimated egress. A failed
partial restore still incurs transferred bytes and request usage.

## Stage 3 — Create the new RunPod network volume

1. Sign in to [RunPod](https://console.runpod.io/).
2. Add enough balance for the network volume’s first billing period and at least one hour of the GPU
   configuration intended for validation. RunPod says an on-demand Pod requires at least one hour’s
   worth of credit.
3. Open **Storage**.
4. Choose **New Network Volume**.
5. Name it `hjepa-vwm-recovered-YYYYMMDD`.
6. Choose **US-MO-1** to reproduce the old datacenter and use
   `https://s3api-us-mo-1.runpod.io`. If US-MO-1 is unavailable, choose another datacenter listed in
   RunPod’s current S3-compatible API table and replace both region/endpoint everywhere below.
7. Enter the calculated capacity in GB. Use **Standard** storage unless a measured workload justifies
   high-performance storage.
8. Choose **Create Network Volume**.
9. Copy the new volume ID from the Storage page into a private migration note. It will not be
   `4hzrwzk8ja`.
10. Confirm the location, size, tier, hourly/monthly estimate, and status before proceeding.

Create a fresh RunPod S3 key under **Settings → S3 API Keys** and store its `user_...` access key and
one-time `rps_...` secret in a password manager. Name it `temporary-aws-restore-YYYYMMDD`.

## Stage 4 — Relaunch the temporary AWS transfer host

Use the same `us-east-2` Amazon Linux 2023, `t3.small`, 8 GiB `gp3`, no-inbound security group, and
Session Manager procedure from the emergency guide. Reuse `HjepaRunpodRescueEc2Role`; it already has
read access to the recovery bucket. Do not launch in another region because same-region S3-to-EC2
transfer is free.

Connect through Session Manager. Install tools:

```bash
sudo dnf install -y tmux unzip
```

```bash
curl https://rclone.org/install.sh | sudo bash
```

Start the persistent restore shell:

```bash
tmux new -s hjepa-restore
```

Disable on-disk rclone configuration:

```bash
export RCLONE_CONFIG=/dev/null
```

Enter the AWS bucket and new RunPod volume ID:

```bash
read -r -p "Exact AWS backup bucket: " AWS_BUCKET
```

```bash
export AWS_BUCKET
```

```bash
read -r -p "New RunPod network-volume ID: " NEW_RUNPOD_VOLUME_ID
```

```bash
export NEW_RUNPOD_VOLUME_ID
```

Create a control directory:

```bash
export RESTORE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
```

```bash
export RESTORE_DIR="$HOME/hjepa-restore-$RESTORE_ID"
```

```bash
mkdir -p "$RESTORE_DIR"
```

Configure AWS from the instance role:

```bash
export RCLONE_CONFIG_AWS_TYPE=s3
```

```bash
export RCLONE_CONFIG_AWS_PROVIDER=AWS
```

```bash
export RCLONE_CONFIG_AWS_ENV_AUTH=true
```

```bash
export RCLONE_CONFIG_AWS_REGION=us-east-2
```

```bash
export RCLONE_CONFIG_AWS_NO_CHECK_BUCKET=true
```

Read the new RunPod access key without embedding it in history:

```bash
read -r -p "New RunPod S3 access key (user_...): " NEW_RUNPOD_S3_ACCESS_KEY
```

Read the secret without echoing it:

```bash
read -r -s -p "New RunPod S3 secret (rps_...): " NEW_RUNPOD_S3_SECRET_KEY; printf '\n'
```

Configure the new endpoint:

```bash
export RCLONE_CONFIG_NEWRUNPOD_TYPE=s3
```

```bash
export RCLONE_CONFIG_NEWRUNPOD_PROVIDER=Other
```

```bash
export RCLONE_CONFIG_NEWRUNPOD_ENV_AUTH=false
```

```bash
export RCLONE_CONFIG_NEWRUNPOD_ACCESS_KEY_ID="$NEW_RUNPOD_S3_ACCESS_KEY"
```

```bash
export RCLONE_CONFIG_NEWRUNPOD_SECRET_ACCESS_KEY="$NEW_RUNPOD_S3_SECRET_KEY"
```

```bash
export RCLONE_CONFIG_NEWRUNPOD_REGION=us-mo-1
```

```bash
export RCLONE_CONFIG_NEWRUNPOD_ENDPOINT=https://s3api-us-mo-1.runpod.io
```

```bash
export RCLONE_CONFIG_NEWRUNPOD_FORCE_PATH_STYLE=true
```

```bash
export RCLONE_CONFIG_NEWRUNPOD_NO_CHECK_BUCKET=true
```

Verify the two remotes:

```bash
rclone listremotes
```

Expected output includes `aws:` and `newrunpod:`.

Verify the AWS identity:

```bash
aws sts get-caller-identity
```

Verify the source prefix is readable:

```bash
rclone lsf "aws:${AWS_BUCKET}/volume" --max-depth 1
```

Verify the new volume is empty:

```bash
rclone lsf "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --max-depth 1
```

The successful output must be empty. If any path appears, stop and verify the volume ID. If every
object is test state that you personally created on this brand-new target, explicitly account for
and remove only those exact objects in the RunPod UI/S3 client, or create another empty volume. Do
not begin the restore with extras because the destination-side integrity check must remain exact.

## Stage 5A — Recommended POSIX-aware restore

Define the restore function. Its six exclusions are intentional and must remain identical for every
copy/check pass:

```bash
restore_posix_aware() { rclone copy "aws:${AWS_BUCKET}/volume" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --exclude '/data/ssv2/train/**' --exclude '/data/ssv2/validation/**' --exclude '/data/ssv2_tiny/train/**' --exclude '/data/ssv2_tiny/validation/**' --exclude '/data/ego4d_tiny/train/**' --exclude '/data/ego4d_tiny/validation/**' --size-only --create-empty-src-dirs --transfers 4 --checkers 8 --s3-upload-concurrency 2 --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$RESTORE_DIR/restore-posix-aware.log" --log-level INFO --stats 30s --stats-one-line -P; }
```

Run it:

```bash
restore_posix_aware
```

Print the status immediately:

```bash
printf 'last rclone exit status: %s\n' "$?"
```

Rerun the same function until it exits zero, then require a second zero pass. Never change it to
`sync`.

Verify the non-link object set using exactly the same filters:

```bash
rclone check "aws:${AWS_BUCKET}/volume" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --exclude '/data/ssv2/train/**' --exclude '/data/ssv2/validation/**' --exclude '/data/ssv2_tiny/train/**' --exclude '/data/ssv2_tiny/validation/**' --exclude '/data/ego4d_tiny/train/**' --exclude '/data/ego4d_tiny/validation/**' --size-only --checkers 8 --combined "$RESTORE_DIR/check-posix-aware-size.txt" --log-file "$RESTORE_DIR/check-posix-aware-size.log" --log-level INFO
```

Run the full read comparison with the same filters:

```bash
rclone check "aws:${AWS_BUCKET}/volume" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --exclude '/data/ssv2/train/**' --exclude '/data/ssv2/validation/**' --exclude '/data/ssv2_tiny/train/**' --exclude '/data/ssv2_tiny/validation/**' --exclude '/data/ego4d_tiny/train/**' --exclude '/data/ego4d_tiny/validation/**' --download --checkers 4 --combined "$RESTORE_DIR/check-posix-aware-download.txt" --log-file "$RESTORE_DIR/check-posix-aware-download.log" --log-level INFO --stats 30s --stats-one-line -P
```

The checks must exit zero. Only then copy the emergency manifests into a separate root on the new
volume so the Pod can rebuild exact membership without AWS credentials. Copying them earlier would
correctly make the data-only destination check report extra files:

```bash
rclone copy "aws:${AWS_BUCKET}/recovery-manifests" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/recovery-manifests" --size-only --transfers 2 --checkers 4 --retries 20 --low-level-retries 50 --log-file "$RESTORE_DIR/restore-manifests.log" --log-level INFO -P
```

Upload the restore logs to AWS:

```bash
aws s3 cp "$RESTORE_DIR" "s3://${AWS_BUCKET}/recovery-manifests/restore-${RESTORE_ID}/" --recursive --region us-east-2
```

Continue at Stage 6.

## Stage 5B — Literal object restore alternative

Use this instead of 5A only when capacity covers all destination logical bytes and regular files at
formerly symlinked paths are acceptable.

Define the full restore:

```bash
restore_all_objects() { rclone copy "aws:${AWS_BUCKET}/volume" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --size-only --create-empty-src-dirs --transfers 4 --checkers 8 --s3-upload-concurrency 2 --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$RESTORE_DIR/restore-all.log" --log-level INFO --stats 30s --stats-one-line -P; }
```

Run and repeat to two consecutive zero passes:

```bash
restore_all_objects
```

Perform a full read check:

```bash
rclone check "aws:${AWS_BUCKET}/volume" "newrunpod:${NEW_RUNPOD_VOLUME_ID}" --download --checkers 4 --combined "$RESTORE_DIR/check-all-download.txt" --log-file "$RESTORE_DIR/check-all-download.log" --log-level INFO --stats 30s --stats-one-line -P
```

Require exit zero. Then copy the emergency manifests onto the new volume; they are outside the AWS
`volume/` prefix and are therefore not included by the literal object copy:

```bash
rclone copy "aws:${AWS_BUCKET}/recovery-manifests" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/recovery-manifests" --size-only --transfers 2 --checkers 4 --retries 20 --low-level-retries 50 --log-file "$RESTORE_DIR/restore-manifests.log" --log-level INFO -P
```

Upload the restore reports to AWS:

```bash
aws s3 cp "$RESTORE_DIR" "s3://${AWS_BUCKET}/recovery-manifests/restore-${RESTORE_ID}/" --recursive --region us-east-2
```

Skip link creation in Stage 8 for any path that is already a valid regular video; the project’s
provenance code can accept identical bytes at the same relative paths, but the physical structure
is not an exact recreation.

## Stage 6 — Close the object-transfer host

After the applicable read check exits zero:

```bash
unset NEW_RUNPOD_S3_ACCESS_KEY NEW_RUNPOD_S3_SECRET_KEY RCLONE_CONFIG_NEWRUNPOD_ACCESS_KEY_ID RCLONE_CONFIG_NEWRUNPOD_SECRET_ACCESS_KEY
```

Revoke the temporary restore S3 key in RunPod Settings. Terminate the EC2 instance from the EC2
console. Keep the IAM role and AWS backup until science validation passes.

## Stage 7 — Deploy a Pod with the recovered volume

1. In RunPod, open **Pods** and choose **Deploy**.
2. Choose **Secure Cloud**; network volumes are available there.
3. Select the new network volume before choosing the GPU. The volume constrains GPU options to its
   datacenter.
4. Choose a GPU/RAM combination appropriate for the current experiment guide. For reconstruction
   alone, use the cheapest compatible Pod; final encoder/training validation requires CUDA.
5. Choose the project’s normal PyTorch/CUDA template. Give the container enough temporary disk for
   packages but do not create a second persistent volume disk.
6. Confirm that the network volume will mount at `/workspace`.
7. Deploy On-Demand. RunPod volumes must be attached during deployment and cannot be attached later
   to an already-created Pod.
8. Open the Pod’s web terminal.

In the **Pod terminal**, verify location:

```bash
pwd
```

Change to the mount:

```bash
cd /workspace
```

Verify the mount and capacity:

```bash
df -hT /workspace
```

List top-level entries without traversing millions of files:

```bash
find /workspace -mindepth 1 -maxdepth 1 -printf '%y %f\n' | sort
```

Expected roots are documented in
[`../SHARED_REFERENCE/01_FORENSIC_VOLUME_MAP.md`](../SHARED_REFERENCE/01_FORENSIC_VOLUME_MAP.md).
`recovery-manifests/` is a new control root and is intentionally not part of the historical topology.

## Stage 8 — Reconstruct exact symlink membership

First reconstruct recorded directory paths, including empty directories that have no file object.
This applies to both restore modes. Locate the source directory inventory:

```bash
find /workspace/recovery-manifests -name source-directory-paths.csv -type f -print
```

Interpret the output literally:

- **Exactly one path:** use it below.
- **More than one path:** use the path inside the intended emergency `SNAPSHOT_ID`; do not combine
  inventories from different source moments.
- **No path:** the backup predates this package or the emergency inventory never completed. Skip the
  inventory commands below, run the documented fallback, and label the result **not an exact
  filesystem reconstruction** because unknown empty directories cannot be inferred from S3 objects.

Transfer `scripts/rebuild_directories_from_inventory.py` with the other helpers. When an inventory
exists, set its exact path:

```bash
read -r -p "Paste the full source-directory-paths.csv path: " SOURCE_DIRECTORY_INVENTORY
```

Dry-run it:

```bash
python3 /workspace/rebuild_directories_from_inventory.py --inventory "$SOURCE_DIRECTORY_INVENTORY" --workspace /workspace --dry-run
```

Require zero unsafe paths, collisions, and duplicates. Run it:

```bash
python3 /workspace/rebuild_directories_from_inventory.py --inventory "$SOURCE_DIRECTORY_INVENTORY" --workspace /workspace
```

Check it:

```bash
python3 /workspace/rebuild_directories_from_inventory.py --inventory "$SOURCE_DIRECTORY_INVENTORY" --workspace /workspace --check
```

The helper deliberately does not recreate RunPod's service-owned `.s3compat_uploads` tree. Directory
names are reconstructed; arbitrary old modes, owners, timestamps, ACLs, xattrs, and hard-link
relationships remain unavailable unless a POSIX-aware archive survived.

If no source directory inventory exists, create only the canonical roots required by current code:

```bash
mkdir -p /workspace/hierarchal-jepa-flow-world-model /workspace/data/ssv2/train /workspace/data/ssv2/validation /workspace/data/ssv2_tiny/train /workspace/data/ssv2_tiny/validation /workspace/data/ego4d/train /workspace/data/ego4d/validation /workspace/data/ego4d_tiny/train /workspace/data/ego4d_tiny/validation /workspace/ssv2_raw /workspace/ego4d_raw /workspace/ckpt /workspace/checkpoints /workspace/stats /workspace/preflight /workspace/hf_cache /workspace/logs /workspace/archive /workspace/recovery-manifests
```

That fallback makes current paths available; it does not recover arbitrary empty directories, old
directory metadata, or undocumented topology.

The rest of this stage applies to the POSIX-aware restore. It uses emergency
`source-size-path.csv` as the exact link-membership record. The link helper never replaces existing
regular files; default behavior is fail-safe.

On the Mac, locate this package’s helpers:

```text
NETWORK_VOLUME_RECOVERY/scripts/rebuild_links_from_inventory.py
NETWORK_VOLUME_RECOVERY/scripts/rebuild_tiny_from_manifest.py
NETWORK_VOLUME_RECOVERY/scripts/rebuild_directories_from_inventory.py
```

Transfer the helper or helpers needed for the selected reconstruction path to the Pod using the SSH procedure in
`AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md`, or push/clone the current repository revision containing this
recovery package. Do not paste secrets into the script.

On the Pod, locate the source inventory copied in Stage 5A:

```bash
find /workspace/recovery-manifests -name source-size-path.csv -type f -print
```

If there is exactly one, store its path:

```bash
read -r -p "Paste the full source-size-path.csv path: " SOURCE_INVENTORY
```

Dry-run link reconstruction:

```bash
python3 /workspace/rebuild_links_from_inventory.py --inventory "$SOURCE_INVENTORY" --workspace /workspace --dry-run
```

The summary must report only the six recognized link families, zero missing targets, zero unsafe
paths, and the expected approximate counts. If the source inventory contains no link-view keys, stop
and use the manifest/official-split fallback in the next chapter.

Create the links:

```bash
python3 /workspace/rebuild_links_from_inventory.py --inventory "$SOURCE_INVENTORY" --workspace /workspace
```

Run it once more in check-only mode:

```bash
python3 /workspace/rebuild_links_from_inventory.py --inventory "$SOURCE_INVENTORY" --workspace /workspace --check
```

Never pass a “replace regular files” option merely to make the command succeed. Quarantine and inspect
collisions first.

## Stage 9 — Repair tracked executable modes without changing content

Object storage may not preserve Unix execute bits. Transfer the package helper
`scripts/restore_git_modes.py` to the Pod alongside the link helper.

Inspect the recovered repository before any pull/reset:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

```bash
git rev-parse --show-toplevel
```

```bash
git status --short
```

```bash
git fsck --full
```

Save the status and commit before changing modes:

```bash
git rev-parse HEAD > /workspace/recovery-manifests/restored-repo-head.txt
```

```bash
git status --short > /workspace/recovery-manifests/restored-repo-status-before-mode-repair.txt
```

Restore only modes recorded in the Git index:

```bash
python3 /workspace/restore_git_modes.py --repo /workspace/hierarchal-jepa-flow-world-model
```

Do not run `git reset --hard`, `git clean`, or an unconditional `git pull`. The restored Pod checkout
can contain unpushed commits, ignored scientific outputs, or a dirty diff not present on this Mac.
Compare and preserve both checkouts before reconciliation.

## Stage 10 — Recreate credentials outside the volume

Recreate these only as needed, preferably beneath `/root` or through environment injection—not in the
Git checkout or `/workspace` backup:

- W&B login/API key;
- Hugging Face token accepted for any gated model repository;
- GitHub SSH/token credentials;
- EGO4D/AWS credentials if raw data must be reacquired;
- any RunPod API key used for automation.

Rotate credentials that were found in the restored object snapshot. Never copy old `/root/.aws`,
`.netrc`, or plaintext tokens into the network volume.

## Stage 11 — Bootstrap software, then validate

Read and follow the current repository-owned setup entry points in order:

1. `AGENT_FILES/AGENTS.md`;
2. `AGENT_FILES/SETUPS/NEW_POD.md` for dependencies and editable install;
3. `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md` for path checks;
4. [`07_VALIDATE_EXPERIMENT_READINESS.md`](07_VALIDATE_EXPERIMENT_READINESS.md) for the recovery
   gates;
5. the exact KANBAN `GUIDE.md` for the experiment to launch.

Use the current requirements and pinned encoder revisions. Do not silently substitute `main`, change
dataset paths, regenerate whitening statistics, or resume a checkpoint before provenance checks
pass.

## Stage 12 — Keep AWS until the new backup strategy is proven

After full filesystem and science readiness passes:

1. Make a new tar/checksum snapshot from the reconstructed POSIX volume if long-term exact symlink,
   mode, and Git preservation is required.
2. Copy that snapshot to at least one independent location.
3. Test-extract it in a disposable directory and compare inventory/checksums.
4. Only then consider S3 Standard-IA or Deep Archive using the
   [AWS cost analysis](../PERSPECTIVE_1_BEFORE_DELETION/02_AWS_ACCOUNT_SECURITY_AND_COSTS.md).
5. Never delete the AWS object-form snapshot on the same day as the first successful training smoke.
