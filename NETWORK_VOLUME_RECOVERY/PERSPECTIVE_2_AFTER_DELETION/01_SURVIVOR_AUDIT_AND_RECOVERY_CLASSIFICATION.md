# Survivor audit and recovery classification

Run this on the Mac before creating, restoring, or “cleaning” a replacement volume. The purpose is
to find every independent source of old bytes and freeze evidence before repair work changes it.

## Stage 0 — establish the irreversible fact

Do not classify the volume as deleted from an authentication error. Use all three checks:

1. In RunPod, open **Storage → Network Volumes** and search for ID `4hzrwzk8ja`.
2. If the row is absent, create/use a valid **S3 API key** and distinguish `NoSuchBucket` from
   `AccessDenied` by following the source-error section of
   [`../SHARED_REFERENCE/04_TROUBLESHOOTING_ALL_PATHS.md`](../SHARED_REFERENCE/04_TROUBLESHOOTING_ALL_PATHS.md#3-runpod-source-errors).
3. Open a RunPod support ticket with the volume ID, account email, datacenter `US-MO-1`, endpoint,
   warning/deletion timestamps, and exact error. Ask whether the volume is terminated, not merely
   detached or payment-blocked.

RunPod's published answer for a terminated volume is that the data cannot be recovered. Support may
clarify state; this handbook does not promise an undocumented provider snapshot.

## Stage 1 — create an evidence directory on the Mac

Open Terminal on the Mac and create a new timestamped directory outside the Git repository:

```bash
export HJEPA_SURVIVOR_AUDIT="$HOME/Desktop/HJEPA_VOLUME_SURVIVOR_AUDIT_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$HJEPA_SURVIVOR_AUDIT"
printf '%s\n' "$HJEPA_SURVIVOR_AUDIT"
```

Record the old identity without any credentials:

```bash
printf '%s\n' \
  'old_volume_id=4hzrwzk8ja' \
  'old_datacenter=US-MO-1' \
  'old_signing_region=us-mo-1' \
  'old_endpoint=https://s3api-us-mo-1.runpod.io' \
  'old_mount=/workspace' \
  "audit_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$HJEPA_SURVIVOR_AUDIT/old-volume-identity.txt"
```

Do not put access keys, secrets, cookies, `.env` contents, or credential-file contents in this
directory.

## Stage 2 — freeze the Mac repository state

The Mac worktree is a survivor and is currently newer than the remote base in uncommitted/untracked
ways. A Git clone alone is insufficient.

Enter the local repository:

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
```

Record identities and the complete path-level status:

```bash
git rev-parse HEAD > "$HJEPA_SURVIVOR_AUDIT/mac-head.txt"
git branch --show-current > "$HJEPA_SURVIVOR_AUDIT/mac-branch.txt"
git remote -v | sed -E 's#(https?://)[^/@]+@#\1REDACTED@#' > "$HJEPA_SURVIVOR_AUDIT/mac-remotes.txt"
git submodule status --recursive > "$HJEPA_SURVIVOR_AUDIT/mac-submodules.txt"
git status --porcelain=v2 --untracked-files=all > "$HJEPA_SURVIVOR_AUDIT/mac-status-porcelain-v2.txt"
git status --short --ignored > "$HJEPA_SURVIVOR_AUDIT/mac-status-with-ignored.txt"
```

Create a bundle of all committed refs/objects:

```bash
git bundle create "$HJEPA_SURVIVOR_AUDIT/HJEPA-VWM-all-refs.bundle" --all
git bundle verify "$HJEPA_SURVIVOR_AUDIT/HJEPA-VWM-all-refs.bundle"
```

Capture the staged/index delta from `HEAD`, including binary, mode, and deletion changes:

```bash
git diff --binary --cached HEAD > "$HJEPA_SURVIVOR_AUDIT/mac-index.patch"
git apply --stat --allow-empty "$HJEPA_SURVIVOR_AUDIT/mac-index.patch"
```

Capture the unstaged worktree delta from the index separately. Keeping two patches is what preserves
which changes were staged:

```bash
git diff --binary > "$HJEPA_SURVIVOR_AUDIT/mac-worktree.patch"
git apply --stat --allow-empty "$HJEPA_SURVIVOR_AUDIT/mac-worktree.patch"
```

Capture every untracked, non-ignored file—including this recovery package—without changing the
worktree:

```bash
git ls-files --others --exclude-standard -z | tar -czf "$HJEPA_SURVIVOR_AUDIT/mac-untracked-files.tar.gz" --null -T -
tar -tzf "$HJEPA_SURVIVOR_AUDIT/mac-untracked-files.tar.gz" > "$HJEPA_SURVIVOR_AUDIT/mac-untracked-archive-members.txt"
```

Ignored files are intentionally not swept into that archive because this repository ignores `.env`,
W&B queues, logs, checkpoints, and other potentially secret/large state. Review the ignored-path
report manually. For each ignored item that is scientifically valuable and safe to retain, copy it
to a separate **encrypted** backup or list its location in the survivor ledger. Never print `.env`
or credential contents.

Hash the evidence files:

```bash
(cd "$HJEPA_SURVIVOR_AUDIT" && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | xargs -0 shasum -a 256 > SHA256SUMS)
```

Keep this directory on the Mac and copy it to an independent disk/cloud location. The ground-up
guide applies the index patch with `--index`, then the worktree patch without it, then the untracked
archive. That order reproduces the current Mac content and staging state over the verified base
commit.

## Stage 3 — audit AWS and other object-storage survivors

For every AWS account you legitimately control:

1. Sign in to the AWS console.
2. Open **S3 → Buckets**.
3. Search for a bucket shaped like `hjepa-runpod-4hzrwzk8ja-...` and any other project backup name.
4. Open candidate buckets without editing them.
5. Look for `volume/` and `recovery-manifests/<SNAPSHOT_ID>/`.
6. Record bucket name, AWS account ID, region, versioning status, object count/bytes shown by the
   available inventory, and every rescue log/inventory/check filename.
7. Download only the small `recovery-manifests/` evidence to the Mac audit directory. Do not start a
   whole restore yet.

If a configured AWS CLI v2 profile exists, identify it without printing secrets:

```bash
aws configure list-profiles
```

List buckets for the chosen profile after replacing only the profile name:

```bash
aws s3api list-buckets --profile REPLACE_WITH_PROFILE --query 'Buckets[].Name' --output text
```

For a candidate bucket, read its region and versioning state:

```bash
aws s3api get-bucket-location --bucket REPLACE_WITH_BUCKET --profile REPLACE_WITH_PROFILE
aws s3api get-bucket-versioning --bucket REPLACE_WITH_BUCKET --profile REPLACE_WITH_PROFILE
```

Measure only the candidate `volume/` prefix. This can take time and incur LIST requests:

```bash
aws s3 ls "s3://REPLACE_WITH_BUCKET/volume/" --recursive --summarize --profile REPLACE_WITH_PROFILE > "$HJEPA_SURVIVOR_AUDIT/aws-volume-listing.txt"
```

Classify the candidate:

- **verified complete**: rescue copy exited zero on complete passes, source/destination count+bytes
  matched, path/size check passed, and preferably full-download check passed;
- **complete-looking but unverified**: object tree/inventory appears complete, but one or more proof
  reports are missing;
- **partial**: logs or inventory identify missing/different/error keys, or the copy stopped early;
- **manifest-only**: evidence exists but `volume/` bytes do not;
- **not a backup**: only W&B metadata, path lists, or checksum text exists.

Never discard a partial candidate. Perspective 2 can restore exact surviving objects and rebuild
other components separately.

Repeat the same read-only audit for Backblaze, Google Cloud, Dropbox, NAS, or any other destination
you knowingly used. Do not grant this guide authority to search accounts or data unrelated to the
project.

## Stage 4 — audit local and external-disk survivors

List mounted external volumes on the Mac:

```bash
ls -la /Volumes > "$HJEPA_SURVIVOR_AUDIT/mounted-volumes.txt"
diskutil list > "$HJEPA_SURVIVOR_AUDIT/diskutil-list.txt"
```

In Finder, inspect only project-related folders on those disks. Look for names containing:

```text
runpod-volume-4hzrwzk8ja
HJEPA
ssv2_raw
ego4d_raw
phase1_step
recovery-manifests
source-size-path.csv
```

Also inspect `~/Downloads`, `~/Desktop`, and project-specific archive locations for:

- the two official SSv2 video ZIPs and label package;
- EGO4D `ego4d.json`, tier manifest, UID lists, or downloaded source videos;
- `.pt` checkpoints/statistics;
- prior project tarballs, Git bundles, patches, rsync copies, or rclone logs.

Record paths, byte sizes, modified times, and SHA-256 values. Do not move or rename candidates during
the audit. A path list or checksum without its corresponding byte file is evidence only, not a
recoverable artifact.

## Stage 5 — audit GitHub and other Git survivors

The audited remote branch currently resolves to the known base commit. Recheck it:

```bash
git ls-remote https://github.com/ShashwatM3/HJEPA-VWM.git refs/heads/phase1-v0.2-frozen-encoder | tee "$HJEPA_SURVIVOR_AUDIT/github-remote-branch.txt"
```

Expected audited SHA:

```text
771cbba077d9f846bdf7a7dd48e12dbf29d54b49
```

Check GitHub branches, tags, pull requests, releases, Actions artifacts, gists, and forks you control
for Pod-originated commits or bundles. A pushed commit is recoverable. A W&B `git.commit` field proves
an identity but does not contain an unpushed worktree.

## Stage 6 — audit W&B cloud state without assuming file uploads

In the W&B website:

1. Open the entity and project used by HJEPA-VWM (the repository default project is `hjepa-vwm`).
2. In **Runs**, export or record run IDs, names, state, creation time, config, Git commit, and final
   checkpoint path/SHA summary fields.
3. Open each important run's **Files** tab. Record the exact filenames actually present.
4. Open **Artifacts** and inspect every version's **Files** tab. Download explicit model/checkpoint,
   whitening, provenance, or probe artifacts that really contain bytes.
5. Distinguish a regular artifact from a **reference artifact**; a reference can point to an external
   object without storing its bytes in W&B.
6. Save downloaded files under the Mac audit directory and hash them.

Current project code explicitly uploads run-provenance and optional whitening/probe reports. It does
not upload training checkpoints. Treat a checkpoint as recovered from W&B only after downloading a
real `.pt` file and validating its SHA; a displayed path or summary SHA alone is not the file.

An offline run directory that existed only under the deleted volume is irrecoverable. `wandb sync`
can upload an offline directory only if those local bytes survived somewhere else.

## Stage 7 — audit dataset and model access

These checks answer whether a new derivation is possible; they do not recover old bytes.

### SSv2

Open the official Qualcomm Something-Something V2 download page. Confirm that the account can view
the two video downloads, label package, research-use agreement, and instructions. Do not download
from an unlicensed mirror to preserve an old filename.

### EGO4D

Open the official EGO4D license portal. Record whether the user has a current approval email. Access
credentials expire after 14 days but can be renewed through EGO4D's process. Never copy the access
key/secret into the survivor ledger.

### Hugging Face

Confirm browser access to the three exact repositories and, for gated DINOv3, that the account shows
approved access. `hf auth whoami` may be used after login; do not print tokens.

The exact repository/revision pairs are listed in the root recovery README. A model page existing at
`main` does not prove that the pinned 40-character commit remains downloadable; the later real smoke
does that.

## Stage 8 — assign a disposition to every artifact family

Copy [`../templates/SURVIVOR_LEDGER.md`](../templates/SURVIVOR_LEDGER.md) into the audit directory:

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
cp NETWORK_VOLUME_RECOVERY/templates/SURVIVOR_LEDGER.md "$HJEPA_SURVIVOR_AUDIT/SURVIVOR_LEDGER.md"
```

Fill one row per family with one of:

- exact bytes recovered;
- exact bytes present but not yet verified;
- partial bytes;
- recomputable new derivation;
- reacquirable licensed source;
- metadata only;
- irrecoverable;
- access blocked/pending.

Minimum families: both checkpoint roots, stats, preflight, logs, repo/.git, W&B offline state, HF
cache, SSv2 raw/full/tiny/labels, EGO4D raw/manifests/full/tiny, archive, and unknown top-level paths
from any old inventory.

## Stage 9 — select the path and freeze the audit

- Choose **Case A** only for a complete backup candidate. If proof is incomplete, state that and run
  the restore into a new volume without altering the backup.
- Choose **Case B** when any meaningful old bytes survive but the set is incomplete.
- Choose **Case C** only when no usable old-volume byte copy exists and current source access is
  sufficient for a fresh build.
- Choose **Case D** for each blocked essential lane; continue with accessible lanes rather than
  pretending the blocker does not exist.

Recompute the Mac audit hash list after completing the ledger:

```bash
(cd "$HJEPA_SURVIVOR_AUDIT" && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | xargs -0 shasum -a 256 > SHA256SUMS)
```

Make a second independent copy of the finished audit directory before beginning restoration.
