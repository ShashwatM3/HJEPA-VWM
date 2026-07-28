# Troubleshooting and disaster cases for both recovery perspectives

Preserve the exact command, UTC time, exit status, last 200 log lines, affected path, source/destination
IDs, client versions, and whether a retry changed the result. Never “fix” recovery by deleting the
destination or suppressing integrity errors.

## 1. Fast decision tree

```text
Can RunPod list bucket 4hzrwzk8ja?
├── NoSuchBucket / volume absent
│   └── Contact RunPod immediately; if terminated, only already copied AWS/local objects survive.
├── Authentication/signature error
│   └── Fix the separate RunPod S3 key, endpoint, region, path style, or host clock.
├── Pagination/502/timeout
│   └── Wait briefly, retry non-deleting copy with conservative concurrency; preserve successes.
└── Yes
    ├── Can AWS destination be listed/written?
    │   ├── No → fix EC2 role/bucket/region; do not alter source.
    │   └── Yes → continue prioritized copy, then whole copy.
    └── Copy completes but verification differs
        ├── Missing destination keys → rerun copy; inspect per-key errors.
        ├── Extra destination keys → distinguish old versions/control prefixes; never auto-delete.
        ├── Same path, different size → copy that path again and full-read check.
        └── Hash only differs → account for multipart ETags, then use --download comparison.
```

## 2. AWS account creation/payment failures

### Signup cannot continue without payment method

AWS requires a valid method for both Free and Paid plans. For AWS India, try only methods officially
offered on the signup/payment screen. An eligible saved card can receive a refundable INR 2
verification. Net banking/UPI support for bills does not guarantee that every signup screen accepts
it for initial verification.

Do not create false identities or repeated accounts. If activation cannot happen before RunPod’s
deadline, use the local/external-disk fallback immediately.

### Account stays “pending activation”

AWS says activation can take up to 24 hours. Check spam, finish every identity/payment/support-plan
step, and use account/billing support. Do not wait idle if source deletion is imminent—begin a local
copy in parallel when storage is available.

### Free plan closes or warns of expiry

Upgrade to Paid before the earlier of six months or credit exhaustion, or move the backup first. AWS
says closed Free-plan data is retained for 90 days, but reopening/downloading requires a Paid upgrade.
After 90 days it is erased. A Budget alert does not extend this.

## 3. RunPod source errors

### `AccessDenied`, `InvalidAccessKeyId`, or signature mismatch

Check all of these:

- the credential is from **Settings → S3 API Keys**, not the ordinary RunPod API key;
- access key is the displayed `user_...` ID and secret is the one-time `rps_...` value;
- endpoint is exactly `https://s3api-us-mo-1.runpod.io`;
- signing region is `us-mo-1`;
- volume ID is exactly `4hzrwzk8ja`;
- rclone provider is `Other` and path-style access is true;
- no stale same-named remote/config file overrides environment variables;
- host UTC time is synchronized.

Inspect time without secrets:

```bash
timedatectl status
```

RunPod rejects requests more than one hour out of sync. Do not use rclone debug/header dumps in a
shared log; verbose authentication traces can expose sensitive context.

### `NoSuchBucket` or volume missing in Storage

Recheck the ID and logged-in RunPod account. Test `ListBuckets` with the new S3 key. If the ID truly
does not exist, capture the response and open RunPod support immediately. RunPod’s published policy
says terminated network-volume data cannot be recovered. AWS cannot reconstruct objects never
copied.

### Listing hangs, reports the same continuation token, or takes minutes

This is a documented RunPod behavior for directories above roughly 10,000 files or 10 GB. The first
list computes/caches MD5 ETags for files created outside the S3 API.

Actions:

1. Leave a progressing request alone.
2. If the client exits with the same-next-token pagination error, wait briefly and retry.
3. Copy high-value prefixes independently so one huge listing is not the only work unit.
4. Keep `--fast-list` off.
5. Keep every successfully copied destination object.

### HTTP 502, reset, timeout, or transient EOF

Rerun the same `rclone copy`. The guide already uses high low-level retries, 15-second retry sleeps,
and conservative four-object concurrency. If errors persist:

- lower `--transfers` from 4 to 2;
- lower `--checkers` from 8 to 4;
- retry the exact failed prefix or path;
- verify EC2 outbound internet and DNS;
- do not increase concurrency blindly.

RunPod’s AWS CLI guidance suggests standard retry mode and at least ten attempts for 502s. Rclone’s
equivalent retry policy is already stronger in the runbook.

### A file larger than 500 MB fails

RunPod requires multipart upload **to** its volume for files over 500 MB and supports maximum files up
to 4 TB. The emergency direction is RunPod GET → AWS PUT, so rclone’s AWS multipart behavior matters
primarily on the destination. On reverse restore, retain rclone multipart support and do not disable
it. Inspect free target capacity because RunPod does not preflight capacity for every copy/part.

### Object name contains `#` or another special character

RunPod notes such names may require URL encoding. Rclone normally handles S3 key escaping. If one path
alone fails, record its exact escaped/unescaped representation and use `rclone copyto` rather than
renaming it. Renaming changes the filesystem identity.

## 4. AWS destination/EC2 errors

### Session Manager says instance is not connected

Verify:

- `HjepaRunpodRescueEc2Role` is the instance profile;
- `AmazonSSMManagedInstanceCore` is attached;
- Amazon Linux 2023/SSM Agent is running;
- the instance has outbound HTTPS via public IPv4/default internet route or SSM VPC endpoints;
- security-group outbound rules permit HTTPS.

No inbound port is required. Do not open SSH to the entire internet.

### AWS `head-bucket` or rclone returns AccessDenied

Confirm bucket spelling, region `us-east-2`, account, instance-role ARN, and both policy resource ARNs.
The bucket-level ARN has no `/*`; the object-level ARN does. Required operations are list/location on
the bucket and get/put/multipart operations on objects. Block Public Access does not block an
authorized same-account role.

### `aws` is missing on Amazon Linux

Install AWS CLI v2 from AWS’s official Linux installer rather than using an unknown package:

```bash
curl https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
```

```bash
unzip -q /tmp/awscliv2.zip -d /tmp
```

```bash
sudo /tmp/aws/install
```

For an ARM/Graviton instance use AWS’s `awscli-exe-linux-aarch64.zip` instead; the recommended
`t3.small` is x86_64.

### Mac AWS CLI crashes with a Python/`pyexpat` error

This was observed during the audit. Do not debug it during a deletion emergency. Use EC2 or reinstall
official AWS CLI v2 exactly as shown in the emergency fallback, then require `aws --version` to pass.

### EC2 root disk fills

The data should stream and never land on disk. Check for accidentally redirected object content,
enormous debug logs, or package caches:

```bash
df -h /
```

Do not delete a log before uploading it to `recovery-manifests/`. Increase the EBS volume if necessary
or rotate uploaded logs; never redirect the dataset to `$HOME`.

## 5. Copy exits nonzero

Rclone exit nonzero means at least one unresolved error in that pass; it does not roll back completed
objects.

Inspect only the relevant log tail:

```bash
tail -n 200 "$RECOVERY_DIR/copy-all.log"
```

Count current error lines for orientation:

```bash
grep -En 'ERROR|Failed to|corrupt|couldn.t' "$RECOVERY_DIR/copy-all.log" | tail -n 200
```

Fix the cause and rerun the same `copy`. Because the guide uses size-only resume, existing same-size
objects skip quickly. After a zero pass, full-read verification is what proves content.

Do not use `--ignore-errors` to declare success. Do not clear the destination between retries.

## 6. Inventory/check mismatch

### Destination appears to have extra objects

Make sure comparison targets `aws:<bucket>/volume`, not the bucket root containing
`recovery-manifests/`. Versioning does not make old versions appear in ordinary current-key listings,
but S3 console object totals/Storage Lens can include noncurrent versions and incomplete multipart
parts.

If real current extras exist under `volume/`, preserve and inventory them. They might come from an
earlier pass/source state. Never use `sync` to erase them during disaster recovery.

### Counts match but bytes differ

Find different paths with `rclone check --size-only --combined`. Recopy each from source with
`copyto`, then rerun the whole non-deleting pass. A zero-byte directory marker can differ from a file;
classify it before action.

### Sizes match but hashes differ

Multipart ETags are not whole-object MD5s. Check whether rclone exposes a common MD5 metadata hash. If
hash comparability is unclear, use `rclone check --download`, which compares byte streams rather than
ETag strings. A `*` from the **download** check is a real content difference; `!` is a read/check
error that must be resolved before deciding whether bytes differ.

For a confirmed content difference during the emergency copy, copy the relative key exactly as it
appears after the status marker in `check-download-combined.txt`. Do not include a leading slash:

```bash
read -r -p "Exact differing relative key: " BAD_KEY
```

Force only that object to be read from RunPod and written as the current AWS version:

```bash
rclone copyto "runpod:4hzrwzk8ja/${BAD_KEY}" "aws:${AWS_BUCKET}/volume/${BAD_KEY}" --ignore-times --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$RECOVERY_DIR/forced-recopy.log" --log-level INFO -P
```

Store the path in an exact raw file list, which safely handles spaces and rclone filter characters:

```bash
printf '%s\n' "$BAD_KEY" > "$RECOVERY_DIR/forced-recopy-key.txt"
```

Prove that one object by reading both byte streams again:

```bash
rclone check runpod:4hzrwzk8ja "aws:${AWS_BUCKET}/volume" --download --files-from-raw "$RECOVERY_DIR/forced-recopy-key.txt" --checkers 1 --combined "$RECOVERY_DIR/forced-recopy-check.txt" --log-file "$RECOVERY_DIR/forced-recopy-check.log" --log-level INFO
```

Require exit zero and an `=` line for that key, then repeat the complete Gate D check. AWS bucket
versioning retains the replaced destination version. If the forced copy itself fails, preserve both
versions/logs and diagnose the source read or destination write; do not declare the same-size object
valid.

### Source changes while copying

Training is stopped under the stated scenario, but API or human writes could still occur. Run two
whole-copy passes to zero, generate the source inventory after the last copy, then rerun the copy and
inventory. Record the UTC interval. Versioning on AWS preserves overwritten destination versions.

## 7. Source disappears during transfer

Stop retry storms and protect the partial AWS destination.

1. Upload all local copy logs/manifests immediately.
2. Record the last successful source access UTC time.
3. Run `rclone size` on the AWS `volume/` prefix.
4. Compare any completed source inventory with destination inventory.
5. Produce lists of known present and missing paths.
6. Copy the AWS bucket to a second independent destination if affordable.
7. Contact RunPod with volume ID, account, datacenter, timestamps, and errors.
8. Classify missing artifacts using
   [`../PERSPECTIVE_2_AFTER_DELETION/06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md`](../PERSPECTIVE_2_AFTER_DELETION/06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md).

Never discard a partial copy. Checkpoints/manifests/code may be fully recoverable even if a dataset
prefix is incomplete.

## 8. Reverse restore failures

### New RunPod volume runs out of space

Stop the copy; completed files remain. Increase the volume size in RunPod (volumes can grow, not
shrink), wait for the new capacity, and rerun. If symlink materialization caused the estimate to
explode, use the POSIX-aware exclusions and rebuild links instead of uploading duplicated view bytes.

Do not delete arbitrary prefixes to make room.

### Wrong new volume ID received files

Stop immediately, record both volume IDs, and inspect both. Do not issue bulk delete—RunPod does not
support `DeleteObjects` anyway, and the wrong volume may contain unrelated state. Create/identify a
clean target and rerun only after human verification.

### AWS objects were moved to Glacier/Deep Archive

Archived objects must be restored to temporary readable copies before rclone can GET them. Initiate
an S3 restore for all needed keys, wait for completion (hours for Deep Archive), choose a retention
window long enough for transfer and verification, and account for restore requests/retrieval/temp-copy
cost. Do not start the RunPod billing clock until readable copies are ready.

### Reverse check reports missing excluded paths

In POSIX-aware mode the six link views are intentionally absent from the S3-to-S3 comparison. Ensure
the exact same exclude list was used for copy and check. Their **membership** is validated after POSIX
link reconstruction with `rebuild_links_from_inventory.py`.

### Reverse full-read check finds a same-size content difference

Set `BAD_KEY` to the exact relative key from the reverse combined report, then force-copy only that
key from AWS into the new RunPod volume:

```bash
read -r -p "Exact differing relative key: " BAD_KEY
```

```bash
rclone copyto "aws:${AWS_BUCKET}/volume/${BAD_KEY}" "newrunpod:${NEW_RUNPOD_VOLUME_ID}/${BAD_KEY}" --ignore-times --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$RESTORE_DIR/forced-recopy.log" --log-level INFO -P
```

Repeat the applicable POSIX-aware or literal `rclone check --download`. If the key is inside one of
the six intentionally excluded link views, do not upload it in POSIX-aware mode; validate/rebuild its
link membership on the attached Pod instead.

## 9. Filesystem/symlink failures on the restored Pod

### Link points at the old path but is broken

The old absolute paths should still begin `/workspace`. Check whether the corresponding raw/chunk
target exists and is decodable. Do not rewrite link text to a different root unless the intended
volume layout changed. Restore the target first, then rerun link check.

### Link path is a regular file

This is expected after literal object restore. Determine whether it is:

- a real video with expected size/frame count (scientifically usable materialization);
- a tiny text object containing the old link target (not usable as video);
- a zero-byte directory marker;
- a collision with unrelated content.

Do not automatically replace regular files. The reconstruction helper refuses collisions by design.
Quarantine, checksum, and retain AWS before any conversion.

### Helper says missing target

Use the inventory path and expected target mapping it prints. For SSv2 the target should be the raw
`.webm`; for EGO4D tiny it should be the full split `.mp4`. A missing target means the unique-byte
corpus is incomplete. Recopy/rebuild that target; creating a dangling link only hides the problem.

### Counts differ by a few files

Do not accept “close enough.” Compare exact path sets against the emergency CSV and manifests. Common
causes are hidden partial files, stale subset entries, a missing source target, or a new regenerated
selection. Each changes provenance.

## 10. Git and mode failures

### `git fsck` reports corrupt/missing objects

Archive the recovered tree first. Compare `.git/objects` with AWS. Fetching from GitHub can replace
pushed objects but not remote-only commits. Save refs, reflogs, stashes, worktree files, and status;
then repair in a clone/quarantine, not in place.

### Every shell script lost execute permission

Use `restore_git_modes.py`, which applies modes from the Git index without replacing content. Files
not tracked by Git need their old mode from an archive/inventory or must be reviewed individually.

### `git pull` would overwrite local changes

Stop. The dirty Pod tree is evidence. Save a patch, untracked archive, refs, and bundle; compare with
the dirty Mac checkout; merge deliberately. Never reset or clean to make setup faster.

## 11. ML validation failures

### Decord cannot open a restored video

Check size, `file`, the AWS/source download check, and FFmpeg decode. A tiny link-text object often
shows as ASCII rather than WebM/MP4. Recopy the exact source object or restore the raw target and link.

### EGO4D file is not 48 frames / ~12 fps / shorter side 256

It is not an exact project chunk. Do not use it for exact provenance. Restore the correct object or
rederive with the locked FFmpeg recipe, labeling it new derivation.

### Encoder smoke redownloads or reports another revision

Confirm `HF_HOME` and `--hf-cache-dir` are `/workspace/hf_cache`, alias is correct, and output revision
equals the pinned 40-character SHA. Authenticate to gated DINOv3 if necessary. Never bypass with
`main`.

### Whitening envelope rejection

Read the mismatch field. Wrong encoder, dataset fingerprint, transform seed, clip budget, preprocessing
version, epsilon contract, or tensor metadata is a real scientific mismatch. Find the right restored
file or compute a new envelope under a new path; do not edit metadata to make it load.

### Checkpoint resume rejection

Use the originating guide’s full flags. Architecture width/slots/blocks, encoder, dataset, whitening,
reconstruction mode, sampler/RNG, and optimizer all matter. Do not use transfer/legacy/reset flags
unless a new experiment plan explicitly authorizes the changed semantics.

### CUDA out of memory in resource preflight

Verify exact GPU, no competing process, batch size, encoder precision, frame microbatch, attention
implementation, architecture, and cache. Follow the guide’s declared tuning/failure policy. Do not
silently reduce a scientific batch or change geometry and call it the same arm.

### W&B works online but checkpoint is missing

This is expected: the training code does not upload checkpoint bytes. W&B’s path/SHA is evidence of
what existed, not a recovery source.

## 12. Unexpected AWS cost

Check, in order:

- EC2 instance still running;
- EBS volume or public IPv4 retained;
- S3 current and noncurrent version bytes;
- incomplete multipart uploads;
- repeated PUT/LIST/GET/checksum/Inventory requests;
- storage-class transition/retrieval/minimum-duration charges;
- internet egress during restore;
- tax.

Budgets alert but do not stop charges. Terminate temporary compute after uploading logs. Do not delete
the only bucket as a cost-control reflex. If money is genuinely unavailable, use AWS support/Free-plan
timeline and migrate a verified second copy before closure.

## 13. Evidence package for support

Provide support only non-secret material:

- account email/ID through the authenticated support form, not public chat;
- RunPod volume ID and datacenter;
- AWS bucket ARN and region (not access keys);
- UTC timestamps;
- exact endpoint and operation name;
- sanitized client version and error text;
- key path or prefix, object size, request ID if present;
- copy/check logs with authorization headers and secrets removed;
- proof of billing/deletion notice.

Never send RunPod S3 secret, AWS secret, W&B/HF token, EGO4D credential, `.env`, or full credential
files.
