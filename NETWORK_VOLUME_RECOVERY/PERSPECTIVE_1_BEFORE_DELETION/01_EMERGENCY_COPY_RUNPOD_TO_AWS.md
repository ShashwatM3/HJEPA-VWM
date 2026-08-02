# Emergency copy: old RunPod volume to private AWS S3

Use this while `s3://4hzrwzk8ja/` still exists. This procedure does **not** launch a RunPod Pod and
does not modify or delete the source. It streams source objects through a temporary AWS EC2 machine
into a private S3 bucket. The EC2 disk does not need to be as large as the volume.

The recommended path is EC2 rather than this Mac because:

- the transfer may involve more than 600,000 paths and must survive a closed laptop lid or browser;
- EC2 has a stable datacenter connection to the destination S3 service;
- an EC2 instance role supplies temporary AWS credentials—no AWS secret is copied to the machine;
- the AWS CLI currently installed on this Mac fails before startup due to a Python 3.14/`pyexpat`
  error.

## Stage 0 — Know the hard blocker before doing anything else

AWS requires a valid payment method even for its Free plan. For an AWS India account, an eligible
card verification can temporarily charge INR 2 and then refund it. AWS India also supports net
banking and UPI for bills, but the sign-up screen itself decides which method it will accept for
identity/payment verification. There is no legitimate command or setting that bypasses this.

If AWS will not activate in time, immediately copy the RunPod volume to a sufficiently large local or
borrowed external disk using the fallback near the end of this guide. A bucket name and endpoint
alone cannot preserve a volume after RunPod terminates it.

Do **not** do any of the following:

- delete or rename the RunPod volume;
- run a source-side `rm`, `move`, `sync`, or delete command;
- launch a Pod merely to gain filesystem access;
- revoke the RunPod S3 key until all copying and checking is complete;
- make either endpoint public;
- wait for an exact source inventory before copying high-value prefixes.

## Stage 1 — Create and secure the AWS account

Skip to Stage 2 only if a usable AWS account already exists.

1. On the Mac, open [AWS account signup](https://portal.aws.amazon.com/billing/signup).
2. Enter an email address that you control long-term. This becomes the AWS **root user** email.
3. Enter a descriptive account name such as `HJEPA recovery`.
4. Verify the email address using the code AWS sends.
5. Create a unique root password and store it in a password manager.
6. Choose **Personal** unless the account legally belongs to a business.
7. Enter the real contact address and a phone number capable of receiving SMS now.
8. Add the valid payment method the signup flow requests. For an India-issued eligible card, expect a
   refundable INR 2 verification transaction. Do not confuse this with an S3 storage charge.
9. Complete the SMS/phone identity challenge.
10. Choose **Basic Support**; paid support is unnecessary for this rescue.
11. Choose the **Free plan** if the screen offers Free versus Paid and the immediate goal is to avoid
    out-of-pocket AWS charges. Record the plan expiry date. The Free plan closes when its credits are
    exhausted or six months elapse, whichever happens first, so it is not a permanent archive unless
    upgraded before then.
12. Wait for the activation email. AWS says activation normally takes minutes but can take up to 24
    hours.
13. Sign in as **Root user** at [the AWS console](https://console.aws.amazon.com/).
14. Click the account name at the top right, choose **Security credentials**, find **Multi-factor
    authentication (MFA)**, choose **Assign MFA device**, and register a passkey or authenticator.
15. Store MFA recovery information safely. Do not create a root access key.
16. Copy the 12-digit AWS account ID from the account menu into a private note; it is used in the
    bucket name.

AWS requires root MFA within 35 days, but doing it now prevents the rescue bucket from depending on a
password alone.

## Stage 2 — Create the private destination bucket

This runbook uses AWS region `us-east-2` (US East, Ohio) and puts the old filesystem contents under
the destination prefix `volume/`. Control files go under `recovery-manifests/`.

1. In the AWS console’s top region selector, choose **US East (Ohio) us-east-2**.
2. Search for **S3**, open it, choose **Buckets**, and choose **Create bucket**.
3. Choose **General purpose** bucket.
4. Confirm region **US East (Ohio) us-east-2**.
5. Enter a globally unique name in this form, replacing the digits with the real account ID:
   `hjepa-runpod-4hzrwzk8ja-123456789012-use2`.
6. Leave **Object Ownership** at **ACLs disabled (recommended)**.
7. Leave all four **Block Public Access** boxes selected. Acknowledge the warning only if AWS asks;
   do not deselect any box.
8. Enable **Bucket Versioning**. Versioning protects overwritten destination objects; it does not
   protect the RunPod source. Old versions also cost money, so do not repeatedly overwrite objects
   for no reason.
9. Under default encryption, choose **Server-side encryption with Amazon S3 managed keys (SSE-S3)**.
   AWS already encrypts new S3 objects by default, but this explicit choice records intent and avoids
   KMS-key charges/permissions.
10. Do not enable public access, ACLs, Transfer Acceleration, Object Lock, or Requester Pays.
11. Choose **Create bucket**.
12. Open the new bucket, choose **Properties**, and verify **Bucket Versioning: Enabled**, **Default
    encryption: SSE-S3**, and **Block all public access: On**.
13. Record the exact bucket name. Every later `${AWS_BUCKET}` placeholder means this exact string.

### Add an incomplete-multipart cleanup rule

Interrupted large uploads leave billable partial parts unless cleaned up.

1. In the bucket, open **Management** and choose **Create lifecycle rule**.
2. Name it `abort-incomplete-multipart-after-7-days`.
3. Choose **Apply to all objects in the bucket** and acknowledge the scope.
4. Select only **Delete expired object delete markers or incomplete multipart uploads** if combined
   in the current UI, then specifically enable **Delete incomplete multipart uploads**.
5. Enter `7` days after initiation.
6. Do not add expiration or storage-class transitions.
7. Save the rule and verify it is Enabled.

## Stage 3 — Create a least-privilege EC2 transfer role

The instance needs permission to write and verify this one bucket. It does not receive permission to
delete objects.

1. Search for **IAM**, open it, choose **Roles**, then **Create role**.
2. For trusted entity, choose **AWS service**.
3. For use case, choose **EC2**, then **Next**.
4. Search for and select the AWS-managed policy `AmazonSSMManagedInstanceCore`. This permits browser
   terminal access through Systems Manager; it does not grant S3 data access.
5. Choose **Next**, name the role `HjepaRunpodRescueEc2Role`, and create it.
6. Open the new role, choose **Add permissions**, then **Create inline policy**.
7. Choose the **JSON** editor.
8. Replace its contents with the policy below, substituting the exact bucket name in both resource
   strings.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListOnlyTheRecoveryBucket",
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:ListBucket",
        "s3:ListBucketMultipartUploads"
      ],
      "Resource": "arn:aws:s3:::hjepa-runpod-4hzrwzk8ja-123456789012-use2"
    },
    {
      "Sid": "WriteAndVerifyRecoveryObjectsWithoutDelete",
      "Effect": "Allow",
      "Action": [
        "s3:AbortMultipartUpload",
        "s3:GetObject",
        "s3:ListMultipartUploadParts",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::hjepa-runpod-4hzrwzk8ja-123456789012-use2/*"
    }
  ]
}
```

9. Choose **Next**, name it `HjepaRunpodRescueBucketAccess`, and create the policy.
10. Back on the role Permissions tab, confirm that exactly the SSM managed policy and this inline S3
    policy are present.

## Stage 4 — Launch the temporary transfer machine

1. Search for **EC2**, open it, and confirm the console region is **US East (Ohio)**.
2. Choose **Launch instance**.
3. Name it `hjepa-runpod-rescue`.
4. Select the latest **Amazon Linux 2023 AMI** shown as an AWS-owned image.
5. Choose `t3.small`. The copy streams data, so an 8 GiB root disk is sufficient; `t3.small` gives
   more breathing room for manifests than a 1 GiB-memory micro instance. AWS credits apply if the
   account/instance is eligible, but verify the launch-page estimate.
6. For key pair, choose **Proceed without a key pair** because this guide uses Session Manager. If
   the console refuses, create a key pair and save it privately, but do not open SSH to the world.
7. Under network settings, use the default VPC and a public subnet with **Auto-assign public IP:
   Enable** so the instance can reach RunPod and AWS endpoints.
8. Create or select a security group with **no inbound rules**. Session Manager uses outbound HTTPS;
   inbound port 22 is unnecessary.
9. Leave the root volume at 8 GiB `gp3`. The dataset is never staged there.
10. Expand **Advanced details** and set **IAM instance profile** to
    `HjepaRunpodRescueEc2Role`.
11. Leave user data blank and choose **Launch instance**.
12. Wait until EC2 reports both instance status checks passed. This commonly takes a few minutes.
13. Select the instance, choose **Connect**, choose **Session Manager**, then **Connect**.

If Session Manager says the instance is not connected, wait two minutes and retry. If it still fails,
verify the instance profile, public IPv4 address, default subnet route to an internet gateway, and
outbound HTTPS. Do not solve it by opening inbound SSH to `0.0.0.0/0`.

## Stage 5 — Prepare the transfer shell

You are now in the **EC2 Session Manager terminal**, not the Mac and not RunPod. Every command through
Stage 12 runs in this terminal unless explicitly stated otherwise.

Install the terminal multiplexer and unzip utility:

```bash
sudo dnf install -y tmux unzip
```

Install rclone using its official installer:

```bash
curl https://rclone.org/install.sh | sudo bash
```

Confirm rclone starts:

```bash
rclone version
```

Confirm the AWS CLI starts (Amazon Linux includes it):

```bash
aws --version
```

Start a persistent terminal session:

```bash
tmux new -s hjepa-rescue
```

From this point, detach without stopping the copy by pressing `Ctrl-B`, releasing both keys, then
pressing `D`. Reconnect later with Session Manager and run `tmux attach -t hjepa-rescue`.

Create a timestamped control directory:

```bash
export SNAPSHOT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
```

```bash
export RECOVERY_DIR="$HOME/hjepa-recovery-$SNAPSHOT_ID"
```

```bash
mkdir -p "$RECOVERY_DIR"
```

Tell rclone not to read or write a credential file:

```bash
export RCLONE_CONFIG=/dev/null
```

## Stage 6 — Create a temporary RunPod S3 key

This part is done in the **Mac browser**, while leaving the EC2 terminal open.

1. Sign in to the [RunPod console](https://console.runpod.io/).
2. Open **Settings**.
3. Expand **S3 API Keys**. This is not the normal RunPod API-key section.
4. Choose **Create an S3 API key** and name it `temporary-volume-rescue-2026-07-23`.
5. Copy both displayed values into a password manager entry:
   - the access key/user ID, normally beginning `user_`;
   - the secret, normally beginning `rps_`.
6. The secret appears only once. Do not paste either value into this repository, chat, a screenshot,
   an EC2 tag, or an AWS policy.

Return to the **EC2 tmux terminal**. Read the access key without placing its value in shell history:

```bash
read -r -p "RunPod S3 access key (user_...): " RUNPOD_S3_ACCESS_KEY
```

Read the secret without echoing it to the screen or history:

```bash
read -r -s -p "RunPod S3 secret (rps_...): " RUNPOD_S3_SECRET_KEY; printf '\n'
```

Configure the in-memory RunPod remote one variable at a time:

```bash
export RCLONE_CONFIG_RUNPOD_TYPE=s3
```

```bash
export RCLONE_CONFIG_RUNPOD_PROVIDER=Other
```

```bash
export RCLONE_CONFIG_RUNPOD_ENV_AUTH=false
```

```bash
export RCLONE_CONFIG_RUNPOD_ACCESS_KEY_ID="$RUNPOD_S3_ACCESS_KEY"
```

```bash
export RCLONE_CONFIG_RUNPOD_SECRET_ACCESS_KEY="$RUNPOD_S3_SECRET_KEY"
```

```bash
export RCLONE_CONFIG_RUNPOD_REGION=us-mo-1
```

```bash
export RCLONE_CONFIG_RUNPOD_ENDPOINT=https://s3api-us-mo-1.runpod.io
```

```bash
export RCLONE_CONFIG_RUNPOD_FORCE_PATH_STYLE=true
```

```bash
export RCLONE_CONFIG_RUNPOD_NO_CHECK_BUCKET=true
```

Configure an in-memory AWS remote that uses the EC2 instance role:

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

Enter the destination bucket name:

```bash
read -r -p "Exact AWS bucket name: " AWS_BUCKET
```

```bash
export AWS_BUCKET
```

Check that rclone sees both in-memory remotes; this prints names, not secrets:

```bash
rclone listremotes
```

Expected lines:

```text
aws:
runpod:
```

## Stage 7 — Prove both endpoints without making a source-wide listing

Confirm that the instance role—not a stored AWS key—is active:

```bash
aws sts get-caller-identity
```

The returned ARN should include `HjepaRunpodRescueEc2Role`.

Confirm the destination bucket exists and the role can see it:

```bash
aws s3api head-bucket --bucket "$AWS_BUCKET" --region us-east-2
```

Success produces no output and exit status zero.

List only the source top level. Be patient: RunPod may compute/cache ETags on first access.

```bash
rclone lsf runpod:4hzrwzk8ja --max-depth 1 --log-file "$RECOVERY_DIR/source-top-level.log" --log-level INFO
```

Expected names include some of `ckpt/`, `checkpoints/`, `data/`, `ssv2_raw/`, `ego4d_raw/`,
`hf_cache/`, `preflight/`, `stats/`, `logs/`, `archive/`, and
`hierarchal-jepa-flow-world-model/`. Missing expected names are evidence to record, not a reason to
erase anything.

If the result is `AccessDenied` or `InvalidAccessKeyId`, fix the RunPod S3 key. If the result is
`NoSuchBucket`, the volume may already be terminated—capture the full error and contact RunPod
support immediately, but understand that RunPod’s documented policy says terminated data cannot be
recovered.

## Stage 8 — Define the safe, reusable copy commands

The functions below always use `rclone copy`, which skips matching destination files and does not
delete destination extras. `--size-only` avoids hundreds of thousands of expensive/slow metadata
requests during rescue. The later full-read check detects same-size corruption.

Define the prefix-copy function:

```bash
rescue_prefix() { local prefix="$1"; local log_name="${prefix//\//_}"; rclone copy "runpod:4hzrwzk8ja/${prefix}" "aws:${AWS_BUCKET}/volume/${prefix}" --size-only --create-empty-src-dirs --transfers 4 --checkers 8 --s3-upload-concurrency 2 --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$RECOVERY_DIR/copy-${log_name}.log" --log-level INFO --stats 30s --stats-one-line -P; }
```

Define a one-file copy function for manifests. A missing historical file will return nonzero without
affecting other files:

```bash
rescue_file() { local object="$1"; local log_name="${object//\//_}"; rclone copyto "runpod:4hzrwzk8ja/${object}" "aws:${AWS_BUCKET}/volume/${object}" --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$RECOVERY_DIR/copy-${log_name}.log" --log-level INFO -P; }
```

Define the whole-volume copy function:

```bash
rescue_all() { rclone copy runpod:4hzrwzk8ja "aws:${AWS_BUCKET}/volume" --size-only --create-empty-src-dirs --transfers 4 --checkers 8 --s3-upload-concurrency 2 --retries 20 --low-level-retries 50 --retries-sleep 15s --contimeout 30s --timeout 10m --log-file "$RECOVERY_DIR/copy-all.log" --log-level INFO --stats 30s --stats-one-line -P; }
```

Do not add `--delete-*`, do not change `copy` to `sync`, and do not add `--fast-list`. RunPod warns
about pagination on huge directories, and `--fast-list` also holds the complete key set in memory.

## Stage 9 — Rescue irreplaceable small state first

Run each command even if a previous historical prefix was absent. An absent prefix generally reports
“directory not found”; an authentication, source-read, or destination-write error must be diagnosed.

Current checkpoints:

```bash
rescue_prefix ckpt
```

Legacy/default checkpoints:

```bash
rescue_prefix checkpoints
```

Whitening tensors:

```bash
rescue_prefix stats
```

Preflight evidence/scripts:

```bash
rescue_prefix preflight
```

Historical archive:

```bash
rescue_prefix archive
```

The entire Pod Git checkout, including `.git`, ignored logs, W&B queues, and any dirty files:

```bash
rescue_prefix hierarchal-jepa-flow-world-model
```

Volume-level logs:

```bash
rescue_prefix logs
```

EGO4D selection manifest directory:

```bash
rescue_prefix ego4d_raw/manifests
```

Attempt each known single-file provenance record:

```bash
rescue_file ego4d_raw/ego4d.json
```

```bash
rescue_file ego4d_raw/video_540ss_manifest.csv
```

```bash
rescue_file ego4d_raw/manifests/selection_manifest.json
```

```bash
rescue_file data/ssv2/labels.json
```

```bash
rescue_file data/ssv2_tiny/manifest.json
```

```bash
rescue_file data/ego4d/chunk_manifest.json
```

```bash
rescue_file data/ego4d_tiny/manifest.json
```

Rescue the two tiny views. They are small in logical path count relative to the full sets, although a
RunPod S3 implementation that follows links may materialize their target videos:

```bash
rescue_prefix data/ssv2_tiny
```

```bash
rescue_prefix data/ego4d_tiny
```

At this point, upload the small logs already created so they survive even if the EC2 instance is
lost:

```bash
aws s3 cp "$RECOVERY_DIR" "s3://${AWS_BUCKET}/recovery-manifests/${SNAPSHOT_ID}/" --recursive --region us-east-2
```

## Stage 10 — Rescue the unique large datasets, then everything

SSv2 raw videos are the unique targets behind the full symlink farm and are likely smaller than the
generated EGO4D corpus, so copy them first:

```bash
rescue_prefix ssv2_raw
```

Copy the exact generated EGO4D chunk corpus next:

```bash
rescue_prefix data/ego4d
```

Now copy all of `ego4d_raw`, catching UID lists and any original downloads that remain:

```bash
rescue_prefix ego4d_raw
```

Run the whole-volume pass. It catches the full SSv2 view, model cache, all unknown/unreferenced files,
and anything missed above:

```bash
rescue_all
```

When it returns, print its exit status immediately:

```bash
printf 'last rclone exit status: %s\n' "$?"
```

Status `0` means that pass finished without unresolved transfer errors. A nonzero status does not
invalidate successful destination objects. Inspect the end of the log:

```bash
tail -n 100 "$RECOVERY_DIR/copy-all.log"
```

Retry the exact same non-deleting whole-volume function after fixing transient errors:

```bash
rescue_all
```

Because `--size-only` skips already copied matching sizes, each pass concentrates on missing/failed
objects. Continue until a full pass exits zero. Then run one additional full pass and require zero
again; this catches source objects that appeared during the first traversal.

Upload current logs again:

```bash
aws s3 cp "$RECOVERY_DIR" "s3://${AWS_BUCKET}/recovery-manifests/${SNAPSHOT_ID}/" --recursive --region us-east-2
```

## Stage 11 — Generate durable source and destination inventories

Do this only after the copy is under way or complete. A first RunPod listing can be slow because
RunPod computes and caches MD5 ETags for files created outside its S3 API.

Record source aggregate size and object count as JSON:

```bash
rclone size runpod:4hzrwzk8ja --json > "$RECOVERY_DIR/source-size.json"
```

Record destination aggregate size and object count as JSON:

```bash
rclone size "aws:${AWS_BUCKET}/volume" --json > "$RECOVERY_DIR/destination-size.json"
```

Record a standards-compliant CSV of source size and path:

```bash
rclone lsf runpod:4hzrwzk8ja -R --files-only --format sp --csv > "$RECOVERY_DIR/source-size-path.csv"
```

Record destination size and path:

```bash
rclone lsf "aws:${AWS_BUCKET}/volume" -R --files-only --format sp --csv > "$RECOVERY_DIR/destination-size-path.csv"
```

Record the modification time exposed by each backend together with size/path:

```bash
rclone lsf runpod:4hzrwzk8ja -R --files-only --format tsp --csv > "$RECOVERY_DIR/source-time-size-path.csv"
```

```bash
rclone lsf "aws:${AWS_BUCKET}/volume" -R --files-only --format tsp --csv > "$RECOVERY_DIR/destination-time-size-path.csv"
```

These are backend-observed timestamps. They can help forensics but are not proof that the original
POSIX `mtime`, `ctime`, or birth time survived the S3 layer.

Record source directory paths separately, including empty directories exposed by RunPod:

```bash
rclone lsf runpod:4hzrwzk8ja -R --dirs-only --format p --csv > "$RECOVERY_DIR/source-directory-paths.csv"
```

Record directory markers/views visible at the destination:

```bash
rclone lsf "aws:${AWS_BUCKET}/volume" -R --dirs-only --format p --csv > "$RECOVERY_DIR/destination-directory-paths.csv"
```

S3 has no intrinsic POSIX directory object. A source empty directory can therefore appear only as a
directory marker or reconstruction record. Preserve the source directory-path inventory even if the
destination directory list differs; it is the evidence needed to recreate empty directories later.

Attempt a source MD5/size/path inventory. Empty, `ERROR`, or `UNSUPPORTED` hashes are legitimate
metadata outcomes and do not prove content loss:

```bash
rclone lsf runpod:4hzrwzk8ja -R --files-only --hash MD5 --format shp --csv > "$RECOVERY_DIR/source-md5-size-path.csv"
```

Record the destination equivalent:

```bash
rclone lsf "aws:${AWS_BUCKET}/volume" -R --files-only --hash MD5 --format shp --csv > "$RECOVERY_DIR/destination-md5-size-path.csv"
```

Record the client versions and UTC timestamp without credentials:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ > "$RECOVERY_DIR/completed-at-utc.txt"
```

```bash
rclone version > "$RECOVERY_DIR/rclone-version.txt"
```

```bash
aws --version > "$RECOVERY_DIR/aws-cli-version.txt" 2>&1
```

Upload the inventories immediately:

```bash
aws s3 cp "$RECOVERY_DIR" "s3://${AWS_BUCKET}/recovery-manifests/${SNAPSHOT_ID}/" --recursive --region us-east-2
```

## Stage 12 — Prove the copy in increasing-strength order

### Gate A: aggregate count and bytes

Print both summaries:

```bash
python3 -m json.tool "$RECOVERY_DIR/source-size.json"
```

```bash
python3 -m json.tool "$RECOVERY_DIR/destination-size.json"
```

The source and `volume/` destination counts and bytes must match. A destination count larger by the
number of `recovery-manifests/` files means the wrong destination root was measured; measure only
`aws:${AWS_BUCKET}/volume`.

### Gate B: path and size comparison

Run rclone’s read-only size check and save a combined report:

```bash
rclone check runpod:4hzrwzk8ja "aws:${AWS_BUCKET}/volume" --size-only --checkers 8 --combined "$RECOVERY_DIR/check-size-combined.txt" --log-file "$RECOVERY_DIR/check-size.log" --log-level INFO
GATE_B_STATUS="$?"
printf '%s\n' "$GATE_B_STATUS" > "$RECOVERY_DIR/check-size-exit-status.txt"
test "$GATE_B_STATUS" -eq 0
```

Exit zero and only lines beginning `=` in the combined report are required. Meanings are:

- `=` identical under that check;
- `+` missing on destination;
- `-` extra on destination;
- `*` different;
- `!` read/check error.

If this gate fails, rerun `rescue_all`, then repeat Gate A and Gate B.

### Gate C: remote hash check

Run the ordinary non-mutating check:

```bash
rclone check runpod:4hzrwzk8ja "aws:${AWS_BUCKET}/volume" --checkers 4 --combined "$RECOVERY_DIR/check-hash-combined.txt" --log-file "$RECOVERY_DIR/check-hash.log" --log-level INFO
GATE_C_STATUS="$?"
printf '%s\n' "$GATE_C_STATUS" > "$RECOVERY_DIR/check-hash-exit-status.txt"
```

This compares common hashes where the backends expose them. Multipart-upload ETags are not always a
whole-object MD5, so Gate C is stronger than size alone but is not universally conclusive.

### Gate D: full byte-stream comparison

After all copy passes are complete, download-read both remotes on the fly and compare their content:

```bash
rclone check runpod:4hzrwzk8ja "aws:${AWS_BUCKET}/volume" --download --checkers 4 --combined "$RECOVERY_DIR/check-download-combined.txt" --log-file "$RECOVERY_DIR/check-download.log" --log-level INFO --stats 30s --stats-one-line -P
GATE_D_STATUS="$?"
printf '%s\n' "$GATE_D_STATUS" > "$RECOVERY_DIR/check-download-exit-status.txt"
test "$GATE_D_STATUS" -eq 0
```

This does not store another full copy on EC2, but it reads every byte from both endpoints and can take
as long as another transfer. It is the strongest available object-content proof when hashes are
missing. If the deletion deadline is imminent, finish all copy passes before starting it.

If Gate D reports a `*` for a same-size object, `rescue_all` will intentionally keep skipping that
object because it uses `--size-only`. Follow **Troubleshooting → Sizes match but hashes differ** to
force-copy only the named key with `copyto --ignore-times`, then repeat Gate D. Do not add
`--ignore-times` to the whole-volume function: that would needlessly rewrite hundreds of thousands
of already verified objects and create noncurrent S3 versions.

Upload final proof reports:

```bash
aws s3 cp "$RECOVERY_DIR" "s3://${AWS_BUCKET}/recovery-manifests/${SNAPSHOT_ID}/" --recursive --region us-east-2
```

In the S3 console, open `recovery-manifests/<SNAPSHOT_ID>/` and visually confirm the JSON, CSV, log,
version, and combined-check files exist.

## Stage 13 — Close the emergency safely

Only after at least Gate A and Gate B pass, and preferably Gate D:

1. In RunPod **Settings → S3 API Keys**, revoke the temporary rescue key.
2. In the EC2 terminal, erase the in-memory source credential variables:

```bash
unset RUNPOD_S3_ACCESS_KEY RUNPOD_S3_SECRET_KEY RCLONE_CONFIG_RUNPOD_ACCESS_KEY_ID RCLONE_CONFIG_RUNPOD_SECRET_ACCESS_KEY
```

3. Detach tmux with `Ctrl-B`, then `D`.
4. In EC2, select `hjepa-runpod-rescue`, choose **Instance state → Terminate instance**, and confirm.
   Termination deletes the temporary root disk and stops compute/EBS charges for that instance.
5. Do **not** delete the S3 bucket, `volume/`, manifests, EC2 role, or old RunPod volume merely because
   the first copy appears present.
6. In AWS Billing, inspect current spend and credit balance. Set calendar reminders at 30, 60, 90,
   120, and 150 days to decide whether to upgrade the Free plan or move the backup. Never let the
   six-month Free-plan closure delete the only copy.
7. Rotate any Hugging Face, W&B, AWS, GitHub, EGO4D, or other credential discovered under
   `volume/`, even though the bucket is private.

## Fallback A — Run the bridge on the Mac

Use this only if EC2/Session Manager is unavailable and the Mac can stay powered, awake, online, and
connected for the entire rescue. The actual copy commands and verification gates are identical.

Repair the broken AWS CLI with the official AWS CLI v2 macOS package:

```bash
curl https://awscli.amazonaws.com/AWSCLIV2.pkg -o /tmp/AWSCLIV2.pkg
```

```bash
sudo installer -pkg /tmp/AWSCLIV2.pkg -target /
```

```bash
aws --version
```

Install rclone:

```bash
brew install rclone
```

```bash
rclone version
```

Create a dedicated temporary IAM user with the same bucket policy if the account has no configured
human CLI access. Create exactly one CLI access key, run `aws configure --profile hjepa-rescue`, and
delete that IAM access key immediately after the transfer. This is less safe than EC2’s temporary
instance credentials; never use a root access key.

Prevent sleep while the transfer terminal is open:

```bash
caffeinate -dimsu
```

Open another Terminal tab, start `tmux`, reproduce Stages 5–13 there, and add this AWS-profile
variable before configuring the `aws:` rclone remote:

```bash
export AWS_PROFILE=hjepa-rescue
```

Do not close the `caffeinate` process until the transfer finishes.

## Fallback B — Copy to a local/external disk when AWS activation is blocked

Do not improvise a one-command copy. Execute the dedicated end-to-end local runbook:

[`04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md`](04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md)

It includes disk identity, case-sensitivity and capacity gates, isolated `volume/` and manifest
roots, priority copies, two full passes, inventories, full-byte verification, credential revocation,
safe ejection, and the exact later restore route. Do not format or erase an existing disk to create
space without separately confirming that its contents are expendable.

## Why AWS DataSync is not the recommended shortcut

AWS DataSync advertises agentless Enhanced-mode transfers from supported S3-compatible sources into
S3. RunPod is not a validated provider, and the APIs do not line up cleanly: DataSync documentation
expects bucket-location and object-tag behaviors depending on task configuration, while RunPod
explicitly does not implement `GetBucketLocation` or any object-tagging operations. RunPod also lacks
bulk `DeleteObjects`. A test might work with tags disabled, but it is not a reliable, officially
validated emergency path. rclone needs only the list/head/get/multipart operations RunPod documents.
