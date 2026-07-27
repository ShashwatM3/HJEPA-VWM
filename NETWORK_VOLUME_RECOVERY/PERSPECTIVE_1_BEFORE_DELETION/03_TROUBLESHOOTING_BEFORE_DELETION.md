# Troubleshoot the rescue before deletion

Use this file only while old volume `4hzrwzk8ja` may still be readable. The priority is preserving
bytes, not restoring a Pod or making the old experiment runnable. Keep every successful destination
object and every log while diagnosing.

The exhaustive command-level cases are in
[`../SHARED_REFERENCE/04_TROUBLESHOOTING_ALL_PATHS.md`](../SHARED_REFERENCE/04_TROUBLESHOOTING_ALL_PATHS.md).
This file supplies the before-deletion order of operations.

## First five minutes

1. Do not delete, resize, rename, detach, or recreate the old volume.
2. Do not use `sync`, `move`, `purge`, `delete`, `--delete-*`, or a destination cleanup command.
3. In RunPod **Storage**, capture a screenshot showing the volume row, ID, datacenter, size, status,
   and warning. Record the screenshot's UTC time.
4. Make a new separate RunPod **S3 API key**. The ordinary RunPod API key will not work.
5. Test only a top-level listing. An authentication failure is not proof of deletion.
6. Start copying the irreplaceable roots from Stage 9 of
   [`01_EMERGENCY_COPY_RUNPOD_TO_AWS.md`](01_EMERGENCY_COPY_RUNPOD_TO_AWS.md) before attempting a
   complete source inventory.

Use this classification:

| Exact observation | Meaning | Immediate action |
|---|---|---|
| Volume row exists and authenticated list works | Source is alive | Start priority copies now |
| Row exists; `AccessDenied`/signature error | Credential/configuration failure | Fix key, region, endpoint, path style, and clock |
| Row absent; authenticated list works | UI/account-view issue, but bytes still exist | Copy now; open support ticket in parallel |
| Valid key returns `NoSuchBucket` and row is absent | Likely terminated or wrong ID/endpoint | Recheck identity once, contact RunPod, protect partial destinations |
| Listing times out/502s after initially working | Transient/list-scale failure | Retry a non-deleting prefix copy at lower concurrency |

## If RunPod authentication fails

Confirm each literal value:

```text
bucket/volume ID: 4hzrwzk8ja
endpoint:         https://s3api-us-mo-1.runpod.io
signing region:   us-mo-1
provider:         Other
path style:       true
```

Then check, in order:

1. The access key begins with the value RunPod displays for the S3 key and the secret is its
   one-time secret—not an ordinary API token.
2. The key belongs to the same RunPod account as the volume.
3. No saved rclone remote is overriding the environment-only remote.
4. The transfer host's UTC clock is correct. RunPod documents rejection when clock skew exceeds one
   hour.
5. Retry a shallow `rclone lsf`, not a recursive source-wide operation.

Never paste the key, secret, debug headers, or credential environment into a support ticket. Include
only the access-key ID if RunPod support explicitly requests its identifier; never include the
secret.

## If AWS signup or activation is blocked

AWS requires a valid payment method even for the Free plan. There is no technical bypass. Do not let
that consume the deletion window.

Use the external-disk fallback in the emergency guide when all of these are true:

- the old volume is still listable;
- AWS is not active or the payment screen cannot be completed;
- an APFS, exFAT, or other suitable private disk has more free bytes than the source plus safety
  headroom;
- the Mac can remain powered, awake, and connected until copy and verification finish.

Before starting, record disk identity and free space:

```bash
diskutil list
```

```bash
df -h /Volumes/REPLACE_WITH_DESTINATION_DISK
```

Do not select the macOS system volume by accident. Use the exact mounted external-disk path from
`diskutil`/Finder, then execute
[`04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md`](04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md) verbatim.

## If the EC2 browser terminal will not connect

Do not open inbound SSH to `0.0.0.0/0`. Verify:

1. the instance is running in `us-east-2` and both EC2 status checks pass;
2. `HjepaRunpodRescueEc2Role` is attached as the instance profile;
3. `AmazonSSMManagedInstanceCore` is attached to that role;
4. the subnet gives the instance outbound internet access;
5. the security group permits outbound HTTPS;
6. two minutes have elapsed for SSM registration.

If SSM still fails, launch a replacement transfer instance with the documented configuration while
the first remains untouched for later inspection. Do not spend the rescue window redesigning the
network.

## If the copy is slow, hangs, or exits nonzero

An rclone error does not undo files already copied. Preserve the destination and rerun the same
`copy` function.

1. Save the UTC time and exit status immediately.
2. Read the last 200 log lines.
3. If errors are 502/reset/timeout/list pagination failures, reduce `--transfers` from 4 to 2 and
   `--checkers` from 8 to 4.
4. Retry the exact priority prefix first.
5. Rerun the whole non-deleting copy after the priority roots succeed.
6. Require two complete zero-exit passes if the source survives long enough.

RunPod documents expensive first listings for large directories because it may calculate/cache
ETags. A progressing first list is not necessarily stuck. Do not add `--fast-list` to this rescue.

## If counts match but verification does not

Use the following hierarchy:

1. path and logical size;
2. a common real hash when both endpoints expose one;
3. `rclone check --download` for a byte-stream comparison.

Do not treat an S3 multipart ETag as a whole-file MD5. A same-size object is not proven identical
until a comparable checksum or full read passes. For a confirmed differing key, use the single-key
`copyto` procedure in the shared troubleshooting guide, retain the older AWS version, and rerun the
complete check.

## If the old source disappears during the copy

Stop recursive retry storms. The source cannot be repaired from AWS, but the partial destination may
contain the most valuable state.

1. Do not delete or restart the AWS bucket.
2. Upload the transfer logs and every completed source inventory to `recovery-manifests/`.
3. Inventory the current AWS `volume/` prefix.
4. Record the last successful source access UTC time and the first `NoSuchBucket` time.
5. Compare any captured source path list with the destination and generate explicit present/missing
   lists.
6. Open a RunPod support ticket with volume ID, datacenter, account email, warning timestamps, and
   request IDs/errors—but no secrets.
7. Switch to Perspective 2 Case B. Never call a partial destination a complete backup.

## Support evidence checklist

Keep this evidence even if support resolves the issue:

- volume ID `4hzrwzk8ja`, datacenter `US-MO-1`, endpoint, and account email;
- warning/deletion timestamps and current UTC time;
- exact operation, HTTP/status text, request ID if present, and client version;
- copy/check exit statuses and last 200 sanitized log lines;
- source and destination object/byte totals if available;
- list of affected prefixes or keys;
- confirmation that no source delete/move/sync operation was run.

Never send AWS secret keys, RunPod secrets, W&B/Hugging Face/EGO4D tokens, `.env`, cookies, or full
debug authentication traces.
