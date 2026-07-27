# Perspective 1 — rescue the volume before RunPod deletes it

This perspective applies only while network volume `4hzrwzk8ja` still exists. The goal is to move
every readable object to an independent destination before RunPod's deletion clock wins.

RunPod's July 2026 billing documentation says storage may be deleted when an account runs out of
funds and cannot be recovered afterward. RunPod's S3-compatible API can access the network volume
without launching a Pod, so lack of GPU/Pod credit does **not** prevent a rescue. A separate RunPod
S3 API key is still required.

## Bottom-line recommendation

Use this order:

1. If a valid AWS payment method and account activation are available in time, stream the volume
   through a temporary EC2 machine into a private, versioned S3 bucket by following
   [`01_EMERGENCY_COPY_RUNPOD_TO_AWS.md`](01_EMERGENCY_COPY_RUNPOD_TO_AWS.md).
2. If AWS signup is blocked or delayed, execute
   [`04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md`](04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md) and copy
   directly from the RunPod S3 endpoint to a sufficiently large external disk. Do not wait for AWS
   while the only copy approaches deletion.
3. Copy checkpoints, repository state, manifests, and provenance first; then unique dataset bytes;
   then the entire bucket. A partial rescue of irreplaceable state is materially better than zero.
4. Verify counts/paths/sizes, then full content while the source remains. Revoke the temporary
   RunPod S3 key only after the strongest feasible source-versus-destination check.

AWS cannot recover a volume after RunPod terminates it. Opening an AWS account is useful only if the
source is still readable or another copy already exists.

## Factors for choosing the destination

| Factor | Private AWS S3 via EC2 | External/local disk | Mac as AWS bridge |
|---|---|---|---|
| Recommended rank | 1 | 2 when AWS activation is blocked | 3 |
| Requires a running RunPod Pod | No | No | No |
| Requires RunPod balance | The existing volume must remain alive; no new Pod charge | Same | Same |
| Requires RunPod S3 key | Yes | Yes | Yes |
| Requires money/payment method now | AWS requires a valid payment method even on Free plan; eligible credits may cover usage | A disk with enough free space must already be available | AWS account plus an awake, reliable Mac |
| Source-to-destination path | RunPod → EC2 stream → S3; dataset is not staged on EC2 disk | RunPod → Mac/external filesystem | RunPod → Mac stream → S3 |
| Survives laptop sleep/outage | Yes, after EC2/tmux starts | No | No |
| Storage elasticity | S3 grows automatically | Fixed physical capacity | S3 grows automatically |
| Integrity tooling | Strong: inventories plus remote/full-download checks | Strong if the disk remains healthy and readable | Same checks, weaker host uptime |
| POSIX fidelity | Object bytes only; links/modes must later be reconstructed | Depends on destination filesystem and S3's source representation | Object bytes only |
| New deletion clock | Free-plan account closes at six months or exhausted credits unless upgraded | Hardware-loss risk; no account-expiry clock | Same AWS clock |
| Ongoing cost | S3 storage/requests; later AWS egress; short EC2/IPv4 runtime | Usually no cloud bill; disk purchase if needed | S3 plus no EC2 runtime, but laptop cost/uptime |
| Operational difficulty | Moderate, but the runbook handholds every console and command step | Lower setup, higher local-capacity risk | Highest credential and uptime burden |

Do not use a public bucket, CloudShell for a days-long bridge, `aws s3 sync` at this key count, or an
unvalidated DataSync shortcut. RunPod warns that list/sync behavior can be slow or fail for large
trees, while the runbook uses resumable, non-deleting rclone copy/check primitives.

## Price and cash stakes

Read [`02_AWS_ACCOUNT_SECURITY_AND_COSTS.md`](02_AWS_ACCOUNT_SECURITY_AND_COSTS.md) before deciding
the account plan or storage class. The useful baseline, before credits and tax, is:

- AWS S3 Standard in `us-east-2`: `$0.023/GB-month` for the first 50 TB;
- AWS PUT/COPY/POST/LIST: `$0.005/1,000` requests;
- AWS GET/other: `$0.004/10,000` requests;
- internet ingress into AWS: `$0`;
- later AWS outbound: global first 100 GB/month free if unused, then `$0.09/GB` in the first 10 TB
  tier used by this estimate;
- RunPod network volume: `$0.07/GB-month` for the first 1 TB, `$0.05/GB-month` beyond it; RunPod says
  direct S3-compatible API use does not change that pricing.

For `B` GB and `N` objects, the first raw S3 month is approximately:

```text
B × 0.023 + ceil(N / 1,000) × 0.005
```

At the repository's planning example of 611,000 objects, PUT requests are about `$3.06`; storage is
the larger recurring variable. A 500 GB copy is roughly `$14.56` for the first S3 month before EC2,
verification GETs, tax, credits, and old versions. The authenticated live inventory—not the example—
must supply the real `B` and `N`.

Eligible genuinely new AWS customers currently receive `$100` signup credit and can earn up to
another `$100`, but AWS still requires a valid payment method and does not promise eligibility to
existing/former customers. The Free plan ends at the earlier of six months or depleted credits; it
is an emergency landing zone, not permanent free storage.

## Exactness and stakes

The rescue can preserve every byte exposed as an S3 object, including unknown files. It does not by
itself prove a POSIX clone. RunPod does not document preservation of:

- symbolic-link identity/text;
- owner/group or arbitrary Unix mode bits;
- hard-link relationships;
- empty directories and all timestamps/xattrs.

This project uses six large symlink views, so Perspective 2 rebuilds them from the source inventory,
official split files, and tiny manifests. Tracked executable modes are recoverable from Git. Unknown
non-Git POSIX metadata is exact only if a POSIX-aware archive already exists inside the rescued
objects.

## Time factors

The theoretical lower bound for `B` GB over `R` Gbit/s is:

```text
hours >= B × 8 / R / 3,600
```

At 100 Mbit/s, 500 GB is at least 11.1 hours and 1,000 GB is at least 22.2 hours before listing,
small-object, retry, and verification overhead. A full byte check can take another copy-duration.
Therefore:

- begin high-value prefix copies before a whole-tree inventory;
- let the EC2 tmux session run after closing the browser;
- prioritize completing copy passes over Gate D if deletion is imminent;
- never interpret a nonzero retry pass as loss of objects already written to S3.

## Execution order

1. Choose exactly one destination immediately:
   - if AWS signup/payment activation is viable, read
     [`02_AWS_ACCOUNT_SECURITY_AND_COSTS.md`](02_AWS_ACCOUNT_SECURITY_AND_COSTS.md), then execute
     [`01_EMERGENCY_COPY_RUNPOD_TO_AWS.md`](01_EMERGENCY_COPY_RUNPOD_TO_AWS.md);
   - if AWS is blocked, execute
     [`04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md`](04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md).
2. For errors, use
   [`03_TROUBLESHOOTING_BEFORE_DELETION.md`](03_TROUBLESHOOTING_BEFORE_DELETION.md).
3. After a verified copy exists and the old volume is gone, restore from
   [AWS](../PERSPECTIVE_2_AFTER_DELETION/02_RESTORE_A_VERIFIED_AWS_BACKUP.md) or a
   [local disk](../PERSPECTIVE_2_AFTER_DELETION/02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md).

## Exit criteria

Perspective 1 is complete only when all of these are recorded:

- source and destination object counts and logical bytes;
- source and destination path/size inventories;
- source and destination directory-path inventories, with source empty directories retained as
  reconstruction evidence;
- zero unresolved copy errors on two complete non-deleting passes;
- zero missing/different keys in the size check;
- full byte-stream check result, or an explicit deletion-deadline reason why it could not finish;
- exact destination identity (AWS bucket/region/prefix or local disk/root/snapshot), RunPod volume
  ID/endpoint, tool versions, and UTC time;
- temporary RunPod S3 key revoked;
- for the AWS route, EC2 is terminated and plan/credit-expiry reminders exist; for the local route,
  the disk is safely ejected and a second independent copy is scheduled/made;
- the surviving AWS/local copy is retained until a new RunPod volume passes every science-readiness
  gate and another independent backup exists.
