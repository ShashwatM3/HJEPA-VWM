# AWS feasibility, account logistics, security, and cost model

Facts and prices in this chapter were checked on **2026-07-23**. AWS prices are region-, tier-, tax-,
and time-dependent. The worked numbers use a general-purpose S3 bucket in **US East (Ohio),
`us-east-2`**, and should be rechecked in the AWS calculator immediately before a later restore.

## 1. Feasibility verdict

### Can a new AWS account receive the old RunPod network volume?

**Yes, if all four conditions hold:**

1. RunPod has not terminated network volume `4hzrwzk8ja`.
2. The RunPod account can still create/use a separate S3 API key.
3. AWS accepts and activates the new account, including a valid payment method and identity check.
4. A client such as rclone copies every source object to private S3 and verifies the result.

No running RunPod Pod is required. RunPod documents direct S3-compatible access as a way to manage a
network volume without launching a Pod, and says RunPod Pod traffic has no ingress/egress fee.

### Can AWS recover it after RunPod deletes it?

**No.** AWS has no privileged connection to RunPod’s underlying disks. The endpoint exposes only the
currently existing volume. RunPod explicitly states that when unpaid storage is eventually
terminated, its data cannot be recovered.

### Does AWS S3 become the new RunPod volume?

**No.** S3 is object storage, not an attachable POSIX `/workspace` filesystem. It is the durable
intermediate backup. Later, create a new RunPod network volume in an S3-supported datacenter and copy
`s3://<aws-bucket>/volume/...` back through the new RunPod volume’s S3 endpoint. Then reconstruct and
audit symlinks/modes on a Pod.

### Does “new AWS account” mean guaranteed free credits?

No. A genuinely new AWS customer can choose the current Free or Paid account plan and is offered
$100 signup credit plus the opportunity to earn up to $100 more through console activities. AWS says
an existing or former AWS customer is ineligible for the Free plan/credits, credits cannot transfer
between accounts, and a valid payment method is mandatory. Creating duplicate identities/accounts to
evade eligibility rules is not a valid strategy.

## 2. New-account logistics for an India-based user

AWS’s official standalone-account flow requires:

- a root email address not already attached to an AWS account;
- an account name, contact name, address, and reachable phone;
- a valid payment method before signup can continue;
- possible SMS/phone identity verification;
- selection of a support plan;
- an activation email, which normally arrives in minutes but can take up to 24 hours.

For an AWS India account:

- AWS India supports Visa, Mastercard, American Express, and RuPay credit/debit cards;
- eligible-card saving/verification charges INR 2 and AWS says it is refunded in 5–7 business days;
- net banking and UPI can pay AWS India bills; UPI AutoPay is also supported;
- card/UPI automatic-payment rules and limits follow Indian regulations;
- billing is handled by the applicable AWS India seller of record and amounts/taxes can appear in
  INR;
- PAN and GST details can be added in Tax Settings when applicable.

“I have no dollars” is not itself a blocker: AWS India can bill in INR, and credits can cover eligible
usage. “I have no valid method AWS accepts” **is** a blocker to account creation. Even the Free plan
requires one.

## 3. Free plan versus Paid plan

| Property | Free account plan | Paid account plan |
|---|---|---|
| Signup credit | $100, plus up to $100 earnable | Same current credit offer for eligible new customers |
| Out-of-pocket service charges during plan | AWS says no charges accrue to the payment method under the plan; eligible use consumes credits | Pay-as-you-go beyond credits or for ineligible services |
| Service access | Selected services/features | All services/features |
| End condition | Earlier of six months or credits exhausted | Account remains open when credits end |
| What happens at end | Account closes and resource/data access is lost | Ordinary billing continues |
| Recovery window after Free-plan closure | AWS retains data for 90 days; upgrading is required to reopen/download | Not applicable |
| Credit expiry after upgrade | Remaining credits continue but expire 12 months after account creation, subject to terms | Same |

The Free plan is reasonable for an emergency rescue when cash is unavailable, but it creates a
second deletion clock. Put the six-month deadline in multiple calendars. Upgrade before expiry or
move the only copy elsewhere. Do not rely on the 90-day post-closure retention window: access during
that window requires upgrading to Paid, and after 90 days AWS erases the account content.

Joining AWS Organizations or setting up Control Tower can automatically upgrade the Free plan and
expire credits under current rules. This recovery account needs neither.

## 4. Security design used by this runbook

### Identity

- Root user: console-only break-glass identity with MFA; no access keys.
- Temporary EC2 instance: assumes `HjepaRunpodRescueEc2Role` and receives short-lived AWS
  credentials from instance metadata.
- RunPod: a separate temporary S3 API access key and secret, held only in the tmux shell environment.
- Long-term human use: set up IAM Identity Center or another federated/temporary-credential path after
  the emergency. AWS recommends temporary credentials and least privilege.

### S3 bucket

- General-purpose S3 bucket in `us-east-2`.
- ACLs disabled and all Block Public Access controls enabled.
- SSE-S3 default encryption.
- Versioning enabled so accidental overwrites retain prior versions.
- No website hosting, public policy, presigned-public workflow, Requester Pays, or cross-account
  access.
- Source files stored under `volume/`; inventories/logs under `recovery-manifests/`.
- Instance role can list, put, get, and manage its multipart uploads, but cannot delete objects.

Versioning is protection, not free storage. Every noncurrent version is billed. Rclone’s rescue copy
normally skips matching files, so repeated successful passes should not create versions for
unchanged objects. Inspect the bucket’s noncurrent bytes before enabling any lifecycle deletion.

### Dataset and credential confidentiality

SSv2 and EGO4D access/distribution terms still apply after a cloud copy. Keep the bucket private and
do not grant public or anonymous access. If a source `.env`, token, or credential was accidentally
inside `/workspace`, the whole-volume strategy intentionally preserves it rather than silently
dropping unknown state. Rotate that secret after discovery.

## 5. Exact July 2026 baseline prices used here

The AWS public price-list data for S3 `us-east-2`, fetched on 2026-07-23, reports:

| Item | Price before credits/tax |
|---|---:|
| S3 Standard storage, first 50 TB/month | $0.023 per GB-month |
| PUT, COPY, POST, or LIST | $0.005 per 1,000 requests |
| GET and other requests | $0.004 per 10,000 requests ($0.0004/1,000) |
| Internet data ingress into AWS | $0 |
| Internet data transfer out, first 100 GB/month across eligible AWS services/regions | $0 if the global free pool is unused |
| Data transfer out after that, first 10 TB/month in this region | $0.09 per GB |
| S3 Standard-IA storage | $0.0125 per GB-month, plus its minimum-size/duration and retrieval rules |
| S3 Glacier Deep Archive advertised starting storage | $0.00099 per GB-month |
| Lifecycle transition to Deep Archive | $0.05 per 1,000 objects |

RunPod’s current standard network-volume prices are:

| Tier | Price |
|---|---:|
| First 1 TB | $0.07 per GB-month |
| Additional storage beyond 1 TB | $0.05 per GB-month |
| Direct network-volume S3 API use | No pricing change according to RunPod S3 API docs |
| RunPod Pod data ingress/egress | $0 according to RunPod Pod pricing docs |

AWS prices exclude applicable taxes. EC2 transfer-host, EBS root-disk, public IPv4, S3 Inventory,
checksum Batch Operations, and archive restore operations are separate when used. The EC2 launch
screen/official calculator is the authority for the few hours of temporary compute because runtime
is unknown.

## 6. Cost equations

Let:

- `B` = logical data size in billed GB;
- `N` = number of destination objects;
- `V` = GB restored from AWS to RunPod in a single calendar month;
- `F` = unused portion of AWS’s global 100 GB/month outbound allowance, between 0 and 100;
- `C` = available AWS promotional credit.
- `T` = the billed-GB cutoff RunPod applies to its documented “first 1 TB” tier.

Approximate first-month S3 Standard cost before credits/tax:

```text
storage          = B × $0.023
initial PUTs     = ceil(N / 1,000) × $0.005
verification GET = ceil(N / 10,000) × $0.004 for each AWS GET pass
AWS ingress      = $0
```

The emergency full-download verification runs on EC2 in the same AWS region as S3, so S3-to-EC2
regional data transfer is free; request and temporary EC2 costs remain.

Approximate later AWS-to-RunPod restore egress before a waiver/credits/tax:

```text
billable outbound GB = max(0, V - F)
outbound cost        = billable outbound GB × $0.09   # while within first 10 TB tier
GET request cost     = ceil(N / 10,000) × $0.004
```

Approximate new RunPod volume monthly storage:

```text
if B <= T:  B × $0.07
if B > T:   T × $0.07 + (B - T) × $0.05
```

RunPod’s public page says “under/over 1 TB” but does not define in that pricing text whether the
billing cutoff is 1,000 or 1,024 billed GB. Use the amount shown by the RunPod creation screen for an
exact quote; do not silently choose a unit convention for a real funding decision.

### Transfer-time planning

For measured end-to-end throughput `R` in gigabits/second, the ideal lower bound for one copy is:

```text
copy hours >= B_GB × 8 / R_Gbps / 3,600
```

Examples before protocol/listing/object overhead: 500 GB at 100 Mbps is about 11.1 hours; 1,000 GB at
100 Mbps is about 22.2 hours; 1,000 GB at 1 Gbps is about 2.2 hours. Hundreds of thousands of small
objects and RunPod’s first-list ETag work can dominate, so these are lower bounds, not completion
promises. Gate D reads both endpoints fully and can take roughly another copy-duration or more. Keep
the EC2/tmux bridge alive and include its runtime/public-IPv4 cost until reports are uploaded.

Promotional credits reduce eligible AWS charges until exhausted, but never change the raw usage
measurement. The Free plan also ends once its credit balance reaches zero.

## 7. Worked estimates

The repository suggests more than 611,000 possible paths, but the live manifest determines `N` and
`B`. Using **611,000 objects** only as a planning example:

| Logical size | S3 Standard storage/month | Initial 611k PUTs | Raw first-month S3 subtotal* | Later outbound to RunPod** | New RunPod volume/month |
|---:|---:|---:|---:|---:|---:|
| 100 GB | $2.30 | about $3.06 | about $5.36 | $0 | $7.00 |
| 500 GB | $11.50 | about $3.06 | about $14.56 | about $36.00 | $35.00 |
| 1,024 GB | $23.55 | about $3.06 | about $26.61 | about $83.16 | $71.20–$71.68† |
| 2,048 GB | $47.10 | about $3.06 | about $50.16 | about $175.32 | $122.40–$122.88† |

\* Excludes GET/LIST repetitions, EC2/EBS/IPv4, tax, old versions, and promotional credits.

\** Assumes the full 100 GB outbound allowance is unused, one-month transfer, the $0.09 tier, and no
approved migration-off credit. The restore GET requests for 611,000 objects add only about $0.25;
egress bytes dominate.

† Range uses a 1,000–1,024 billed-GB interpretation of RunPod’s public “first 1 TB” threshold. The
RunPod console quote is authoritative.

At 1,024 GB, the initial $100 credit would cover roughly 4.2 months of S3 Standard storage if there
were no other usage, but the Free plan’s six-month maximum still applies. This is a planning ratio,
not a promise about credit eligibility or tax.

## 8. How to reduce later AWS egress legally

AWS currently offers eligible customers a credit for data transfer out when moving all data off AWS
or, after discussion, all data off a particular service. This is not automatic.

Before restoring the only backup to RunPod:

1. Keep the AWS account active and in good standing.
2. Open AWS Support and request **“free data transfer to move off AWS.”**
3. State the AWS region, S3 bucket, total bytes, object count, intended RunPod destination, and that
   this is a complete migration of the recovery dataset.
4. Wait for written approval **before transferring**.
5. Follow the approved byte/credit window. Current official policy gives 60 days after approval to
   complete the move.
6. Delete remaining AWS data/workloads or close the account within the approved conditions only
   after the RunPod restore is fully verified and another backup exists.

AWS says accounts with under 100 GB are expected to use the existing 100 GB/month free allowance and
are not eligible for extra credit. Approval is discretionary and covers data-transfer-out charges,
not storage, requests, compute, or tax. Never delete the AWS copy merely to qualify before the new
copy passes filesystem and science validation.

## 9. Storage-class decision

### During rescue and until full validation: S3 Standard

Keep everything in Standard while copying, checking, restoring, and diagnosing. It has immediate
access and no archive restore workflow. Request fees are small relative to risking inaccessible data.

### After a verified independent copy exists: possibly Standard-IA

Standard-IA can reduce storage from $0.023 to $0.0125 per GB-month in `us-east-2`, but it adds:

- retrieval charges;
- a 30-day minimum storage-duration charge;
- a 128 KB minimum billable object size;
- lifecycle transition requests (currently $0.01 per 1,000 for Standard-IA).

AWS announced on 2026-07-16 that objects may now transition into Standard-IA/One Zone-IA immediately
instead of waiting 30 days in Standard; that change does not eliminate Standard-IA’s own size,
duration, and retrieval economics. With hundreds of thousands of possibly tiny link/metadata keys,
calculate object-size distribution before transitioning.

### Do not blindly Deep Archive the raw object tree

Deep Archive looks like roughly $1/TB-month, but this tree’s high object count changes the answer:

- 611,000 lifecycle transitions cost about **$30.55** once at $0.05/1,000;
- every archived object carries 40 KB metadata overhead—about 24.4 GB for 611,000 objects;
- 8 KB/object of that overhead is billed at S3 Standard and 32 KB at archive rates;
- every object has a 180-day minimum storage duration;
- standard retrieval can take up to about 12 hours and bulk up to 48 hours;
- restore requests, retrieval bytes, temporary restored copies, and later internet egress cost extra;
- restoring hundreds of thousands of objects is operationally harder than restoring a few archives.

Deep Archive becomes sensible only after a new POSIX volume is validated and a **secondary**
tar-based archival representation has been created, checksummed, test-restored, and documented. A
tar archive preserves link text and Unix metadata much better than independent object keys and
reduces request count. Keep the verified object-form snapshot until the archive round-trip is proven.

Do not use One Zone-IA for the only backup; its lower resilience is a poor trade for irreplaceable
research state.

## 10. Cost controls that do and do not protect data

Do now:

1. Open **Billing and Cost Management → Budgets → Create budget**.
2. Create a monthly cost budget at a low amount such as `$1` with alerts at 50%, 80%, and 100% to an
   email you read.
3. Create a second alert around the expected monthly S3 amount plus a small margin.
4. Enable Free Tier/credit-balance notifications.
5. Review **Bills**, S3 `TimedStorage-ByteHrs`, request charges, EC2, EBS, and public IPv4 after the
   first day.
6. Terminate the EC2 transfer instance immediately after reports are uploaded.
7. Retain the incomplete-multipart abort rule.
8. Schedule plan-expiry and credit-expiry reminders outside AWS.

An AWS Budget sends alerts; it does **not** stop S3 charges or protect the bucket from Free-plan
closure. A hard “shut everything off at $X” automation can delete or block the only backup and is not
used here.

## 11. Why the transfer methods rank as they do

| Method | Verdict | Reason |
|---|---|---|
| rclone on temporary EC2 in destination region | **Recommended** | Streams endpoint-to-endpoint, resumes by rerun, no local full-size disk, instance role, explicit non-delete copy/check tools. |
| rclone on Mac | Good fallback | Same semantics, but laptop uptime/network and temporary AWS credentials are weaker. |
| RunPod S3 → local disk → AWS | Good when AWS activation is delayed | Two independent copies, but requires enough local storage and doubles transfer work. |
| AWS CLI direct endpoint-to-endpoint | Not direct | One CLI invocation has one endpoint context; it cannot natively use RunPod as source and AWS as destination without local staging/script logic. |
| `aws s3 sync` against RunPod | Avoid at this scale | RunPod explicitly warns `sync`/listing can fail on 10,000+ files or complex trees. |
| AWS DataSync Enhanced other-cloud → S3 | Not relied upon | RunPod is not validated and lacks APIs DataSync workflows can expect, including bucket-location and object-tagging behavior. |
| CloudShell as long-running bridge | Not recommended | Convenient but has session/storage constraints and is a poor home for a huge, days-long transfer. |
| Launching a RunPod Pod and `rsync` | Impossible under stated budget | Needs RunPod compute credit and does not solve AWS account setup. |

## 12. Integrity economics

S3 validates supported upload checksums in transit, but an S3 multipart ETag is not necessarily a
whole-object MD5. This is why the rescue uses several independent facts:

1. successful per-object copy with retries;
2. source/destination aggregate counts and bytes;
3. persistent key/size/hash inventories;
4. ordinary remote hash check where common hashes exist;
5. `rclone check --download`, which reads both objects and compares bytes on the fly.

At 611,000 objects, one AWS GET pass costs about $0.25 in request fees; same-region S3-to-EC2 transfer
is free. The full check’s main cost is EC2 runtime, not S3 request money. AWS S3 Batch checksum can
also compute stored-object checksums, but it is not a source-versus-destination proof and currently
adds a per-GB computation charge. Use it only as an optional later audit.

S3 Inventory can produce scheduled object reports and is useful for long retention, but the rclone
source manifest is the irreplaceable baseline because AWS Inventory knows only the destination.
