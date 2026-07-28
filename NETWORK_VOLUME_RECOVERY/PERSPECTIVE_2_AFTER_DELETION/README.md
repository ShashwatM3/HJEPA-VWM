# Perspective 2 — the old RunPod volume has been deleted

This perspective begins only after network volume `4hzrwzk8ja` is no longer readable. RunPod states
that a terminated volume cannot be recovered. The first job is therefore not “recreate everything”;
it is to identify which independent sources survived and assign an honest recovery class.

## Choose the surviving-source case

Run [`01_SURVIVOR_AUDIT_AND_RECOVERY_CLASSIFICATION.md`](01_SURVIVOR_AUDIT_AND_RECOVERY_CLASSIFICATION.md)
before creating or overwriting a new volume.

| Case | What survived | Best attainable outcome | Primary path |
|---|---|---|---|
| A | A complete AWS/local object backup plus rescue inventories | Exact object restoration; known filesystem topology can then be reconstructed | [AWS restore](02_RESTORE_A_VERIFIED_AWS_BACKUP.md) or [local-disk restore](02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md) |
| B | A partial object backup, some local files, W&B artifacts, or another Pod/disk | Exact recovery for those bytes; derive/reacquire other eligible assets; explicitly list losses | [`06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md`](06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md) |
| C | No old-volume bytes, but this Mac, GitHub, dataset access, W&B cloud runs, and model access survive | Functionally equivalent **fresh-start** volume for current experiments; no historic exact resume | [`03_REBUILD_FROM_GROUND_UP.md`](03_REBUILD_FROM_GROUND_UP.md) |
| D | No backup and an essential dataset/model entitlement is unavailable | Only the accessible subset of experiments can be rebuilt; the blocked lane remains impossible | Component guides plus an explicit blocker record |

Do not mix Cases A and C silently. If one checkpoint came from AWS and one whitening tensor was
recomputed, label them separately with their own source and SHA-256.

## Factors for evaluating the approaches

| Factor | Complete backup restore | Partial-backup recovery | Ground-up, no backup |
|---|---|---|---|
| Historic checkpoint recovery | Yes, if checkpoint objects passed rescue | Only individually surviving checkpoint bytes | No |
| Exact resume | Possible after provenance/encoder/dataset gates | Per-checkpoint; missing dependencies can still block | No; start fresh runs |
| Dataset byte identity | Yes for backed-up real files | Mixed | SSv2 source bytes may match the current official distribution; generated EGO4D chunks are a new derivation unless environment/source identity is identical |
| Old dirty Pod worktree | Yes if whole repo object tree survived coherently | Maybe | Only Mac/GitHub state; Pod-only changes are lost |
| Old logs/offline W&B | Yes if backed up | Maybe | Cloud-synced W&B only; unsynced queue lost |
| Licensing dependency | Restore remains subject to license but does not need redownload authorization to obtain its own private backup | Needed for missing assets | Required for both dataset reacquisition and gated models |
| Network transfer | AWS→RunPod size of backup | Surviving bytes plus reacquisition | Qualcomm ~19.4 GB SSv2 download; project EGO4D selection ~290 GB transient across four batches; model downloads |
| RunPod compute | Validation Pod; restore itself can use S3 API without a Pod | Validation/rebuild Pod | Pod needed for EGO4D encoding and all ML validation |
| Difficulty | Moderate | Highest; every artifact has a different disposition | Long but deterministic if access succeeds |
| Best evidence | Old/new inventories and full content checks | Per-artifact SHA/source ledger | New manifests, tool versions, fingerprints, and readiness gates; never claim old identity |

For Case A, choose the transfer source with these factors:

| Factor | Verified AWS S3 backup | Verified local/external-disk backup |
|---|---|---|
| AWS account needed | Yes | No |
| AWS restore charge | Requests plus internet egress unless allowance/approved credit covers it | None |
| Mac must stay awake during upload | No; EC2/tmux bridge persists | Yes |
| Independent-copy durability | Multi-AZ S3 durability, but Free-plan/credit expiry must be managed | One physical disk can fail or be lost; make a second copy |
| Upload bottleneck | AWS-region host to RunPod | Home/office Mac upstream bandwidth |
| Operational ease | More account/IAM setup; resilient unattended transfer | Less cloud setup; more physical-disk and uptime responsibility |
| RunPod costs after upload | New volume plus validation Pod | Same |
| Runbook | [`02_RESTORE_A_VERIFIED_AWS_BACKUP.md`](02_RESTORE_A_VERIFIED_AWS_BACKUP.md) | [`02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md`](02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md) |

## Price stakes

### Restoring an AWS backup

The dominant variable is internet egress from AWS to RunPod. Using the July 2026 baseline and
assuming the global 100 GB/month allowance is otherwise unused:

```text
estimated AWS outbound = max(0, restored_GB - 100) × $0.09
```

This applies only while the transfer remains in the first 10 TB pricing tier used by the runbook.
AWS offers eligible customers a preapproved migration-off credit; request it **before** transferring.
It is discretionary, has conditions, and does not cover storage, requests, compute, or tax.

The new RunPod volume must fit the literal restore plus headroom. Do not use a generic 300 GB size
for Case A; calculate it from the destination inventory because the old volume may be much larger.

### Restoring a local/external-disk backup

There is no AWS bill. The cash costs are the replacement RunPod volume, validation Pod, electricity,
and any disk that was purchased. The practical cost is time: the Mac must remain powered, awake,
connected to the rescue disk, and online for the full upload and byte check. A 500 GB upload over a
100 Mbit/s upstream has an ideal lower bound of about 11.1 hours per pass before protocol and
small-file overhead; verification can require another full read. Use the local restore guide's
tmux/`caffeinate` procedure and never erase the only disk after one successful upload.

### Ground-up current-project rebuild

The project's established batched EGO4D process uses a 300 GB network volume: one ~72 GB raw batch
coexists with accumulated ~50–90 GB generated chunks and the rest of the project. At RunPod's
published first-tier rate, 300 GB is approximately `$21/month` before tax and any console-specific
minimum/rounding. The volume cannot be shrunk later.

The EGO4D guide estimates ~2.5–6 hours for four download/encoding batches, plus setup and validation;
actual Pod cost is `live hourly Pod price × running hours`. GPU prices and availability are dynamic,
so record the console quote immediately before deployment. SSv2 extraction itself does not need a
GPU, but the final encoder/training gates do.

AWS is optional in a pure ground-up rebuild. It is required only if it is the surviving backup or is
chosen as the new independent backup. Dataset owners do not charge a cloud-storage fee in these
guides, but internet, disk, RunPod storage/compute, tax, and future backup costs still apply.

## Ease and recommended order

1. A complete verified backup is easiest and preserves the most. Restore it first; do not redownload
   a dataset over known-good historic bytes.
2. A partial backup is harder than either extreme because identity must be decided artifact by
   artifact. Preserve it read-only and create a ledger before filling gaps.
3. A ground-up rebuild has the clearest procedure but the weakest historic recovery. Use it only
   after proving no old byte copy exists.
4. If SSv2, EGO4D, or DINOv3 access is denied, continue rebuilding the accessible lanes. Record the
   blocked lane; do not substitute an unrelated mirror or model while keeping the old name.

## What “run any experiment” means here

At the end of Perspective 2, the strongest supportable promise is:

- every experiment implemented and documented in the audited July 2026 repository has its declared
  dataset, pinned encoder, code, cache, statistics/preflight requirements, credentials, and writable
  checkpoint directory;
- applicable historic checkpoints load and pass exact compatibility checks;
- all current tests, data/provenance gates, real encoder smokes, resource preflight, W&B identity,
  and a disposable end-to-end training run pass.

It cannot promise compatibility with future code that does not yet exist. Current code implements
Phase-1 coarse training; `FineFlow`, pixel generation, an inference sampler, and later phases are not
implemented and therefore are not volume-recovery targets.

## Execution order

1. [`01_SURVIVOR_AUDIT_AND_RECOVERY_CLASSIFICATION.md`](01_SURVIVOR_AUDIT_AND_RECOVERY_CLASSIFICATION.md)
2. Exactly one primary route:
   - [`02_RESTORE_A_VERIFIED_AWS_BACKUP.md`](02_RESTORE_A_VERIFIED_AWS_BACKUP.md), or
   - [`02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md`](02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md), or
   - [`03_REBUILD_FROM_GROUND_UP.md`](03_REBUILD_FROM_GROUND_UP.md)
3. Component work as applicable:
   - [`04_REBUILD_SSV2.md`](04_REBUILD_SSV2.md)
   - [`05_REBUILD_EGO4D.md`](05_REBUILD_EGO4D.md)
   - [`06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md`](06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md)
4. [`07_VALIDATE_EXPERIMENT_READINESS.md`](07_VALIDATE_EXPERIMENT_READINESS.md)
5. [`08_TROUBLESHOOTING_AFTER_DELETION.md`](08_TROUBLESHOOTING_AFTER_DELETION.md)

Keep the old AWS/local backup and every survivor unchanged until validation passes and a second
independent backup of the reconstructed volume exists.
