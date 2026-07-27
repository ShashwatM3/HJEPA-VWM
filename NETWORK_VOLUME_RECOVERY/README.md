# HJEPA-VWM network-volume recovery decision hub

Audit date: **2026-07-23 (Asia/Kolkata)**. Provider, account, pricing, dataset-access, and tooling
claims were checked against official or project-primary sources on that date.

This package deliberately separates two different disasters. Do not blend their instructions:

1. [`PERSPECTIVE_1_BEFORE_DELETION/`](PERSPECTIVE_1_BEFORE_DELETION/README.md) is for a volume that
   still exists. Its job is to rescue every readable byte before RunPod terminates it.
2. [`PERSPECTIVE_2_AFTER_DELETION/`](PERSPECTIVE_2_AFTER_DELETION/README.md) is for a volume that is
   already gone. Its job is either to restore a surviving backup or build a new, honest derivation
   from the Mac, GitHub, W&B, licensed datasets, and model repositories.

The old volume identity is:

| Field | Value |
|---|---|
| RunPod network-volume ID / S3 bucket | `4hzrwzk8ja` |
| RunPod datacenter / signing region | `US-MO-1` / `us-mo-1` |
| RunPod S3 endpoint | `https://s3api-us-mo-1.runpod.io` |
| Pod mount point | `/workspace` |
| Mapping | `s3://4hzrwzk8ja/x` ↔ `/workspace/x` |

## Decide which perspective applies

Use the RunPod console first. Open **Storage → Network Volumes** and look for ID `4hzrwzk8ja`.

| Observation | Meaning | Go here |
|---|---|---|
| The volume row exists | Bytes may still be readable; the deletion warning is not deletion | [Perspective 1 immediately](PERSPECTIVE_1_BEFORE_DELETION/README.md) |
| The row is absent, but a properly authenticated S3 `HeadBucket`/top-level list succeeds | The volume still exists even if the UI view was misleading | [Perspective 1 immediately](PERSPECTIVE_1_BEFORE_DELETION/README.md) |
| `AccessDenied`, `InvalidAccessKeyId`, or signature mismatch | Authentication is broken; this does **not** prove deletion | Fix the key with [shared troubleshooting](SHARED_REFERENCE/04_TROUBLESHOOTING_ALL_PATHS.md#3-runpod-source-errors), then decide |
| The row is absent and RunPod support confirms termination, or a valid S3 key returns `NoSuchBucket` | RunPod documents terminated storage as unrecoverable | [Perspective 2](PERSPECTIVE_2_AFTER_DELETION/README.md) |

Do not spend the deletion window building a perfect inventory before copying. Perspective 1 copies
irreplaceable prefixes first, then the entire bucket, then inventories and verifies it.

## What “recovery” can honestly mean

These labels are used consistently throughout the package:

| Label | Meaning |
|---|---|
| **Exact object restoration** | The same object key and bytes came from a verified backup. |
| **Exact filesystem reconstruction** | The same bytes plus the required path, symlink target, and tracked executable mode were reconstructed and audited. S3 alone does not prove this. |
| **Functionally equivalent derivation** | Current code and science gates pass, but at least one byte stream, software environment, selection, or generated artifact differs from the lost volume. |
| **Fresh start** | The project can launch new runs, but historic continuation state is absent. |
| **Irrecoverable** | No surviving source contains the information. A label, SHA, W&B metric, or manifest is not a substitute for missing model/video bytes. |

If the old volume is deleted and no byte backup exists, the following cannot be regenerated:

- training checkpoint tensors, optimizer state, sampler/RNG state, and exact-resume position;
- dirty or unpushed Pod-only Git state, stashes, unreachable Git objects, and ignored Pod-only files;
- unsynced W&B offline queues, original stdout/stderr logs, and old preflight evidence;
- exact generated EGO4D chunk bytes unless the exact inputs and encoding environment survived;
- dataset video bytes when the owner no longer grants the user access.

W&B is not assumed to contain checkpoints. Current `train.py` uploads run provenance and optional
whitening inputs, and records a final checkpoint path/SHA, but it does not call `log_artifact()` for
the checkpoint bytes.

## Folder map

```text
NETWORK_VOLUME_RECOVERY/
├── README.md
├── VALIDATION_REPORT.md                 verified scope, tests, and honest execution limits
├── PERSPECTIVE_1_BEFORE_DELETION/
│   ├── README.md                         factor/decision overview
│   ├── 01_EMERGENCY_COPY_RUNPOD_TO_AWS.md
│   ├── 02_AWS_ACCOUNT_SECURITY_AND_COSTS.md
│   ├── 03_TROUBLESHOOTING_BEFORE_DELETION.md
│   └── 04_EMERGENCY_COPY_RUNPOD_TO_LOCAL_DISK.md
├── PERSPECTIVE_2_AFTER_DELETION/
│   ├── README.md                         factor/decision overview
│   ├── 01_SURVIVOR_AUDIT_AND_RECOVERY_CLASSIFICATION.md
│   ├── 02_RESTORE_A_VERIFIED_AWS_BACKUP.md
│   ├── 02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md
│   ├── 03_REBUILD_FROM_GROUND_UP.md
│   ├── 04_REBUILD_SSV2.md
│   ├── 05_REBUILD_EGO4D.md
│   ├── 06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md
│   ├── 07_VALIDATE_EXPERIMENT_READINESS.md
│   └── 08_TROUBLESHOOTING_AFTER_DELETION.md
├── SHARED_REFERENCE/
│   ├── 01_FORENSIC_VOLUME_MAP.md
│   ├── 02_CODE_REFERENCED_PATH_CATALOG.md
│   ├── 03_OFFICIAL_SOURCES.md
│   └── 04_TROUBLESHOOTING_ALL_PATHS.md
├── scripts/
├── templates/
└── tests/
```

The shared forensic map is code-derived; it is not a claim that the inaccessible bucket was listed.
Only Perspective 1's authenticated live inventory can discover unreferenced personal files.

## Current project identity used by this audit

| Item | Audited value |
|---|---|
| Local branch | `phase1-v0.2-frozen-encoder` |
| Remote base commit | `771cbba077d9f846bdf7a7dd48e12dbf29d54b49` |
| GitHub | `https://github.com/ShashwatM3/HJEPA-VWM.git` |
| V-JEPA2 revision | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` |
| SigLIP2 revision | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` |
| DINOv3 revision | `5931719e67bbdb9737e363e781fb0c67687896bc` |
| Transformers | `4.57.6` |

The Mac worktree is dirty and contains untracked experiment records. Therefore the remote commit is
only a base. The after-deletion guide snapshots the Mac's staged binary patch, unstaged binary
patch, and untracked files before using GitHub; blindly cloning only the remote would omit current
work and a single combined patch would lose the index/staging state.

## Included conservative tools

The helper scripts compare inventories, rebuild safe directory paths and known symlink views,
restore tracked Git modes, and audit a reconstructed volume. They refuse path traversal, symlink
escape, and content collisions. Run their tests from the repository root:

```bash
python3 -m unittest discover -s NETWORK_VOLUME_RECOVERY/tests -v
```

Copy [`templates/RECOVERY_SIGNOFF.md`](templates/RECOVERY_SIGNOFF.md) for each attempt and fill it
from logs and hashes, not memory.

## Safety rules that apply to both perspectives

- Never use `rclone sync`, `aws s3 sync --delete`, `rclone purge`, or a deletion flag in a rescue or
  restore.
- Never put RunPod, AWS, Hugging Face, W&B, GitHub, or EGO4D secrets in Git, chat, screenshots,
  shell command arguments, or `/workspace` backup manifests.
- Keep licensed SSv2/EGO4D data private and comply with their current terms.
- Preserve evidence before repairing it. Do not overwrite a partial backup, old manifest, old Git
  tree, or checkpoint merely to make a validation command pass.
- Do not call a recomputation “the recovered original.” Record old and new SHA-256 identities.
- No guide can guarantee every future, unwritten experiment. The final validation contract covers
  current code, all four current dataset choices, all three current encoder lanes, applicable
  historic checkpoints, provenance, resource preflight, and a disposable training smoke.

Read [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) for the exact mechanical checks, the one
environment-limited repository test result, and the operations that require your private accounts.
The evidence index is [`SHARED_REFERENCE/03_OFFICIAL_SOURCES.md`](SHARED_REFERENCE/03_OFFICIAL_SOURCES.md).
