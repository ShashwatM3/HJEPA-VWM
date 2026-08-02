# Troubleshoot restoration and ground-up rebuilding after deletion

Use this only after the old volume is no longer readable. RunPod cannot recreate terminated bytes;
troubleshooting now means locating survivors, restoring them without damage, and identifying honest
new derivations.

For low-level RunPod, AWS, rclone, POSIX, Git, video, encoder, checkpoint, and W&B cases, also use
[`../SHARED_REFERENCE/04_TROUBLESHOOTING_ALL_PATHS.md`](../SHARED_REFERENCE/04_TROUBLESHOOTING_ALL_PATHS.md).

## Start by reclassifying, not retrying blindly

| Observation | Correct class | Next file |
|---|---|---|
| AWS copy has every object and rescue proof passes | Case A | [`02_RESTORE_A_VERIFIED_AWS_BACKUP.md`](02_RESTORE_A_VERIFIED_AWS_BACKUP.md) |
| Local/external-disk copy has every object and rescue proof passes | Case A | [`02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md`](02B_RESTORE_A_VERIFIED_LOCAL_BACKUP.md) |
| Some old bytes survived | Case B | [`06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md`](06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md) |
| No old bytes survived, but all required source entitlements survive | Case C | [`03_REBUILD_FROM_GROUND_UP.md`](03_REBUILD_FROM_GROUND_UP.md) |
| A dataset/model entitlement is missing | Case D for that lane | Record blocker; rebuild other lanes |

An AWS bucket containing only `recovery-manifests/`, W&B metrics, hashes, or path lists is not a byte
backup. A checkpoint path plus SHA does not contain checkpoint tensors.

## Mac survivor-evidence failures

### The audit shell variable is empty

Find the already-created directory in Finder or list only the expected Desktop names:

```bash
find "$HOME/Desktop" -maxdepth 1 -type d -name 'HJEPA_VOLUME_SURVIVOR_AUDIT_*' -print
```

Set the exact selected path; do not create a new empty directory under the old name:

```bash
export HJEPA_SURVIVOR_AUDIT="$HOME/Desktop/REPLACE_WITH_EXACT_EXISTING_DIRECTORY"
```

Then run `shasum -a 256 -c "$HJEPA_SURVIVOR_AUDIT/SHA256SUMS"` from the Mac.

### The untracked tar is empty or missing files

Compare its member list with `mac-status-porcelain-v2.txt`. Git's untracked archive intentionally
excludes ignored files. Inspect `mac-status-with-ignored.txt`; recover valuable ignored logs,
checkpoints, W&B queues, and `.pt` files separately without copying secrets into the public Git tree.

Do not regenerate the archive after changing the worktree and call it the original snapshot. Create
a new timestamped audit and retain both.

## Restore-from-AWS failures

### New volume is too small

Stop the transfer. Grow the same new volume in the RunPod console, wait for the new capacity, verify
`df -hT /workspace`, and rerun the identical non-deleting copy. RunPod allows growth, not shrinkage.
Do not remove arbitrary prefixes to make the estimate fit.

### AWS objects are archived

Restore the required S3 objects to temporary readable copies before starting RunPod compute. Wait
until every object reports a completed restore and choose a restore-retention window long enough for
transfer plus a full byte check. Deep Archive retrieval can take hours and costs requests,
retrieval, and temporary Standard-class storage.

### Symlink paths became regular files

This can be a scientifically valid materialized view, but it is not the original POSIX structure.
Inspect file size and video decodability first. Never replace regular files automatically. In
POSIX-aware mode, preserve/quarantine collisions and use the inventory reconstruction helper only
after every unique target exists.

## Git reconstruction failures in the ground-up path

### The remote no longer has the audited base commit

Use the frozen bundle, not a newer branch tip:

```bash
git bundle list-heads /workspace/recovery-input/mac-audit/HJEPA-VWM-all-refs.bundle
```

Create a repository from the bundle in a new empty path, then check out exact commit
`771cbba077d9f846bdf7a7dd48e12dbf29d54b49`. Do not overwrite a partially reconstructed repository.

### An index/worktree `git apply --check` fails

Require the exact base SHA before diagnosing:

```bash
git -C /workspace/hierarchal-jepa-flow-world-model rev-parse HEAD
```

If it is not the audited SHA, stop. If it is correct, preserve the error, both patch SHA-256 values,
Git version, and affected paths. Apply into a second clone for investigation. The index patch must
be applied first with `--index`; the worktree patch must be applied second without it. Do not use
`--reject`, three-way guessing, or hand-edits and then claim exact Mac reconstruction.

### Reconstructed `git status` differs from the captured Mac status

Diff the two porcelain-v2 files. Common causes are omitted untracked files, mode normalization,
line-ending conversion, or applying over the wrong base. Preserve both trees. A successful content
build may still be possible, but exact Mac-tree status has not passed.

## SSv2 rebuild failures

| Failure | Meaning | Safe response |
|---|---|---|
| Qualcomm account cannot access downloads | Entitlement/access blocker | Contact the publisher; do not use an unlicensed mirror |
| Not exactly two video ZIPs | Incomplete/changed package set | Re-read current official download page and instruction PDF |
| Concatenated TGZ will not list | ZIP extraction order/part set/corruption is wrong | Verify original package hashes; re-extract into a clean staging directory |
| Raw count is not exactly `220847` | Extraction incomplete, duplicate/nested layout, or package changed | Compare filenames and archive listing; do not build split links yet |
| Train/validation counts differ from `168913`/`24777` | Wrong JSON, duplicate IDs, missing targets, or wrong split name | Stop; inspect official JSON and helper error |
| Tiny count differs from `4002`/`348` | Seed/samples-per-class/code differs | Require current `make_subset.py`, seed 42, 23/2 per class |

Never manufacture missing videos, silently skip IDs, or relabel a partial set as full SSv2.

## EGO4D rebuild failures

### Credentials return expired/forbidden

EGO4D credentials expire after 14 days. Renew through the official license flow and replace only the
credentials. Do not change the frozen selection manifest or UID lists merely because a key expired.

### Raw download fills the volume

Stop before the filesystem reaches zero free bytes. The canonical workflow is one batch only:
download batch → validate all selected UIDs → chunk → validate generated outputs → record evidence →
delete only that verified raw batch. Do not download all four batches simultaneously.

### FFmpeg/chunk counts or geometry differ

Keep `.part.mp4` and failed logs as evidence, but never count them as data. Verify the exact current
command contract: four seconds, 12 fps, 48 frames, H.264, CRF 27, shorter side 256, no audio. A new
FFmpeg build may produce different bytes while meeting geometry; label it a new derivation and save
the build/version record.

### Final counts are outside the project guide's ranges

Do not relax the acceptance window. Compare selected source UIDs, batch completion records,
`chunk_manifest.json`, failed-source list, and split isolation. Reprocess only a proven incomplete
source UID; do not move a UID between train and validation.

## Hugging Face encoder failures

1. Confirm `HF_HOME=/workspace/hf_cache`.
2. Confirm `hf auth whoami` succeeds without printing the token.
3. For DINOv3, confirm the account has accepted the current gate/license.
4. Require the exact full revision from the root README.
5. Run one encoder smoke at a time.

Do not change a pinned revision to `main`, substitute a similarly named model, or copy a partial
cache directory and call it complete. If the exact revision is withdrawn, the corresponding historic
lane is blocked unless a verified cache survived.

## W&B and checkpoint misconceptions

- A W&B run page proves uploaded run metadata/history, not every local file.
- `final_checkpoint_path` and a SHA summary are references/evidence, not checkpoint bytes.
- A W&B reference artifact can point to an external object without storing it in W&B.
- An offline run can be synced only if its local `.wandb`/run directory survived.
- Treat a checkpoint as recovered only after downloading a real `.pt`, hashing it, loading it on
  CPU, and passing project provenance/resume gates.

Never create a zero-byte/dummy checkpoint or reuse an old filename for a newly trained model.

## Readiness gate failures

Fix the first failed gate in
[`07_VALIDATE_EXPERIMENT_READINESS.md`](07_VALIDATE_EXPERIMENT_READINESS.md); later success cannot
cancel an earlier identity failure.

| Failed gate | What must remain blocked |
|---|---|
| dataset fingerprint/count/decode | Any run using that dataset |
| encoder revision/fingerprint | Any run using that encoder or bound whitening/checkpoint |
| whitening envelope | Only whitened runs bound to that artifact |
| checkpoint provenance/resume | Historic resume from that checkpoint; fresh runs may remain valid |
| exact resource preflight | The specific paid run configuration |
| W&B identity | Launch as a deliberately labeled non-science smoke only |
| disposable training smoke | All paid scientific launches |

For a no-backup rebuild, the old inventory and historic checkpoint gates are explicitly `N/A—old
bytes lost`, not PASS. Current datasets, encoders, provenance, resources, W&B, and disposable
training remain mandatory.

## Do not use these apparent shortcuts

- `git reset --hard`, `git clean`, or unconditional `git pull` over survivor/recovered work;
- changing immutable model revisions to `main`;
- accepting approximate dataset counts;
- copying licensed data from an unverified public mirror;
- regenerating a whitening tensor under an old filename without new provenance;
- deleting the only AWS/local survivor after one smoke test;
- declaring “recovered” solely because training starts.

The finish condition is the signed readiness matrix plus an independent backup—not a visually
plausible `/workspace` tree.
