# HJEPA-VWM recovery sign-off

- Recovery operator:
- Started UTC:
- Completed UTC:
- Perspective used: 1 before deletion / 2 after deletion
- Recovery case: live rescue / A complete backup / B partial / C ground-up / D blocked lane
- Claimed outcome: exact object restore / exact filesystem reconstruction / functional derivation / fresh start
- Survivor-ledger path:
- Independent audit-copy paths:
- Old RunPod volume ID: `4hzrwzk8ja`
- Old RunPod datacenter: `US-MO-1`
- New RunPod volume ID:
- New RunPod datacenter:
- AWS bucket:
- Emergency snapshot ID:
- Restore ID:
- Restore mode: POSIX-aware / literal-object

## Old-volume disposition

- RunPod console observation and UTC:
- Authenticated S3 observation and UTC:
- RunPod support confirmation/case ID:
- Last successful old-volume read UTC:
- Deletion/termination confirmed by:
- Old source inventory available: yes / no
- If no inventory, reason:

## Object rescue proof

Mark every unavailable field `N/A` and state why; never turn absence of proof into PASS.

- Source objects:
- Source bytes:
- AWS `volume/` objects:
- AWS `volume/` bytes:
- Final copy exit status:
- Size check exit status:
- Hash check exit status:
- Full-download check exit status:
- Missing paths:
- Different paths:
- Error paths:
- Source directory-path inventory:
- Destination directory-path inventory:
- Empty-directory reconstruction disposition:

## Reverse restore proof

- Restored objects/bytes excluding link views:
- Reverse copy exit status:
- Reverse size check exit status:
- Reverse full-download check exit status:
- Link reconstruction check exit status:

## Ground-up derivation proof

- Mac Git-bundle SHA-256/verification:
- Mac index-patch SHA-256/`--index` application check:
- Mac worktree-patch SHA-256/application check:
- Mac untracked-archive SHA-256/member-list check:
- Reconstructed status versus captured Mac status:
- SSv2 official package filenames/sizes/SHA-256 manifest:
- SSv2 exact raw/train/validation/tiny counts:
- EGO4D access approval/credential-expiry record:
- EGO4D metadata/tier/selection/batch manifest identities:
- EGO4D final train/validation/tiny counts:
- Runtime `pip freeze`, FFmpeg, CUDA, and image/template evidence:
- Permanently lost artifact list/ledger location:

## Project identities

- Restored Git SHA:
- Restored Git dirty status archived at:
- SSv2 fingerprint old/new:
- SSv2 tiny fingerprint old/new:
- EGO4D fingerprint old/new:
- EGO4D tiny fingerprint old/new:
- V-JEPA2 revision/fingerprint:
- SigLIP2 revision/fingerprint:
- DINOv3 revision/fingerprint:

## Artifact disposition

Select exactly one primary disposition per row: exact-restored / topology-reconstructed / new-derivation / lost / blocked / not-applicable.

| Artifact/prefix | Primary disposition | Source | Old SHA/inventory | New SHA/inventory | Evidence/notes |
|---|---|---|---|---|---|
| SSv2 raw | | | | | |
| SSv2 full view | | | | | |
| SSv2 tiny | | | | | |
| EGO4D chunks | | | | | |
| EGO4D manifests | | | | | |
| EGO4D tiny | | | | | |
| `ckpt/` | | | | | |
| `checkpoints/` | | | | | |
| `stats/` | | | | | |
| `preflight/` | | | | | |
| `hf_cache/` | | | | | |
| repo `.git`/dirty state | | | | | |
| repo logs/W&B queues | | | | | |
| archive/logs | | | | | |

## Readiness gates

| Gate | PASS/FAIL/N/A | Evidence path | Notes / mandatory N/A reason |
|---|---|---|---|
| Mount/topology | | | |
| Automated volume audit | | | |
| Links/targets | | | |
| Manifests/counts | | | |
| Decode/geometry | | | |
| Git identity | | | |
| Runtime/CUDA | | | |
| Compile/tests/offline smokes | | | |
| Three real encoder smokes | | | |
| Dataset/run provenance | | | |
| Whitening | | | |
| Checkpoint exact-resume | | | |
| Exact resource preflight | | | |
| W&B | | | |
| Disposable end-to-end smoke | | | |

## Remaining risks and deadlines

- AWS Free/Paid plan:
- AWS plan expiry:
- AWS credit expiry/balance:
- RunPod balance notification configured:
- Next independent backup location:
- Credentials rotated:
- Known deviations/losses:

## Final declarations

- [ ] No password, token, secret key, cookie, `.env` content, or credential file was stored in the evidence package.
- [ ] No artifact is labeled exact from path/name/size alone when a stronger identity check was available.
- [ ] Every `N/A` readiness gate has a written reason and does not conceal a required current-run gate.
- [ ] The source backup/survivor remains unchanged and retained.
- [ ] A second independent backup of the reconstructed current state exists and was test-read.

I confirm that no artifact labeled “restored” or “exact” has an unresolved identity mismatch.

- Signed/date:
