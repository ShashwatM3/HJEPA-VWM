# Validate filesystem and ML experiment readiness

This chapter turns “it looks restored” into explicit gates. A gate is **PASS**, **FAIL**, or **NOT
APPLICABLE**—never “probably.” Save every result below `/workspace/recovery-manifests/readiness/`.

Passing these gates supports the current repository’s documented dataset, encoder, checkpoint,
whitening, probe, and training lanes. It cannot guarantee a future experiment with new dependencies
or data that does not yet exist.

## Gate 0 — Freeze the recovery evidence

Enable pipeline failure propagation in this Pod shell so a failing validation command cannot be
hidden by a successful `tee`. Run this again after opening any new shell or tmux pane:

```bash
set -o pipefail
```

Create the directory:

```bash
mkdir -p /workspace/recovery-manifests/readiness
```

Record UTC start time:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ | tee /workspace/recovery-manifests/readiness/started-at.txt
```

Copy into a private note or `RECOVERY_SIGNOFF.md`:

- old RunPod volume ID `4hzrwzk8ja`;
- new RunPod volume ID;
- surviving-backup type and identity: AWS bucket/snapshot prefix or local disk/rescue snapshot;
- source/destination object count and bytes;
- emergency copy/check exit statuses;
- restore mode: POSIX-aware or literal-object;
- reverse-copy/check exit statuses;
- old and restored Git commit/status;
- which artifacts are restored, reconstructed, rederived, or lost.

Route-specific **PASS**:

- Case A/B restore: rescue/restore manifests and logs exist at the surviving backup and under
  `/workspace/recovery-manifests`; every claimed exact object has no unresolved missing/different/
  error result outside deliberately excluded link views.
- Case C ground-up: the two-copy Mac survivor audit verifies under `/workspace/recovery-input`, the
  survivor ledger explicitly says the old inventory is unavailable, and every new derivation has a
  build manifest. An old source-object comparison is `N/A—old volume terminated without inventory`,
  never PASS.
- Case D: use the Case C rule for rebuilt lanes and record each blocked lane. A blocked required
  lane prevents whole-project readiness even when accessible lanes pass.

## Gate 1 — Mount and top-level contract

Confirm `/workspace` is a mounted filesystem with sufficient writable capacity:

```bash
findmnt /workspace | tee /workspace/recovery-manifests/readiness/findmnt-workspace.txt
```

```bash
df -hT /workspace | tee /workspace/recovery-manifests/readiness/df-workspace.txt
```

Record top-level types:

```bash
find /workspace -mindepth 1 -maxdepth 1 -printf '%y\t%f\n' | sort | tee /workspace/recovery-manifests/readiness/top-level.tsv
```

Expected historical roots are:

```text
archive
checkpoints
ckpt
data
ego4d_raw
hf_cache
hierarchal-jepa-flow-world-model
logs
preflight
ssv2_raw
stats
```

Some may legitimately be absent if the live source manifest proves they never existed. The new
`recovery-manifests` and `recovery-input` roots are intentional recovery controls. Any other root
must be inventoried, not deleted.

Test a new noncritical write, force it to disk, hash it, and retain it as mount evidence:

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib, os
p = Path('/workspace/recovery-manifests/readiness/write-test.bin')
data = os.urandom(1024 * 1024)
with p.open('wb') as f:
    f.write(data)
    f.flush()
    os.fsync(f.fileno())
print(p, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
PY
```

**PASS:** correct mount, expected roots accounted for, free space satisfies the intended run’s
checkpoint/cache margin, and the write/fsync test succeeds.

## Gate 2 — Automated restored-volume audit

Make the recovery helpers available at `/workspace`. In a ground-up rebuild, copy them from the
reconstructed repository:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
cp NETWORK_VOLUME_RECOVERY/scripts/audit_restored_volume.py /workspace/
cp NETWORK_VOLUME_RECOVERY/scripts/rebuild_directories_from_inventory.py /workspace/
cp NETWORK_VOLUME_RECOVERY/scripts/rebuild_links_from_inventory.py /workspace/
cp NETWORK_VOLUME_RECOVERY/scripts/rebuild_ssv2_from_splits.py /workspace/
cp NETWORK_VOLUME_RECOVERY/scripts/rebuild_tiny_from_manifest.py /workspace/
```

For a historic backup restore whose repository predates this package, transfer the same five files
from the Mac by following `AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md`. Basic proxied SSH does not support
SCP/SFTP; use the guide's public-IP full-SSH path or move them through the new volume's S3 endpoint.
Then require all five files:

```bash
test -s /workspace/audit_restored_volume.py
test -s /workspace/rebuild_directories_from_inventory.py
test -s /workspace/rebuild_links_from_inventory.py
test -s /workspace/rebuild_ssv2_from_splits.py
test -s /workspace/rebuild_tiny_from_manifest.py
```

Run the read-only audit:

```bash
python3 /workspace/audit_restored_volume.py --workspace /workspace --json-out /workspace/recovery-manifests/readiness/volume-audit.json
```

Review the human summary and JSON. It reports known roots, file/link/broken-link counts, key manifests,
raw/full/tiny video counts, checkpoints, stats, cache snapshot clues, and repository state without
hashing every video byte.

**PASS:** no required manifest is silently missing, no broken known-dataset link exists, and every
count discrepancy is reconciled against the old source manifest.

## Gate 3 — Exact symlink and target contract

Record every known dataset link and target:

```bash
find /workspace/data/ssv2 /workspace/data/ssv2_tiny /workspace/data/ego4d_tiny -type l -printf '%p\t%l\n' 2>/dev/null | sort > /workspace/recovery-manifests/readiness/dataset-symlinks.tsv
```

Record broken known-dataset links:

```bash
find /workspace/data/ssv2 /workspace/data/ssv2_tiny /workspace/data/ego4d_tiny -xtype l -printf '%p\t%l\n' 2>/dev/null | sort | tee /workspace/recovery-manifests/readiness/broken-dataset-symlinks.tsv
```

The broken-link report must be empty.

For POSIX-aware restore that used inventory reconstruction, check exact membership again with the
source inventory:

```bash
python3 /workspace/rebuild_links_from_inventory.py --inventory /workspace/recovery-manifests/REPLACE_WITH_SNAPSHOT/source-size-path.csv --workspace /workspace --check
```

If the source inventory did not expose link keys, run check mode on the exact fallback helpers used
in [`06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md`](06_RECOVER_PARTIAL_BACKUPS_AND_ARTIFACTS.md).
For full SSv2 rebuilt from official split JSON:

```bash
python3 /workspace/rebuild_ssv2_from_splits.py --train-json /workspace/ssv2_raw/REPLACE_WITH_TRAIN_JSON --validation-json /workspace/ssv2_raw/REPLACE_WITH_VALIDATION_JSON --workspace /workspace --check
```

For tiny membership rebuilt from the preserved manifests:

```bash
python3 /workspace/rebuild_tiny_from_manifest.py --dataset ssv2 --manifest /workspace/data/ssv2_tiny/manifest.json --workspace /workspace --check
```

```bash
python3 /workspace/rebuild_tiny_from_manifest.py --dataset ego4d --manifest /workspace/data/ego4d_tiny/manifest.json --workspace /workspace --check
```

If a tiny manifest was lost and the subset was rederived, check the newly written manifest with the
same command but record the result as **rederived**, not exact historical membership.

For literal object restore, regular files under the old link views are allowed only when they are
real video bytes and the later dataset provenance matches. Record that the physical topology differs.

**PASS:** source membership is complete, every link resolves to the intended raw/full target, or each
documented materialized replacement is a valid identical video at the same relative dataset path.

## Gate 4 — Manifest and count consistency

Show the SSv2/tiny manifests without dumping the large label map:

```bash
python3 - <<'PY' | tee /workspace/recovery-manifests/readiness/ssv2-manifest-summary.txt
import json
from pathlib import Path
labels = json.loads(Path('/workspace/data/ssv2/labels.json').read_text())
tiny = json.loads(Path('/workspace/data/ssv2_tiny/manifest.json').read_text())
print('labels type/count:', type(labels).__name__, len(labels))
print('tiny seed:', tiny.get('seed'))
print('tiny source_root:', tiny.get('source_root'))
for split, info in tiny.get('splits', {}).items():
    print(split, info.get('count'), info.get('classes'), info.get('min_per_class'), info.get('max_per_class'))
PY
```

Show EGO4D manifest consistency:

```bash
python3 - <<'PY' | tee /workspace/recovery-manifests/readiness/ego4d-manifest-summary.txt
import json
from pathlib import Path
selection = json.loads(Path('/workspace/ego4d_raw/manifests/selection_manifest.json').read_text())
chunks = json.loads(Path('/workspace/data/ego4d/chunk_manifest.json').read_text())
tiny = json.loads(Path('/workspace/data/ego4d_tiny/manifest.json').read_text())
print('selection totals:', selection.get('totals'))
print('chunk splits:', chunks.get('splits'))
print('tiny seed/caps:', tiny.get('seed'), tiny.get('per_video_cap'))
for split, info in tiny.get('splits', {}).items():
    print('tiny', split, info.get('count'), info.get('source_videos'))
PY
```

Re-run the project’s strict EGO4D split-isolation/file-manifest validation from
`AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md` Stage 4D. The established broad expected range is 150k–190k
train chunks and 12k–25k validation chunks, but the old exact manifest/source inventory—not the
range—is the recovery authority.

**PASS:** exact old counts/membership and manifest records agree; EGO4D source UIDs never cross
train/validation; tiny subsets match their old manifest or are explicitly labeled rederived.

## Gate 5 — Decode and geometry audit

This reads a deterministic first/middle/last sample from every split with Decord:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

```bash
python3 - <<'PY' | tee /workspace/recovery-manifests/readiness/video-decode-audit.txt
from pathlib import Path
from decord import VideoReader, cpu

roots = {
    'ssv2/train': Path('/workspace/data/ssv2/train'),
    'ssv2/validation': Path('/workspace/data/ssv2/validation'),
    'ssv2_tiny/train': Path('/workspace/data/ssv2_tiny/train'),
    'ssv2_tiny/validation': Path('/workspace/data/ssv2_tiny/validation'),
    'ego4d/train': Path('/workspace/data/ego4d/train'),
    'ego4d/validation': Path('/workspace/data/ego4d/validation'),
    'ego4d_tiny/train': Path('/workspace/data/ego4d_tiny/train'),
    'ego4d_tiny/validation': Path('/workspace/data/ego4d_tiny/validation'),
}
for name, root in roots.items():
    paths = sorted([*root.glob('*.webm'), *root.glob('*.mp4')])
    assert paths, f'{name}: empty'
    picks = sorted({0, len(paths)//2, len(paths)-1})
    for i in picks:
        p = paths[i]
        vr = VideoReader(str(p), ctx=cpu(0), num_threads=1)
        assert len(vr) > 0, p
        frame = vr[0].asnumpy()
        assert frame.ndim == 3 and frame.shape[2] == 3, (p, frame.shape)
        fps = float(vr.get_avg_fps())
        if name.startswith('ego4d'):
            assert len(vr) == 48, (p, len(vr))
            assert 11.5 <= fps <= 12.5, (p, fps)
            assert min(frame.shape[:2]) == 256, (p, frame.shape)
        print(name, p.name, len(vr), f'{fps:.3f}', frame.shape)
print('deterministic decode audit OK')
PY
```

The assertion `48 frames/12 fps/shorter side 256` applies to generated EGO4D chunks. SSv2 source
geometry/frame counts vary and are validated by successful decode plus provenance.

For stronger corruption detection, run a batched audit over every video or compare every file through
the old/new full-download check. Three samples per split alone are not a checksum proof.

**PASS:** deterministic samples decode, EGO4D contract holds, and the object-level byte proof covers
the complete corpus.

## Gate 6 — Repository identity and remote-only state

Enter the exact expected path:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

Record Git state before any synchronization:

```bash
git rev-parse HEAD | tee /workspace/recovery-manifests/readiness/git-head.txt
```

```bash
git status --short | tee /workspace/recovery-manifests/readiness/git-status.txt
```

```bash
git fsck --full 2>&1 | tee /workspace/recovery-manifests/readiness/git-fsck.txt
```

```bash
git stash list | tee /workspace/recovery-manifests/readiness/git-stashes.txt
```

List ignored scientific/runtime state:

```bash
git status --short --ignored | awk '$1 == "!!" || /wandb|logs|checkpoints|archive|\.pt/' > /workspace/recovery-manifests/readiness/git-ignored-runtime.txt
```

Compare this with the Mac checkout before fetching/pulling. Preserve unpushed refs, stashes, untracked
files, and dirty patches. Do not use `reset --hard` or `clean`.

**PASS:** `git fsck` reports no corruption; the intended commit and dirty state are recorded; all
remote-only state is separately backed up or intentionally reconciled.

## Gate 7 — Runtime and CUDA

Record versions:

```bash
python3 - <<'PY' | tee /workspace/recovery-manifests/readiness/runtime.txt
import platform, sys
import torch, transformers
print('python:', sys.version)
print('platform:', platform.platform())
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda runtime:', torch.version.cuda)
print('cudnn:', torch.backends.cudnn.version())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY
```

Record NVIDIA state:

```bash
nvidia-smi | tee /workspace/recovery-manifests/readiness/nvidia-smi.txt
```

Record the complete installed environment:

```bash
python3 -m pip freeze --all > /workspace/recovery-manifests/readiness/pip-freeze.txt
```

**PASS:** CUDA is true, expected GPU is visible, package imports work, and material runtime differences
from old provenance are reviewed.

## Gate 8 — Repository compile, unit tests, and offline smokes

Run the repository’s mandatory compile set:

```bash
python3 -m py_compile config.py data.py encoders.py models.py losses.py diagnostics.py provenance.py train.py whiten_stats.py rank_probe.py drift_probe.py make_subset.py select_ego4d_uids.py chunk_ego4d.py make_ego4d_subset.py
```

Run all tests:

```bash
pytest -q 2>&1 | tee /workspace/recovery-manifests/readiness/pytest.txt
```

Run model smoke:

```bash
python3 -c "from models import smoke_test_models; smoke_test_models()" 2>&1 | tee /workspace/recovery-manifests/readiness/models-smoke.txt
```

Run the repository’s model-side encoder smoke:

```bash
python3 -c "from models import smoke_test_encoder; smoke_test_encoder()" 2>&1 | tee /workspace/recovery-manifests/readiness/models-encoder-smoke.txt
```

Run diagnostics smoke:

```bash
python3 -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()" 2>&1 | tee /workspace/recovery-manifests/readiness/diagnostics-smoke.txt
```

Run default dataloader smoke:

```bash
python3 -c "from data import smoke_test_dataloader; smoke_test_dataloader()" 2>&1 | tee /workspace/recovery-manifests/readiness/default-dataloader-smoke.txt
```

Run the synthetic Stage 0 forward/backward sanity check:

```bash
python3 train.py --stage0-only 2>&1 | tee /workspace/recovery-manifests/readiness/stage0.txt
```

**PASS:** every command exits zero. Do not interpret skipped tests caused by missing required GPU
packages as a full pass; inspect pytest’s skip report.

## Gate 9 — Real frozen-encoder revisions

Use the exact shared cache:

```bash
export HF_HOME=/workspace/hf_cache
```

V-JEPA2:

```bash
python3 encoders.py --smoke --encoder vjepa2_vitl16 --batch-size 1 --hf-cache-dir /workspace/hf_cache 2>&1 | tee /workspace/recovery-manifests/readiness/encoder-vjepa2.txt
```

SigLIP2:

```bash
python3 encoders.py --smoke --encoder siglip2_vitb16 --batch-size 1 --hf-cache-dir /workspace/hf_cache 2>&1 | tee /workspace/recovery-manifests/readiness/encoder-siglip2.txt
```

DINOv3:

```bash
python3 encoders.py --smoke --encoder dinov3_vitb16 --batch-size 1 --hf-cache-dir /workspace/hf_cache 2>&1 | tee /workspace/recovery-manifests/readiness/encoder-dinov3.txt
```

Each output must show the expected immutable revision from the
[recovery decision hub](../README.md#current-project-identity-used-by-this-audit), finite
expected-shape features,
and zero trainable encoder parameters. A smoke that downloads a different revision is a failure even
if its tensor shape matches.

**PASS:** all three current encoder lanes pass on CUDA at pinned commits.

## Gate 10 — Dataset/run provenance

Materialize a fresh no-step provenance file for each dataset needed. Start with the two tiny sets:

```bash
mkdir -p /workspace/preflight/recovery
```

```bash
python3 train.py --data ssv2_tiny --encoder vjepa2_vitl16 --hf-cache-dir /workspace/hf_cache --preflight-only --provenance-out /workspace/preflight/recovery/ssv2_tiny_vjepa2.json
```

```bash
python3 train.py --data ego4d_tiny --encoder vjepa2_vitl16 --hf-cache-dir /workspace/hf_cache --preflight-only --provenance-out /workspace/preflight/recovery/ego4d_tiny_vjepa2.json
```

For full datasets, the provenance builder inventories relative paths, resolved sizes, Decord frame
counts, and manifests. It can take substantial time; run in tmux and do not skip it before a full run:

```bash
python3 train.py --data ssv2 --encoder vjepa2_vitl16 --hf-cache-dir /workspace/hf_cache --preflight-only --provenance-out /workspace/preflight/recovery/ssv2_vjepa2.json
```

```bash
python3 train.py --data ego4d --encoder vjepa2_vitl16 --hf-cache-dir /workspace/hf_cache --preflight-only --provenance-out /workspace/preflight/recovery/ego4d_vjepa2.json
```

Repeat with the actual experiment encoder. Compare a recovered old preflight/run provenance against
the new file:

```bash
python3 train.py --compare-provenance /workspace/preflight/REPLACE_WITH_OLD.json /workspace/preflight/recovery/REPLACE_WITH_NEW.json
```

**PASS:** common-field comparison passes for exact reconstruction, or any new identity is explicitly
accepted as a new experiment rather than an exact resume.

## Gate 11 — Whitening artifacts

For every whitened experiment, inspect the exact file named in its KANBAN guide:

```bash
python3 whiten_stats.py --inspect /workspace/stats/REPLACE_WITH_EXACT_STATS.pt 2>&1 | tee /workspace/recovery-manifests/readiness/whitening-inspect.txt
```

Then run that guide’s no-step preflight with all exact flags, including `--whiten-features`,
`--whiten-stats-path`, encoder alias, dataset, expected clip count, epsilon, architecture, and seed.
The implementation intentionally rejects shape-compatible but wrong identities.

**PASS:** each needed file loads and its payload fingerprint/identity matches the exact experiment.
Non-whitened experiments mark this gate NOT APPLICABLE.

## Gate 12 — Checkpoint readability and exact-resume proof

Inventory without loading:

```bash
find /workspace/ckpt /workspace/checkpoints -type f -name '*.pt' -printf '%s\t%p\n' 2>/dev/null | sort -k2 | tee /workspace/recovery-manifests/readiness/checkpoints.tsv
```

For every checkpoint to be resumed:

1. Verify its SHA-256 against a saved W&B summary/manifest when available.
2. Read its originating KANBAN `GUIDE.md` and reconstruct **all** architecture/data/encoder/whitening
   flags exactly.
3. Run that guide’s provenance/resource preflights.
4. Create a new recovery-smoke checkpoint directory; never point the smoke at the source directory.
5. Resume for exactly one additional step with W&B disabled or an explicitly named non-science run.
6. Require successful checkpoint load, dataset/encoder/whitener identity checks, optimizer/sampler/RNG
   restore, one finite step, and a new checkpoint in the smoke directory.

Example shape only—do not paste it until every `REPLACE_...` value is taken from the exact guide:

```bash
WANDB_MODE=disabled python3 train.py REPLACE_WITH_EXACT_ORIGINATING_FLAGS --resume /workspace/ckpt/REPLACE_WITH_RUN/phase1_stepREPLACE.pt --steps REPLACE_WITH_ONE_MORE_THAN_SAVED_STEP --checkpoint-dir /workspace/ckpt/recovery_resume_smoke/REPLACE_WITH_RUN
```

Never add `--allow-legacy-checkpoint`, `--allow-dataset-transfer`, or `--reset-optimizer` to force a
pass. Each authorizes a meaningful change.

**PASS:** every checkpoint required for future work passes an isolated exact-resume step. Missing
historical checkpoints that are not needed are recorded, not ignored.

## Gate 13 — Exact resource preflight for the next experiment

Run the next KANBAN guide’s declared commands in its declared order. Its resource preflight performs
one exact forward/backward/optimizer/diagnostic recipe and writes provenance. A generic baseline
example is:

```bash
python3 train.py --data ssv2_tiny --encoder vjepa2_vitl16 --hf-cache-dir /workspace/hf_cache --resource-preflight --provenance-out /workspace/preflight/recovery/baseline_resource_preflight.json
```

The actual experiment may require a different encoder, dataset, batch size, precision, frame
microbatch, attention implementation, whitening file, reconstruction mode, latent geometry, or
checkpoint. Use the guide, not the generic baseline.

**PASS:** exact resource preflight exits zero and its reported peak memory leaves the guide’s required
margin.

## Gate 14 — W&B and non-science launch labeling

Check local W&B status without printing the key:

```bash
wandb status 2>&1 | tee /workspace/recovery-manifests/readiness/wandb-status.txt
```

If old `wandb/` offline runs survived, list before syncing:

```bash
find /workspace/hierarchal-jepa-flow-world-model/wandb -maxdepth 2 -type d -name 'offline-run-*' -print 2>/dev/null | tee /workspace/recovery-manifests/readiness/wandb-offline-runs.txt
```

Use the exact project/entity and `--require-wandb` for real launches. Name recovery-only runs clearly,
for example `Recovery validation · EGO4D data smoke · 2 steps`; do not let them masquerade as
scientific arms.

**PASS:** required W&B identity/network works and any offline queues are reconciled without duplicate
or mislabeled runs.

## Gate 15 — Disposable end-to-end train smoke

After the exact resource preflight, run a short disposable smoke in its own checkpoint directory.
For a minimal baseline data path:

```bash
WANDB_MODE=disabled python3 train.py --data ssv2_tiny --encoder vjepa2_vitl16 --hf-cache-dir /workspace/hf_cache --steps 2 --checkpoint-dir /workspace/ckpt/recovery_smoke_ssv2_vjepa2
```

For EGO4D:

```bash
WANDB_MODE=disabled python3 train.py --data ego4d_tiny --encoder vjepa2_vitl16 --hf-cache-dir /workspace/hf_cache --steps 2 --checkpoint-dir /workspace/ckpt/recovery_smoke_ego4d_vjepa2
```

These baseline smokes do not replace the exact flags required by a SigLIP2, DINOv3, whitening,
reconstruction, capacity, memory, or latent-shape arm. Run an exact two-step smoke for each distinct
configuration family before its first real launch.

**PASS:** data workers, decoder, frozen encoder, trainable modules, optimizer, diagnostics,
checkpoint write, and clean process exit all succeed with finite values.

## Gate 16 — Final launch hygiene

Before a real run:

```bash
pgrep -af 'python.*train.py|whiten_stats.py|chunk_ego4d.py' || true
```

```bash
nvidia-smi
```

```bash
df -h /workspace
```

Create a unique tmux session and unique checkpoint directory. Run only the exact current KANBAN
`GUIDE.md` command. Stop immediately on any declared guide failure condition.

## Readiness matrix

| Capability | Additional gate beyond common 0–10 |
|---|---|
| Fresh unwhitened SSv2 run | SSv2 provenance + exact encoder resource preflight + smoke |
| Fresh unwhitened EGO4D run | EGO4D manifest/isolation/provenance + exact encoder preflight + smoke |
| Fresh whitened run | Exact standalone whitening envelope + whitened no-step/resource preflight |
| Exact checkpoint resume | Checkpoint hash + exact flags + isolated one-step resume |
| Drift/rank analysis | Exact checkpoint + probe manifest + matching encoder feature cache or honest recomputation |
| New encoder lane | Not covered merely by restored cache; requires code/tests/pinned-revision preflight |
| Future new dataset/experiment | Not covered; add a new provenance and resource contract |

## Final sign-off

Recovery is stage 100% for the **current documented project** only when:

- object copy and reverse-copy proofs pass;
- topology/link targets and old dataset membership are accounted for;
- strict dataset and encoder fingerprints match old provenance where exactness is claimed;
- every required checkpoint and whitening envelope loads under its exact contract;
- all tests, three real encoder smokes, Stage 0, exact resource preflight, and disposable train smoke
  pass;
- credentials are recreated safely and W&B requirements work;
- AWS or the verified local rescue disk remains as an independent backup;
- all deviations/losses are explicitly recorded.

Write the completion time:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ | tee /workspace/recovery-manifests/readiness/completed-at.txt
```
