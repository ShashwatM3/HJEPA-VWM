# Recover partial backups, links, caches, and experiment artifacts

Use this only after object restoration. “Rebuild” has different meanings:

- **exact reconstruction**: same bytes, membership, and provenance can be recovered;
- **scientifically equivalent reconstruction**: current project identity checks pass, although
  storage representation differs (for example, a regular video replacing a symlink);
- **new derivation**: an artifact is recomputed from surviving inputs and must receive new provenance;
- **irrecoverable**: no surviving source contains enough information.

Never label a new derivation as the old artifact.

## 1. Triage before changing anything

Run these in the recovered Pod, saving output under `/workspace/recovery-manifests/post-restore/`.

Create the report directory:

```bash
mkdir -p /workspace/recovery-manifests/post-restore
```

Record mounted capacity:

```bash
df -hT /workspace | tee /workspace/recovery-manifests/post-restore/df-workspace.txt
```

Record top-level names and types:

```bash
find /workspace -mindepth 1 -maxdepth 2 -printf '%y\t%p\n' | sort > /workspace/recovery-manifests/post-restore/top-two-levels.txt
```

Record link counts without following them:

```bash
find /workspace -type l -printf '%p\t%l\n' > /workspace/recovery-manifests/post-restore/all-symlinks.tsv
```

Record broken links:

```bash
find /workspace -xtype l -printf '%p\t%l\n' > /workspace/recovery-manifests/post-restore/broken-symlinks.tsv
```

Record video-file counts by known directory:

```bash
for d in /workspace/ssv2_raw/20bn-something-something-v2 /workspace/data/ssv2/train /workspace/data/ssv2/validation /workspace/data/ssv2_tiny/train /workspace/data/ssv2_tiny/validation /workspace/data/ego4d/train /workspace/data/ego4d/validation /workspace/data/ego4d_tiny/train /workspace/data/ego4d_tiny/validation; do printf '%s\t' "$d"; find "$d" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l; done | tee /workspace/recovery-manifests/post-restore/known-video-counts.tsv
```

The final command’s counts are orientation only; the dedicated validation chapter performs safer
semantic checks. Do not delete unexpected files to make counts match.

## 2. Recovery decision matrix

| Missing thing | Minimum surviving source | Recovery status |
|---|---|---|
| SSv2 full/tiny links | Raw videos plus source inventory or official split metadata | Exact membership and intended links can be rebuilt |
| SSv2 raw video bytes | Materialized full-view regular videos or original licensed distribution | Possibly recoverable; otherwise irrecoverable |
| SSv2 `labels.json` | Original split/label JSON | Deterministically regenerable |
| EGO4D tiny links | Full chunk corpus plus source inventory or tiny manifest | Exact membership can be rebuilt |
| EGO4D full chunks | Exact AWS objects | Exact bytes recoverable |
| Missing EGO4D chunks | Licensed raw videos plus selection/metadata | New derivation; may not be byte-identical |
| Checkpoint `.pt` | Exact AWS object or another independent copy | Exact only; cannot be recomputed |
| Whitening `.pt` | Exact object, or exact dataset+encoder+settings | Exact object recoverable; otherwise new derivation |
| HF cache snapshot | Exact object or still-available pinned revision and access | Operationally reproducible, subject to gating/network |
| Probe feature cache | Exact object or exact dataset+encoder+probe manifest | Recomputable but expensive; mark new derivation |
| Old preflight JSON/log | Exact object | Old evidence cannot be recreated; only rerun as new evidence |
| Offline W&B queue | Exact repo `wandb/` directory | Irrecoverable if never synced and not backed up |
| Dirty/unpushed Pod code | Restored worktree/`.git`, patch, bundle, or another checkout | Irrecoverable if none survives |

## 3. SSv2 reconstruction

### Case A — raw videos and source inventory survived

This is the preferred path. Required targets should exist at:

```text
/workspace/ssv2_raw/20bn-something-something-v2/<id>.webm
```

Locate the emergency inventory:

```bash
find /workspace/recovery-manifests -name source-size-path.csv -type f -print
```

Transfer `NETWORK_VOLUME_RECOVERY/scripts/rebuild_links_from_inventory.py` from the Mac to
`/workspace/rebuild_links_from_inventory.py`, then dry-run:

```bash
python3 /workspace/rebuild_links_from_inventory.py --inventory /workspace/recovery-manifests/REPLACE_WITH_SNAPSHOT/source-size-path.csv --workspace /workspace --family ssv2 --dry-run
```

Replace only the snapshot path, then require zero missing/unsafe/collision entries. Create links:

```bash
python3 /workspace/rebuild_links_from_inventory.py --inventory /workspace/recovery-manifests/REPLACE_WITH_SNAPSHOT/source-size-path.csv --workspace /workspace --family ssv2
```

Check them:

```bash
python3 /workspace/rebuild_links_from_inventory.py --inventory /workspace/recovery-manifests/REPLACE_WITH_SNAPSHOT/source-size-path.csv --workspace /workspace --family ssv2 --check
```

This recovers exact full and tiny membership even if the original split JSON is absent because the
pre-deletion inventory contains every historical relative path.

### Case B — raw videos survived, source inventory has no link keys

Use the original SSv2 train/validation JSON beneath `/workspace/ssv2_raw/labels/`. First list candidate
JSON names without changing them:

```bash
find /workspace/ssv2_raw -maxdepth 5 -type f -name '*.json' -print | sort
```

Inspect only keys/shapes, not the entire large file:

```bash
python3 - /workspace/ssv2_raw/REPLACE_WITH_TRAIN_JSON <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
x = json.loads(p.read_text())
print(type(x).__name__, len(x))
print(x[0] if isinstance(x, list) and x else list(x)[:10])
PY
```

The common official record shape is a list containing `id` and `template`. Do not guess if the
actual shape differs. Transfer and use `scripts/rebuild_ssv2_from_splits.py` with the exact two files:

```bash
python3 /workspace/rebuild_ssv2_from_splits.py --train-json /workspace/ssv2_raw/REPLACE_WITH_TRAIN_JSON --validation-json /workspace/ssv2_raw/REPLACE_WITH_VALIDATION_JSON --workspace /workspace --dry-run
```

Create the full links and `labels.json`:

```bash
python3 /workspace/rebuild_ssv2_from_splits.py --train-json /workspace/ssv2_raw/REPLACE_WITH_TRAIN_JSON --validation-json /workspace/ssv2_raw/REPLACE_WITH_VALIDATION_JSON --workspace /workspace
```

If the preserved tiny manifest exists, rebuild its exact recorded membership with the recovery
helper. Transfer `scripts/rebuild_tiny_from_manifest.py` to `/workspace`, dry-run it, then create and
check the links:

```bash
python3 /workspace/rebuild_tiny_from_manifest.py --dataset ssv2 --manifest /workspace/data/ssv2_tiny/manifest.json --workspace /workspace --dry-run
```

```bash
python3 /workspace/rebuild_tiny_from_manifest.py --dataset ssv2 --manifest /workspace/data/ssv2_tiny/manifest.json --workspace /workspace
```

```bash
python3 /workspace/rebuild_tiny_from_manifest.py --dataset ssv2 --manifest /workspace/data/ssv2_tiny/manifest.json --workspace /workspace --check
```

Only when the old tiny manifest is absent, create a **new deterministic derivation** from the
repository:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

```bash
python3 make_subset.py --data-root /workspace/data --train-per-class 23 --val-per-class 2 --seed 42
```

If an old manifest exists but is corrupt or cannot be satisfied, copy it into
`/workspace/recovery-manifests/post-restore/` with its SHA-256 before choosing to generate a new
manifest. Never overwrite the only old selection record merely to make the new subset command run.

### Case C — only materialized full-view regular videos survived

Check that the files are actual decodable videos, not tiny link-text objects:

```bash
find /workspace/data/ssv2/train -maxdepth 1 -type f -name '*.webm' -print | sort | sed -n '1,10p' | while IFS= read -r sample; do file "$sample"; done
```

Spot-decode them before attempting deduplication. If they contain the complete union of train and
validation IDs, they can seed a new raw root. Do not move or delete them in place. First allocate
enough free space, copy one split into a new quarantine raw root, verify content hashes, then add the
other split. Only after the raw union and an independent AWS copy are verified should regular view
files be replaced by links. Because that replacement is destructive, use the helper’s default
collision refusal and handle it in a separately reviewed migration—not an automatic rescue step.

If space is limited, same-filesystem hard links can avoid duplicated bytes, but hard-link behavior on
RunPod’s filesystem must be tested with a disposable file first. A hard-link transformation is not
the old symlink topology and must be documented.

### Case D — raw bytes and decodable materialized views are both absent

The dataset cannot be reconstructed from `labels.json`, manifests, checkpoints, or W&B. Those contain
identities/metadata, not video bytes. Reacquire the original licensed 20BN Something-Something V2
distribution, record the received package SHA-256 values, validate both ZIPs and the extracted exact
counts, extract into the canonical raw path, then follow Case B. Qualcomm's current page does not
publish a checksum table that this guide can truthfully call an expected hash. If the distribution
is no longer available to this user, exact SSv2 recovery is impossible.

## 4. EGO4D reconstruction

### Case A — full generated chunks survived

Do not re-encode them. Verify their manifest first, then rebuild tiny links from the emergency
inventory:

```bash
python3 /workspace/rebuild_links_from_inventory.py --inventory /workspace/recovery-manifests/REPLACE_WITH_SNAPSHOT/source-size-path.csv --workspace /workspace --family ego4d --dry-run
```

```bash
python3 /workspace/rebuild_links_from_inventory.py --inventory /workspace/recovery-manifests/REPLACE_WITH_SNAPSHOT/source-size-path.csv --workspace /workspace --family ego4d
```

If the inventory omitted tiny paths but the old `data/ego4d_tiny/manifest.json` survived, transfer
`scripts/rebuild_tiny_from_manifest.py` to `/workspace` and reconstruct its recorded membership:

```bash
python3 /workspace/rebuild_tiny_from_manifest.py --dataset ego4d --manifest /workspace/data/ego4d_tiny/manifest.json --workspace /workspace --dry-run
```

```bash
python3 /workspace/rebuild_tiny_from_manifest.py --dataset ego4d --manifest /workspace/data/ego4d_tiny/manifest.json --workspace /workspace
```

```bash
python3 /workspace/rebuild_tiny_from_manifest.py --dataset ego4d --manifest /workspace/data/ego4d_tiny/manifest.json --workspace /workspace --check
```

If neither inventory membership nor the old manifest survives, regenerate the intended deterministic
subset as a new derivation:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

```bash
python3 make_ego4d_subset.py --data-root /workspace/data --train-chunks 4000 --val-chunks 350 --per-video-cap 10 --seed 42
```

### Case B — chunks are missing or corrupt

This is a new derivation, not guaranteed byte-identical. Exact old chunks depended on source bytes,
FFmpeg behavior/build, timing, privacy intervals, and encoding parameters. Preserve the old
`selection_manifest.json` and `chunk_manifest.json`; they are the reconstruction specification.

The canonical full acquisition procedure remains
`AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md`. Execute its dashboard/license, batch, verification, and repair
blocks literally. The condensed recovery sequence is:

1. Obtain/renew official EGO4D access. Credentials are time-limited and must stay outside
   `/workspace`.
2. Ensure at least the guide’s required peak free space. The established 210-hour batched recipe
   estimates roughly 72 GB transient raw per batch and 50–90 GB permanent chunks; measure current
   reality before resizing.
3. Install the `ego4d` CLI and system `ffmpeg`.
4. Restore/reacquire `/workspace/ego4d_raw/ego4d.json` and
   `/workspace/ego4d_raw/video_540ss_manifest.csv`.
5. Prefer the surviving `/workspace/ego4d_raw/manifests/selection_manifest.json` and its UID/batch
   files. If missing, create a **new** selection with the current script and record that identity
   changed.
6. For each of four batches: download raw `video_540ss`, count and decode-check raw files, run the
   idempotent chunker, decode-check processed chunks, inspect the cumulative manifest, and only then
   delete that batch’s transient originals.
7. Verify source-UID split isolation and final chunk ranges.
8. Rebuild `ego4d_tiny`.

If selection must be regenerated, use:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

```bash
python3 select_ego4d_uids.py --metadata /workspace/ego4d_raw/ego4d.json --download-manifest /workspace/ego4d_raw/video_540ss_manifest.csv --target-hours 210 --val-fraction 0.10 --batches 4 --seed 42 --out-dir /workspace/ego4d_raw/manifests
```

For a downloaded batch, use the exact established chunker contract:

```bash
python3 chunk_ego4d.py --raw-dir /workspace/ego4d_raw/v2/video_540ss --manifest /workspace/ego4d_raw/manifests/selection_manifest.json --metadata /workspace/ego4d_raw/ego4d.json --out-root /workspace/data/ego4d --chunk-seconds 4 --fps 12 --shorter-side 256 --crf 27 --workers 2
```

Require the chunker’s final `failed: 0`, 48 frames/chunk, approximately 12 fps, shorter side 256,
valid H.264 decode, and no source UID in both splits. Never delete raw batch files before those checks.

## 5. Frozen encoder snapshots

If `/workspace/hf_cache` survived, retain it and run offline-capable smoke checks before downloading
anything. The current immutable revisions are listed in the
[recovery decision hub](../README.md#current-project-identity-used-by-this-audit).

Set the shared cache:

```bash
export HF_HOME=/workspace/hf_cache
```

Check the three adapters one at a time on CUDA:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

```bash
python3 encoders.py --smoke --encoder vjepa2_vitl16 --batch-size 1 --hf-cache-dir /workspace/hf_cache
```

```bash
python3 encoders.py --smoke --encoder siglip2_vitb16 --batch-size 1 --hf-cache-dir /workspace/hf_cache
```

```bash
python3 encoders.py --smoke --encoder dinov3_vitb16 --batch-size 1 --hf-cache-dir /workspace/hf_cache
```

If a snapshot is missing, authenticate to Hugging Face outside the volume as required and rerun the
same command. The adapter requests its pinned commit. Do not pass `--revision main`. Record the newly
resolved snapshot path and compare the printed revision/fingerprint with old checkpoint provenance.

If the pinned commit is no longer downloadable and not backed up, affected fresh/resume experiments
cannot honestly reproduce the old encoder substrate.

## 6. Checkpoints

Search both roots:

```bash
find /workspace/ckpt /workspace/checkpoints -type f -name '*.pt' -printf '%s\t%p\n' 2>/dev/null | sort -k2 > /workspace/recovery-manifests/post-restore/checkpoints.tsv
```

Checkpoint absence is irrecoverable unless another byte copy exists. W&B’s final path/SHA does not
contain weights. Do not create an empty file at an expected path or rename a different arm’s
checkpoint to satisfy a guide.

For each intended resume, use the exact guide’s `--resume` and preflight command. The loader validates
encoder/dataset/provenance, sampler, optimizer, reconstruction, and whitening contracts. Never use
`--allow-legacy-checkpoint`, `--allow-dataset-transfer`, or `--reset-optimizer` merely to bypass a
recovery mismatch; those flags express deliberate scientific changes.

## 7. Whitening statistics

Inspect a restored envelope without training:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

```bash
python3 whiten_stats.py --inspect /workspace/stats/REPLACE_WITH_EXACT_FILE.pt
```

The restored envelope must match the experiment’s encoder fingerprint, dataset fingerprint,
transform seed, clip count, preprocessing version, tensor geometry, and eigensolver settings.

If absent, generate new statistics only using the exact KANBAN guide. A representative invocation is:

```bash
python3 whiten_stats.py --data REPLACE_WITH_DATASET --split train --encoder REPLACE_WITH_ENCODER --hf-cache-dir /workspace/hf_cache --batch-size 8 --max-clips 12800 --seed 42 --output /workspace/stats/REPLACE_WITH_NEW_PROVENANCE_PATH.pt --device cuda
```

Never overwrite the historical path with a non-matching derivation. Resume checkpoints trained with
whitening normally embed the whitener, so a strict resume may work without the standalone stats file;
fresh whitened runs still need the correct external envelope.

## 8. Probe caches, preflights, logs, and W&B

- `logs/drift_probe/encoder_features_*.pt`: recompute only with the exact probe manifest, dataset, and
  pinned encoder. Save under a new tag if any identity changes.
- preflight JSON: rerun the corresponding KANBAN guide and write a new dated file. It does not recreate
  historical evidence.
- PNG/JSON diagnostic reports: can be regenerated only when their input checkpoint/cache survives.
- volume/repo logs: cannot be reconstructed from metrics; preserve partial logs and label gaps.
- repo `wandb/`: run `wandb sync` only after inspecting which runs are offline and reauthenticating.
  Do not sync the same queue under a new scientific identity by accident.
- W&B online data can restore logged metrics/artifacts that were explicitly uploaded, not local
  checkpoints, unlogged probe caches, or stderr.

## 9. Repository and Python environment

Before replacing a damaged Pod checkout, make a byte copy or tar of it, save `git fsck`, refs,
statuses, stashes, and untracked files, and compare it with this Mac. A checkpoint’s `git_commit` plus
dirty flag cannot recreate the dirty patch.

If the checkout is irreparably corrupt, create a fresh clone at the required historical path and
checkout the recorded commit. Then copy only reviewed remote-only/ignored state from quarantine. Do
not merge by blindly overwriting the Mac or Pod tree.

The old environment is not completely reconstructable without a saved `pip freeze`/container digest.
Install current `requirements.txt`, record a new environment snapshot, and treat changed dependency
versions as a provenance change:

```bash
python3 -m pip install -r /workspace/hierarchal-jepa-flow-world-model/requirements.txt
```

```bash
python3 -m pip freeze --all > /workspace/recovery-manifests/post-restore/pip-freeze.txt
```

```bash
python3 - <<'PY' > /workspace/recovery-manifests/post-restore/runtime.txt
import platform, sys
import torch, transformers
print('python', sys.version)
print('platform', platform.platform())
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('cuda_runtime', torch.version.cuda)
print('cudnn', torch.backends.cudnn.version())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
PY
```

## 10. Final honesty rules

- Exact video object restored and content-checked: call it **restored**.
- Symlink rebuilt from exact old membership to the exact restored target: call it **reconstructed**.
- Artifact recomputed under unchanged, verified inputs: call it **rederived**, with a new timestamp and
  hash.
- Dataset reacquired/re-encoded or selection changed: create a new dataset provenance identity and do
  not exact-resume old checkpoints unless the code’s explicit transfer policy and experiment plan
  authorize it.
- Missing checkpoint/dirty diff/offline queue with no surviving copy: call it **lost**. There is no
  technically honest workaround.
