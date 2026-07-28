# Forensic map of the expected `/workspace` volume

This is a code-derived map, not a claim that the old volume was live-listed. The old volume’s actual
manifest must be generated in the emergency guide. Repository implementation and current experiment
guides take precedence where old setup prose describes a historical path.

## 1. Expected top-level topology

```text
/workspace/
├── hierarchal-jepa-flow-world-model/     Git checkout and repo-local ignored runtime state
├── data/
│   ├── ssv2/                             full SSv2 split view
│   ├── ssv2_tiny/                        deterministic SSv2 smoke subset
│   ├── ego4d/                            generated full EGO4D chunk corpus
│   └── ego4d_tiny/                       deterministic EGO4D smoke subset
├── ssv2_raw/                             unique SSv2 source videos, labels, possible archives
├── ego4d_raw/                            EGO4D metadata/manifests; transient raw downloads
├── ckpt/                                 current per-run checkpoint directories
├── checkpoints/                          default and historical checkpoint root
├── stats/                                standalone whitening envelopes
├── preflight/                            resource/provenance preflight outputs and launch helpers
├── hf_cache/                             pinned Hugging Face model snapshots
├── logs/                                 volume-level logs
└── archive/                              retired/legacy artifacts retained for provenance
```

No code reference to another required top-level `/workspace` root was found. That does **not** prove
there are no extra personal or historical files. The whole source bucket is copied precisely to
capture those unknowns.

### Service and interrupted-write leftovers

The volume can also contain objects that are not experiment inputs but still explain an interrupted
operation:

- RunPod documents a hidden `.s3compat_uploads/` directory for multipart-upload parts/metadata. It
  should be cleaned after `CompleteMultipartUpload` or `AbortMultipartUpload`, but a failed upload can
  leave it temporarily. Inventory/rescue its readable objects as evidence, but do not manually
  recreate the service-owned directory on a replacement volume or treat a part as a finished artifact.
- `chunk_ego4d.py` writes sibling `*.part.mp4` files and atomically renames only successful encodes.
  A crash can leave a partial file even though normal failure handling removes it.
- checkpoint, whitening, cache, and JSON writers use hidden sibling names shaped like
  `.<destination>.<random>.tmp` before atomic replacement. A process/container crash can leave one.
- downloader/package-manager caches, editor swap files, core dumps, and personally created files are
  possible even though current code does not require a named root for them.

The emergency copy preserves any such readable key because it is safer to classify it after rescue.
The readiness checks never count `.part`, `.tmp`, or multipart parts as valid checkpoints, manifests,
or videos.

## 2. Artifact criticality table

| Location | What it contains | Who reads it | Criticality | Can it be recreated? |
|---|---|---|---|---|
| `/workspace/data/ssv2` | `train/`, `validation/`, `labels.json` | `data.py`, provenance, all SSv2 runs | Required path | Links can be rebuilt only if raw bytes and split metadata survive. `labels.json` must survive or be regenerated from the official labels. |
| `/workspace/ssv2_raw` | Original `.webm` videos, source labels, possibly downloaded ZIP archives | SSv2 symlink trees/rebuild | **Irreplaceable in practice** | Only by reacquiring the licensed/original SSv2 distribution. Preserve first. |
| `/workspace/data/ssv2_tiny` | 4,002 train + 348 validation links and `manifest.json` | smoke runs and fingerprints | Reproducible but provenance-sensitive | `make_subset.py` recreates it deterministically from the full split using seed 42 and 23/2 samples per class. Preserve the manifest anyway. |
| `/workspace/data/ego4d` | Real generated four-second H.264 MP4 chunks and `chunk_manifest.json` | full EGO4D training/provenance | **Irreplaceable for exact resume** | Conceptually regenerable from licensed raw videos, but byte identity depends on the exact source and FFmpeg build/settings. Preserve. |
| `/workspace/data/ego4d_tiny` | ~4,000 train + ~350 validation links and `manifest.json` | smoke runs/provenance | Reproducible but provenance-sensitive | `make_ego4d_subset.py` can rebuild from the full chunks. |
| `/workspace/ego4d_raw` | `ego4d.json`, tier CSV, UID lists, `selection_manifest.json`, other selection/download provenance | EGO4D acquisition/rebuild | **Essential provenance** | Some metadata can be reacquired after EGO4D authorization; the exact selection record should be preserved. |
| `/workspace/ckpt` | Per-experiment `.pt` checkpoints | resume, probes, comparisons | **Irreplaceable** | Not from current `train.py`; only a separately uploaded byte artifact or other backup can restore one. |
| `/workspace/checkpoints` | default/legacy checkpoints | old guides/default config | **Irreplaceable if used** | No. Preserve even if `ckpt/` is current. |
| `/workspace/stats` | strict encoder/dataset-bound whitening `.pt` envelopes | fresh whitened runs | Important/occasionally required | Can be recomputed if exact dataset, encoder revision, transform, seed, clip count, and eigensolver contract survive. Expensive and not guaranteed identical across changed software. |
| `/workspace/preflight` | JSON evidence and some run helper scripts | human launch gates/audit | Scientific evidence | Can be rerun, but old results document exactly what was tested before a run. Preserve. |
| `/workspace/hf_cache` | frozen-encoder configs and weights at pinned commits | all real encoder adapters | Operationally important | Redownloadable only while repositories, revisions, credentials, and network remain available. DINOv3 access can be gated. Exact cache path participates in the current encoder specification. |
| `/workspace/logs` | shell/training logs | diagnosis/human | Important evidence | No exact recreation. Metrics in W&B are not a replacement for stderr/stdout. |
| `/workspace/archive` | historical artifacts | retrospective analysis | Unknown-to-important | Treat as irreplaceable until inspected. |
| repo `.git` | commit objects, refs, worktree index | Git/provenance | Important | GitHub restores pushed commits only. Remote-only commits/stashes/objects do not return. |
| repo source files | possibly dirty/unpushed source | training | **Irreplaceable if remote-only** | This Mac has a dirty checkout, but it may not match the Pod. Preserve both and compare; never overwrite one blindly. |
| repo `logs/` | probe caches/reports, default preflight/whitening outputs | probes/humans | Important and ignored by Git | Some reports recompute; large encoder feature caches are costly. Preserve. |
| repo `wandb/` | local W&B run metadata and offline queues | `wandb sync` | Important if unsynced | No, if the run never reached W&B servers. |

## 3. SSv2 layout in detail

### 3.1 Unique source bytes

The expected unique source is:

```text
/workspace/ssv2_raw/20bn-something-something-v2/<video-id>.webm
```

The publisher's current page specifies exactly 220,847 `.webm` videos and a 19.4 GB total download
for the split TGZ distribution. Historical downloaded ZIPs may also remain in
`/workspace/ssv2_raw`. Extracted size, archive filenames/sizes, and exact old-volume totals must come
from the live inventory; do not infer them from the publisher's download-size figure.

Source label JSON is expected somewhere below `/workspace/ssv2_raw/labels/`, historically including
train and validation records. Those split records are required to recreate the full split symlink
view if the view itself is missing.

### 3.2 Full training view

Expected paths:

```text
/workspace/data/ssv2/train/<video-id>.webm
/workspace/data/ssv2/validation/<video-id>.webm
/workspace/data/ssv2/labels.json
```

Official expected counts are exactly 168,913 train links and 24,777 validation links. The
links are absolute and normally resolve into `/workspace/ssv2_raw/20bn-something-something-v2/`.
Consequences:

- copying only `/workspace/data/ssv2` can leave a completely unusable dataset if link targets were
  not materialized by the S3 interface;
- restoring the same absolute `/workspace/...` target paths makes surviving link text valid again;
- if S3 turns each link into a regular file containing the target video bytes, training can still
  work, but the restored volume consumes much more physical space;
- if S3 stores link text as tiny objects, those objects must not be mistaken for video bytes.

`labels.json` maps video ID to class. Split membership is encoded by the train/validation directory
view and/or the original official split JSON, not by that class map alone.

### 3.3 Tiny view

`make_subset.py` creates absolute symlinks under `/workspace/data/ssv2_tiny`, with exactly 4,002
training examples and 348 validation examples for the current official 174-class split. The
deterministic contract is seed 42, 23 train examples per class, and 2 validation examples per class.
Its `manifest.json` records the selection and should be treated as provenance even though the subset
is reproducible.

## 4. EGO4D layout in detail

### 4.1 Full generated chunks

Expected output:

```text
/workspace/data/ego4d/train/*.mp4
/workspace/data/ego4d/validation/*.mp4
/workspace/data/ego4d/chunk_manifest.json
```

The project’s established preprocessing contract targets about 170,000 train chunks and 18,000
validation chunks. Chunks are four seconds, 12 fps, H.264, no audio, shorter side 256, CRF 27. These
are generated files, not symlinks. The files are the exact bytes consumed by training, so they and
`chunk_manifest.json` take priority over temporary original downloads.

### 4.2 Selection and acquisition records

Preserve all of `/workspace/ego4d_raw`, especially:

```text
/workspace/ego4d_raw/ego4d.json
/workspace/ego4d_raw/video_540ss_manifest.csv
/workspace/ego4d_raw/manifests/selection_manifest.json
```

The directory can also contain train/validation/batch UID lists. Some older prose refers to a
slightly different tier-manifest location; the current working-tree experiment configuration points
at `/workspace/ego4d_raw/video_540ss_manifest.csv` and the selection manifest shown above. Preserve
both any historical and current copies rather than “cleaning up” the discrepancy.

The raw download staging directory `/workspace/ego4d_raw/v2/video_540ss/` is designed to be drained
after successful chunking. If it contains files, they are still valuable and must be copied; an empty
directory there is normal.

### 4.3 Tiny view

`/workspace/data/ego4d_tiny/{train,validation}` is normally an absolute-symlink subset of the full
chunk corpus with about 4,000/350 clips and a `manifest.json`. It is reproducible only if the full
chunks and their valid split membership survive.

## 5. Checkpoints: why every byte matters

Current experiments generally write into one directory per run below `/workspace/ckpt/`; historical
defaults also use `/workspace/checkpoints/`. The code and KANBAN guides reference many families,
including `inv009` through `inv017`, encoder-pair arms, slot-capacity arms, internal-memory arms,
latent-shape sweeps, and EGO4D arms. Examples include:

```text
/workspace/ckpt/inv016_capacity_ego4d_nc128/phase1_step15000.pt
/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/phase1_step15000.pt
/workspace/ckpt/inv016_unwhitened_memory_m1024/phase1_step15000.pt
/workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var/phase1_step10000.pt
/workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d/phase1_step15000.pt
/workspace/ckpt/encoder_pair/dino3b/phase1_step15000.pt
/workspace/ckpt/encoder_pair/siglip2b/phase1_step15000.pt
/workspace/checkpoints/phase1_step*.pt
```

These examples are **not** a whitelist. Copy all of both roots. A modern checkpoint can include:

- bottleneck and EMA bottleneck weights;
- flow model and decoder weights;
- optimizer tensors;
- exact RNG and sampler state needed for deterministic resume;
- resolved configuration and training step;
- dataset and run provenance;
- encoder specification, revision, and fingerprint;
- W&B run identity;
- reconstruction mean/tracker state;
- an embedded fixed feature whitener and identity when whitening is active.

Frozen encoder weights are intentionally not duplicated inside each checkpoint. A checkpoint can
therefore survive while resume still fails because the exact encoder snapshot/cache is absent.

## 6. Whitening, preflight, diagnostics, and model cache

Known standalone whitening examples include:

```text
/workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt
/workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt
/workspace/stats/inv016_encoder_substrate/siglip2_vitb16_ego4d_train_seed42.pt
```

The envelope is bound to encoder fingerprint, dataset fingerprint, preprocessing version, transform
seed, clip count, tensor geometry, and eigensolver settings. A same-shaped tensor from the wrong lane
is rejected by design.

Known preflight families include DINOv3 real-adapter/pinned-alias smokes, encoder-pair exact/resource
preflights, many investigation-016 arms, and investigation-017 V-JEPA2/latent-shape helpers. Some are
JSON records; some are executable shell helpers. Copy the entire directory.

Repo-local ignored output can include:

```text
logs/preflight/*.json
logs/whiten/*.pt
logs/drift_probe/manifest_*.json
logs/drift_probe/encoder_features_*.pt
logs/drift_probe/*.json
logs/drift_probe/*.png
wandb/
```

Feature caches may be on the order of a gigabyte. W&B artifact logging for probes is opt-in and does
not necessarily include cached feature tensors or every PNG.

The expected shared Hugging Face cache is `/workspace/hf_cache`. Current pinned revisions are:

| Alias | Repository | Pinned commit |
|---|---|---|
| `vjepa2_vitl16` | `facebook/vjepa2-vitl-fpc64-256` | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` |
| `siglip2_vitb16` | `google/siglip2-base-patch16-256` | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` |
| `dinov3_vitb16` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | `5931719e67bbdb9737e363e781fb0c67687896bc` |

Do not replace those with `main`. The resolved cache directory is represented in the current encoder
specification/fingerprint, and DINOv3 may require an accepted Hugging Face access gate.

## 7. Git, environment, and secrets

The expected repository path is intentionally the historical spelling:

```text
/workspace/hierarchal-jepa-flow-world-model
```

Preserve its `.git` directory and working tree. Checkpoint provenance stores a commit SHA and a dirty
boolean, but **not the dirty diff**. A dirty or unpushed Pod checkout cannot be reconstructed from the
checkpoint or GitHub.

This repository ignores, among other things, `.env`, `wandb/`, `/logs/`, `/run.log`, `checkpoints/`,
`archive/`, `*.pt`, `ssh.txt`, temporary paths, and error logs. Git is therefore intentionally unable
to restore much of the scientific/runtime state.

The Python requirements are mostly lower bounds rather than a lockfile; only selected packages are
pinned. Run provenance captures important fields such as Python/platform, Torch, Transformers,
CUDA/cuDNN, GPU, and Git, but it is not a complete `pip freeze`. Exact environment reconstruction is
not guaranteed if no environment snapshot exists on the old volume.

Normal Pod-local credentials often live outside the network volume under `/root/.aws`,
`/root/.config/wandb`, or `/root/.netrc`; they will not be rescued through the volume S3 API and should
be recreated, not copied. Conversely, a credential accidentally stored under `/workspace` is part of
the whole-volume backup. Keep AWS private and rotate any such credential after discovery.

## 8. Dataset identity and symlink consequences

`provenance.py` inventories dataset-relative paths, resolved byte sizes, Decord frame counts, and
manifest hashes. It resolves symlinks while measuring the target, but the fingerprint is based on the
relative dataset path/size/frame contract—not the absolute symlink target text. Therefore:

- a restored regular video file at the same relative path can satisfy the scientific dataset
  identity even if it was formerly a symlink;
- a broken link, a tiny object containing link text, a different transcode, or a changed frame count
  cannot;
- full EGO4D completeness and UID/count contracts remain strict;
- an exact manifest and successful project provenance check are more meaningful than matching only
  a directory count.

## 9. Scale estimate and why it changes the rescue method

Using documented counts, the volume can expose roughly:

- 220,847 SSv2 raw videos;
- 193,690 full SSv2 split entries;
- 4,350 SSv2 tiny entries;
- about 188,000 EGO4D chunks;
- about 4,350 EGO4D tiny entries;
- plus code, cache, checkpoints, manifests, logs, and diagnostics.

That is plausibly **more than 611,000 filesystem paths/object keys** if the RunPod S3 layer exposes
each link/path. It is an estimate, not a live count. RunPod officially warns that recursive listing
or `sync` can be slow or fail above 10,000 files or 10 GB. The emergency runbook uses:

1. small irreplaceable prefixes first;
2. a resumable whole-volume `rclone copy` with no deletion semantics;
3. repeated passes until zero copy errors;
4. source and destination inventories;
5. metadata/size checks followed by a full read comparison when time permits.

## 10. What only a live source inventory can answer

The following remain deliberately marked unknown until the old bucket is listed:

- exact object count and logical byte total;
- exact volume capacity and free space;
- which historical checkpoint/run directories actually exist;
- whether raw download ZIPs and transient EGO4D source videos remain;
- whether repository logs/W&B queues or remote-only Git objects exist;
- whether personal/unreferenced files exist;
- how RunPod’s S3 layer represents this volume’s particular symlinks;
- whether any source key is already unreadable;
- the exact deletion deadline.

The manifests and logs produced by the next guide convert these unknowns into recorded facts.
