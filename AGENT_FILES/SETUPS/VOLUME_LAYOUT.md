# VOLUME_LAYOUT.md — RunPod network volume structure

> **Audience:** Coding agents and human operators.
> **Purpose:** Single source of truth for what exists on the RunPod network volume **today**, what the **target** layout is after first-time setup, and how that layout maps to code paths in `config.py`.
>
> **Operator how-to:** migration commands and SSH workflow live in [`SETUP.md`](SETUP.md) Path A (steps A7–A12). This file describes **what** the volume should look like, not step-by-step pod operations.

**Provenance:** Current-state counts and paths come from a volume inspection recorded in `CHAT.md` (2025-06-06/07). Re-verify on the pod before acting — symlink counts and folder names may differ slightly on your volume.

---

## 1. Design principle — code and data are siblings

The git repo holds **code only**. Datasets, checkpoints, and Hugging Face cache live **next to** the repo under `/workspace/`, not inside it. This keeps `git clone` / `git pull` safe and makes paths predictable for `config.py`.

| Location | In git? | Persists when pod stops? |
|---|---|---|
| `/workspace/hierarchal-jepa-flow-world-model/` | Yes (cloned) | Yes (on network volume) |
| `/workspace/data/` | No | Yes |
| `/workspace/ckpt/` | No | Yes |
| `/workspace/checkpoints/` | No (legacy) | Yes |
| `/workspace/ssv2_raw/` | No | Yes |
| `/workspace/ego4d_raw/` | No | Yes (transient: raw batches deleted after chunking; `manifests/` + `ego4d.json` kept) |
| `/workspace/hf_cache/` | No | Yes |
| `/workspace/archive/` | No | Yes |

> **Checkpoints:** real runs write to **`/workspace/ckpt/<run_tag>/`** via `--checkpoint-dir` (per-run subdir, mandatory for parallel sweeps — see [`KANBAN/PHASE_1/investigation_007/GUIDE.md`](../../KANBAN/PHASE_1/investigation_007/GUIDE.md)). The `config.py` default `/workspace/checkpoints/` is **legacy** — it holds only early single-run `phase1_step*.pt` files and is no longer written to.

RunPod mounts the network volume at `/workspace` by default. SSH and pod ops: [`SETUP.md`](SETUP.md), [`GUIDES/MLOPS.md`](../../GUIDES/MLOPS.md).

---

## 2. Historical state (pre–v0 setup — migration long done)

> **⚠️ HISTORICAL — no longer reflects the volume.** Migration completed in June 2026; the live layout is Section 3. Kept only as a record of the starting point.

This is what the volume looked like **before** Path A migration and **before** v0 code is deployed. Legacy Python from a prior implementation may still be present — **disregard it**; build fresh from phase docs.

```
/workspace/                                              (~38G used; volume has headroom)
├── hf_cache/                                            ← exists; essentially empty (~512 B)
├── hierarchal-jepa-flow-world-model/                    ← OLD repo (~697 MB)
│   ├── data/something-something-v2/                     ← prepared SSv2 (WRONG LOCATION for v0)
│   │   ├── train/          → 168,913 symlinks to .webm
│   │   ├── validation/     → 24,777 symlinks
│   │   └── labels.json     → 34 MB (video_id → class label)
│   ├── checkpoints/
│   │   └── stage1_final.pt → 533 MB (legacy; archive or ignore)
│   └── [legacy .py files: config.py, data.py, models.py, train.py, …]  ← DISREGARD
└── ssv2_raw/                                            ← raw Qualcomm SSv2 (~37 GB)
    ├── 20bn-something-something-v2/                     ← 220,847 extracted .webm files (~19 GB)
    ├── 20bn-something-something-v2-00.zip               ← 9.4 GB (redundant if extract exists)
    ├── 20bn-something-something-v2-01.zip               ← 8.8 GB (redundant if extract exists)
    └── labels/labels/
        ├── train.json
        ├── validation.json
        └── labels.json
```

### What exists vs what v0 still needs

| Item | Status on volume | v0 action |
|---|---|---|
| Raw `.webm` videos | **Exists** — `/workspace/ssv2_raw/20bn-something-something-v2/` | Read-only; do not move or re-encode |
| Full SSv2 symlink layout | **Exists** — but nested under old repo path | **One-time migrate** → `/workspace/data/ssv2/` ([`SETUP.md`](SETUP.md) A8) |
| `ssv2_tiny` smoke subset | **Does not exist yet** on a fresh volume | **Create once** via `make_subset.py` ([`SETUP.md`](SETUP.md) A12; see [`AGENTS.md`](../AGENTS.md) §5) |
| v0 Python codebase | **Does not exist** (only legacy code) | Agent implements Phases 1–3; human clones fresh repo ([`SETUP.md`](SETUP.md) A9) |
| Checkpoints dir (sibling) | **Does not exist** at `/workspace/checkpoints/` | `mkdir` during setup; v0 writes `phase1_step*.pt`, etc. |
| `hf_cache` populated | **Empty** | Fills in Phase 3 when `sd-vae-ft-mse` downloads |
| Legacy `stage1_final.pt` | **Exists** inside old repo | Move to `archive/` so training never loads it by mistake |

**Important:** Prepared SSv2 data is **already built** — migration is an instant `mv` of symlink directories, not a re-download or re-encode. Symlinks continue to point at files under `/workspace/ssv2_raw/`.

---

## 3. Target state (after first-time setup + Phase 1 data work)

After [`SETUP.md`](SETUP.md) Path A (steps A8, A9, A12) and Phase 1 `make_subset.py`, the volume should match this layout:

```
/workspace/
├── hierarchal-jepa-flow-world-model/          ← git repo — CODE ONLY (v0 implementation)
│   ├── config.py, data.py, models.py, …
│   ├── make_subset.py
│   ├── train.py
│   └── AGENT_FILES/                           ← planning docs (also in git)
│
├── data/                                      ← all datasets (sibling to repo)
│   ├── ssv2/                                  ← full SSv2 (migrated from old nested path)
│   │   ├── train/          → ~168,913 symlinks → /workspace/ssv2_raw/.../*.webm
│   │   ├── validation/     → ~24,777 symlinks
│   │   └── labels.json
│   ├── ssv2_tiny/                             ← smoke subset (created by make_subset.py)
│   │   ├── train/          → ~4,000 symlinks (stratified, 23/class × 174 classes)
│   │   ├── validation/     → ~350 symlinks (2/class)
│   │   └── manifest.json   → selected video_ids, seed, per-class counts
│   ├── ego4d/                                 ← EGO4D chunk corpus (created by chunk_ego4d.py)
│   │   ├── train/          → ~170k REAL .mp4 files (4s, 12fps, 256px shorter side)
│   │   ├── validation/     → ~18k REAL .mp4 files (split by SOURCE video — no leakage)
│   │   └── chunk_manifest.json → cumulative counts + ffmpeg params (rescanned each run)
│   └── ego4d_tiny/                            ← smoke subset (created by make_ego4d_subset.py)
│       ├── train/          → ~4,000 symlinks into data/ego4d/train (≤10 per source video)
│       ├── validation/     → ~350 symlinks
│       └── manifest.json   → seed, per-video selected chunk names
│
├── ssv2_raw/                                  ← raw .webm (unchanged, read-only)
│   └── 20bn-something-something-v2/
│
├── ego4d_raw/                                 ← EGO4D CLI downloads (TRANSIENT raw batches)
│   ├── manifests/                             ← UID/batch files + selection_manifest.json (KEEP)
│   ├── video_540ss_manifest.csv               ← authoritative downloadable-tier inventory (KEEP)
│   ├── ego4d.json                             ← official metadata (KEEP; CLI writes it here)
│   └── v2/
│       └── video_540ss/                       ← raw 540ss batch, deleted after chunk+verify
│
├── ckpt/                                      ← ACTIVE training outputs (sibling to repo)
│   ├── inv015_whiten_abs_recon/              ← one subdir per run (via --checkpoint-dir)
│   │   └── phase1_step15000.pt               ← naming: phase1_step{step}.pt
│   ├── inv012_sharp_slot_recon_only/
│   └── … (one dir per experiment/sweep tag)
│
├── checkpoints/                               ← LEGACY (config.py default; no longer written)
│   └── phase1_step*.pt                        ← early single-run files (Jun 2026), kept for history
│
├── archive/                                   ← retired artifacts (e.g. legacy stage1_final.pt)
│
└── hf_cache/                                  ← Hugging Face cache (Phase 3 VAE download)
```

### Dataset directory contract

All four dataset roots share the **same internal shape** — a `train/` and `validation/`
directory of short, independently sampleable clip files:

```
<dataset_root>/
├── train/           ← one video file per clip (.webm symlink for ssv2*, REAL .mp4 for ego4d,
├── validation/        .mp4 symlink for ego4d_tiny)
└── labels.json         ← full SSv2 only
    or manifest.json    ← ssv2_tiny / ego4d_tiny (subset provenance)
    or chunk_manifest.json ← ego4d (chunker provenance)
```

Key difference: `data/ego4d/` holds **real files** (new 4-second H.264 encodes produced by
`chunk_ego4d.py` from `ego4d_raw`), not symlinks — the raw EGO4D videos are hours long, so
clip files must be created, unlike SSv2 whose raw files are already clip-sized.

Training selects which root via CLI: `python train.py --data ssv2_tiny` (default, ~4–5 hr
smoke), `--data ssv2`, `--data ego4d_tiny`, or `--data ego4d`. No code edits required to
switch. EGO4D build procedure: [`AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md`](../KNOWLEDGE/ego4d/GUIDE.md).

---

## 4. Path defaults — what the code expects

Hardcoded in `config.py` (see [`AGENTS.md`](../AGENTS.md) §12):

| Symbol | Default path | Purpose |
|---|---|---|
| Repo root | `/workspace/hierarchal-jepa-flow-world-model` | Working directory for `train.py` |
| `DATA_ROOT` | `/workspace/data` (override: env `JEPA_DATA_ROOT`) | Parent of both datasets |
| `SSV2_FULL` | `{DATA_ROOT}/ssv2` | Full Something-Something V2 |
| `SSV2_TINY` | `{DATA_ROOT}/ssv2_tiny` | Stratified smoke subset |
| `EGO4D` | `{DATA_ROOT}/ego4d` | EGO4D chunk corpus (`cfg.data.ego4d_root`) |
| `EGO4D_TINY` | `{DATA_ROOT}/ego4d_tiny` | EGO4D smoke subset (`cfg.data.ego4d_tiny_root`) |
| `CHECKPOINT_DIR` | `/workspace/checkpoints` (default, **legacy**) | Saved `.pt` checkpoints. Real runs override via `--checkpoint-dir /workspace/ckpt/<tag>/` (`config.py:309`, applied at `train.py:1440`). |
| `HF_CACHE_DIR` | `/workspace/hf_cache` | `diffusers` VAE cache (Phase 3) |

**Local dev:** set `JEPA_DATA_ROOT` to a local folder that mirrors `data/ssv2` and `data/ssv2_tiny` structure. No environment auto-detection in code.

**Raw video resolution:** symlinks in `data/ssv2/train/` point at absolute paths under `/workspace/ssv2_raw/`. `make_subset.py` resolves source symlinks and creates new symlinks to the same underlying `.webm` files — no copying.

---

## 5. Migration map (current → target)

One-time human steps on the pod ([`SETUP.md`](SETUP.md) A8). Not agent code changes.

| From (current) | To (target) | Operation |
|---|---|---|
| `…/hierarchal-jepa-flow-world-model/data/something-something-v2/` | `/workspace/data/ssv2/` | `mv` (instant; symlinks preserved) |
| — | `/workspace/data/ssv2_tiny/` | `python make_subset.py` (Phase 1 script) |
| — | `/workspace/checkpoints/` | `mkdir -p` |
| `…/hierarchal-jepa-flow-world-model/checkpoints/stage1_final.pt` | `…/archive/` or ignore | `mv` (optional safety) |
| Old repo without v0 `train.py` | Renamed or replaced | Fresh `git clone` of HJEPA-VWM v0 |

After migration, **do not** store datasets or checkpoints inside the git repo directory.

---

## 6. Who does what

| Task | Who | When |
|---|---|---|
| Inspect volume (`ls`, symlink counts) | Human or agent (read-only SSH) | Before first train |
| Migrate SSv2 to `/workspace/data/ssv2` | Human ([`SETUP.md`](SETUP.md) A8) | Once per volume |
| Implement `make_subset.py` | Agent (Phase 1) | Once in codebase |
| Run `make_subset.py` on pod | Human ([`SETUP.md`](SETUP.md) A12) | Once per volume |
| Hardcode paths in `config.py` | Agent (Phase 1) | In git |
| Write checkpoints to `/workspace/checkpoints/` | `train.py` at runtime | Every training run |
| Populate `hf_cache/` | `train.py` / `eval.py` (Phase 3) | First VAE download |

---

## 7. Verification commands (run on pod)

```bash
# Volume mounted
df -h /workspace

# Full SSv2 after migration
test -f /workspace/data/ssv2/labels.json && echo "ssv2 OK"
find /workspace/data/ssv2/train -maxdepth 1 -type l | wc -l      # expect ~168913

# Tiny subset after make_subset.py
test -f /workspace/data/ssv2_tiny/manifest.json && echo "ssv2_tiny OK"
find /workspace/data/ssv2_tiny/train -maxdepth 1 -type l | wc -l  # expect ~4000

# Raw backing files
find /workspace/ssv2_raw/20bn-something-something-v2 -maxdepth 1 -name '*.webm' | wc -l  # expect ~220847

# EGO4D chunk corpus (after AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md Stage 4)
test -f /workspace/data/ego4d/chunk_manifest.json && echo "ego4d OK"
find /workspace/data/ego4d/train -maxdepth 1 -name '*.mp4' | wc -l       # expect ~165k-175k
find /workspace/data/ego4d/validation -maxdepth 1 -name '*.mp4' | wc -l  # expect ~15k-20k

# EGO4D tiny subset (after make_ego4d_subset.py)
test -f /workspace/data/ego4d_tiny/manifest.json && echo "ego4d_tiny OK"
find /workspace/data/ego4d_tiny/train -maxdepth 1 -type l | wc -l        # expect ~4000

# v0 code present
test -f /workspace/hierarchal-jepa-flow-world-model/train.py && echo "v0 code OK"
```

---

## 8. Related docs

| Doc | Role |
|---|---|
| [`VOLUME_LAYOUT.md`](VOLUME_LAYOUT.md) | **This file** — current vs target volume tree, path contract |
| [`SETUP.md`](SETUP.md) | Operator steps: SSH, clone, migrate, train |
| [`GUIDES/MLOPS.md`](../../GUIDES/MLOPS.md) | RunPod SSH, W&B, checkpoints |
| [`../AGENTS.md`](../AGENTS.md) | Path defaults, data semantics, preprocessing |
| [`../AGENT-BEHAVIOUR/PROTOCOL.md`](../AGENT-BEHAVIOUR/PROTOCOL.md) §3 | Deployment target summary for agents |
