# Code-referenced `/workspace` path catalog

Audit date: **2026-07-23**. This appendix records the cleaned path families found by searching the
current repository, setup documentation, knowledge documents, scripts, tests, and every KANBAN
experiment record containing a literal `/workspace/` reference. It complements the live object
inventory created by the emergency-copy guide; it is not a substitute for that inventory.

## 1. How to interpret this catalog

- **Active contract** means current code or the current experiment launch path reads or writes it.
- **Historical/compatibility** means an older guide or run record names it. Preserve it even if
  current defaults have moved.
- **Template** means shell variables, wildcards, or placeholder text expand into multiple actual
  paths. A template is not itself an expected filename.
- A directory being referenced proves that code expects or may create it; it does not prove that the
  old volume actually contains it.
- The final search found literal `/workspace` references in 125 files (836 matching lines). Dynamic paths built from
  configuration values cannot all appear as complete string literals, so their resolved families
  are listed below from the relevant code and launch scripts.

The reproducible literal-reference search is:

```bash
rg -n '/workspace/' --glob '!NETWORK_VOLUME_RECOVERY/**' --glob '!.git/**'
```

Do not use the result as a copy allow-list. The rescue copies the entire bucket because personal
files, ignored files, dynamically named outputs, and remote-only state may have no code reference.

## 2. Canonical active roots

| Path | Status | Contract |
|---|---|---|
| `/workspace/hierarchal-jepa-flow-world-model` | Active | Remote Git worktree. The historical spelling `hierarchal` is intentional. |
| `/workspace/data` | Active default | `JEPA_DATA_ROOT`; dataset loader resolves dataset names below it. |
| `/workspace/data/ssv2` | Active | Full SSv2 split view. |
| `/workspace/data/ssv2_tiny` | Active | Deterministic SSv2 subset. |
| `/workspace/data/ego4d` | Active | Full generated EGO4D chunk corpus. |
| `/workspace/data/ego4d_tiny` | Active | Deterministic EGO4D subset. |
| `/workspace/ssv2_raw` | Active rebuild dependency | Unique SSv2 video bytes and source metadata. |
| `/workspace/ego4d_raw` | Active provenance/rebuild dependency | EGO4D metadata, selection records, and transient download staging. |
| `/workspace/checkpoints` | Active default | `config.py` default checkpoint directory. |
| `/workspace/ckpt` | Active experiment convention | Per-run output root used by current KANBAN launch guides. |
| `/workspace/stats` | Active | Standalone whitening-statistics envelopes. |
| `/workspace/preflight` | Active | Exact/resource preflight evidence and remote helper scripts. |
| `/workspace/hf_cache` | Active | Shared Hugging Face cache and pinned encoder snapshots. |
| `/workspace/logs` | Historical/current special case | Volume-level logs explicitly used by at least one run; most current run logs are repo-local `logs/`. |
| `/workspace/archive` | Historical retention | Old artifacts whose significance cannot safely be inferred from age. |

## 3. Dataset and provenance paths

### 3.1 SSv2

Active/reference paths:

```text
/workspace/data/ssv2/
/workspace/data/ssv2/train/
/workspace/data/ssv2/validation/
/workspace/data/ssv2/labels.json
/workspace/data/ssv2_tiny/
/workspace/data/ssv2_tiny/train/
/workspace/data/ssv2_tiny/validation/
/workspace/data/ssv2_tiny/manifest.json
/workspace/ssv2_raw/
/workspace/ssv2_raw/20bn-something-something-v2/
/workspace/ssv2_raw/20bn-something-something-v2/<video-id>.webm
```

The full and tiny split directories are normally absolute-symlink views. Other label JSON and ZIP
paths below `/workspace/ssv2_raw` are acquisition-dependent and may not have a single canonical
literal name in the code. Preserve the entire root.

Historical repo-local layout found in old prose:

```text
/workspace/hierarchal-jepa-flow-world-model/data/something-something-v2/
/workspace/hierarchal-jepa-flow-world-model/data/something-something-v2/train/
```

That historical path must not be silently substituted for the active `/workspace/data/ssv2` path.
If it exists in the live manifest, preserve it as separate evidence.

### 3.2 EGO4D

Active/reference paths:

```text
/workspace/data/ego4d/
/workspace/data/ego4d/train/
/workspace/data/ego4d/validation/
/workspace/data/ego4d/chunk_manifest.json
/workspace/data/ego4d_tiny/
/workspace/data/ego4d_tiny/train/
/workspace/data/ego4d_tiny/validation/
/workspace/data/ego4d_tiny/manifest.json
/workspace/ego4d_raw/
/workspace/ego4d_raw/ego4d.json
/workspace/ego4d_raw/video_540ss_manifest.csv
/workspace/ego4d_raw/manifests/
/workspace/ego4d_raw/manifests/selection_manifest.json
/workspace/ego4d_raw/manifests/train_uids.txt
/workspace/ego4d_raw/manifests/val_uids.txt
/workspace/ego4d_raw/manifests/batch_<batch-number>_uids.txt
/workspace/ego4d_raw/v2/video_540ss/
/workspace/ego4d_raw/v2/video_540ss/<uid>.mp4
```

Historical/acquisition variants also named in project documents:

```text
/workspace/ego4d_raw/v2/ego4d.json
/workspace/ego4d_raw/v2/video_540ss/manifest.csv
/workspace/ego4d_raw/manifests/all_uids.txt
```

Some old prose truncates or abbreviates manifest names in examples. The authoritative current
configuration is the selection manifest plus top-level `video_540ss_manifest.csv`; nevertheless,
all variants must be retained because they may establish how an older corpus was selected.

## 4. Checkpoint path families

Every file under both `/workspace/checkpoints` and `/workspace/ckpt` is irreplaceable unless a human
has independently proved otherwise. The following list records concrete directory names or templates
present in the codebase; it is deliberately **not** a whitelist.

### 4.1 Default and legacy root

```text
/workspace/checkpoints/
/workspace/checkpoints/phase1_step*.pt
/workspace/checkpoints/phase1_step7500.pt
/workspace/checkpoints/phase1_step15000.pt
/workspace/hierarchal-jepa-flow-world-model/checkpoints/stage1_final.pt
```

### 4.2 Early capacity and regularization sweeps

```text
/workspace/ckpt/L0.05_D256x2_nc64
/workspace/ckpt/L0.05_D256x2_nc128
/workspace/ckpt/L0.05_D256x2_nc256
/workspace/ckpt/L0.05_D512x2_nc32
/workspace/ckpt/L0.05_D512x4_nc32
/workspace/ckpt/L0.1_D256x2_nc32
/workspace/ckpt/L0.2_D256x2_nc32
/workspace/ckpt/L0.2_D512x2_nc64
/workspace/ckpt/L0.5_D256x2_nc32
/workspace/ckpt/L1.0_D256x2_nc32
/workspace/ckpt/sigreg0.3_D512x4_nc32
/workspace/ckpt/sigreg1.0_D512x4_nc32
/workspace/ckpt/sigreg3.0_D512x4_nc32
/workspace/ckpt/sigreg10.0_D512x4_nc32
```

### 4.3 Investigation and smoke directories

```text
/workspace/ckpt/ego4d_smoke
/workspace/ckpt/ego4d_run037_ab
/workspace/ckpt/ssv2_regression_smoke
/workspace/ckpt/encoder_pair_smoke/dino3b
/workspace/ckpt/encoder_pair_smoke/siglip2b
/workspace/ckpt/encoder_full_smoke/dino3b
/workspace/ckpt/encoder_full_smoke/siglip2b
/workspace/ckpt/encoder_pair/dino3b
/workspace/ckpt/encoder_pair/siglip2b
/workspace/ckpt/inv009_run1_sigreg6_full
/workspace/ckpt/inv009_run2_sigreg5_residual
/workspace/ckpt/inv010_residual_230096d
/workspace/ckpt/inv011_cosine_residual
/workspace/ckpt/inv011_new_recon_loss
/workspace/ckpt/inv011_original_recon_nopred
/workspace/ckpt/inv011_fixed_position_decoder
/workspace/ckpt/inv011_fixed_position_present_recon
/workspace/ckpt/inv011_present_recon_only
/workspace/ckpt/inv012_sharp_slot_recon_cov0p01
/workspace/ckpt/inv012_sharp_slot_recon_only
/workspace/ckpt/inv013_residual_recon_only
/workspace/ckpt/inv015_ae_latent_stack_whiten
/workspace/ckpt/inv015_whiten_abs_recon
/workspace/ckpt/inv015_whiten_abs_recon_geom
/workspace/ckpt/inv015_whiten_abs_recon_cov_var
```

### 4.4 Investigation 011 present-only geometry sweep

Root/template:

```text
/workspace/ckpt/inv011_present_only_geometry_sweep/<run-tag>
```

Named variants found in guides and run records:

```text
po_geom_sig5_cov0p003
po_geom_sig5_cov0p01
po_geom_sig7p5_cov0
po_geom_sig7p5_cov0p003
po_geom_sig7p5_cov0p01
po_geom_sig10_cov0
po_geom_sig10_cov0p003
po_geom_sig10_cov0p01
po_geom_sig12p5_cov0
po_geom_sig12p5_cov0p003
run_042_po_geom_sig7p5_cov0
run_043_po_geom_sig5_cov0p003
run_044_po_geom_sig10_cov0p003
run_045_po_geom_sig12p5_cov0
run_046_po_geom_sig10_cov0
run_047_po_geom_sig5_cov0p01
run_048_po_geom_sig7p5_cov0p003
run_049_po_geom_sig10_cov0p01
run_050_po_geom_sig7p5_cov0p01
run_051_po_geom_sig12p5_cov0p003
```

Both naming schemes occur in the repository. Do not collapse one into the other.

### 4.5 Investigation 016 families

```text
/workspace/ckpt/inv016_capacity_preflight_<dataset>_nc128
/workspace/ckpt/inv016_capacity_ego4d_nc128
/workspace/ckpt/inv016_internal_memory_preflight_m1024
/workspace/ckpt/inv016_unwhitened_memory_m512
/workspace/ckpt/inv016_unwhitened_memory_m1024
/workspace/ckpt/inv016_unwhitened_m512_cov_var
/workspace/ckpt/inv016_unwhitened_m512_cov_var_preflight
/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512
/workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var
/workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var_preflight
/workspace/ckpt/inv016_unwhitened_siglip2_m512_no_geometry_regularizers
/workspace/ckpt/inv016_siglip2_whiten_abs_recon_cov_var_ego4d
/workspace/ckpt/inv016_siglip2_whiten_abs_recon_cov_var_ego4d_smoke
/workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d
/workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d_recon1
```

### 4.6 Investigation 017 dynamic latent-shape families

The current sweep script creates:

```text
/workspace/ckpt/inv017_<encoder>_raw_shape_n<n-c>_d<d-c>_s42
```

`<encoder>` is `vjepa2`, `siglip2`, or `dinov3`. The intended shapes are `16x512`, `64x128`,
`16x128`, `16x256`, `32x128`, `32x512`, `64x256`, and `64x512`; the DINOv3 lane also includes the
`32x256` center. Each completed directory is expected to contain `phase1_step15000.pt` and
`run_provenance.json`.

### 4.7 Exact checkpoint filenames explicitly tested by guides

```text
/workspace/ckpt/ego4d_smoke/phase1_step500.pt
/workspace/ckpt/ego4d_run037_ab/phase1_step15000.pt
/workspace/ckpt/ssv2_regression_smoke/phase1_step100.pt
/workspace/ckpt/encoder_pair/dino3b/phase1_step15000.pt
/workspace/ckpt/encoder_pair/siglip2b/phase1_step15000.pt
/workspace/ckpt/inv011_cosine_residual/phase1_step7500.pt
/workspace/ckpt/inv016_capacity_ego4d_nc128/phase1_step15000.pt
/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/phase1_step15000.pt
/workspace/ckpt/inv016_unwhitened_memory_m512/phase1_step15000.pt
/workspace/ckpt/inv016_unwhitened_memory_m1024/phase1_step15000.pt
/workspace/ckpt/inv016_unwhitened_m512_cov_var/phase1_step15000.pt
/workspace/ckpt/inv016_unwhitened_siglip2_m512_cov_var/phase1_step10000.pt
/workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d/phase1_step15000.pt
/workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d_recon1/phase1_step15000.pt
```

Other generic references include `/workspace/ckpt/phase1_step2500.pt`,
`/workspace/checkpoints/phase1_stepXXXX.pt`, and `phase1_step*.pt`. A concrete numbered path in a
guide is an expected milestone, not evidence that earlier or later checkpoint files are disposable.

## 5. Statistics and preflight paths

### 5.1 Whitening statistics

```text
/workspace/stats/encoder_pair_ssv2/
/workspace/stats/encoder_pair_ssv2/dinov3_vitb16_ssv2_train_seed42.pt
/workspace/stats/encoder_pair_ssv2/siglip2_vitb16_ssv2_train_seed42.pt
/workspace/stats/inv016_bottleneck_capacity/
/workspace/stats/inv016_encoder_substrate/
/workspace/stats/inv016_encoder_substrate/siglip2_vitb16_ego4d_train_seed42.pt
```

Historical repo-local whitening path:

```text
/workspace/hierarchal-jepa-flow-world-model/logs/whiten/whiten_stats_ssv2_train_seed42.pt
```

### 5.2 Preflight evidence and helpers

```text
/workspace/preflight/dinov3/pinned_alias_smoke.json
/workspace/preflight/dinov3/real_adapter_smoke.json
/workspace/preflight/encoder_pair/dino_exact.json
/workspace/preflight/encoder_pair/dino_resource.json
/workspace/preflight/encoder_pair/siglip_exact.json
/workspace/preflight/encoder_pair/siglip_resource.json
/workspace/preflight/encoder_pair/siglip_unwhitened_resource.json
/workspace/preflight/inv016_bottleneck_capacity/
/workspace/preflight/inv016_encoder_substrate/
/workspace/preflight/inv016_internal_memory_width/
/workspace/preflight/inv016_unwhitened_m512_cov_var/ego4d_m512_cov_var_resource.json
/workspace/preflight/inv016_unwhitened_siglip2_m512_cov_var/ego4d_siglip2_m512_cov_var_resource.json
/workspace/preflight/inv016_unwhitened_siglip2_m512_no_geometry_regularizers/ego4d_siglip2_m512_no_geometry_regularizers_resource.json
/workspace/preflight/inv017_vjepa2_direct/RUN_VJEPA2_ARMS_ONLY.sh
/workspace/preflight/inv017_vjepa2_direct/RUN_VJEPA2_UNTRIED_ARMS_ONLY.sh
```

Several run scripts build further preflight filenames from variables. Preserve the whole root and
do not assume that JSON is the only valuable extension: the investigation-017 files above are remote
helper scripts copied out of a working tree and have recorded SHA-256 identities.

## 6. Repository-local and ignored runtime state

The remote repository root has the following explicit or inferred-from-code runtime children:

```text
/workspace/hierarchal-jepa-flow-world-model/.git/
/workspace/hierarchal-jepa-flow-world-model/train.py
/workspace/hierarchal-jepa-flow-world-model/logs/
/workspace/hierarchal-jepa-flow-world-model/wandb/
/workspace/hierarchal-jepa-flow-world-model/archive/
/workspace/hierarchal-jepa-flow-world-model/train.log
```

Current run scripts commonly write relative paths such as `logs/<tag>.log`; after `cd` to the repo,
those resolve below the repository even when the complete `/workspace` path never appears in source.
The same applies to probe reports and caches described in the forensic map. Git ignores much of this
state.

One explicit volume-level log is:

```text
/workspace/logs/inv016_unwhitened_siglip2_m512_no_geometry_regularizers.log
```

## 7. Encoder cache contract

The literal cache root is always:

```text
/workspace/hf_cache/
```

Hugging Face creates its own nested `hub/models--.../snapshots/<commit>/`, blobs, refs, and lock
layout below this root. Those generated names are not individually hard-coded. Preserve every key;
do not retain only the three snapshot directories while discarding blobs or metadata.

## 8. Placeholders that must not be treated as literal objects

The search also finds examples such as:

```text
/workspace/ckpt/$tag
/workspace/ckpt/${tag}
/workspace/ckpt/PASTE_RUN_TAG
/workspace/ckpt/inv016_unwhitened_memory_m${width}
/workspace/ckpt/inv016_capacity_preflight_${dataset}_nc128
/workspace/ego4d_raw/manifests/batch_${BATCH}_uids.txt
/workspace/checkpoints/phase1_step*.pt
/workspace/checkpoints/phase1_stepXXXX.pt
```

These document expansion rules or human placeholders. Do not create files literally containing `$`,
`*`, or `XXXX` merely to make the catalog appear complete. The live S3 inventory is the authority
for the concrete expansions that actually survived.

## 9. Final preservation rule

The complete preservation set is:

```text
every readable key below s3://4hzrwzk8ja/
```

This catalog answers which paths the repository is known to care about and how to prioritize them.
Only the source manifest from
[`../PERSPECTIVE_1_BEFORE_DELETION/01_EMERGENCY_COPY_RUNPOD_TO_AWS.md`](../PERSPECTIVE_1_BEFORE_DELETION/01_EMERGENCY_COPY_RUNPOD_TO_AWS.md)
answers what was actually on this specific volume at the time of rescue.
