# Phase 1 experiment configuration

The training interface is deliberately split into:

- one versioned YAML recipe containing the complete experiment;
- five scientific CLI overrides for the axes most frequently changed in recent KANBAN work;
- optional operator flags for resume, preflight, W&B identity, and output paths.

The canonical recipe is [`configs/train.yaml`](configs/train.yaml).

## Why these five CLI hyperparameters

Investigations 012–017 were reviewed as one recent research window. The repeated paid-run axes were:

| CLI axis retained | Recent evidence |
|---|---|
| `--data` | Investigation 015’s SSv2 recipe transferred to EGO4D in Investigation 016. |
| `--encoder` | Investigation 016 compared V-JEPA2, SigLIP 2, and DINOv3; Investigation 017 registered three encoder lanes. |
| `--n-c` | Investigation 016 swept 32/64/128 slots; Investigation 017 registered 16/32/64. |
| `--d-c` | Investigation 017 made external slot width a primary 128/256/512 factorial axis. |
| `--bottleneck-mixer-dim` | Investigation 016 compared internal widths 512 and 1024 and selected 512. |

Other parameters did change during those investigations, but they are coupled recipe choices:

- whitening must agree with its statistics artifact and target space;
- reconstruction target/mode must agree with reconstruction weights and diagnostics;
- variance, covariance, SIGReg, and slot losses form a geometry bundle;
- decoder shape, learning rates, schedules, AGC, and diagnostic cadence define the training
  background against which a sweep is interpreted.

Those settings now belong in YAML so a run cannot accidentally mix half of one recipe with half of
another through a long shell command.

## Normal usage

Use the checked-in recipe without scientific overrides:

```bash
python train.py --config configs/train.yaml
```

Override only the active shape/encoder axes:

```bash
python train.py \
  --config configs/train.yaml \
  --data ego4d \
  --encoder dinov3_vitb16 \
  --n-c 16 \
  --d-c 512 \
  --bottleneck-mixer-dim 512
```

`--config` defaults to `configs/train.yaml`, so it may be omitted:

```bash
python train.py --encoder siglip2_vitb16 --n-c 64 --d-c 256
```

Configuration resolution is:

```text
config.py dataclass defaults
  -> selected YAML
  -> the five explicitly supplied scientific CLI overrides
  -> operator-only CLI overrides
  -> final validation
```

Every supplied YAML key is checked against the experiment schema. Duplicate keys, unknown keys,
legacy/ignored fields, audit-owned fields, and wrong scalar types fail before model construction.
The YAML’s SHA-256 and resolved path are recorded in the resolved configuration for auditability.
Both are excluded from scientific parity and resume identity: those checks bind the resolved
scientific values, so a comment, W&B label, machine path, or explicitly permitted dataset transfer
cannot create a false mismatch.

## Scientific CLI surface

| Flag | YAML field overridden | Allowed values |
|---|---|---|
| `--data` | `data.dataset` | `ssv2`, `ssv2_tiny`, `ego4d`, `ego4d_tiny` |
| `--encoder` | `encoder.alias` | `vjepa2_vitl16`, `siglip2_vitb16`, `dinov3_vitb16` |
| `--n-c` | `model.n_c` | Positive integer |
| `--d-c` | `model.d_c` | Positive integer divisible by `model.f_c_heads` |
| `--bottleneck-mixer-dim` | `model.bottleneck_mixer_dim` | Positive integer divisible by `model.bottleneck_cross_attn_heads` |

Changing the three shape axes makes checkpoints shape-incompatible. Do not resume across them.

Former scientific flags such as `--lambda-var`, `--lambda-recon`, `--steps`,
`--present-recon-only`, `--whiten-features`, `--decoder-dim`, and learning-rate flags are rejected
by `argparse`; edit their YAML fields instead.

## Operator CLI surface

These flags do not define the scientific recipe. They remain available because they identify or
control a particular execution:

| Flag | Purpose |
|---|---|
| `--config PATH` | Select the YAML recipe. |
| `--resume PATH` | Override `runtime.resume`. |
| `--stage0-only` | Override `runtime.mode` with Stage 0. |
| `--preflight-only` | Override `runtime.mode` with provenance-only preflight. |
| `--resource-preflight` | Override `runtime.mode` with the exact resource preflight. |
| `--provenance-out PATH` | Override `runtime.provenance_out`. |
| `--compare-provenance LEFT RIGHT` | Compare two provenance JSON files and exit. |
| `--require-wandb` | Make W&B failures fatal for this execution. |
| `--wandb-entity`, `--wandb-project`, `--wandb-group`, `--wandb-name`, `--wandb-run-id` | Override YAML W&B identity. |
| `--checkpoint-dir PATH` | Override the YAML output directory; mandatory to isolate parallel runs. |
| `--reset-optimizer` | Resume model state with a new optimizer. |
| `--allow-dataset-transfer` | Explicitly allow a resume dataset fingerprint change. |
| `--allow-legacy-checkpoint` | Permit a legacy checkpoint lacking current identity fields. |

## YAML structure

### Top level

```yaml
seed: 42
debug_shapes: true
checkpoint_dir: /workspace/ckpt/my_run
```

`JEPA_DATA_ROOT` remains the narrow environment override for the dataset parent. It is intentionally
not hard-coded into the checked-in YAML, so local and RunPod layouts can use the same recipe.

### Encoder

```yaml
encoder:
  alias: vjepa2_vitl16
  revision: null
  input_frames: 8
  input_height: 256
  input_width: 256
  precision: bf16
  frame_microbatch: 8
  attention_implementation: sdpa
  hf_cache_dir: /workspace/hf_cache
```

The input contract remains fixed at 8×256×256. A non-null revision must be an immutable
40-character Hub commit SHA. A CLI `--encoder` override may replace a YAML alias only when
`encoder.revision` is `null`; this prevents a commit SHA pinned for one repository from being
silently carried into another.

### Model

```yaml
model:
  n_c: 32
  d_c: 256
  bottleneck_mixer_dim: 256
  bottleneck_convnext_blocks: 2
  bottleneck_cross_attn_heads: 8
  bottleneck_latent_blocks: 3
  f_c_blocks: 6
  f_c_dim: 256
  f_c_heads: 8
  condition_dropout: 0.10
  decoder_dim: 256
  decoder_blocks: 2
  decoder_heads: 8
```

The legacy V-JEPA geometry fields remain dataclass defaults for old checkpoints/tests and are not
listed as normal experiment settings. Runtime encoder geometry comes from `EncoderSpec`.

### Training, objectives, and schedules

All former training hyperparameters live under `train`:

```yaml
train:
  global_batch: 64
  stage1_steps: 15000
  max_steps: 15000

  lr_bottleneck: 0.0001
  lr_coarse_flow: 0.0002
  lr_decoder: 0.0001
  warmup_steps: 1500
  total_latent_steps: 105000
  adam_betas: [0.9, 0.95]
  weight_decay: 0.05

  grad_clip: 0.5
  grad_skip_threshold: 150.0
  agc_enabled: true
  agc_lambda_bottleneck: 0.20
  agc_lambda_coarse_flow: 0.10
  agc_lambda_decoder: 0.20
  agc_eps: 0.001
  instability_warn_grad_norm: 30.0
  instability_warn_l_flow: 1.0

  ema_m_start: 0.996
  ema_m_end: 0.9999
  ema_schedule_steps: 105000

  lambda_var: 0.10
  var_floor_std_target: 1.0
  lambda_cov: 0.0
  lambda_slot: 0.0
  lambda_sigreg: 0.0
  sigreg_warmup_steps: 2000

  lambda_recon: 0.0
  lambda_recon_pred: 0.0
  recon_loss_mode: cosine
  recon_warmup_steps: 2000
  recon_residual_target: false
  recon_mean_momentum: 0.99
  present_recon_only: false
  predict_residual: false

  whiten_features: false
  whiten_stats_path: ""
  whiten_expected_clips: 12800
  whiten_eps: 0.0001

  horizon_k: 4
  frame_stride: 2
  precision: bf16
  log_every: 50
  diag_every: 500
  checkpoint_every: 2500
```

`max_steps` stops the loop. `stage1_steps` remains the LR schedule endpoint, so change them together
when intentionally changing the full schedule. Short smoke runs normally change only `max_steps`
and therefore remain inside the original schedule.

### Data

```yaml
data:
  dataset: ssv2_tiny
  num_workers: 8
  pin_memory: true
  ego4d_selection_manifest: /workspace/ego4d_raw/manifests/selection_manifest.json
  ego4d_download_manifest: /workspace/ego4d_raw/video_540ss_manifest.csv
```

### Runtime and W&B

```yaml
runtime:
  mode: train
  resume: null
  provenance_out: null
  compare_provenance: null
  require_wandb: false
  reset_optimizer: false
  allow_dataset_transfer: false
  allow_legacy_checkpoint: false

wandb:
  entity: null
  project: hjepa-vwm
  group: null
  name: null
  run_id: null
```

Allowed runtime modes are `train`, `stage0`, `preflight`, and `resource_preflight`.

## Migrating an old command

Old:

```bash
python train.py \
  --data ego4d \
  --steps 15000 \
  --encoder vjepa2_vitl16 \
  --batch-size 64 \
  --horizon-k 12 \
  --present-recon-only \
  --lambda-recon 1 \
  --lambda-var 0.5 \
  --lambda-cov 0.01 \
  --bottleneck-mixer-dim 512 \
  --n-c 32 \
  --d-c 256 \
  --decoder-dim 512 \
  --decoder-blocks 4
```

New:

1. Put steps, batch, horizon, mode, losses, and decoder settings in a copied/versioned YAML recipe.
2. Run only the current experiment axes:

```bash
python train.py \
  --config configs/my_ego4d_recipe.yaml \
  --data ego4d \
  --encoder vjepa2_vitl16 \
  --n-c 32 \
  --d-c 256 \
  --bottleneck-mixer-dim 512
```

There is no second hidden YAML/JSON/TOML experiment loader. `config.py` supplies typed fallback
defaults; the selected YAML is the editable run recipe; W&B and checkpoints record the resolved
result.
