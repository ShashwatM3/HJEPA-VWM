# Phase 1 experiment configuration

Phase 1 has exactly one experiment configuration file:
[`configs/train.yaml`](configs/train.yaml).

`train.py` resolves that file on every invocation. There is no `--config` flag and no second
experiment YAML to select, copy, or keep synchronized.

## Configuration order

```text
config.py fallback defaults
  -> configs/train.yaml
  -> five explicitly supplied scientific CLI overrides
  -> operator-only CLI overrides
  -> final validation
```

The YAML file is the editable source of truth for the complete experiment background. It is divided
into named categories, and every editable value has an inline comment describing its categorical
choices, hard constraint, or reasonable tuning range.

The loader rejects duplicate keys, unknown keys, ignored legacy fields, audit-owned fields, and
wrong scalar types before model construction. Documented categorical choices and hard
range/order constraints are validated before paid work starts. `encoder.frame_microbatch: null`
selects the tested value after the encoder hot override (V-JEPA2/SigLIP 2 = 8, DINOv3 = 32), and
the resolved integer is what provenance records. The YAML's resolved path and SHA-256 are recorded
for auditability but excluded from scientific parity and resume identity; parity binds the
resolved scientific values.

## Five scientific CLI overrides

Investigations 012–017 were reviewed as one recent research window. The primary repeated experiment
axes—especially in Investigations 016 and 017—were:

| Flag | YAML field overridden | Recent evidence |
|---|---|---|
| `--data` | `data.dataset` | SSv2-to-EGO4D transfer in Investigation 016 |
| `--encoder` | `encoder.alias` | V-JEPA2, SigLIP 2, and DINOv3 lanes in Investigations 016–017 |
| `--n-c` | `model.n_c` | 32/64/128, followed by 16/32/64 |
| `--d-c` | `model.d_c` | 128/256/512 latent-width factorial |
| `--bottleneck-mixer-dim` | `model.bottleneck_mixer_dim` | 512/1024 internal-width comparison |

Whitening, reconstruction mode/target, geometry losses, decoder shape, schedules, optimizer, and
diagnostic cadence remain together in the YAML recipe because they are coupled background choices.

Normal launch:

```bash
python train.py
```

Launch with hot experiment axes:

```bash
python train.py \
  --data ego4d \
  --encoder dinov3_vitb16 \
  --n-c 16 \
  --d-c 512 \
  --bottleneck-mixer-dim 512
```

Changing the three shape axes makes checkpoints shape-incompatible. Do not resume across them.
When `encoder.revision` is non-null, `--encoder` may not change the YAML alias because that would
carry a repository-specific commit SHA into another encoder.

## Operator CLI controls

These controls identify or operate one execution; they do not add scientific hyperparameters:

| Flag | Purpose |
|---|---|
| `--resume PATH` | Override `runtime.resume`. |
| `--stage0-only` | Override `runtime.mode` with Stage 0. |
| `--preflight-only` | Override `runtime.mode` with provenance-only preflight. |
| `--resource-preflight` | Override `runtime.mode` with the exact resource preflight. |
| `--provenance-out PATH` | Override `runtime.provenance_out`. |
| `--compare-provenance LEFT RIGHT` | Compare two provenance JSON files and exit. |
| `--require-wandb` | Make W&B failures fatal for this execution. |
| `--wandb-entity`, `--wandb-project`, `--wandb-group`, `--wandb-name`, `--wandb-run-id` | Override the YAML W&B identity. |
| `--checkpoint-dir PATH` | Override the output directory; isolate parallel runs. |
| `--reset-optimizer` | Resume model state with a new optimizer. |
| `--allow-dataset-transfer` | Explicitly allow a guarded dataset fingerprint change. |
| `--allow-legacy-checkpoint` | Permit a legacy checkpoint lacking current identity fields. |

## Starting a new experiment

1. Edit the appropriate categorized values in `configs/train.yaml`.
2. Review the inline allowed-value/range comments and keep coupled settings coherent.
3. Commit the YAML change so the exact recipe is versioned.
4. Launch with `python train.py` plus only the hot axes that define the comparison arm.

Former scientific flags such as `--steps`, `--lambda-var`, `--lambda-recon`,
`--present-recon-only`, `--whiten-features`, `--decoder-dim`, and learning-rate flags are rejected.
Their corresponding values are edited only in `configs/train.yaml`.
