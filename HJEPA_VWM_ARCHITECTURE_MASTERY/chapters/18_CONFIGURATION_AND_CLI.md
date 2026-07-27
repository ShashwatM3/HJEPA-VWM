# 18 — Configuration and CLI

## Configuration layers

`Config` contains:

```text
Config
├── EncoderConfig
├── ModelConfig
├── TrainConfig
├── DataConfig
├── seed
├── hf_cache_dir synchronization
├── debug_shapes
└── checkpoint_dir
```

Construction starts from dataclass defaults. `train.py` applies CLI overrides, then
`finalize_training_config` validates/normalizes relationships before any module is built.

## Encoder config

| Field | Default |
|---|---|
| alias | `vjepa2_vitl16` |
| revision | `None` → registry's immutable tested revision |
| input frames | 8 |
| height/width | 256/256 |
| precision | `bf16` |
| frame microbatch | 8 |
| attention implementation | `sdpa` |
| HF cache | `/workspace/hf_cache` |

`revision=None` never means mutable Hub `main`.

## Model config

| Field | Default |
|---|---:|
| `n_c` | 32 |
| `d_c` | 256 |
| bottleneck internal `M` | 256 |
| ConvNeXt blocks | 2 |
| bottleneck cross heads | 8 |
| latent blocks | 3 |
| flow blocks/heads | 6/8 |
| condition dropout | 0.10 |
| decoder width/blocks/heads | 256/2/8 |

Legacy V-JEPA geometry fields (`d_e`, tubelet, patch, derived `n_ctx`) exist for old checkpoint/tests.
Production detailed geometry comes from the resolved `EncoderSpec`, not those fields.

`f_c_dim=256` is a retained config field, but current `CoarseFlow` uses external `d_c` as its width.
Do not treat `f_c_dim` as an independent working architecture knob.

## Training config

The full numeric default ledger is in [17](17_EXACT_NUMBER_ATLAS.md). Key mode defaults:

```text
present_recon_only      false
predict_residual        false
recon_residual_target   false
whiten_features         false
recon_loss_mode         cosine
all optional loss weights except lambda_var = 0
```

## Data config

```text
JEPA_DATA_ROOT, default /workspace/data
dataset, default ssv2_tiny
workers 8
pin_memory true
```

EGO selection and authoritative download-tier manifest paths are explicit config, not discovered
heuristically.

## Core command flags

| Flag | Parser default | Effect |
|---|---|---|
| `--data` | `ssv2_tiny` | one of four dataset roots |
| `--steps` | 15000 | loop `max_steps`, not schedule denominator |
| `--resume PATH` | none | strict continuation source |
| `--seed` | 42 | all named seed streams |
| `--stage0-only` | false | one real sanity update and EMA/freeze checks |
| `--batch-size` | config 64 | physical/global batch in this single-process trainer |

## Encoder flags

| Flag | Choices/default |
|---|---|
| `--encoder` | `vjepa2_vitl16`, `siglip2_vitb16`, `dinov3_vitb16`; V-JEPA default |
| `--encoder-revision` | registry default if omitted |
| `--encoder-precision` | `fp32` or `bf16` |
| `--encoder-frame-microbatch` | config 8 |
| `--encoder-attention-implementation` | config `sdpa` |
| `--hf-cache-dir` | `/workspace/hf_cache` |

An arbitrary revision is validated as immutable. BF16 requires CUDA.

## Preflight and provenance flags

| Flag | Meaning |
|---|---|
| `--resource-preflight` | execute one exact forward/backward/optimizer/diagnostic recipe and write provenance |
| `--preflight-only` | resolve/validate provenance without training update |
| `--provenance-out PATH` | explicit JSON destination |
| `--compare-provenance LEFT RIGHT` | compare encoder-independent common identity and exit |

`--compare-provenance` is an alternate command path; it does not build/train the model.

## W&B flags

| Flag | Meaning |
|---|---|
| `--require-wandb` | make init/log/finalization fatal on failure |
| `--wandb-entity` | tracking entity |
| `--wandb-project` | default `hjepa-vwm` |
| `--wandb-group` | controlled comparison group |
| `--wandb-name` | human-readable compliant name |
| `--wandb-run-id` | explicit continuation/new ID |

Credentials are provided by environment/login, never these provenance fields.

## Resume-policy flags

| Flag | Consequence |
|---|---|
| `--reset-optimizer` | keep fresh AdamW instead of loading compatible state |
| `--allow-dataset-transfer` | permit only dataset-owned provenance differences |
| `--allow-legacy-checkpoint` | enable weaker historical reader |

These are scientific changes, not convenience fixes.

## Cadence flags

| Flag | Config default |
|---|---:|
| `--log-every` | 50 |
| `--diag-every` | 500 |

Checkpoint cadence is config-only at 2,500 in the current CLI.

## Loss/task flags

| Flag | Default/effect |
|---|---|
| `--horizon-k` | 4 decoded-frame indices |
| `--lambda-var` | 0.10 |
| `--lambda-cov` | 0 |
| `--lambda-slot` | 0 |
| `--lambda-sigreg` | 0 |
| `--sigreg-warmup-steps` | 2,000 |
| `--lambda-recon` | 0; present feature anchor |
| `--lambda-recon-pred` | 0; predicted-future feature anchor |
| `--recon-loss-mode` | `cosine`; alternative `relative_mse` |
| `--recon-residual-target` | feature target `e-mean` |
| `--recon-mean-momentum` | 0.99 |
| `--recon-warmup-steps` | 2,000 |
| `--present-recon-only` | skip target and `F_c` |
| `--predict-residual` | flow target is EMA temporal delta |

## Whitening flags

| Flag | Default/effect |
|---|---|
| `--whiten-features` | off |
| `--whiten-stats-path` | required when on |
| `--whiten-eps` | `1e-4` |
| `--whiten-expected-clips` | 12,800 |

Whitened and raw reconstruction losses are not numerically comparable.

## Optimizer flags

| Flag | Default |
|---|---:|
| `--lr-bottleneck` | `1e-4` |
| `--lr-coarse-flow` | `2e-4` |
| `--lr-decoder` | `1e-4` |
| `--no-agc` | AGC otherwise enabled |
| `--agc-lambda-bottleneck` | 0.20 |
| `--agc-lambda-coarse-flow` | 0.10 |
| `--grad-skip-threshold` | 150 |

Decoder AGC factor 0.20 and global clip 0.5 are currently config-only.

Peak LRs are reconstructed from current config after resume, not read from a checkpoint's scheduled
group LR. Otherwise cosine decay would be applied twice.

## Architecture flags

| Flag | Default | Compatibility |
|---|---:|---|
| `--bottleneck-mixer-dim` | 256 | changes most of `B` |
| `--bottleneck-latent-blocks` | 3 | changes `B` depth |
| `--n-c` | 32 | changes queries, flow sequence, decoder memory |
| `--d-c` | 256 | changes `B` output, all `F_c`, decoder input |
| `--decoder-dim` | 256 | changes `D` |
| `--decoder-blocks` | 2 | changes `D` |
| `--checkpoint-dir` | `/workspace/checkpoints` | real runs use unique `/workspace/ckpt/tag` |

Do not resume across shape-changing architecture flags.

## Validation/normalization rules

`finalize_training_config` enforces:

- `M>0` and divisible by eight bottleneck cross heads;
- `N_c>0`;
- `D_c>0` and divisible by eight flow heads;
- batch greater than one;
- frame microbatch positive;
- decoder LR positive;
- valid reconstruction mode;
- present-only requires positive present reconstruction;
- present-only zeroes prediction reconstruction and disables temporal residual with a message;
- residual feature target requires an active reconstruction anchor;
- mean momentum is inside `(0,1)`;
- whitening requires stats path, positive epsilon, and positive exact clip expectation.

Encoder construction adds input geometry, alias, revision, precision, and layout checks.

## A shipped-default command

The shortest invocation:

```bash
python train.py
```

means V-JEPA2, SSv2 tiny, full-prediction, `k=4`, 15k steps, variance floor only, no reconstruction,
raw/unwhitened features, optional W&B behavior, and legacy checkpoint destination. It is not the
recommended paid-run command because it lacks `--require-wandb` and a unique checkpoint directory.

## Example recipe interpretation

```bash
python train.py \
  --data ego4d \
  --encoder dinov3_vitb16 \
  --present-recon-only \
  --bottleneck-mixer-dim 512 \
  --n-c 32 --d-c 256 \
  --lambda-recon 1 \
  --lambda-var 0.5 \
  --lambda-cov 0.01 \
  --checkpoint-dir /workspace/ckpt/example
```

This is a present-representation experiment. `k`, `F_c` prediction gates, and target encoding are
irrelevant at runtime. It does not become a full world-model success because it uses DINO or EGO.

## Hidden interaction traps

- `--steps 2000` does not make a 2k cosine schedule.
- Passing `--whiten-stats-path` without `--whiten-features` does not activate whitening.
- Passing prediction reconstruction with present-only is normalized back to zero.
- `--predict-residual` and `--recon-residual-target` act in different spaces.
- `--lambda-cov 0` still computes/logs `L_cov`.
- `--lambda-sigreg 0` still computes `L_SIG` with an isolated generator.
- A decoder always exists and is in AdamW, but with no recon forward its gradients are `None`.
- `f_c_dim` legacy config does not independently override current `F_c` width.
