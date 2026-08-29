# Two-step rollout locked experiment runbook

This runbook prepares Investigation 23. It does not authorize training until the committed SHA,
Run-60 checkpoint, EGO4D inventory, tests, provenance parity, and GPU preflight all pass.

## Locked identities

- Branch: `exp/two-step-rollout-loss`; use the final commit containing this runbook.
- Run-60 checkpoint:
  `/workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt`
- Required checkpoint SHA-256:
  `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1`
- Required EGO4D dataset fingerprint recorded for the matched Investigation 19/20 lineage:
  `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c`.
  Stop if fresh provenance produces a different fingerprint; do not override it.
- W&B group: `inv023_two_step_rollout`.
- Control name: `Investigation 23 · Rollout endpoint consistency · Control, rollout weight 0`.
- Treatment name:
  `Investigation 23 · Rollout endpoint consistency · Treatment, rollout weight 0.10`.

The shared recipe is `configs/experiments/two_step_rollout_base.yaml`. The two leaf YAML files
inherit it. Operational checkpoint directories and W&B names differ; the only scientific
difference is `train.lambda_rollout` (`0.0` versus `0.10`). The focused config test enforces this.

## Gates

Run the focused test and lint commands in the existing environment:

```bash
.venv/bin/pytest -q tests/test_experiment_config.py tests/test_rollout_loss.py tests/test_two_step_rollout_protocol.py
.venv/bin/ruff check config.py diagnostics.py train.py evaluate_two_step_rollout.py preflight_two_step_rollout.py tests/test_experiment_config.py tests/test_rollout_loss.py tests/test_two_step_rollout_protocol.py
```

Verify the checkpoint bytes:

```bash
sha256sum /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt
```

Materialize both no-step provenance files. These commands load/validate identities but do not train:

```bash
python train.py --config configs/experiments/two_step_rollout_control.yaml --preflight-only --provenance-out /workspace/forensics/inv023/control_provenance.json
python train.py --config configs/experiments/two_step_rollout_treatment.yaml --preflight-only --provenance-out /workspace/forensics/inv023/treatment_provenance.json
```

The generic provenance comparator expects identical scientific fields and therefore will reject the
intentional rollout-weight difference. Pair equivalence is instead enforced by
`test_paired_configs_differ_only_in_rollout_weight`; compare the two provenance files manually for
dataset fingerprint, encoder identity, fixed checkpoint, frozen hashes, data order, and fresh
`coarse_flow_init_hash` before launch.

## Exact bounded GPU preflight

Run on CUDA before either arm. It allocates `(64,64,512)` present/future tensors, builds the bf16
autocast graph with one teacher-forced and two rollout `F_c` calls, calls backward, reports peak
allocated/reserved CUDA memory, and creates no optimizer, checkpoint, or training loop:

```bash
python preflight_two_step_rollout.py --config configs/experiments/two_step_rollout_treatment.yaml
```

CPU execution is refused. If batch 64 OOMs, stop. Select one common reduced batch by changing both
`train.global_batch` and `protocol.common_global_batch` once in the shared base YAML, rerun tests and
the treatment-shaped GPU preflight, commit that protocol revision, and use it for **both** arms.
Never reduce only the treatment batch.

## Exact training commands (do not run during implementation)

Control:

```bash
PYTHONUNBUFFERED=1 python train.py --config configs/experiments/two_step_rollout_control.yaml
```

Treatment:

```bash
PYTHONUNBUFFERED=1 python train.py --config configs/experiments/two_step_rollout_treatment.yaml
```

Expected checkpoints:

- `/workspace/ckpt/inv023_two_step_rollout_control/phase1_step2500.pt`
- `/workspace/ckpt/inv023_two_step_rollout_control/phase1_step5000.pt`
- `/workspace/ckpt/inv023_two_step_rollout_treatment/phase1_step2500.pt`
- `/workspace/ckpt/inv023_two_step_rollout_treatment/phase1_step5000.pt`

Use separate deterministic tmux sessions/logs and run at most one arm per GPU. Do not resume into a
nonempty output directory unless exact checkpoint provenance has been validated.

## Exact checkpoint evaluation commands

```bash
python evaluate_two_step_rollout.py --config configs/experiments/two_step_rollout_control.yaml --checkpoints /workspace/ckpt/inv023_two_step_rollout_control/phase1_step2500.pt /workspace/ckpt/inv023_two_step_rollout_control/phase1_step5000.pt --output /workspace/forensics/inv023/control_rollout_evaluation.json
```

```bash
python evaluate_two_step_rollout.py --config configs/experiments/two_step_rollout_treatment.yaml --checkpoints /workspace/ckpt/inv023_two_step_rollout_treatment/phase1_step2500.pt /workspace/ckpt/inv023_two_step_rollout_treatment/phase1_step5000.pt --output /workspace/forensics/inv023/treatment_rollout_evaluation.json
```

Each `hjepa-two-step-rollout-evaluation-v1` JSON records checkpoint path/SHA, resolved config hash,
`F_c` model hash, ordered fixed-batch sample IDs and hash, dataset fingerprint, solver grid, full
1/2/4/8 normal/shuffled/zero endpoint results, copy/batch-mean ratios, displacement cosine/norm
ratio, validity, condition-shuffle degradation, and teacher-forced endpoint/velocity metrics.

W&B remains focused: existing optimization/stability/collapse metrics are retained; diagnostic
cadence adds only stable `eval/rollout_{4,8}_copy_ratio` and
`eval/rollout_{4,8}_condition_shuffle_degradation`. The JSON is authoritative for the complete grid.

## Decisions and stops

- **Success:** treatment genuine 4/8-step endpoint/copy and endpoint/batch-mean ratios are below 1,
  beat control at both 2500 and 5000, retain positive conditioning degradation, improve direction
  (provisional cosine ≥0.30 and ≥+0.05 versus control), avoid near-zero/overscaled displacement,
  and worsen teacher-forced error by no more than 20% without instability.
- **Partial success:** persistent matched improvement at both checkpoints but one absolute gate is
  missed. Exposure contributes but is insufficient.
- **Failure:** rollout training loss falls without fixed-batch 4/8-step improvement, both arms move
  equally, or treatment worsens rollout, conditioning, or teacher-forced behavior materially.
- **Stop/invalid:** any NaN, skipped update, instability warning, frozen-state/hash change, dataset
  fingerprint mismatch, checkpoint/config mismatch, or sustained treatment-only gradient/AGC
  collapse. One isolated AGC clip is diagnostic, not by itself a stop.

## Locked 15,000-step confirmation continuation

The completed 5,000-step arms remain immutable. The confirmation protocol references their
2,500/5,000 checkpoints by path and SHA-256, restores the exact step-5,000
model/optimizer/RNG/sampler state, and writes only to new confirmation directories.

- Control source: /workspace/ckpt/inv023_two_step_rollout_control/phase1_step5000.pt,
  SHA-256 fa7bf9546a834d6db7f3d80f25fd0cb0a42e1dba662930f08900a42e27d2a429.
- Treatment source: /workspace/ckpt/inv023_two_step_rollout_treatment/phase1_step5000.pt,
  SHA-256 0da168b31fcaf008a3488c3930d205115ebe94495c1fe4097f255fc1b386090f.
- New checkpoints per arm: 5500, 7500, 10000, 15000.
- Deterministic evaluations: referenced original 5000, then new 7500, 10000, 15000.
- Inspect every 50-step log from 5350 through 5450; preserve the step-5500 checkpoint.
- Confirmation resumes set runtime.resume_wandb_run false; each arm creates a new W&B run
  whose first training log is global step 5000.

No-step validation:

    python train.py --config configs/experiments/two_step_rollout_confirmation_control.yaml --preflight-only --provenance-out /workspace/forensics/inv023_confirmation/control_provenance.json
    python train.py --config configs/experiments/two_step_rollout_confirmation_treatment.yaml --preflight-only --provenance-out /workspace/forensics/inv023_confirmation/treatment_provenance.json

Sequential launch commands (do not run without explicit approval):

    PYTHONUNBUFFERED=1 python train.py --config configs/experiments/two_step_rollout_confirmation_control.yaml
    PYTHONUNBUFFERED=1 python train.py --config configs/experiments/two_step_rollout_confirmation_treatment.yaml

Deterministic confirmation evaluations:

    python evaluate_two_step_rollout.py --config configs/experiments/two_step_rollout_confirmation_control.yaml --checkpoints /workspace/ckpt/inv023_two_step_rollout_control/phase1_step5000.pt /workspace/ckpt/inv023_two_step_rollout_confirmation_control/phase1_step7500.pt /workspace/ckpt/inv023_two_step_rollout_confirmation_control/phase1_step10000.pt /workspace/ckpt/inv023_two_step_rollout_confirmation_control/phase1_step15000.pt --output /workspace/forensics/inv023_confirmation/control_rollout_evaluation.json
    python evaluate_two_step_rollout.py --config configs/experiments/two_step_rollout_confirmation_treatment.yaml --checkpoints /workspace/ckpt/inv023_two_step_rollout_treatment/phase1_step5000.pt /workspace/ckpt/inv023_two_step_rollout_confirmation_treatment/phase1_step7500.pt /workspace/ckpt/inv023_two_step_rollout_confirmation_treatment/phase1_step10000.pt /workspace/ckpt/inv023_two_step_rollout_confirmation_treatment/phase1_step15000.pt --output /workspace/forensics/inv023_confirmation/treatment_rollout_evaluation.json
