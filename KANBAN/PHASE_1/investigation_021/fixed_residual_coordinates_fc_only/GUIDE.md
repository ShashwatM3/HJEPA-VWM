# Guide — launch the fixed-coordinate residual experiment

This is the complete remote-shell procedure **after** finishing
`AGENT_FILES/SETUPS/NEW_POD.md` through step 5 (`wandb login`). It launches one full 15,000-step
scientific run on exactly one GPU. It does not launch a smoke ablation.

Run every command below inside the same remote SSH shell unless a block explicitly says it is a
monitoring command for a later connection. If the shell disconnects before launch, reconnect and
repeat the environment-export command before continuing.

Do not use `--resume` for the initial launch. This experiment must warm-start from the registered
Investigation-019 present-only checkpoint so `B`, `D`, fresh `B_EMA=B`, fresh `F_c`, optimizer,
schedule, sampler, RNG, and W&B state reproduce the intended step-0 boundary.

## Fixed experiment identity

Implementation prerequisite:

```text
0c1d343ce19258d5d575ea17a1654a46b32abf85
```

Source checkpoint:

```text
/workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt
```

Source SHA-256:

```text
931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1
```

W&B display name:

```text
Investigation 21 · Fixed residual coordinates · Coarse flow only, DINOv3 64 slots by 512
```

## 1. Enter the repository and export the locked identities

Remote SSH shell — run this command:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
```

Remote SSH shell — run this one `export` command:

```bash
export HF_HOME=/workspace/hf_cache INV021_BRANCH=phase1-v0.2-frozen-encoder INV021_IMPLEMENTATION_COMMIT=0c1d343ce19258d5d575ea17a1654a46b32abf85 INV021_SOURCE_CHECKPOINT=/workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt INV021_SOURCE_SHA256=931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1 INV021_SOURCE_WANDB_ID=60yaqw6d INV021_SOURCE_FEATURE_FINGERPRINT=963cf988fb16b1e3971f952ade8afe90b29cef7dfd7103e4f312dada5c0eb415 INV021_SOURCE_DATASET_FINGERPRINT=df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c INV021_ENCODER=dinov3_vitb16 INV021_N_C=64 INV021_D_C=512 INV021_M=512 INV021_OUTPUT=/workspace/ckpt/inv021_fixed_residual_coordinates/fc_only INV021_PREFLIGHT=/workspace/hierarchal-jepa-flow-world-model/logs/inv021_preflight INV021_LOG=/workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log INV021_TMUX=inv021_fc_only INV021_WANDB_GROUP=inv021_fixed_residual_coordinates INV021_WANDB_NAME='Investigation 21 · Fixed residual coordinates · Coarse flow only, DINOv3 64 slots by 512'
```

## 2. Synchronize a clean, reproducible checkout

`NEW_POD.md` already cloned and checked out the branch. This step updates it without resetting or
cleaning over remote work.

Remote SSH shell — run this command:

```bash
bash -lc 'set -Eeuo pipefail; cd /workspace/hierarchal-jepa-flow-world-model; git status --short --branch; git fetch origin; git checkout phase1-v0.2-frozen-encoder; git pull --ff-only origin phase1-v0.2-frozen-encoder; git merge-base --is-ancestor 0c1d343ce19258d5d575ea17a1654a46b32abf85 HEAD; git rev-parse HEAD; git status --short --branch; test -z "$(git status --porcelain)"'
```

Stop if the checkout is dirty, the pull is not fast-forwardable, or the implementation commit is
not an ancestor of `HEAD`. Do not use `git reset --hard`, `git clean`, or hand-copy Python files.
Record the printed `HEAD`; that is the candidate launch commit.

## 3. Verify the one-GPU host, data, dependencies, and idle target

The experiment uses only logical GPU 0 even if the pod exposes more devices.

Remote SSH shell — run this command:

```bash
bash -lc 'set -Eeuo pipefail; test "$(nvidia-smi -L | wc -l)" -ge 1; nvidia-smi -L; test -d /workspace/data/ego4d/train; test -d /workspace/data/ego4d/validation; test -f /workspace/data/ego4d/chunk_manifest.json; test -f /workspace/ego4d_raw/manifests/selection_manifest.json; test -s /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt; python3 -c "import torch, transformers, decord, wandb; assert transformers.__version__ == \"4.57.6\"; print(\"DEPENDENCIES_OK\", torch.__version__, transformers.__version__)"; tmux list-sessions 2>/dev/null || true; pgrep -af "python.*train.py" || true; nvidia-smi'
```

An unrelated live training process or occupied GPU is a scheduling conflict. Inspect it; do not
kill it. Continue only when GPU 0 is available for this run.

Confirm that the new scientific mode is present:

```bash
python3 train.py --help | grep -E -- '--optimization-scope|--temporal-target|--warm-start-from|--resource-preflight'
```

## 4. Run local correctness gates on the pod

Run the full suite:

```bash
python3 -m pytest -q
```

Require all tests to pass. The implementation commit was locally verified at 254 passing tests;
the exact count may increase on a newer clean launch commit, but failures are not acceptable.

Run the model smoke:

```bash
python3 -c "from models import smoke_test_models; smoke_test_models()"
```

Run the diagnostic smoke:

```bash
python3 -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

Run the real pinned DINOv3 adapter smoke on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python3 encoders.py --smoke --encoder dinov3_vitb16 --batch-size 1 --hf-cache-dir /workspace/hf_cache
```

Do not run `--stage0-only --optimization-scope fc_only`: the implementation deliberately rejects
Stage 0 for a mode that requires a learned warm start. The exact no-step and one-step warm-started
preflights below replace that generic sanity path.

## 5. Verify the registered source and effective recipe

This gate reads the checkpoint and current YAML without mutating paid-run state.

Remote SSH shell — run this command:

```bash
python3 - <<'PY'
import os
from pathlib import Path

import torch

from config import load_experiment_config
from provenance import optimization_contract, sha256_file
from train import EXPERIMENT_CONFIG_PATH, finalize_training_config

path = Path(os.environ["INV021_SOURCE_CHECKPOINT"])
digest = sha256_file(path)
assert digest == os.environ["INV021_SOURCE_SHA256"], digest
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
assert checkpoint["schema"] == "hjepa-phase1-checkpoint-v2"
assert checkpoint["next_step"] == 15000
assert checkpoint["wandb_run_id"] == os.environ["INV021_SOURCE_WANDB_ID"]
assert checkpoint["feature_fingerprint"] == os.environ["INV021_SOURCE_FEATURE_FINGERPRINT"]
assert checkpoint["dataset_identity"]["fingerprint"] == os.environ["INV021_SOURCE_DATASET_FINGERPRINT"]
source = checkpoint["config"]
assert source["encoder"]["alias"] == os.environ["INV021_ENCODER"]
assert source["model"]["n_c"] == int(os.environ["INV021_N_C"])
assert source["model"]["d_c"] == int(os.environ["INV021_D_C"])
assert source["model"]["bottleneck_mixer_dim"] == int(os.environ["INV021_M"])
assert source["model"]["bottleneck_latent_blocks"] == 3
assert source["model"]["decoder_dim"] == 512
assert source["model"]["decoder_blocks"] == 4
assert source["train"]["present_recon_only"] is True
assert source["train"]["whiten_features"] is False
assert source["train"]["recon_residual_target"] is False
assert "bottleneck" in checkpoint and "decoder" in checkpoint

cfg = load_experiment_config(EXPERIMENT_CONFIG_PATH).config
cfg.data.dataset = "ego4d"
cfg.encoder.alias = os.environ["INV021_ENCODER"]
cfg.model.n_c = int(os.environ["INV021_N_C"])
cfg.model.d_c = int(os.environ["INV021_D_C"])
cfg.model.bottleneck_mixer_dim = int(os.environ["INV021_M"])
cfg.train.predict_residual = True
cfg.train.optimization_scope = "fc_only"
finalize_training_config(cfg)
assert not cfg.train.present_recon_only
assert cfg.train.lambda_recon == 1.0 and cfg.train.lambda_recon_pred == 0.0
assert cfg.train.recon_loss_mode == "cosine" and not cfg.train.recon_residual_target
assert cfg.train.lambda_var == 0.5 and cfg.train.lambda_cov == 0.01
assert cfg.train.lambda_sigreg == 0.0 and cfg.train.lambda_slot == 0.0
assert not cfg.train.whiten_features
assert cfg.train.horizon_k == 12 and cfg.train.frame_stride == 2
assert cfg.train.global_batch == 64
assert cfg.train.max_steps == cfg.train.stage1_steps == 15000
assert cfg.train.warmup_steps == 1500 and cfg.train.lr_coarse_flow == 0.0001
assert cfg.model.f_c_blocks == 6 and cfg.model.f_c_heads == 8
assert optimization_contract(cfg) == {
    "scope": "fc_only",
    "trainable_modules": ["F_c"],
    "frozen_modules": ["B", "B_EMA", "D"],
    "ema_updates": False,
    "optimized_objective_terms": ["L_flow"],
}
print("INV021_SOURCE_AND_RECIPE_OK", digest)
PY
```

## 6. Materialize and inspect exact no-step provenance

First require fresh preflight paths. Never overwrite old evidence.

```bash
bash -lc 'set -Eeuo pipefail; test ! -e /workspace/hierarchal-jepa-flow-world-model/logs/inv021_preflight; mkdir -p /workspace/hierarchal-jepa-flow-world-model/logs/inv021_preflight'
```

Materialize the exact run without an optimizer step:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 train.py --data ego4d --encoder dinov3_vitb16 --n-c 64 --d-c 512 --bottleneck-mixer-dim 512 --warm-start-from /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt --temporal-target residual --optimization-scope fc_only --preflight-only --provenance-out /workspace/hierarchal-jepa-flow-world-model/logs/inv021_preflight/run_provenance.json --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm --wandb-group inv021_fixed_residual_coordinates --wandb-name 'Investigation 21 · Fixed residual coordinates · Coarse flow only, DINOv3 64 slots by 512'
```

The complete EGO4D identity opens every clip to bind decoded frame counts. Several minutes of CPU
and storage activity with an idle GPU can be normal. Do not kill it merely because output is quiet.

Inspect the resulting identity:

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["INV021_PREFLIGHT"]) / "run_provenance.json"
p = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "scope": "fc_only",
    "trainable_modules": ["F_c"],
    "frozen_modules": ["B", "B_EMA", "D"],
    "ema_updates": False,
    "optimized_objective_terms": ["L_flow"],
}
assert p["schema"] == "hjepa-run-provenance-v1"
assert p["optimization_contract"] == expected
assert p["common"]["optimization_contract"] == expected
assert set(p["frozen_state_hashes"]) == {"B", "B_EMA", "D"}
assert p["frozen_state_hashes"] == p["common"]["frozen_state_hashes"]
assert all(len(value) == 64 for value in p["frozen_state_hashes"].values())
assert p["resolved_config"]["train"]["optimization_scope"] == "fc_only"
assert p["resolved_config"]["train"]["predict_residual"] is True
assert p["dataset_identity"]["fingerprint"] == os.environ["INV021_SOURCE_DATASET_FINGERPRINT"]
assert p["feature_fingerprint"] == os.environ["INV021_SOURCE_FEATURE_FINGERPRINT"]
assert p["warm_start"]["checkpoint_sha256"] == os.environ["INV021_SOURCE_SHA256"]
assert p["warm_start"]["source_wandb_run_id"] == os.environ["INV021_SOURCE_WANDB_ID"]
assert p["warm_start"]["target_bottleneck_policy"] == "fresh_exact_copy_of_loaded_online_bottleneck"
assert p["warm_start"]["component_state_hashes"]["bottleneck"] == p["warm_start"]["component_state_hashes"]["target_bottleneck"]
print("INV021_PROVENANCE_OK", p["common_identity"], p["common"]["trainable_init_hash"], p["frozen_state_hashes"])
PY
```

## 7. Run the exact one-step resource and immutability gate

This is the only pre-launch training step. It exercises the full batch, two encoder passes,
warm-started fixed representation, residual target, backward pass, `F_c`-only optimizer, AGC,
clipping, diagnostic batch, and frozen-state hash verification on GPU 0.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 train.py --data ego4d --encoder dinov3_vitb16 --n-c 64 --d-c 512 --bottleneck-mixer-dim 512 --warm-start-from /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt --temporal-target residual --optimization-scope fc_only --resource-preflight --provenance-out /workspace/hierarchal-jepa-flow-world-model/logs/inv021_preflight/resource.json 2>&1 | tee /workspace/hierarchal-jepa-flow-world-model/logs/inv021_preflight/resource.log
```

Require the command to exit zero. Then inspect its measured contract:

```bash
python3 - <<'PY'
import json
import math
import os
from pathlib import Path

path = Path(os.environ["INV021_PREFLIGHT"]) / "resource.json"
p = json.loads(path.read_text(encoding="utf-8"))
contract = p["optimization_contract"]
metrics = p["resource_preflight"]["metrics"]
assert contract["scope"] == "fc_only"
assert contract["trainable_modules"] == ["F_c"]
assert contract["frozen_modules"] == ["B", "B_EMA", "D"]
assert contract["ema_updates"] is False
assert contract["optimized_objective_terms"] == ["L_flow"]
assert p["frozen_state_hashes"] == p["common"]["frozen_state_hashes"]
assert p["resolved_config"]["train"]["predict_residual"] is True
assert metrics["prediction_active"] == 1.0
assert metrics["grad_skipped"] == 0.0
assert math.isfinite(metrics["loss"]) and math.isfinite(metrics["L_flow"])
assert abs(metrics["loss"] - metrics["L_flow"]) < 1e-7
assert p["resource_preflight"]["batch_size"] == 64
assert p["resource_preflight"]["total_peak_memory_bytes"] > 0
print("INV021_RESOURCE_OK", p["resource_preflight"])
PY
```

An OOM, source/provenance failure, frozen-state failure, nonfinite metric, or skipped step is a hard
stop. Do not lower batch size, change the model, or silently switch GPU while keeping this run
identity.

## 8. Prove output names are unused

Remote SSH shell — run this collision gate:

```bash
bash -lc 'set -Eeuo pipefail; test ! -e /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only; test ! -e /workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log; if tmux has-session -t inv021_fc_only 2>/dev/null; then exit 1; fi; mkdir -p /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only /workspace/hierarchal-jepa-flow-world-model/logs'
```

If this fails, inspect the existing artifacts/session. Do not delete or overwrite them. Determine
whether they are this registered run before deciding between monitoring, exact resume, or a new
operational suffix recorded in the KANBAN.

## 9. Launch the paid run on GPU 0

This is one `tmux` command. Training stays in the foreground inside the detached session, and
`pipefail` makes a failed Python process fail the session rather than being hidden by `tee`.

```bash
tmux new-session -d -s inv021_fc_only "cd /workspace/hierarchal-jepa-flow-world-model && set -Eeuo pipefail && export HF_HOME=/workspace/hf_cache && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 train.py --data ego4d --encoder dinov3_vitb16 --n-c 64 --d-c 512 --bottleneck-mixer-dim 512 --warm-start-from /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt --temporal-target residual --optimization-scope fc_only --checkpoint-dir /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only --provenance-out /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only/run_provenance.json --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm --wandb-group inv021_fixed_residual_coordinates --wandb-name 'Investigation 21 · Fixed residual coordinates · Coarse flow only, DINOv3 64 slots by 512' --require-wandb 2>&1 | tee /workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log"
```

## 10. Prove launch immediately

Confirm the session exists:

```bash
tmux list-sessions
```

Confirm the exact training process exists:

```bash
pgrep -af "python.*train.py.*optimization-scope fc_only"
```

Confirm GPU 0 is occupied by the intended process:

```bash
nvidia-smi
```

Inspect the live pane:

```bash
tmux capture-pane -p -t inv021_fc_only -S -160
```

Inspect the persistent log:

```bash
tail -n 160 /workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log
```

Wait boundedly for persisted provenance:

```bash
timeout 1800 bash -lc 'until test -s /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only/run_provenance.json; do sleep 5; done'
```

Re-run the provenance assertions against the paid run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("/workspace/ckpt/inv021_fixed_residual_coordinates/fc_only/run_provenance.json")
p = json.loads(path.read_text(encoding="utf-8"))
assert p["optimization_contract"] == {
    "scope": "fc_only",
    "trainable_modules": ["F_c"],
    "frozen_modules": ["B", "B_EMA", "D"],
    "ema_updates": False,
    "optimized_objective_terms": ["L_flow"],
}
assert p["resolved_config"]["train"]["predict_residual"] is True
assert set(p["frozen_state_hashes"]) == {"B", "B_EMA", "D"}
assert p["frozen_state_hashes"] == p["common"]["frozen_state_hashes"]
print("INV021_PAID_PROVENANCE_OK", p["common_identity"], p["tracking_identity"])
PY
```

Wait boundedly for the first logged update:

```bash
timeout 1800 bash -lc 'until grep -q "step=0" /workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log; do sleep 5; done'
```

Inspect it:

```bash
grep -m 1 "step=0" /workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log
```

Require all of the following before calling the launch valid:

- a fresh W&B ID/URL with the exact display name and group;
- global step `0`, `prediction_active=1`, finite nonzero `L_flow`, and `grad_skipped=0`;
- logged `loss` equals `L_flow` up to float serialization;
- provenance says residual target, `fc_only`, only `F_c` trainable, no EMA updates, and exact
  `B/B_EMA/D` frozen hashes;
- source checkpoint SHA matches the registered value;
- only GPU 0 is used by this experiment;
- log and checkpoint paths match this guide.

Add the W&B ID and clean launch commit to `DESCRIPTION.md`/`OBSERVATIONS.md` only after these checks.

## 11. Early scientific and validity tripwires

At diagnostic steps, these fixed-representation quantities should stay constant up to numerical
tolerance on the deterministic validation population:

```text
coarse_copy_loss
c_std_mean
c_dead_dim_frac
c_cross_video_cosine
c_effective_rank
c_plus_effective_rank
L_recon_present
L_recon_cplus
L_recon_video_gap
```

Only `F_c`-dependent quantities should learn materially:

```text
L_flow
coarse_model_loss
coarse_vs_copy_ratio
coarse_vs_batch_mean_ratio
L_recon_chat
```

Stop and preserve evidence if frozen hashes change, the mode is wrong, metrics become nonfinite,
skipped updates spiral, the process writes outside its registered paths, or provenance/source
identity fails. Do **not** stop only because early prediction ratios are poor; the registered test
is the complete 15,000-step trajectory.

## 12. Monitor from any later SSH connection

Session, PID, and GPU:

```bash
tmux list-sessions; pgrep -af "python.*train.py.*optimization-scope fc_only" || true; nvidia-smi
```

Latest log:

```bash
tail -n 120 /workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log
```

Latest tmux pane:

```bash
tmux capture-pane -p -t inv021_fc_only -S -160
```

Checkpoint inventory:

```bash
find /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort
```

Once the W&B ID is known, export it in that monitoring shell:

```bash
export INV021_WANDB_ID=REPLACE_WITH_ACTUAL_WANDB_ID
```

Pull the current unsampled report when needed:

```bash
cd /workspace/hierarchal-jepa-flow-world-model && python3 run_history.py --run "$INV021_WANDB_ID" --report
```

## 13. Exact recovery boundary

SSH loss is not a reason to relaunch; reconnect and inspect `inv021_fc_only`. If the process fails
before any checkpoint, preserve the log and diagnose the operational failure. Restart from the
registered warm start only when no scientific update is being discarded and the cause is fixed.

If the process fails after a valid Investigation-021 checkpoint, resume only from its newest
complete checkpoint. Never combine `--resume` with `--warm-start-from`, and never resume from the
Investigation-019 source.

Set the exact continuation checkpoint and the already-created W&B ID:

```bash
export INV021_RESUME_CHECKPOINT=/workspace/ckpt/inv021_fixed_residual_coordinates/fc_only/phase1_stepREPLACE.pt INV021_WANDB_ID=REPLACE_WITH_ACTUAL_WANDB_ID
```

After confirming the old `inv021_fc_only` session is absent, resume on GPU 0:

```bash
tmux new-session -d -s inv021_fc_only "cd /workspace/hierarchal-jepa-flow-world-model && set -Eeuo pipefail && export HF_HOME=/workspace/hf_cache && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 train.py --data ego4d --encoder dinov3_vitb16 --n-c 64 --d-c 512 --bottleneck-mixer-dim 512 --resume '$INV021_RESUME_CHECKPOINT' --temporal-target residual --optimization-scope fc_only --checkpoint-dir /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only --provenance-out /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only/run_provenance.json --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm --wandb-group inv021_fixed_residual_coordinates --wandb-name 'Investigation 21 · Fixed residual coordinates · Coarse flow only, DINOv3 64 slots by 512' --wandb-run-id '$INV021_WANDB_ID' --require-wandb 2>&1 | tee -a /workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log"
```

The strict resume path rechecks the original warm-start identity and frozen hashes before loading
live state. A mismatch is a hard stop, not a reason to reset the optimizer or permit a legacy
checkpoint.

## 14. Prove terminal completion

The run is complete only when W&B says `finished`, the process exits zero, the final logged update
exists, and the final checkpoint/provenance agree.

Confirm no paid process remains and inspect the final log:

```bash
pgrep -af "python.*train.py.*optimization-scope fc_only" || true; tail -n 160 /workspace/hierarchal-jepa-flow-world-model/logs/inv021_fc_only.log
```

Require the final checkpoint:

```bash
test -s /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only/phase1_step15000.pt
```

Verify terminal schema, mode, and immutable tensors:

```bash
python3 - <<'PY'
from pathlib import Path

import torch

from provenance import state_dict_hash

path = Path("/workspace/ckpt/inv021_fixed_residual_coordinates/fc_only/phase1_step15000.pt")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
assert checkpoint["schema"] == "hjepa-phase1-checkpoint-v2"
assert checkpoint["next_step"] == 15000
p = checkpoint["run_provenance"]
assert checkpoint["config"]["train"]["optimization_scope"] == "fc_only"
assert checkpoint["config"]["train"]["predict_residual"] is True
assert p["optimization_contract"]["trainable_modules"] == ["F_c"]
assert p["optimization_contract"]["frozen_modules"] == ["B", "B_EMA", "D"]
assert p["optimization_contract"]["ema_updates"] is False
assert p["optimization_contract"]["optimized_objective_terms"] == ["L_flow"]
expected = p["frozen_state_hashes"]
assert expected == p["common"]["frozen_state_hashes"]
actual = {
    "B": state_dict_hash(checkpoint["bottleneck"]),
    "B_EMA": state_dict_hash(checkpoint["target_bottleneck"]),
    "D": state_dict_hash(checkpoint["decoder"]),
}
assert actual == expected, (actual, expected)
print("INV021_FINAL_CHECKPOINT_OK", checkpoint["wandb_run_id"], expected)
PY
```

Record the final checkpoint checksum:

```bash
sha256sum /workspace/ckpt/inv021_fixed_residual_coordinates/fc_only/phase1_step15000.pt
```

Finally, pull the complete unsampled W&B history and perform Reading Cycle A. Write factual metrics
to `METRIC_READOUT.md`, interpretation to `ANALYSIS.md`, and then update both KANBAN triads and the
Phase-1 index. Never infer the final verdict from process disappearance or `L_flow` alone.
