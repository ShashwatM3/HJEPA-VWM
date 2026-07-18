# GUIDE — run the EGO4D internal-memory sweep

This guide is intentionally simple. It launches one fail-fast `tmux` queue. The queue checks the
largest architecture first, then runs 512 and 1,024 sequentially on the single GPU.

## 1. Fresh pod setup

If this is a fresh pod, execute only Sections 1–5 of
[`AGENT_FILES/SETUPS/NEW_POD.md`](../../../../AGENT_FILES/SETUPS/NEW_POD.md): install packages,
create cache paths, get the code, install requirements, and verify/log into W&B. Do not run the
generic SSv2 examples later in that file.

## 2. Pull the exact published code

Run on the pod:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git status --short --branch
git fetch origin
git checkout phase1-v0.2-frozen-encoder
git pull --ff-only origin phase1-v0.2-frozen-encoder
git status --short --branch
git rev-parse HEAD
python -m pip install -r requirements.txt
pytest -q
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

Both Git status outputs must have no tracked changes. Record the printed SHA. The launching agent
must confirm it equals the locally tested/pushed SHA.

## 3. Notice the missing whitening step

There is nothing to generate, copy, or inspect. Old whitening `.pt` files may remain on the volume;
this experiment ignores them. Do not add any `--whiten-*` flag to the queue.

## 4. Launch the whole sweep

First make sure the queue name is unused and the GPU has no training process:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
tmux has-session -t inv016_internal_memory_queue 2>/dev/null && echo "STOP: queue already exists" || true
pgrep -af '[p]ython.*train.py' || true
nvidia-smi
bash -n KANBAN/PHASE_1/investigation_016/internal_memory_width_sweep/RUN_SWEEP.sh
```

If no `train.py` process is listed and `bash -n` prints nothing, launch exactly once:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
tmux new-session -d -s inv016_internal_memory_queue "bash -lc 'cd /workspace/hierarchal-jepa-flow-world-model && set -o pipefail && bash KANBAN/PHASE_1/investigation_016/internal_memory_width_sweep/RUN_SWEEP.sh 2>&1 | tee logs/inv016_internal_memory_queue.log'"
```

That one command performs, in order:

```text
Stage 0 at M=1024
resource preflight at M=1024 and batch 64
15,000-step EGO4D M=512 run
15,000-step EGO4D M=1024 run
```

`set -Eeuo pipefail` stops the queue at the first failure. The second arm cannot start if Stage 0,
resource preflight, W&B, or the 512 arm fails.

## 5. Verify it actually started

Run these separately:

```bash
tmux list-sessions
```

```bash
tmux capture-pane -p -t inv016_internal_memory_queue -S -120
```

```bash
pgrep -af '[p]ython.*train.py'
```

```bash
nvidia-smi
```

The first expected marker is `STAGE0_START width=1024`. Then require `STAGE0_DONE`, followed by
`RESOURCE_PREFLIGHT_START`. Full EGO4D provenance scans can be quiet and CPU-bound for many minutes;
use the process/I/O checks in the autonomous remote-run guide before calling one hung.

## 6. Monitor without guessing

Queue state:

```bash
tmux capture-pane -p -t inv016_internal_memory_queue -S -160
```

Durable queue log (still available after `tmux` exits):

```bash
tail -n 120 logs/inv016_internal_memory_queue.log
```

Largest gate log:

```bash
tail -n 80 logs/inv016_internal_memory_resource_m1024.log
```

Current arm logs:

```bash
tail -n 80 logs/inv016_unwhitened_memory_m512.log 2>/dev/null || true
tail -n 80 logs/inv016_unwhitened_memory_m1024.log 2>/dev/null || true
```

The 512 arm must end with `ARM_DONE width=512` before `ARM_START width=1024` appears. Final success
is `QUEUE_DONE group=inv016_unwhitened_internal_memory_width`.

## 7. Required run identity

For both W&B runs confirm:

```text
data = ego4d
present_recon_only = true
prediction_active = 0
whiten_features = false
bottleneck_mixer_dim = 512 or 1024
n_c = 32
d_c = 256
lambda_recon = 1.0
lambda_var = lambda_cov = lambda_sigreg = lambda_slot = 0
```

At the first logged step require finite metrics, `grad_skipped=0`, `L_flow=0`, and
`L_recon_pred=0`. Do not stop merely because raw loss differs from historical whitened loss.

## 8. Completion evidence

After `QUEUE_DONE`, run:

```bash
test -f /workspace/ckpt/inv016_unwhitened_memory_m512/phase1_step15000.pt
test -f /workspace/ckpt/inv016_unwhitened_memory_m1024/phase1_step15000.pt
test -f /workspace/ckpt/inv016_unwhitened_memory_m512/run_provenance.json
test -f /workspace/ckpt/inv016_unwhitened_memory_m1024/run_provenance.json
sha256sum /workspace/ckpt/inv016_unwhitened_memory_m512/phase1_step15000.pt
sha256sum /workspace/ckpt/inv016_unwhitened_memory_m1024/phase1_step15000.pt
```

Confirm both W&B states are `finished`, record their IDs/URLs in `DESCRIPTION.md`, then analyze each
with present-only Reading Cycle B and apply [`PLAN.md`](PLAN.md)'s joint loss-plus-gap decision rule.
