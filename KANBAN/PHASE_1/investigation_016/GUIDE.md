# GUIDE — run the EGO4D internal-memory width pair

This is the exact guide for the paired 512/1,024 launch. One fail-fast `tmux` queue checks the
largest architecture, runs 512 to completion, and then runs 1,024 on the same GPU.

## 1. Fresh pod setup

On a fresh pod, execute only Sections 1–5 of
[`AGENT_FILES/SETUPS/NEW_POD.md`](../../../AGENT_FILES/SETUPS/NEW_POD.md): packages, cache paths,
code, Python requirements, and W&B login. Do not run the generic SSv2 examples after Section 5.

## 2. Pull and test the published commit

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

Require a clean tracked worktree and the exact locally tested/pushed SHA.

## 3. Whitening step: skip it

There is no whitening prerequisite. Existing `.pt` statistics stay on the volume but are ignored.
Do not generate/copy/inspect them and do not add any `--whiten-*` flag.

## 4. Launch everything once

Check safety and script syntax:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
tmux has-session -t inv016_internal_memory_queue 2>/dev/null && echo "STOP: queue already exists" || true
pgrep -af '[p]ython.*train.py' || true
nvidia-smi
bash -n KANBAN/PHASE_1/investigation_016/RUN_INTERNAL_MEMORY_SWEEP.sh
```

If there is no existing queue or training process and the GPU is idle, launch exactly once:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
tmux new-session -d -s inv016_internal_memory_queue "bash -lc 'cd /workspace/hierarchal-jepa-flow-world-model && set -o pipefail && bash KANBAN/PHASE_1/investigation_016/RUN_INTERNAL_MEMORY_SWEEP.sh 2>&1 | tee logs/inv016_internal_memory_queue.log'"
```

The queue order is fixed:

```text
Stage 0 at M=1024
resource preflight at M=1024 and batch 64
15,000-step EGO4D M=512 run
15,000-step EGO4D M=1024 run
```

`set -Eeuo pipefail` prevents any later item from starting after a failure.

## 5. Verify and monitor

Run these separately:

```bash
tmux list-sessions
```

```bash
tmux capture-pane -p -t inv016_internal_memory_queue -S -160
```

```bash
pgrep -af '[p]ython.*train.py'
```

```bash
nvidia-smi
```

Durable controller/gate/arm logs:

```bash
tail -n 120 logs/inv016_internal_memory_queue.log
tail -n 80 logs/inv016_internal_memory_resource_m1024.log 2>/dev/null || true
tail -n 80 logs/inv016_unwhitened_memory_m512.log 2>/dev/null || true
tail -n 80 logs/inv016_unwhitened_memory_m1024.log 2>/dev/null || true
```

The expected marker order is `STAGE0_DONE`, `RESOURCE_PREFLIGHT_DONE`, `ARM_DONE width=512`,
`ARM_DONE width=1024`, then `QUEUE_DONE`. Full EGO4D provenance scans can be quiet and CPU-bound;
check process I/O before calling one hung.

## 6. Required identity

Both W&B runs must show:

```text
data = ego4d
present_recon_only = true
prediction_active = 0
whiten_features = false
n_c = 32
d_c = 256
lambda_recon = 1.0
lambda_var = lambda_cov = lambda_sigreg = lambda_slot = 0
```

The only between-run change is `bottleneck_mixer_dim = 512` versus `1024`. At the first logged
step require finite metrics, `grad_skipped=0`, `L_flow=0`, and `L_recon_pred=0`.

## 7. Completion proof

After `QUEUE_DONE`:

```bash
test -f /workspace/ckpt/inv016_unwhitened_memory_m512/phase1_step15000.pt
test -f /workspace/ckpt/inv016_unwhitened_memory_m1024/phase1_step15000.pt
test -f /workspace/ckpt/inv016_unwhitened_memory_m512/run_provenance.json
test -f /workspace/ckpt/inv016_unwhitened_memory_m1024/run_provenance.json
sha256sum /workspace/ckpt/inv016_unwhitened_memory_m512/phase1_step15000.pt
sha256sum /workspace/ckpt/inv016_unwhitened_memory_m1024/phase1_step15000.pt
```

Confirm both W&B states are `finished`, then update the two run records:

- [`run_064_unwhitened_internal_memory_m512/`](run_064_unwhitened_internal_memory_m512/)
- [`run_065_unwhitened_internal_memory_m1024/`](run_065_unwhitened_internal_memory_m1024/)

Apply [`SWEEP_PLAN_information_preservation.md`](SWEEP_PLAN_information_preservation.md)'s joint
loss-plus-gap rule. The gap must improve, not merely stay flat. The current fixed batch supports
only an exact-chunk/within-source preservation claim; global preservation is prohibited.
