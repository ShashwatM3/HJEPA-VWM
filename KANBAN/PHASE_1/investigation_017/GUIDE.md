# GUIDE — Investigation 17 raw-feature latent-shape sweep

This guide launches nine fresh runs sequentially. It never resumes a checkpoint.

## Prerequisites

1. Publish the local implementation and set `INV017_EXPECTED_COMMIT` to that exact SHA.
2. On the GPU host, sync a clean `phase1-v0.2-frozen-encoder` checkout to that SHA.
3. Install requirements, pass the full test suite and model/diagnostic smokes, and verify W&B login.
4. Confirm no other `train.py` process owns the GPU and all nine output paths are empty.

## Launch

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export INV017_EXPECTED_COMMIT=<published-tested-sha>
bash KANBAN/PHASE_1/investigation_017/RUN_LATENT_SHAPE_SWEEP.sh
```

The queue first runs Stage 0 and a physical-batch-64 resource preflight on the largest `64×512`
shape. It then runs the control and eight remaining arms in the order registered in
[`SWEEP_PLAN_latent_shape.md`](SWEEP_PLAN_latent_shape.md).

## W&B identity

```text
entity = smahalanobis-uc-davis
project = hjepa-vwm
group = inv017_raw_latent_shape_cov_var
name = Investigation 17 · Raw latent shape · <N> slots × <D> dimensions
```

## Completion

Each arm must exit zero, finish in W&B, and produce `phase1_step15000.pt` plus
`run_provenance.json` in its unique checkpoint directory. The queue stops on the first failure or
collision; it does not skip ahead or overwrite output.
