# GUIDE — DINOv3 latent-shape sweep

Complete `AGENT_FILES/SETUPS/NEW_POD.md` through Step 5 (`wandb login`). Then, from the clean GPU
checkout at the published tested SHA, with accepted checkpoint access or a valid cached snapshot:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export INV017_EXPECTED_COMMIT=<published-tested-sha>
bash KANBAN/PHASE_1/investigation_017/RUN_LATENT_SHAPE_SWEEP.sh dinov3
```

The script pins revision `5931719e67bbdb9737e363e781fb0c67687896bc`, uses validated frame
microbatch 32, reuses exact center `fiactcw6`, and launches eight fresh non-center runs in group
`inv017_dinov3_latent_shape_cov_var` with names:

```text
Investigation 17 · DINOv3 latent shape · N=<N>, D=<D>
```

Do not resume Run 69: it had no covariance or variance. Do not relaunch the `32×256` seed-42
center: `fiactcw6` already completed that exact scientific configuration.
Do not add a separate test suite, Stage 0, or resource-preflight process unless the human asks.
