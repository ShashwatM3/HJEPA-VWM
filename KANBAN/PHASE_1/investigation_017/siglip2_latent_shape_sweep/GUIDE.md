# GUIDE — SigLIP 2 latent-shape sweep

Complete `AGENT_FILES/SETUPS/NEW_POD.md` through Step 5 (`wandb login`). Then, from the clean GPU
checkout at the published tested SHA:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export INV017_EXPECTED_COMMIT=<published-tested-sha>
bash KANBAN/PHASE_1/investigation_017/RUN_LATENT_SHAPE_SWEEP.sh siglip2
```

The script pins revision `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`, uses frame microbatch 8,
reuses exact prior `32×256` center `ufbeokj2`, and launches only the eight untried shapes in group
`inv017_siglip2_latent_shape_cov_var` with names:

```text
Investigation 17 · SigLIP 2 latent shape · N=<N>, D=<D>
```

Do not rerun or resume Run 67; use its recorded W&B evidence as the center control. Do not reuse
V-JEPA2/DINOv3 output paths.
Do not add a separate test suite, Stage 0, or resource-preflight process unless the human asks.
