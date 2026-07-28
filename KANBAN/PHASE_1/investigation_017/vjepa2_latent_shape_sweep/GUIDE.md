# GUIDE — V-JEPA2 latent-shape sweep

Complete `AGENT_FILES/SETUPS/NEW_POD.md` through Step 5 (`wandb login`). Then, from the clean GPU
checkout at the published sweep SHA, launch the paid queue directly:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export INV017_EXPECTED_COMMIT=<published-tested-sha>
bash KANBAN/PHASE_1/investigation_017/RUN_LATENT_SHAPE_SWEEP.sh vjepa2
```

The script pins revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, uses frame microbatch 8,
reuses exact prior `32×256` center `guiduvjp`, and launches the eight non-center shapes in group
`inv017_vjepa2_latent_shape_cov_var` with names:

```text
Investigation 17 · V-JEPA2 latent shape · N=<N>, D=<D>
```

Do not rerun or resume the Run-66 center; use its recorded W&B evidence as the center control.
`16×512` must start fresh and must not resume crashed W&B `ihiuptdp`. Do not reuse another encoder
lane's output directory.
Do not insert a separate test suite, Stage 0, or resource-preflight job between NEW_POD Step 5 and
this command. Each paid `train.py` process performs its own required provenance validation.
