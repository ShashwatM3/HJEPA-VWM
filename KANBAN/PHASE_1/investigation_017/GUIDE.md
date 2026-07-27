# GUIDE — Investigation 17 three-encoder latent-shape sweep

This guide launches fresh runs only. It never resumes or overwrites a checkpoint.

## Prerequisites

1. Complete `AGENT_FILES/SETUPS/NEW_POD.md` through Step 5, including W&B login.
2. Set `INV017_EXPECTED_COMMIT` to the published sweep SHA in the clean GPU checkout.
3. Confirm full EGO4D is present and no other `train.py` owns the GPU.
4. Execute the selected encoder bundle's command below directly. Do not add a separate test suite,
   Stage 0, or resource-preflight job unless the human explicitly requests one.

## Launch one encoder sweep

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export INV017_EXPECTED_COMMIT=<published-tested-sha>
bash KANBAN/PHASE_1/investigation_017/RUN_LATENT_SHAPE_SWEEP.sh vjepa2
```

Replace `vjepa2` with `siglip2` or `dinov3` for the other documented lanes. The corresponding
bundle `GUIDE.md` records its exact alias/revision/group.

## Launch all three sequentially

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export INV017_EXPECTED_COMMIT=<published-tested-sha>
for encoder_lane in vjepa2 siglip2 dinov3; do
  bash KANBAN/PHASE_1/investigation_017/RUN_LATENT_SHAPE_SWEEP.sh "$encoder_lane"
done
```

The shell exits on the first failed arm. Rerunning a completed lane intentionally
fails on its existing outputs; inspect the failure and choose a new explicit output identity rather
than deleting or resuming silently.

## W&B identity

```text
entity = smahalanobis-uc-davis
project = hjepa-vwm
group = inv017_<encoder>_latent_shape_cov_var
name = Investigation 17 · <official encoder> latent shape · N=<N>, D=<D>
```

Each display name has exactly the three repository-required middle-dot segments.

## Completion

Every arm must exit zero and produce `phase1_step15000.pt` plus `run_provenance.json` in its unique
checkpoint directory. As each W&B run starts, append its ID to the appropriate lane
`OBSERVATIONS.md`; after completion, record its state and Reading Cycle B verdict there.
