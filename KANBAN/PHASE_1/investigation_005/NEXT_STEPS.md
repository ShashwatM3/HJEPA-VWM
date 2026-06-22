# Next steps — Investigation 005

## Immediate (ACTIVE)

[`drawn-elevator-16`](drawn-elevator-16/) appears to be the resume run (~3.5h on W&B).

1. **Pull final metrics from W&B** for `drawn-elevator-16` (`0n5mx3qf`).
2. Update [`drawn-elevator-16/OBSERVATIONS.md`](drawn-elevator-16/OBSERVATIONS.md) with outcome.
3. **Watch:** `grad_skipped`, `coarse_vs_copy_ratio`, `c_effective_rank`, final step count.

If resume not yet launched or failed, use:

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4
```

4. **Abort** if `grad_skipped=1` sustained (>10% over 500 steps).

## On success

- Record final metrics vs PHASE_1 §12 gates
- **Close investigation 005**
- Re-evaluate [investigation_004](investigation_004/) if rank too low
- Proceed toward project Phase 2 per [`AGENT_FILES/PHASES/PHASE_2.md`](../../AGENT_FILES/PHASES/PHASE_2.md)

## On failure

- Do not use post-spike checkpoints from `elated-snowflake-15`
- Consider further LR reduction or optimizer reset on resume
- Do not revert collapse config (`lambda_var=0.5`, `horizon_k=12`) without new evidence
