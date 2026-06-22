# Next steps — Investigation 005

## Immediate (ACTIVE)

1. Confirm checkpoint exists: `ls -lh /workspace/checkpoints/phase1_step7500.pt`
2. Pull branch with `--lr-coarse-flow` / `--lr-bottleneck` CLI (commit `af3f87f`).
3. Launch resume:

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4
```

4. **Watch:** `grad_skipped`, `coarse_vs_copy_ratio`, `c_effective_rank`.
5. **Abort** if `grad_skipped=1` sustained (>10% over 500 steps).

## On success

- Record final metrics vs PHASE_1 §12 gates
- **Close investigation 005**
- Re-evaluate [investigation_004](investigation_004/) if rank too low
- Proceed toward project Phase 2 per [`AGENT_FILES/PHASES/PHASE_2.md`](../../AGENT_FILES/PHASES/PHASE_2.md)

## On failure

- Do not use post-spike checkpoints
- Consider further LR reduction or earlier abort policy
- Do not revert collapse config (`lambda_var=0.5`, `horizon_k=12`) without new evidence
