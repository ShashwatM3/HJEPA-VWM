# Next steps — Investigation 003

Investigation **closed** with winning config:

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5
```

Follow-ups:

1. **[investigation_005](investigation_005/)** — run to 15k and confirm gates (rank plateau, copy ratio hold).
2. **[investigation_004](investigation_004/)** — only if rank plateaus low **after** a stable 15k attempt.
3. Do **not** re-enable `lambda_slot` without new evidence.

Code knobs added during this investigation: `--horizon-k`, `--lambda-var`, `--lambda-cov`,
`--lambda-slot` (slot/cov default off); init fixes non-flagged.
