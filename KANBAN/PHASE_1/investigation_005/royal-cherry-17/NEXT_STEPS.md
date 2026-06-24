# Next steps — royal-cherry-17

## Status

Run **killed** at step 11350. Grad stability **solved**; representation collapse **not**.
Investigation 005 remains **OPEN**.

## Do not

- Resume from any royal-cherry checkpoint (rank ~5.8, copy ratio > 12 at end).
- Treat `grad_skipped=0` as success in isolation.

## Recommended next run (spawn new folder)

**Resume `phase1_step7500.pt` with AGC + halved coarse-flow LR** — combines royal's stability
machinery with drawn's LR conservatism:

```bash
python train.py \
  --data ssv2 \
  --steps 15000 \
  --resume /workspace/checkpoints/phase1_step7500.pt \
  --horizon-k 12 \
  --lambda-var 0.5 \
  --lr-coarse-flow 1e-4 \
  --log-every 50 \
  --diag-every 500
```

**Watch (new):** `L_flow` (abort if > 1.5 for 200 consecutive steps), `agc_Fc_max_ratio`
(abort if median > 200 over 500 steps), plus existing copy/rank/skip metrics.

## If that fails — escalate in order

1. **Tighter AGC:** `--agc-lambda-coarse-flow 0.05` (keep λ_B=0.20).
2. **Optimizer reset on resume** — load weights from 7500 but re-init Adam state (code change).
3. **Fresh 15k from init** with AGC + cerulean config (no resume) — tests whether 7500→8500
   re-traversal is the problem.
4. **Code:** log pre-AGC grad norm; raise `instability_warn` to use `L_flow > 1.0` OR
   `agc_Fc_max_ratio > 50`; optional LR backoff when AGC ratio spikes.

## On success

Close investigation 005; Phase 2 per `AGENT_FILES/PHASES/PHASE_2.md`.
