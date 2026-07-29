# Next steps — investigation_020

1. Completed: verified the registered DINOv3 `64×512`, `M=512` source checkpoint and SHA-256.
2. Completed: published warm start, `--temporal-target`, provenance parity, and the bounded
   parallel dataset-inventory acceleration.
3. Completed: passed the local full suite and the pod's targeted contracts, formatting, source,
   recipe, dependency, and authentication gates.
4. Completed: both resource preflights exited zero from the exact clean launch commit.
5. Completed: proved common source SHA and step-0 initialization hash; only
   `train.predict_residual` differs scientifically.
6. Completed: launched both arms on separate A100s and recorded W&B IDs `3y2hxj5t` and `8r6akjsx`.
7. Monitor both arms through terminal step 15,000 without altering the live recipe.
8. Analyze each run independently with Reading Cycle A, then apply the registered paired decision
   rule in [`SWEEP_PLAN_temporal_target.md`](SWEEP_PLAN_temporal_target.md).
9. If neither arm passes both prediction gates, record the correct failure labels; do not declare
   the lower `L_flow` arm the winner.
10. After selecting a temporal target, register one matched V-JEPA2 control if the encoder temporal
   prior remains decision-relevant.
11. Open a separate follow-up investigation for SIGReg only after the temporal target question is
   answered.
