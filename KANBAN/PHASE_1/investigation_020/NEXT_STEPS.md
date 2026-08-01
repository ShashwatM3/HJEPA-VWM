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
7. Completed: both arms reached terminal step 15,000 without a recipe change or invalid update.
8. Completed: each run has an independent Reading-Cycle-A metric readout and analysis.
9. Completed: neither arm passes either prediction gate, so Investigation 020 closes with no
   winner. Residual is **Healthy rep, no predictor**; full latent is a **Static-`c` trap**.
10. Open a new investigation for a single-variable residual probe: freeze the transferred B and
    B_EMA, train Fc in fixed coordinates, and keep every other identity and setting matched.
11. Defer the V-JEPA control and SIGReg addition until the frozen-code probe establishes that the
    current Fc/objective can learn a fixed target. Do not spend encoder/regularizer compute while
    predictor learnability remains unresolved.
