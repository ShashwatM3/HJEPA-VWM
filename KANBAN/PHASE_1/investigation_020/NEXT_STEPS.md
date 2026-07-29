# Next steps — investigation_020

1. Preserve the registered DINOv3 `64×512`, `M=512` source identity and verify its checkpoint
   SHA-256 on the pod.
2. Publish the implemented warm-start, `--temporal-target`, and provenance-parity plumbing.
3. Run the targeted tests, full `pytest -q`, and relevant smoke checks before publication.
4. Run both resource preflights from that exact clean commit.
5. Verify common provenance: the source checkpoint hash and trainable step-0 initialization hash
   must match; the only scientific difference must be `train.predict_residual`.
6. Launch both arms concurrently on separate GPUs with [`GUIDE.md`](GUIDE.md).
7. Analyze each run independently with Reading Cycle A, then apply the registered paired decision
   rule in [`SWEEP_PLAN_temporal_target.md`](SWEEP_PLAN_temporal_target.md).
8. If neither arm passes both prediction gates, record the correct failure labels; do not declare
   the lower `L_flow` arm the winner.
9. After selecting a temporal target, register one matched V-JEPA2 control if the encoder temporal
   prior remains decision-relevant.
10. Open a separate follow-up investigation for SIGReg only after the temporal target question is
   answered.
