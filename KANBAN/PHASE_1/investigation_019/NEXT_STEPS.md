# Next steps — investigation_019

1. Publish one clean, tested commit with `lambda_var=0.5` and `lambda_cov=0.01`.
2. Execute the DINO-only correction command in [`GUIDE.md`](GUIDE.md) on the four-GPU pod.
3. Verify all four DINOv3 processes, exact covariance-plus-variance W&B identities, resolved
   configs, logs, provenance paths, and GPU assignments.
4. Treat the completed no-geometry V-JEPA2 lane as a separate ablation; rerun V-JEPA2 with
   covariance plus variance before making a three-encoder sweep comparison.
5. After each corrected lane completes, record the four W&B IDs and terminal evidence.
6. Analyze every arm with present-only Reading Cycle B, then choose a Pareto winner separately
   inside each encoder lane using the preregistered decision rule.
7. Do not transfer a shape into full prediction solely because it has the lowest raw
   reconstruction loss; require code dependence, stability, and the geometry guard.
