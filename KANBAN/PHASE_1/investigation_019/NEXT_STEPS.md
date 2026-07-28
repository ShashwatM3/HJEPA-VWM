# Next steps — investigation_019

1. Publish one clean, tested commit containing this investigation and the reconstruction-only YAML.
2. Execute [`GUIDE.md`](GUIDE.md) on the four-GPU pod.
3. Verify all four V-JEPA2 processes, exact W&B identities, logs, provenance paths, and GPU
   assignments. Leave DINOv3 and SigLIP 2 queued behind the completed prior lane.
4. After each lane completes, record the four W&B IDs and terminal evidence before the controller
   advances.
5. Analyze every arm with present-only Reading Cycle B, then choose a Pareto winner separately
   inside each encoder lane using the preregistered decision rule.
6. Do not transfer a shape into full prediction solely because it has the lowest raw
   reconstruction loss; require code dependence, stability, and the geometry guard.
