# GUIDE — Run 64, EGO4D internal memory M=512

This arm is launched first by the parent investigation's exact sequential controller:
[`../GUIDE.md`](../GUIDE.md). Do not launch a second standalone process.

Expected identity:

```text
bottleneck_mixer_dim = 512
checkpoint = /workspace/ckpt/inv016_unwhitened_memory_m512
log = logs/inv016_unwhitened_memory_m512.log
W&B = Investigation 16 · Internal memory width · EGO4D M=512
```

The parent queue creates this run only after the 1,024-wide Stage 0/resource gates pass. It requires
this arm's final checkpoint and provenance before starting the sibling 1,024 run.
