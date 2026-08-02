# GUIDE — Run 65, EGO4D internal memory M=1024

This arm is launched second by the parent investigation's exact sequential controller:
[`../GUIDE.md`](../GUIDE.md). Do not launch it beside the 512 process.

Expected identity:

```text
bottleneck_mixer_dim = 1024
checkpoint = /workspace/ckpt/inv016_unwhitened_memory_m1024
log = logs/inv016_unwhitened_memory_m1024.log
W&B = Investigation 16 · Internal memory width · EGO4D M=1024
```

The same shape is used for Stage 0 and the batch-64 resource preflight. The paid arm starts only if
those gates and the complete 512 run succeed.
