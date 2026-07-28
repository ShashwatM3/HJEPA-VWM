# Next steps — DINOv3 latent-shape sweep

1. Run the direct eight-arm paid queue in [`GUIDE.md`](GUIDE.md); do not insert a separate test
   suite, Stage 0, or resource-preflight job.
2. Record W&B IDs as soon as each arm streams.
3. Reuse `fiactcw6` as the geometry-active `32×256` center; use Run 69 only as the no-geometry
   reference.
4. Repeat a marginal winner and center at a second seed before changing defaults.
