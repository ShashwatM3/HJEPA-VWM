# Next steps — Run 68, DINOv3 whitened internal-memory M=512 covariance plus variance

1. Preserve the launch recipe and qualitative report; do not mark the run complete without verified W&B/checkpoint evidence.
2. Extract the exact W&B identity, terminal state, whitening identity, checkpoint SHA-256, and late-window metrics when available.
3. Use Run 69 as the controlled whitening ablation: keep the late projection, `lambda_var=0.5`, `lambda_cov=0.01`, and all other non-whitening settings matched.
4. Compare geometry and reconstruction carefully. The broader sequence mixed whitening and architecture changes, so only the matched Run 68/69 comparison can isolate whitening.