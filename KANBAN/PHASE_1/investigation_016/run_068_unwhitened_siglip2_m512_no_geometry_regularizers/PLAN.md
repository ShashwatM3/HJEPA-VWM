# Plan — Run 68

1. Confirm the remote repository is on published commit `4c402556e0b85684499f5a111630b7c8bc20ca89`
   and that the two prior Run 66/67 processes are stopped.
2. Run the pinned SigLIP2 CUDA adapter smoke, exact Stage 0, and physical-batch-64 resource
   preflight with both `lambda_var` and `lambda_cov` set to zero.
3. Launch one fresh paid run in its own tmux session with `--require-wandb`, unique log,
   checkpoint, provenance, and W&B identity.
4. Verify resolved config: `whiten_active=0`, `L_var=0`, `L_cov=0`, present-only reconstruction,
   zero predicted/flow loss, finite gradients, and no skipped updates.
5. Monitor checkpoints, W&B history, GPU/process health, and tripwires through step 15,000.

## Stop conditions

Stop and preserve evidence for wrong encoder/revision, whitening or predicted reconstruction
activation, nonzero variance/covariance weights, W&B/provenance failure, OOM, nonfinite values,
repeated gradient skips, or persistent instability. Do not tune the recipe during the run.
