# Next steps — fixed residual coordinates

1. Completed: launch, exact provenance/resource gate, and stable training through step 12,450.
2. Completed: pulled all 250 unsampled W&B rows and wrote `METRIC_READOUT.md` and `ANALYSIS.md`.
3. Treat this run as formally incomplete and scientifically **Healthy rep, no predictor through
   step 12,450**. Do not present it as a finished 15,000-step run.
4. Do not return first to encoder, bottleneck geometry, SIGReg, covariance, slot, or decoder sweeps.
5. Add an offline numerically integrated flow-endpoint evaluation so the learned rollout is
   compared directly with zero residual and batch mean.
6. Register one full-scale fixed-coordinate direct-residual-prediction run. This separates
   predictable information in `c_t` from rectified-flow denoising and time-conditioning difficulty.
7. If direct regression passes, redesign the temporal flow objective. If it fails, move to a larger
   temporal model or a representation explicitly trained for predictable state.
