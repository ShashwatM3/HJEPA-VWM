# Next steps — residual prediction

1. Completed: source, plan, preflight, launch, and W&B registration.
2. Completed: run `3y2hxj5t` finished all 15,000 updates and wrote a hashed final checkpoint.
3. Completed: [`METRIC_READOUT.md`](METRIC_READOUT.md), [`ANALYSIS.md`](ANALYSIS.md), this triad,
   and the parent synthesis now record the terminal result.
4. Close this arm as **Healthy rep, no predictor**; do not claim a target winner because both arms
   fail the registered prediction gates.
5. Register a new investigation for one residual run with the transferred B/B_EMA frozen and Fc
   trained in fixed coordinates. Keep encoder, source checkpoint, seed, data order, target/noise,
   horizon, Fc, schedule, and diagnostics unchanged.
6. Do not combine the frozen-B probe with SIGReg, V-JEPA, prediction-side reconstruction, a new
   horizon, or extra Fc capacity.
