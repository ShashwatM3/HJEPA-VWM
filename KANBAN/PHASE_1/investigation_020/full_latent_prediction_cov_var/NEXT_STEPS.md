# Next steps — full-latent prediction

1. Completed: source, plan, preflight, launch, and W&B registration.
2. Completed: run `8r6akjsx` finished all 15,000 updates and wrote a hashed final checkpoint.
3. Completed: [`METRIC_READOUT.md`](METRIC_READOUT.md), [`ANALYSIS.md`](ANALYSIS.md), this triad,
   and the parent synthesis now record the terminal result.
4. Close this arm as a **Static-`c` trap**. Do not tune the same full-latent recipe: it reduces the
   temporal target while becoming progressively worse than copy.
5. The next causal probe belongs to a new investigation and should freeze B/B_EMA under the
   residual target. Do not combine that probe with encoder, regularizer, horizon, or Fc changes.
