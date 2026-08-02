# Next steps — coarse-flow architecture and metric audit

No paid training run is authorized by this reasoning record.

1. Measure fixed-checkpoint flow loss with identical `z`, `tau`, and `eps` under real, shuffled,
   and null conditioning.
2. Numerically integrate the fixed `F_c` from noise to `Delta_hat`; compare terminal residual MSE
   with zero residual, batch mean, and the current one-step velocity readout.
3. Measure target and prediction per-token mean and norm as functions of `tau`; determine whether
   the final `LayerNorm` without a learned velocity head imposes an active output constraint.
4. Use those results to select one coherent intervention:
   - ignored condition: change the conditioning/output interface;
   - useful condition but bad endpoint geometry: add fixed-decoder prediction reconstruction;
   - good integrated endpoints but misleading velocity ratios: repair the acceptance metric before
     training anything else.
5. Do not change encoder, bottleneck geometry, horizon, flow capacity, and reconstruction objective
   together. The current checkpoint can answer the three audit questions without retraining.
