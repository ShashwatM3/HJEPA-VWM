# Observations — investigation_020

## Registered prior before launch

The current prior is:

1. DINOv3 `60yaqw6d` is the transfer source because its selected `64×512`, `M=512` bottleneck is
   stable, high-rank, source-separated, and substantially more reconstructive and
   correct-code-dependent than the matched V-JEPA2 candidate, despite V-JEPA2's modest rank edge;
2. covariance plus variance is the safest common geometry substrate because it is the only current
   V-JEPA2 recipe that has simultaneously passed the historical common-step reconstruction,
   spread, separation, rank, and correct-code-dependence comparison against pure SIGReg;
3. residual prediction should make `F_c` spend its capacity on temporal change and should avoid
   transporting the large static component of `c_t`;
4. residual prediction does not weaken the copy baseline—the lazy solution becomes
   `Delta_hat=0`—so a ratio near 1 remains a failure;
5. a pretrained present bottleneck may improve the starting representation, but it does not imply
   that the temporal residual is predictable;
6. the maximum-surprise outcome would be a clean, high-rank warm start in which the full-latent arm
   passes both prediction gates while the residual arm again ties zero residual.

## Why covariance plus variance can outperform this project's SIGReg

These are mechanism hypotheses, not yet isolated causal conclusions:

- `variance_floor` flattens each example's entire `(N_c,D_c)` code and measures each coordinate's
  standard deviation across videos. Fixed slot identities shared by every video cannot satisfy it
  by themselves.
- The current `sigreg_loss` instead pools the `B*N_c` slot rows and sketches the distribution over
  `D_c`. Slot-to-slot structure can therefore contribute to its apparent richness even when
  example-to-example spread is weaker.
- The covariance loss deterministically attacks off-diagonal second moments. SIGReg uses 128 random
  projections and at most 512 pooled rows per step, so its geometry gradient is stochastic and more
  diffuse.
- SIGReg asks for the stronger full isotropic-Gaussian law. That can spend capacity on marginal
  normality or static detail unless a predictive objective makes those directions useful.

The matched W&B observation supports the outcome but does not distinguish these mechanisms. A
future SIGReg investigation must be a controlled probe after the temporal target is selected.

## Primary-source boundary

- [VICReg](https://arxiv.org/abs/2105.04906) motivates per-dimension variance and off-diagonal
  covariance as explicit collapse/redundancy controls.
- [LeJEPA](https://arxiv.org/abs/2511.08544) defines its total loss as prediction/alignment plus
  SIGReg, not SIGReg alone.
- [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379) studies alignment plus
  Gaussian regularization under stationary additive-noise transitions and Gaussian latent
  assumptions. Those guarantees do not automatically transfer to this code.

## Run evidence

No Investigation-020 run has launched. Append W&B IDs, terminal states, step-specific metrics, and
Reading-Cycle-A verdicts here after both arms complete. Do not replace the registered prior above.
