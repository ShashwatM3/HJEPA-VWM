# Plan — full-latent full-prediction arm

## Shared implementation

Implement the same warm-start, temporal-target CLI, provenance, and validation work described in
[`../residual_prediction_cov_var/PLAN.md`](../residual_prediction_cov_var/PLAN.md).

The full-latent arm must not use a separate code path for initialization. It calls the same
warm-start function with the same source checkpoint and selects only:

```text
--temporal-target full_latent
```

## Arm-specific contract

```text
flow_target = c_plus
eps_c = N(0,I)
z_c = (1-tau)*eps_c + tau*c_plus
u_target = c_plus - eps_c
```

The online condition `c_t` remains undetached so `L_flow` trains both `B` and `F_c`. `c_plus` stays
detached through `B_EMA`. The diagnostic endpoint is the predicted full future latent, without a
residual add-back.

## Tests

In addition to the shared warm-start tests, prove:

- `--temporal-target full_latent` sets `predict_residual=false`;
- no present-target `B_EMA(e_t)` call is required to form the flow target;
- flow noise remains unit Gaussian;
- the full-latent endpoint is not added to online `c_t`;
- common step-0 module hashes match the residual arm;
- the only allowed scientific provenance difference is `predict_residual`.

Run the same targeted and full test commands listed in the residual plan.

## Locked arm config

Use the parent recipe plus:

```text
temporal_target = full_latent
predict_residual = false
```

## Analysis

Use Reading Cycle A. The primary failure discriminator is:

```text
falling coarse_copy_loss + ratio near 1
    => static-c trap
```

Compare ratios and verdicts to the paired residual arm; do not compare absolute flow losses.
