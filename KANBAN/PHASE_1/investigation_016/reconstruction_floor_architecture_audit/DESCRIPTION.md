# Reconstruction-floor architecture audit

## Status

COMPLETE — analysis-only audit; no new W&B training run was launched from this folder.

## Question

Why do the present-only, whitened latent autoencoder runs settle in the same broad
`L_recon ~= 0.6-0.7` band after moving from SSv2 to EGO4D? Is the limiting mechanism in
the data, target construction, bottleneck/decoder capacity, loss, gradient routing,
optimizer, schedule, or diagnostics?

## Primary target and controls

- Primary historical run: EGO4D run 058, `mvbx96nv`,
  [`ae_latent_stack_whiten_abs_recon_cov_var_ego4d`](../run_058_ae_latent_stack_whiten_abs_recon_cov_var_ego4d/),
  finished at 15,000 steps.
- Matched scientific control: SSv2 run 057, `cdvp6hou`,
  [`ae_latent_stack_whiten_abs_recon_cov_var`](../../investigation_015/run_057_ae_latent_stack_whiten_abs_recon_cov_var/),
  finished at 15,000 steps.
- Live context only: EGO4D weight-1 run 060, `2423b84g`,
  [`ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1`](../run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/),
  read provisionally through W&B step 5,450. It is not assigned a final verdict here.

## Scope

The audit traces the present-only path end to end:

1. EGO4D source selection and 4-second chunking;
2. clip ordering, temporal sampling, crop, and color jitter;
3. the frozen V-JEPA encoder;
4. offline ZCA whitening;
5. the Perceiver-style bottleneck;
6. the fixed-position reconstruction decoder;
7. cosine reconstruction plus variance/covariance objectives;
8. optimizer, clipping, warmups, and LR decay;
9. the fixed-batch reconstruction and geometry diagnostics.

The implementation reference is the current code at commit
`a27cd84dd67783f7ee8e68bb68e7c1a38a309534`. Historical behavior was checked directly
at run-057 commit `5ea4421ff7aa034894dc07ab6270ae766a00d83a` and run-058 commit
`21d2aa8a97e8f60536a681bc5a4324d1a2c28838` using `git show`/`git diff`.

## Deliverables

- [`METRIC_READOUT.md`](METRIC_READOUT.md) records run facts without causal claims.
- [`ANALYSIS.md`](ANALYSIS.md) contains the file-by-file audit, calculations, ranked
  mechanisms, corrected interpretation, and discriminating experiments.
- [`OBSERVATIONS.md`](OBSERVATIONS.md) is the concise verdict.
- [`NEXT_STEPS.md`](NEXT_STEPS.md) gives the ordered research sequence.

## Interpretation boundary

The current strict-pipeline whitening artifact was available and inspected exactly;
run 058 used a legacy, unbound artifact produced by the same whitening method. Exact
eigenvalue numbers in this audit therefore characterize the current EGO4D target
construction and strongly illuminate the design, but they are not represented as a
byte-level measurement of run 058's old file.
