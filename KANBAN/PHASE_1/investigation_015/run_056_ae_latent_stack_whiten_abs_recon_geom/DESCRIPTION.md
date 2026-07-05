# Run 056 — `ae_latent_stack_whiten_abs_recon_geom` (inv015)

## Status

PLANNED — not launched. W&B group `inv015_ae_latent_stack_whiten` (overlays runs 054/055).
Mode: present-only autoencoder (Reading Cycle B). Why this run:
[`HYPOTHESIS.md`](HYPOTHESIS.md). Launch: [`GUIDE.md`](GUIDE.md).

## What this run is

Run 055's exact recipe with the inv011 winning geometry regularizer bundle turned on:

```text
present-only reconstruction, cosine loss
ABSOLUTE (clean) whitened-feature target   (recon_residual_target = FALSE, as in run 055)
FIXED offline feature whitening            (reuses run 054/055 stats file)
Perceiver latent-stack bottleneck          (bottleneck_latent_blocks = 3, default)
+ lambda_var    = 0.5     (was 0.0 in run 055)
+ lambda_sigreg = 5.0     (was 0.0 in run 055), sigreg_warmup_steps = 2000
+ lambda_cov    = 0.01    (was 0.0 in run 055)
  lambda_slot   = 0.0     (unchanged — slot loss was rejected in inv003; not resurrected)
no prediction (F_c inactive), no prediction-side recon
```

The three added weights are the terminal values of inv011 run 047 `po_geom_sig5_cov0p01`
(`az60m6mx`, W&B-verified: `lambda_sigreg=5`, `lambda_cov=0.01`, `lambda_var=0.5`), the
sweep's peak-rank point (`c_effective_rank` 150.8).

## Config delta vs run 055 (`nzz64pl6`)

Everything else identical (seed 42, 15k steps, `horizon_k=12`, LRs 1e-4/1e-4, decoder
512×4, `n_c=32`, `lambda_recon=0.05`, `recon_warmup_steps=2000`, `recon_loss_mode=cosine`,
`present_recon_only`, `whiten_features`, same whitening stats file, `whiten_eps=1e-4`,
`recon_residual_target=FALSE`, `lambda_recon_pred=0`). Add exactly:

```text
+ --lambda-var 0.5
+ --lambda-sigreg 5.0
+ --sigreg-warmup-steps 2000
+ --lambda-cov 0.01
```

## Code changes: NONE

This run requires no pipeline edits. All three regularizers are implemented and flag-gated:

| Term | Loss fn (`losses.py`) | Wiring (`train.train_step`, AGENTS.md §8/§9) | CLI flag (`train.py`) |
|---|---|---|---|
| Variance floor (VICReg V) | `variance_floor` | step 9 (always computed), step 11 (added: `lambda_var * L_var`) | `--lambda-var` (:1126) |
| SIGReg isotropy | `sigreg_loss` | step 9 (per-step generator, fp32), step 12 (`lambda_sigreg * sigreg_scale * L_sigreg` when `>0`) | `--lambda-sigreg` (:1134), `--sigreg-warmup-steps` (:1143) |
| Covariance (VICReg C) | `covariance_floor` | step 9, step 12 (`lambda_cov * L_cov` when `>0`) | `--lambda-cov` (:1104) |

`finalize_training_config` imposes no conflict: `--present-recon-only` only requires
`--lambda-recon > 0` (satisfied at 0.05) and disables the prediction/residual branches; the
non-prediction regularizers "still follow their configured weights" (AGENTS.md §8). No new
config field, no new buffer, no new gradient path.

## Gradient routing (what these terms train)

All three act on the ONLINE abstract latent `c_t` (the trainable `Bottleneck` output),
gradient into `B` only. `F_c` is inactive (present-only) and the decoder `D` is untouched by
them (it is trained solely by `L_recon`). The frozen encoder, `B_EMA`, and the whitener are
never reached — consistent with the §15 invariants.

- **`variance_floor`**: reshapes `c_t` to `(B, N_c·D_c)`, per-coordinate batch std, one-sided
  hinge `max(0, 1.0 − std_j)` averaged over coordinates. Added at full weight from step 0
  (no warmup). Cannot over-inflate; only lifts contracted dims.
- **`covariance_floor`**: pools `c_t` to `(B·N_c, D_c)`, centers, forms the `D_c×D_c`
  covariance, returns `Σ_{i≠j} cov_ij² / D_c`. Added at full weight from step 0.
- **`sigreg_loss`**: pools the SAME `(B·N_c, D_c)` object, subsamples to `max_rows=512`,
  sketches 128 random unit directions, BHEP normality statistic toward `N(0,I)`. Runs in
  FP32 with autocast disabled (WALK_FIXES F2) and draws from a per-step `torch.Generator` so
  logging it never perturbs the global RNG. Scaled by `sigreg_scale`, a linear 0→1 ramp over
  `sigreg_warmup_steps=2000` (protects the zero-init bottleneck; the EMA-lag rationale in the
  config comment does NOT apply here because there is no `F_c`/`c_plus` flow target in
  present-only mode).

## Whitening interaction

The regularizers operate on `c_t`; whitening operates on the encoder features `e` before the
bottleneck. They are mechanically independent (neither sees the other's tensor), so this is a
new equilibrium but not a new interaction path. The whitening stats file is REUSED
(`logs/whiten/whiten_stats_ssv2_train_seed42.pt`) — never recomputed mid-experiment.

## Loss-scale expectations (from run 055's unregularized readouts)

Run 055 logged these terms for-logging-only (weight 0) at convergence: `L_cov≈19.3`,
`L_var≈0.57`, `L_sigreg≈0.009`. At the new weights the added contributions start near
`0.01·19.3 ≈ 0.19` (cov), `0.5·0.57 ≈ 0.28` (var, shrinking toward 0 as std → 1 satisfies the
hinge), and `5·0.009·scale ≈ 0.05` (sigreg at full ramp) — all large relative to the scaled
recon anchor (`0.05·~0.64 ≈ 0.03`) and `L_flow=0`. The geometry bundle therefore dominates
the objective, exactly as in the inv011 sweep (total `loss ≈ 0.037` there). This is expected,
not a red flag: the run is deliberately geometry-shaped. The huge `L_cov` headroom is why
covariance is the dominant rank lever.

## Determinism / reproducibility

Seed 42, same init path as runs 054/055 (train from scratch — do NOT `--resume` any 052/053
checkpoint, rejected by architecture, nor 054/055, to keep the init comparable). SIGReg adds
stochastic projections via its own per-step generator; this is the same stochasticity the
inv008/inv011 sigreg runs carried and does not affect the other gradient paths.
