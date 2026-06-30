# Investigation 011 — Does cosine reconstruction help, and can present-only reconstruction train a rich `c`?

**Status:** OPEN — code switches added; runs not launched.
**Opened:** 2026-06-30
**Closed:** —

## Question

investigation_010 produced the healthiest Phase 1 representation so far, but prediction still tied
the zero-residual/copy baseline. The reconstruction channel may have been undercut by the old
`MSE / Var(e)` objective, which let the decoder game feature norms and made the readout mostly
blind to prediction quality.

This investigation tests two separate reconstruction questions:

- **Run A — cosine reconstruction in the full residual recipe.** Keep the inv010 recipe
  (`λ_sigreg=5`, `λ_var=0.5`, `λ_recon=0.05`, `λ_recon_pred=0.05`, residual prediction,
  decoder `512x4`, `n_c=32`, `k=12`) and switch only the reconstruction formula to
  `--recon-loss-mode cosine`.
- **Run B — present-only reconstruction bottleneck test.** Disable prediction with
  `--present-recon-only` and train only `D(B(e_t)) -> e_t` with the cosine reconstruction loss;
  no `F_c` loss, no residual target, no future `c_hat`, no `D(c_hat) -> e_{t+k}` branch.

## Runs

| Run | Config | Role | Main readout |
|---|---|---|---|
| A | full residual + SIGReg + `--recon-loss-mode cosine` | Does the new recon geometry help the current best recipe? | `coarse_vs_copy_ratio`, rank, `L_recon_chat - L_recon_cplus` |
| B | `--present-recon-only`, `λ_var=0`, `λ_sigreg=0`, `λ_recon=0.05` | Can reconstruction alone make `c_t` rich enough to decode `e_t`? | `L_recon_present`, `c_effective_rank`, `c_cross_video_cosine` |

## Interpretation

- **Run A win:** copy ratio drops below inv010 while rank/cosine/stability stay healthy.
- **Run A neutral:** rank and reconstruction change but copy ratio remains near 1, so prediction is
  still the bottleneck.
- **Run B win:** `L_recon_present` falls materially while `c_effective_rank` rises without collapse,
  proving the bottleneck can carry detailed present information under the new objective.
- **Run B fail:** reconstruction stalls and rank stays low, meaning the present decoder path still
  cannot make `c` information-rich by itself.

## Code switches

- `--recon-loss-mode cosine|relative_mse`
- `--present-recon-only`
- `cfg.train.present_recon_only`
- `losses.reconstruction_loss(..., mode=...)`

## Execution

Use [`GUIDE.md`](GUIDE.md).
