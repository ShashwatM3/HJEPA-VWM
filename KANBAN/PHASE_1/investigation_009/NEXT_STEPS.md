# Next Steps - investigation_009

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

The next branch cleaned optimizer and regularization settings around residual prediction to see whether the result survived a full clean run.

## Closure / Carry-Forward Status

- Status: **CLOSED**.
- Conclusion to carry forward: Residual prediction made c_t more dynamic and healthier, but F_c mostly tied the zero-residual baseline. The failure moved from representation collapse toward predictor learning.
- Runs covered: 034, 035.

## Follow-Up Chain

This investigation feeds into `investigation_010`: clean residual run with optimizer/regularization plumbing. The reason is: Investigation 009 had promising representation movement but still poor forecasting. This branch removed confounds by using the cleaned residual/reconstruction/SIGReg setup.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.

## Original Notes Preserved

# Next steps — investigation_009 (residual prediction + SIGReg substrate)

## Tier 0 — commit the code (prerequisite, user action) ⏳

The residual + var-floor changes are implemented and statically verified (py_compile + ruff clean,
all call sites wired, `predict_residual=False` byte-identical) but **uncommitted** in the working
tree. Per PROTOCOL, commit only when asked:

1. Implemented: `cfg.train.predict_residual`, `--predict-residual`, `losses.residual_target`,
   residual branches in `train.train_step` / `run_diagnostics` / `reconstruction_readouts`,
   residual-aware `diagnostics.coarse_baselines`, and a residual gradient-contract block in
   `models.smoke_test_models`.
2. ⏳ **Commit on `phase1-v0.2-frozen-encoder`** (flag defaults to current behavior) + push, so the
   pod `git pull` gets it. **Run `python -c "from models import smoke_test_models; smoke_test_models()"`
   and `python train.py --stage0-only --predict-residual` on a torch box first** (the residual path
   is exercised there — torch is pod-only locally).

## Tier 1 — launch the 2× A100 wave

Per [`GUIDE.md`](GUIDE.md). Run 1 = `λ_sigreg 6 · λ_var 0.5 · no recon · full-latent`;
Run 2 = `λ_sigreg 5 · λ_var 0.5 · recon 0.05+0.05 · --predict-residual`. Both decoder 512×4,
`n_c=32`, `k=12`, `lr_coarse_flow=1e-4`. Hardening (from the inv007 Wave-2 death): verified `tmux`
detach, **step-0 component print** (confirm the residual `coarse_copy_loss ≈ ‖Δ‖²` and `λ·L` terms
are in range), step-600 tripwire.

After data: record each run's W&B name+id in [`DESCRIPTION.md`](DESCRIPTION.md), read the triad
`coarse_vs_copy_ratio ∧ c_effective_rank ∧ L_recon_chat` against [`OBSERVATIONS.md`](OBSERVATIONS.md).

## Tier 2 — conditional follow-ups (depend on the wave result)

- **Run 2 ratio falls below the full-latent baseline (toward <1)** → residual prediction is a real
  lever → refine: bump **horizon `k`** (the copy-weakening knob residual pairs with), add an
  anti-collapse term on Δ̂, and run a clean residual-vs-full A/B (drop recon).
- **Run 2 ratio stays ≥ 1 / tracks full-latent** → residual alone is insufficient at `k=12` (Δ mostly
  unpredictable) → pivot to the [END_OF_WAVE_2 §2.6](../investigation_007/END_OF_WAVE_2.md) Tier-1
  ladder: horizon-`k` sweep, inverse-dynamics / transition head, multi-step rollout loss. Residual
  prediction likely remains a *component* of those.
- **Run 1 is a clean substrate** (rank healthy, no collapse, prediction merely worse-by-the-law) →
  adopt SIGReg-only as the going-forward base; recon confirmed dead weight for prediction.
- **`L_recon_chat` drops under the residual decode** (unexpected) → reconstruction-with-residuals has
  legs after all → re-open the recon thesis on the residual target.

## What closes investigation_009

The 2-run wave answers (a) is residual prediction a prediction lever, and (b) is SIGReg-only a clean
substrate. Set [`DESCRIPTION.md`](DESCRIPTION.md) status → CLOSED with a Conclusion synthesis in
[`OBSERVATIONS.md`](OBSERVATIONS.md), refresh the [PHASE_1 README](../README.md), and point at
whichever Tier-2 follow-up (almost certainly the horizon/prediction pivot → a probable
**investigation_010**) the result selects.
