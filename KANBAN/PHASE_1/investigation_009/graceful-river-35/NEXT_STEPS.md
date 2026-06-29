# Next steps — graceful-river-35 (Run 2)

This run is the wave's pivotal datapoint and seeds the Tier-2 follow-ups in
[../NEXT_STEPS.md](../NEXT_STEPS.md). It selected the "residual is a real lever on `c`, but
prediction is the new bottleneck" branch: the ratio fell toward 1 (min 0.915) and the representation
became dynamic/healthy, but `F_c` ties copy by predicting `Δ̂ ≈ 0`. Prioritized:

1. **Anti-collapse on `Δ̂` (the deliberately-omitted change #7) — now directly motivated.** A
   variance / norm-matching term forcing `‖Δ̂‖ ≈ ‖Δ‖` would forbid the zero shortcut this run fell
   into. Single most direct follow-up.
2. **Govern the decorrelation to the predictive band.** The predictive window was ρ ≈ 0.77 (step
   ~3500). Hold it by lowering `λ_recon_pred` (the option-3 anchor drives the runaway decorrelation),
   dropping recon entirely for a clean residual A/B, or annealing the horizon `k`.
3. **Sweep horizon `k` with residual prediction** — `k` sets how far `c` must move and how inferable
   that move is; residual + a tuned `k` is the natural pairing.
4. **A dynamics-specific predictor signal** (inverse-dynamics / multi-step rollout) — recon inflates Δ
   with appearance change, not predictable dynamics; an objective rewarding transition structure is
   the deeper fix.

Decision metric forward is unchanged: `coarse_vs_copy_ratio` *decisively and stably* < 1 (→ 0.70),
with `c_effective_rank` and a *moderate* ρ as corroborating signals. These follow-ups almost
certainly open **investigation_010**.
