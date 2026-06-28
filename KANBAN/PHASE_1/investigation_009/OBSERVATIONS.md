# Observations — investigation_009 (residual prediction + SIGReg substrate)

No run data yet. This file registers the **pre-run predictions** the wave is launched on; metrics +
a neutral "reads as match/contradiction" pass are appended as dated sections once the runs land
(mirror the inv008 OBSERVATIONS discipline: measurements first, verdicts deferred). Design rationale
lives in [`DESCRIPTION.md`](DESCRIPTION.md) and [`GUIDE.md`](GUIDE.md).

---

## 2026-06-28 — Registered predictions

λ-extrapolations are log-linear from the inv008 SIGReg points (`fbqgix1x` λ10 → rank 73.3,
`9jxc8i1q` λ3 → rank 34.1).

### Run 1 — SIGReg substrate (λ_sigreg=6.0, λ_var=0.5, no recon, full-latent)

- **P1 `c_effective_rank` ~55** (between λ3=34 and λ10=73) — likely *just under* the 60 gate.
- **P2 `coarse_vs_copy_ratio` > 1, ~6–7** — worse than the no-SIGReg baseline (the inv008 rank↑⟺
  prediction↓ law). Run 1 characterizes a substrate; it is **not** expected to fix prediction.
- **P3 `L_flow` elevated ~0.85–0.90**; `c_std_mean` ~0.90–0.93 (floor active), `c_dead_dim_frac=0`,
  `c_cross_video_cosine` < 0.5 — no collapse.
- **P4 `L_recon_*` are meaningless** for Run 1 — the decoder is never trained (`λ_recon=0`), so the
  diag-cadence readouts decode an init decoder (~1.0). Expected and harmless; ignore them for Run 1.

### Run 2 — residual prediction (λ_sigreg=5.0, λ_var=0.5, recon 0.05+0.05, **residual**)

- **P1 `c_effective_rank` ~45–50** (λ5 substrate).
- **P2 (decision) `coarse_vs_copy_ratio`** — the whole point. Two outcomes:
  - falls **below the full-latent baseline** (toward <1) → residual prediction extracts predictable
    motion → a real lever → build on it (k sweep, anti-collapse on Δ, drop recon for a clean A/B).
  - stays **≥ 1 / tracks full-latent** → residual alone is insufficient (Δ is mostly
    unpredictable at k=12) → the horizon/inverse-dynamics/rollout pivot is the path.
  - Registered prior: **most likely ≥ 1** (residual does not weaken copy; it needs predictable
    motion), but watch for a dip below Run 1 / the inv008 full-latent ratio.
- **P3 `L_recon_chat`** with the residual decode `ĉ=c_t+Δ̂` — likely still pinned at the ~0.585
  recon floor (inv007 recon-blindness); a drop would be the surprising, hypothesis-supporting result.
- **P4 stability:** no cliff, `grad_norm` in band (~2–4), `grad_skipped=0`; `c_std_mean` ~0.92.
  Collapse-watch: `Δ̂ → 0` (ratio → 1 trivially) and `c_cross_video_cosine` climbing.

### Cross-run read

- `coarse_vs_copy_ratio` is **comparable across both runs and vs inv008** (the residual copy baseline
  is the same `‖Δ‖²`; σ-scaling cancels). The headline comparison is **Run 2 ratio vs Run 1 / inv008
  full-latent ratio at matched λ**.
- Absolute `L_flow` / `coarse_copy_loss` are **at natural Δ scale** in Run 2 (smaller than full-latent
  in absolute terms) — read the **ratio**, not the absolutes, across the two parametrizations.

*(Per PROTOCOL: append dated sections for any new data; do not rewrite the above.)*
