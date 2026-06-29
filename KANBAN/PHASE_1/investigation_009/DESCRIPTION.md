# Investigation 009 — Does predicting the temporal residual (Δ) beat the copy baseline?

**Status:** OPEN — **both runs complete** (committed `bc77db6`); analysis written
([`RESULTS_ANALYSIS.md`](RESULTS_ANALYSIS.md)). Awaiting the Tier-2 decision (anti-collapse on `Δ̂`
/ govern ρ — almost certainly → investigation_010).
**Opened:** 2026-06-28 (spun out of [investigation_008](../investigation_008/DESCRIPTION.md))
**Closed:** —

## Question

investigation_008 showed SIGReg breaks the rank-13 ceiling (`c_effective_rank` → 73 at λ=10) but
prediction gets **monotonically worse** (`coarse_vs_copy_ratio` 1.6 → 8.2): the disease is
**temporal, not utilization**. The copy baseline wins because `c` barely moves over the horizon
(`coarse_copy_loss` *falls* as SIGReg fills `d_c` with static appearance detail). This wave is the
first swing at the temporal problem, as **one wave of two runs**:

- **Run 1 — SIGReg substrate.** The current stack with the reconstruction anchor removed and only
  SIGReg as the rank lever (`λ_sigreg=6`). Characterizes the cleanest strong-SIGReg substrate (no
  recon baggage) and what its prediction does.
- **Run 2 — residual prediction.** Keep the full stack (SIGReg + recon) but change `F_c`'s task from
  predicting the full future latent `c_{t+k}` to predicting the **temporal residual**
  Δ = `c_{t+k} − c_t`, reconstructing the future from `ĉ = c_t + Δ̂`.

> Does focusing `F_c` on the *change* Δ extract predictable motion the full-latent flow was wasting
> capacity copying through — i.e. does `coarse_vs_copy_ratio` finally fall toward <1 — and does
> reconstruction-with-residuals help?

## The residual reparametrization — what it does and does NOT do (load-bearing caveat)

Predicting Δ instead of `c_{t+k}` is, in exact arithmetic, a **reparametrization**. The copy
baseline stays `‖c_t − c_{t+k}‖² = ‖Δ‖²` (in residual space it is "predict the zero residual"), so
`coarse_vs_copy_ratio` is **numerically identical at the optimum** and directly comparable across
the two runs. It does **not** weaken the copy baseline. What it changes:

- **Inductive bias / optimization (the upside):** `F_c` no longer spends capacity transporting the
  large static `c_t` through the noisy flow — it focuses entirely on the motion Δ, which can extract
  predictable structure the full-latent flow drowned out.
- **The lazy shortcut relocates (the risk):** "output 0" now ties copy (instead of "parrot `c_t`"),
  so without predictable motion Run 2 asymptotes to ratio ≈ 1 from above, not below.

So Run 2 is a real, standard inductive-bias bet — **not a guaranteed fix**. The lever that actually
*weakens* copy is the horizon `k` (bigger, more-structured Δ); residual prediction is its natural
partner and the leading Tier-1 follow-up ([END_OF_WAVE_2 §2.6](../investigation_007/END_OF_WAVE_2.md)).

## Key decisions

- **Variance floor stays ON (`λ_var=0.5`) in BOTH runs — data-grounded (not the pure-replace arm).**
  W&B (inv008 λ10 `fbqgix1x`) shows that under strong SIGReg the floor's hinge is **continuously
  active**: `L_var ≈ 0.03–0.05` *every logged step*, with `c_std_mean` pinned at **~0.88–0.91 (below
  the 1.0 target the whole run)** because SIGReg flattens the spectrum and pulls per-dim std below 1.
  The floor is **load-bearing, not redundant** — it holds std at ~0.90 instead of lower. (Contrast:
  `cerulean-snow-13`, floor-only, relaxed to `L_var ≈ 0.008` once std reached 1.0.) The pure-replace
  arm (`λ_var=0`) is deferred.
- **Residual mechanics (the changes beyond "update the metric"):**
  - **EMA-both-ends temporal target** Δ = `B_EMA(e_{t+k}) − B_EMA(e_t)` — purely temporal, fully
    detached (not contaminated by the online-vs-EMA gap).
  - **Flow noise scaled to Δ's std** (`losses.residual_target` returns σ): keeps the velocity-target
    SNR at ~50% (Δ is ~0.4× the scale of `c`), keeps the ratio comparable, and keeps SIGReg active in
    the loss balance.
  - **Recon add-back** `ĉ = c_t^online + Δ̂` so the recon gradient still reaches `B` (via the add-back
    and the `F_c` conditioning) and `F_c`.
  - **Copy baseline = zero residual**; `coarse_baselines` / `reconstruction_readouts` are
    residual-aware (`L_recon_cplus` still decodes the *true* `c_plus`).
  - **No anti-collapse term on Δ̂** — deliberately omitted for a clean first test (the
    `Δ̂ → 0` collapse risk is left observable, not pre-empted).
- **Recon kept ON in Run 2** (`λ_recon=0.05` present anchor + `λ_recon_pred=0.05` residual-future
  decode) — testing the reconstruction-with-residuals hypothesis despite inv007's recon-blindness
  (0.585 floor) finding.
- **λ values:** Run 1 `λ_sigreg=6.0` (predicted rank ~55, near but likely under the 60 gate),
  Run 2 `λ_sigreg=5.0` (predicted rank ~50). Both decoder **512×4** (eager-plant-22 background),
  `n_c=32`, `k=12`, `lr_coarse_flow=1e-4`.

## Runs

**One wave, 2-wide on a 2× A100 pod. Control for both = inv008 history (full-latent, on W&B).**

| GPU | Run | λ_sigreg · λ_var · recon · task | Role | W&B name (id) |
|---|---|---|---|---|
| 0 | **Run 1** | 6.0 · 0.5 · off · full-latent | SIGReg-only substrate | `fine-meadow-34` (`xz3nabr9`) |
| 1 | **Run 2** | 5.0 · 0.5 · 0.05 + 0.05 · **residual** | residual prediction + recon | `graceful-river-35` (`jsh6uo7p`) |

**Result (2026-06-28):** Run 2 (residual) reversed the static-`c` disease — `c` became temporally
dynamic (ρ(c_t,c_{t+k}) ~0.9→0.23), reached the richest healthy `c` yet (rank 57.9, cosine 0.17,
std 1.0), and dropped `coarse_vs_copy_ratio` ~6× (6.06 → 1.08, min 0.915@3500) — **but** `F_c`
ties copy by predicting `Δ̂≈0` (ratio not decisively < 1). Run 1 confirmed SIGReg pumps rank (→50)
while freezing `c` (ρ→0.92) and worsening prediction (ratio→6). Bottleneck relocated: "`c` static"
→ "`c`'s motion unpredictable." Full analysis: [`RESULTS_ANALYSIS.md`](RESULTS_ANALYSIS.md).

**Bottom line (pre-data):** decisive on two fronts — (1) does residual prediction move
`coarse_vs_copy_ratio` below the full-latent baseline (and toward <1)? (2) is SIGReg-only a clean
substrate worth carrying forward, with recon confirmed as dead weight?

## Full design, implementation, execution

- **Execution (2× A100 pod → launch → monitor):** [`GUIDE.md`](GUIDE.md)
- **Registered predictions:** [`OBSERVATIONS.md`](OBSERVATIONS.md)
- **Plan / tiers / close conditions:** [`NEXT_STEPS.md`](NEXT_STEPS.md)

## Parent / sibling context

- inv008 SIGReg result (rank↑, prediction↓): [investigation_008/OBSERVATIONS.md](../investigation_008/OBSERVATIONS.md)
- The temporal reframe + Tier-1 ladder: [investigation_007/END_OF_WAVE_2.md](../investigation_007/END_OF_WAVE_2.md) §2.3–2.6
- Code: `--predict-residual` flag; `losses.residual_target`; `train.train_step` /
  `run_diagnostics` / `reconstruction_readouts`; `diagnostics.coarse_baselines`;
  `cfg.train.predict_residual`.
