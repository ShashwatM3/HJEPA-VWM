# Next steps — investigation_008 (SIGReg sweep)

## Tier 0 — implement SIGReg (prerequisite, code change) ✅ DONE (uncommitted)

Implemented in the working tree (spec: [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md) §2):

1. ✅ `losses.sigreg_loss(...)` — BHEP/Epps–Pulley over random projections; pooled
   `c → (N, d_c)`, row subsample (`max_rows=512`), `O(P·S²)` pairwise term, fp32.
2. ✅ `cfg.train.lambda_sigreg = 0.0` (`config.py`, next to `lambda_var`).
3. ✅ `train.py:train_step` — **computes always** (logs `L_sigreg`), **adds only when
   `lambda_sigreg > 0`** (mirrors `lambda_cov`/`lambda_slot`; `λ=0` byte-identical).
   `"L_sigreg"` added to the metrics dict.
4. ✅ `--lambda-sigreg` CLI flag + `cfg.train` override (mirrors `--lambda-var`).
5. ✅ `tests/test_sigreg.py` — ≈0 on `N(0,I)`, strictly higher on low-rank, finite
   gradient, 0 on a degenerate batch, reproducible with a generator. (Skips locally
   where torch is absent; **runs on the pod**.) All changed files byte-compile clean.
6. ⏳ **Commit on `phase1-v0.2-frozen-encoder`** (flag defaults to current behavior) so
   the pod `git pull` gets it — left for the user (PROTOCOL: commit only when asked).
   **Run `pytest tests/test_sigreg.py` on a torch box first** to confirm the BHEP
   thresholds hold before launching.

## Tier 1 — launch the 4× A100 sweep

Per [`GUIDE.md`](GUIDE.md). `λ_sigreg ∈ {0.3, 1.0, 3.0, 10.0}`, all on the eager-plant-22
background (decoder 512×4, `λ_recon=0.05`, `λ_var=0.5`, `n_c=32`, `k=12`). Hardening
(carried over from the inv007 Wave-2 death): verified `tmux` detach, step-600 tripwire,
**plus a step-0 loss-component print** to confirm `λ_sigreg·L_sigreg` is in range before
committing 8h (re-center the λ ladder if not).

After data: record each run's W&B name+id in [`OBSERVATIONS.md`](OBSERVATIONS.md), read
the triad `c_effective_rank ∧ coarse_vs_copy_ratio ∧ L_recon_present` against
[`SIGREG_DESIGN.md`](SIGREG_DESIGN.md) §6.

## Tier 2 — conditional follow-ups (depend on the Wave-1 result)

- **Rank rose, copy_ratio fell** → SIGReg is a real lever: refine λ around the winner;
  try the **pure-replace** arm (`--lambda-var 0`) to confirm SIGReg suffices alone.
- **Rank rose, copy_ratio stayed >1** → utilization was *not* the prediction bottleneck →
  the temporal pivot is unimpeachable → open **investigation_009** on the prediction side
  (horizon `k` sweep / inverse-dynamics / rollout — END_OF_WAVE_2 §2.6 Tier 1).
- **Rank would not move even at λ=10** → the bottleneck `B` architecture binds rank →
  architectural follow-up on `d_c`/`B` (END_OF_WAVE_2 §2.6 Tier 2).

## On hold (not abandoned)

- **inv007 Wave 2 (`n_c` sweep)** — deprioritized; SIGReg attacks the *actually* collapsed
  axis (`d_c`), `n_c` adds slots not dims. See
  [../investigation_007/wave_2/NEXT_STEPS.md](../investigation_007/wave_2/NEXT_STEPS.md).
  Revisit only if inv008 shows latent *capacity* (not utilization) is implicated.

## What closes investigation_008

The `λ_sigreg` sweep answers the rank-ceiling question. Set
[DESCRIPTION.md](DESCRIPTION.md) status → CLOSED with a Conclusion in OBSERVATIONS, refresh
the [PHASE_1 README](../README.md) row, and point at whichever Tier-2 follow-up the result
selects.
