# Next steps — Wave 2

## ⏸️ ON HOLD — superseded by investigation_008 (SIGReg)

**Decision (2026-06-27):** the `n_c` re-run is **deferred**, not cancelled. Wave 1 showed
the *actually* collapsed axis is `d_c` (per-slot dim, rank 13/256), and `n_c` adds slots,
not dims — so it's the lower-value lever. The higher-value next experiment is
[investigation_008](../../investigation_008/DESCRIPTION.md): **SIGReg**, an
isotropic-Gaussian regularizer that attacks `d_c` directly. The 4-GPU re-run below is
fully prepped (GUIDE §3c) and can be revived in one launch **if** inv008 implicates latent
*capacity* (more slots needed) rather than *utilization* (fill existing dims). Until then,
the GPUs go to inv008.

## Tier 0 (PARKED) — re-run the wave (it never produced data) — 4-GPU plan

The first attempt produced step-0 init only; the 5 failed runs are being **deleted from W&B**. The
pod is now 4-GPU (was 5), so we drop exactly one run. Launch command + hardening live in
[`../GUIDE.md`](../GUIDE.md) §3c.

1. **Diagnose the death first** (5 min): `tail -n 50 logs/*.log` on the pod; check RunPod pod
   events for a stop/reclaim; `dmesg | grep -i oom`.
2. **Re-launch 4 runs — drop only `helpful-snow-25` (λ=1.0).** Keep the **full latent ladder
   `n_c` 64 / 128 / 256** (the only untested axis — a 3-point trend, not just bookends) **plus the
   combined "all bigger" run** as the one interaction-effect check. λ=1.0 is the lone cut: Wave 1
   has three clean weight points on a flat −0.005/doubling line and `L_flow` didn't degrade even at
   λ=0.5, so its result is known to two decimals → zero expected information. (If you were forced to
   2 GPUs, the irreducible core is `n_c=64` + `n_c=256`; with 4 GPUs we get the middle rung and the
   interaction run for free.)
3. **Harden the launch** so a silent 6-minute death can't recur:
   - Confirm `tmux` detach (`Ctrl-B D`) *before* disconnecting SSH; **verify `tmux ls`**.
   - Step-600 tripwire: ~15 min in, check each log passed `step.*500` (snippet in GUIDE §3c) — flags
     any run that hasn't logged the first diagnostic.
   - If `n_c=256` OOMs (not expected), lower `global_batch`/`num_workers` in `config.py` (no CLI
     flag) and relaunch that run alone.
4. Cut at the `L_recon_present` plateau (~8–9k); latent runs may reorganize, so watch the curve
   flatten rather than fixing a step.
5. **After data lands:** record each kept run's **new** W&B name + id in its `OBSERVATIONS.md`
   (the attempt-1 ids are dead), then read the `L_recon_present` ∧ `c_effective_rank` ∧
   `coarse_vs_copy_ratio` triad against SWEEP_PLAN §4.

## Tier 1 — the likely pivot (regardless of the n_c re-run)

Per [`../END_OF_WAVE_2.md`](../END_OF_WAVE_2.md) §2 and the external literature it cites, the real
disease is **temporal under-informativeness** (`c` is distinct-per-video but nearly static in time,
so the copy baseline wins). Reconstruction can't fix that. Open a **new investigation** on the
prediction side:

1. **Increase horizon `k`** (sweep `--horizon-k` up from 12) — cheapest test; bigger `‖Δc‖` weakens copy.
2. **Inverse-dynamics / transition auxiliary** (predict what changed between `eₜ` and `e₊`).
3. **Multi-step rollout loss** on `F_c` (V-JEPA 2-AC / SkyJEPA standard).
4. **Explicit anti-copy term** (penalize `ĉ` near `cₜ`).

**Decision metric for everything downstream: `coarse_vs_copy_ratio` → <1.** Reconstruction readouts
are now demoted to diagnostics, not the objective.

## What closes investigation_007

The n_c re-run result (Tier 0) closes the *binding-constraint* question. Whichever way it lands, the
investigation's center of gravity moves to the Tier-1 prediction pivot — likely a **new
investigation_008**. Update [investigation_007/DESCRIPTION.md](../DESCRIPTION.md) status to CLOSED
and point it there once the re-run lands.
