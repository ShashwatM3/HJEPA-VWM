# Observations — Wave 1

Cross-run synthesis of the 5 weight×decoder runs. Per-run detail in each run's
`OBSERVATIONS.md`; the full narrative (tables, mechanism, plots) is in
[`../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md`](../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md) and the
live W&B report linked there. **Numbers below were pulled from W&B, not memory.**

---

## 2026-06-27 — Final read (all runs plateaued at ~9k)

| Run | Axis | **L_recon_present** | cplus | chat | copy_ratio | eff_rank | L_flow |
|---|---|---|---|---|---|---|---|
| [toasty-donkey-21](toasty-donkey-21/) | λ=0.1 | 0.5959 | 0.5954 | 0.6027 | 1.59 | 12.97 | 0.41 |
| [jolly-glade-20](jolly-glade-20/) | λ=0.2 | 0.5915 | 0.5906 | 0.5966 | 1.56 | 12.98 | 0.40 |
| [light-universe-24](light-universe-24/) | λ=0.5 | 0.5855 | 0.5843 | 0.5913 | 1.66 | 12.66 | 0.42 |
| [gallant-dew-22](gallant-dew-22/) | 512×2 | 0.5895 | 0.5917 | 0.5999 | 1.74 | 12.51 | 0.44 |
| [eager-plant-22](eager-plant-22/) | 512×4 | 0.5845 | 0.5850 | 0.5959 | 1.53 | 12.95 | 0.42 |

baseline ([easy-blaze-19](../../investigation_006/easy-blaze-19/)): λ=0.05, 256×2 → ~0.60.

### Three load-bearing conclusions

1. **The floor is inert to weight and decoder.** Total `L_recon_present` spread = **0.0114**
   across a 5× weight range and a 6× decoder-param range. None reached the 0.55 gate. The weight
   axis is monotone but ~−0.005/doubling (would need λ≈30 to break the floor); the two *lowest*
   floors come from *different* axes (eager 512×4 = 0.5845 ≈ light λ=0.5 = 0.5855) — noise around
   a fixed wall. **→ weight-bound and decoder-bound are both ruled out.**

2. **Reconstruction is structurally blind to prediction.** `L_recon_present ≈ L_recon_cplus`
   (decode the *true* future latent) within ~0.001, and `L_recon_chat` (decode the *predicted*
   latent) is only **+0.006…+0.011** higher — while the floor is 0.585. A perfect vs a bad
   prediction reconstruct almost identically → **no reconstruction objective can supervise
   prediction at this floor.** This is the mechanistic reason `easy-blaze-19` (option 3) failed,
   and it holds across every config here.

3. **The floor is a *utilization* limit, not a capacity one.** `c_effective_rank` ≈ 13/256 in
   every run (gate >60), invariant to weight and decoder — `c` leaves ~95% of its slot-dim space
   unused. The under-used axis is `d_c` (per-slot dim), which this wave does not touch — *and
   neither does Wave 2's `n_c` knob.* This is the seed of doubt that frames Wave 2's prediction.

### Health (a real positive)

Every run: `grad_skipped = 0`, `instability_warn = 0`, `c_dead_dim_frac = 0` at every logged
step; `c_cross_video_cosine` 0.72→0.13–0.30 (no Mode-A collapse); `c_slot_diversity_rank`
~15–19/32 (no slot collapse); the historic ~step-8600 Mode-B cliff never fired. The reconstruction
anchor's protective effect is confirmed. One caveat: `coarse_vs_copy_ratio` was the only metric
not fully plateaued (drifting to ~1.5–1.7) but still >1 (loses to copy) and not axis-dependent.

### What this wave decided

It converted "the floor is mysterious" into "the floor is **not** about pressure or decoder
size, and reconstruction can't see prediction anyway." Two of three levers eliminated → the
latent axis (`n_c`) became the sole survivor and the entire content of [Wave 2](../wave_2/). It
also planted the prediction that *even n_c probably can't help*, because the blindness gap (0.01)
is dwarfed by the floor (0.585) at any reachable n_c — see
[Wave 2 OBSERVATIONS](../wave_2/OBSERVATIONS.md).
