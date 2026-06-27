# Observations — eager-plant-22 (decoder 512×4, depth)

## 2026-06-27 — Final read (~step 9100, plateaued)

| Metric | Value | Read |
|---|---|---|
| `L_recon_present` | **0.5845** | **lowest floor of the wave** — yet from a λ=0.05 run, *tying* the 10×-weight run |
| `L_recon_cplus` | 0.5850 | ≈ present |
| `L_recon_chat` | 0.5959 | **+0.0109 over cplus — the largest blindness gap of the wave**, still ≪ the 0.585 floor |
| `coarse_vs_copy_ratio` | 1.53 | lowest of the wave, but still >1 (loses to copy) |
| `c_effective_rank` | 12.95 | ~13 ceiling, unmoved |
| `c_slot_diversity_rank` | 17.8 / 32 | healthy |
| `c_cross_video_cosine` | 0.25 | healthy |
| `L_flow` | 0.42 | baseline-level |

**Trajectory:** warmup drop then asymptote to 0.5845, flat by ~8k (last 1k moves <0.001). Stable
throughout.

## Interpretation

The biggest decoder produces the lowest floor — **but only by 0.005, and from the baseline weight**.
That it *ties the 10× weight run* (light-universe, 0.5855) is the single cleanest proof that neither
axis is the binding constraint: two very different perturbations bottom out at the same ~0.585 wall.

This run is also the **showcase for the blindness finding**. Its `chat − cplus` gap is the wave's
largest (0.0109) — a deeper decoder reconstructs the *true* future latent slightly better, widening
the gap to the *predicted* one — yet 0.011 is still trivial against the 0.585 floor. So even the best
decoder cannot make prediction quality visible to reconstruction. (This is the run plotted in the
W&B report's "present vs cplus vs chat" panel.)

## Connection to the sequence

The last run of [Wave 1](../DESCRIPTION.md) and its decoder ladder's strong end (paired with
[gallant-dew-22](../gallant-dew-22/)). Its rank stuck at ~13 despite 6× decoder params is the
evidence that crystallized the **utilization-limit** reframe: the under-used axis is `d_c`, not
decoder size — and Wave 2's `n_c` knob doesn't touch `d_c` either. That doubt is what makes
[Wave 2](../../wave_2/DESCRIPTION.md) "mostly confirmatory."
