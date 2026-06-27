# Wave 1 Analysis & Wave 2 Prediction — investigation_007 capacity-floor sweep

Grounded in the 5 Wave-1 runs (W&B project `smahalanobis-uc-davis/hjepa-vwm`, group
`inv007_capacity_floor`), pulled via the W&B API at the ~9k cut. Design context:
[`SWEEP_PLAN_decoder_capacity.md`](SWEEP_PLAN_decoder_capacity.md); execution:
[`GUIDE.md`](GUIDE.md).

**Run → config map (Wave 1, weight axis ×3 + decoder axis ×2):**

| Run | run_id | λ_recon | Decoder | n_c | Axis tested |
|---|---|---|---|---|---|
| toasty-donkey-21 | a2trqp9c | 0.1 | 256×2 | 32 | weight (low) |
| jolly-glade-20 | 5x7aoxnn | 0.2 | 256×2 | 32 | weight (mid) |
| light-universe-24 | rju7xsh2 | 0.5 | 256×2 | 32 | weight (aggressive) |
| gallant-dew-22 | 708jrel8 | 0.05 | 512×2 | 32 | decoder width |
| eager-plant-22 | 591mt31k | 0.05 | 512×4 | 32 | decoder depth |

---

# SECTION 1 —— Wave 1 Analysis

## 1.1 The runs (all plateaued, all cut ~8.8–9.25k)

| Run | λ_recon | Decoder | Axis | **L_recon_present** | copy_ratio | eff_rank | L_flow |
|---|---|---|---|---|---|---|---|
| toasty-donkey-21 | 0.1 | 256×2 | weight (low) | 0.5959 | 1.59 | 12.97 | 0.41 |
| jolly-glade-20 | 0.2 | 256×2 | weight (mid) | 0.5915 | 1.56 | 12.98 | 0.40 |
| light-universe-24 | 0.5 | 256×2 | weight (aggr) | **0.5855** | 1.66 | 12.66 | 0.42 |
| gallant-dew-22 | 0.05 | 512×2 | decoder width | 0.5895 | 1.74 | 12.51 | 0.44 |
| eager-plant-22 | 0.05 | 512×4 | decoder depth | **0.5845** | 1.53 | 12.95 | 0.42 |

*(prior baseline easy-blaze-19: λ=0.05, 256×2 → ~0.60)*

All five floors are **plateaued** — the last three logged points move <0.001. The ~9k cut was
correct; running to 15k would not change a single conclusion.

## 1.2 Gate 1 — did the floor move? **No, on either axis.**

The entire spread of `L_recon_present` across all 5 runs is **0.0114** (0.5845 → 0.5959), despite
a 5× weight range and a 6× decoder-param range. Floor ≈ **0.585**, i.e. `c` reconstructs ~**41%**
of the encoder's feature variance — same as the prior runs.

- **Weight axis** is cleanly monotone but hopelessly weak: 0.5959 → 0.5915 → 0.5855
  (λ 0.1→0.2→0.5), about **−0.005 per weight-doubling**. To reach the 0.55 gate from 0.5855 you'd
  need ~6 more doublings — **λ ≈ 30**. The axis is real but useless.
- **Decoder axis**: 512×2 → 0.5895, 512×4 → 0.5845 vs ~0.60 baseline. ~0.01–0.015 of movement from
  3–6× the parameters; depth marginally beats width. The "shared-D pulled two ways" intuition gets,
  at best, a rounding error.
- Tellingly, the **two lowest floors come from *different* axes** (eager 512×4 = 0.5845 and light
  λ=0.5 = 0.5855, basically tied). If either lever genuinely bound the floor, its extreme would
  stand alone. They don't — it's noise around a fixed wall.

## 1.3 The decisive finding — **reconstruction is structurally blind to prediction**

The three recon readouts at the final step:

| Run | present (decode cₜ) | cplus (decode true c₊) | chat (decode *predicted* ĉ) | **chat − cplus** |
|---|---|---|---|---|
| toasty | 0.5959 | 0.5954 | 0.6027 | **+0.0073** |
| jolly | 0.5915 | 0.5906 | 0.5966 | **+0.0060** |
| light | 0.5855 | 0.5843 | 0.5913 | **+0.0070** |
| gallant | 0.5895 | 0.5917 | 0.5999 | **+0.0082** |
| eager | 0.5845 | 0.5850 | 0.5959 | **+0.0109** |

Two things fall out:
1. **present ≈ cplus** (within ~0.001) → reconstructing the *future* from the *true* future latent
   is exactly as hard as reconstructing the present. The floor is temporally invariant — a property
   of `c`'s bandwidth, not of prediction difficulty.
2. **chat − cplus ≈ 0.006–0.011** → swapping a *perfect* future latent for the *actually predicted*
   one costs only ~1–2% of the floor. **The reconstruction objective literally cannot see
   prediction error**, because the ~0.585 floor dwarfs the ~0.01 prediction-induced gap.

This is the mechanistic proof of *why* easy-blaze-19 (option 3, training recon through F_c) was a
negative result — and it's **invariant across weight and decoder**. The decoder rebuilds the same
generic 41%-of-variance blob whether you hand it the true or the predicted latent. No
reconstruction-based objective can supervise prediction at this floor.

## 1.4 The floor is *utilization*-bound — on an axis we aren't sweeping

`c_effective_rank` converges to **~13/256 in every run** — 12.51 (gallant) to 12.98 (jolly), with
the aggressive-weight run actually *lowest* (12.66). It is **invariant to both decoder size and
weight**. `c` isn't bandwidth-starved; it's leaving ~95% of its existing 256-dim slot space unused.
Critically, the collapsed axis is **`d_c` (per-slot dim, rank 13/256)** — and Wave 1's knobs
(weight, decoder) don't touch it, *and neither does Wave 2's `n_c` knob* (which adds slots, not
dims). Hold that thought for Section 2.

## 1.5 Health — every run is clean (a genuine positive)

Across all 5 runs, at every logged step: `grad_skipped = 0`, `instability_warn = 0`,
`c_dead_dim_frac = 0`. `c_cross_video_cosine` collapses from 0.72 (init) to a healthy 0.13–0.30
(videos stay distinct → **no Mode-A representational collapse**), and `c_slot_diversity_rank` holds
~15–19/32 (no slot collapse). The historic **~step-8600 Mode-B cliff never fired** — all ran past
9k cleanly. The reconstruction anchor's predicted protective effect (SWEEP_PLAN §4b) is confirmed.

*One honest caveat:* `coarse_vs_copy_ratio` was the **one metric not fully plateaued** — it drifts
down ~uniformly (a synchronized step-down around 7.5–8k across *all* runs, likely schedule-driven)
and ends at ~1.5–1.7. But it's (a) not axis-dependent and (b) still ~2× the ≤0.70 gate and still
**>1 (losing to copy)**, so more steps wouldn't flip the verdict.

### TL;DR — Section 1
Neither reconstruction **weight** (5×) nor **decoder size** (6×) moves the floor: it sits pinned at
**0.585 (~41% variance) ± 0.01** across all 5 runs, with `c_effective_rank` stuck at ~13/256
regardless of config. The killer finding: reconstructing from a *predicted* vs a *perfect* future
latent differs by only ~0.01 while the floor is 0.585 — so **reconstruction is structurally blind to
prediction**, which is exactly why option-3 failed. The floor is a **utilization** limit on `d_c`,
an axis neither wave sweeps. Training was flawlessly stable throughout (zero grad-skips, zero
collapse, no cliff).

---

# SECTION 2 —— Wave 2 Prediction

Wave 1 eliminated two of three capacity levers and, more damningly, showed the recon objective
can't see prediction at *any* floor near 0.585. That reframes Wave 2 as **mostly confirmatory**.

| Run | Config | Predicted outcome | Breaks 0.55? | Worth running? |
|---|---|---|---|---|
| **n_c=64** | 0.05, 256×2, 64 | floor ~0.57–0.58; eff_rank stays ~13; copy_ratio >1.4 | almost certainly no | **Yes — core test** |
| **n_c=256** | 0.05, 256×2, 256 | floor *maybe* 0.55–0.57 (KV-token effect); eff_rank ~13 absolute → fraction craters; highest collapse risk | possibly grazes it | **Yes — highest value** |
| **n_c=128** | 0.05, 256×2, 128 | interpolates the two above | no | Optional (redundant) |
| **λ=1.0** | 1.0, 256×2, 32 | floor ~0.581 (extrapolated); L_flow ticks up; copy_ratio flat/worse | no (confident) | Low — predictable |
| **combined** | 0.2, 512×2, 64 | floor ~0.575–0.585; levers won't compound | no | Low — confounded |

## 2.1 The latent axis (n_c) is the only real experiment

It's the sole untested lever, but Wave 1 gives a **mechanistic prior against it**: the underused
axis is `d_c` (rank 13/256), and adding *slots* doesn't address *per-slot dim* collapse. The most
likely result is that more slots mostly add **more unused slots** — floor flat, `eff_rank` still
~13, rank *fraction* dropping.

There's one way n_c could move the floor: more slots = more **decoder KV tokens** = more places to
paint detail, lowering recon MSE *cosmetically* without making `c` more information-rich. The
diagnostic that separates the two:
- floor down **∧ eff_rank up ∧ copy_ratio down** → `c` genuinely got richer → latent capacity is
  the lever → Stage 2. *(~10%.)*
- floor down **but eff_rank flat and copy_ratio still >1** → cosmetic KV effect, prediction
  unfixed → **pivot**. *(~30%.)*
- floor flat → utilization confirmed → **pivot**. *(~60%.)*

**Crucial point from §1.3:** even in the optimistic case, n_c can't plausibly drag the floor from
0.585 down to the ~0.01 scale where the prediction gap becomes visible. 8× slots won't take
0.585 → 0.1. So **n_c almost cannot fix prediction even if it lowers the floor** — the blindness is
the binding problem, and it survives any realistic floor reduction.

## 2.2 The other three are confirmation, not discovery

- **λ=1.0**: we have 3 clean weight points; this just verifies the −0.005/doubling extrapolation
  (→ ~0.581) and whether L_flow finally degrades. Most predictable run in the wave.
- **combined**: only meaningful if levers compound — and §1.2/§1.4 say they share one mechanism, so
  they won't. Also confounds n_c=64 with weight+decoder, making it strictly less informative than
  the clean n_c=64 run.
- **n_c=128**: fine as an interpolation point but adds little between 64 and 256.

## 2.3 Recommendation — run a **reduced** Wave 2

Given the GPUs are free you *can* run all 5, but scientifically the wave collapses to **two runs
that matter: `n_c=64` and `n_c=256`** (the latent-axis bookends). Those two decide between "latent
capacity is the lever" and "pivot." The other three are predictable closure.

- **If you want speed/cost discipline:** run just `n_c=64` + `n_c=256` (2 GPUs, one short wave),
  read the floor ∧ eff_rank ∧ copy_ratio triad, and almost certainly **pivot to a horizon/task
  objective** — because the real disease (from §1.3 + the ‖Δc‖/‖c‖~0.38-and-falling
  target-barely-moves problem) is that nothing drives `c` to encode *time-varying* content, which
  reconstruction provably can't fix.
- **If you'd rather have the last word on reconstruction documented:** run all 5; just know 3 of
  them are foregone.

Either way, **the most probable end state of investigation_007 is the pivot**, and Wave 2's job is
to make that pivot evidence-backed rather than assumed.

### TL;DR — Section 2
Only the **n_c runs are real experiments**; λ=1.0 and the combined run are predictable closure
(floor ~0.58, no break). Wave 1's blindness finding means n_c **almost cannot fix prediction even if
it lowers the floor** — you'd need the floor near ~0.01, which 8× slots can't deliver — so the most
likely result (~90%) is floor flat-or-cosmetic, copy_ratio still >1, → **pivot to horizon/task**.
Recommend a **2-run reduced wave (n_c=64 + n_c=256)** to confirm the latent axis, then pivot; run
all 5 only if you want the reconstruction thesis closed on the record.
