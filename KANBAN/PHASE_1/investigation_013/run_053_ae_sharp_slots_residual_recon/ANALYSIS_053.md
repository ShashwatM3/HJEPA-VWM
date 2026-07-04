# Run 053 analysis — `ae_sharp_slots_residual_recon` (`7teohhwc`, inv013)

> Analyzed 2026-07-03 via Reading Cycle B (`GUIDES/READING_EXPERIMENTS.md`), W&B run
> `smahalanobis-uc-davis/hjepa-vwm/7teohhwc`, group `inv013_residual_recon_only`.
> Diagnostic record is complete (diag every 500 steps; all 25 points read, not downsampled).

**TL;DR verdict: the residual reconstruction target did exactly what we designed it to do —
reconstruction is now provably video-specific (`L_recon_video_gap = 0.433` and still growing) — but
the geometry still collapsed anyway, and in fact ended slightly worse than run 052 (rank 10.5 vs
13.4, cosine 0.929 vs 0.906). This is a scientifically decisive negative-on-geometry result, not a
wasted run: it cleanly separates two failure modes we had conflated, and it tells us the next lever
unambiguously.**

Cycle B label: **Low-rank decodable** — with the important nuance that it is *not* template
collapse this time.

## Run status

- State: **crashed** at step **12,150 / 15,000** (~81%), after ~5h56m (~1.75 s/step). There is no
  training-side pathology before death — `grad_skipped=0`, `grad_has_nan=0`, `instability_warn=0`
  for the entire run, `grad_norm` flat at ~0.02–0.03, and every metric was on a smooth trend at the
  last log. This looks like an external kill (pod stopped / process killed), not a blowup. Last
  checkpoint would be step 10,000 (`checkpoint_every=2500`; 12,500 was never reached).
- The missing 2,850 steps don't change the verdict: every geometry metric was degrading
  monotonically and decelerating, and LR was already down to 0.12× (cosine decay tail).
  Extrapolating, it would have ended around rank ~9, cosine ~0.94. Not worth resuming.
- Config verified correct: `present_recon_only=1`, `prediction_active=0`, `L_flow=0`,
  `L_recon_pred=0`, `recon_target_residual=1`, all four geometry lambdas 0, `recon_mean_norm`
  stable ~1624 (tracker initialized and healthy), `recon_scale` reached 1.0 after warmup. AGC never
  clipped on B or D. This is exactly the experiment the GUIDE specified.

## Reading cycle (Cycle B, present-recon-only)

| Q | Question | Pass? | Evidence (@ step 12,000 unless noted) |
|---|----------|-------|---------------------------------------|
| Q1 | Trained alive, right mode? | ✅ | 0 skips, 0 NaNs, grad_norm ~0.028, all mode flags correct; crash at 12,150 was external |
| Q2 | `c_t` alive / video-specific? | ⚠️ split | Geometrically weak: `c_std_mean` **0.255** (target ~1.0), cosine **0.929**. But *informationally* video-specific — see gap below |
| Q3 | Rich latent? | ❌ | `c_effective_rank` **10.5** (target >60; run 052 was 13.4); `c_slot_diversity_rank` **8.5**/32 |
| Q4 | Recon learning content? | ✅ strongly | `L_recon_present` 1.012 → **0.453**, monotone; **`L_recon_shuffled_c` ~0.886 (flat/rising)**; **`L_recon_video_gap` 0 → 0.433, still growing at crash** |
| Q5 | Geometry & content cooperating? | ❌ | After step ~4,000 they moved in *opposite* directions: recon −0.05 while rank −50%, cosine +0.09, std −0.12 |
| Q6 | **Verdict** | | **Low-rank decodable (video-specific, geometrically contracted)** |

## Finding 1 — the residual target worked. The honesty probe proves it.

This is the headline positive result, and it's unambiguous:

- Decoding a video's own `c_t` gives residual-cosine loss **0.453**; decoding a *shuffled* (wrong
  video's) `c_t` gives **0.886**. The wrong latent decodes nearly twice as badly.
- From initialization (1.012), the decoder improved by 0.559 with the correct latent but only 0.126
  with a wrong one — so roughly **77% of everything the decoder learned is conditioned on the
  specific video's latent**. Compare run 052, where we estimated ~85% of the improvement was a
  video-independent template.
- `L_recon_shuffled_c` actually *rose* slightly over training (0.864 → 0.886) while the gap widened
  every single diagnostic step, never plateauing. The decoder became progressively *more* dependent
  on the right code, exactly the incentive we built.

Run 052's specific failure — reconstruction loss satisfiable without any information flowing
through `c_t` — is fixed. The template now earns zero loss and all reconstruction pressure routes
through the bottleneck.

## Finding 2 — geometry collapsed anyway, in two clean phases

The complete diagnostic record shows an expansion phase followed by a long contraction:

- **Steps 0–~4,000 (expansion):** from the zero-gated init (all videos identical, cosine 1.0,
  std 0), the code spread out: cosine fell to its best value **0.842 @ step 4,000**, `c_std_mean`
  rose to its peak **0.378 @ 4,000**, and the video gap grew fastest here.
- **Steps 4,000–12,150 (contraction):** every geometry metric then degraded monotonically — rank
  19.9 → 10.5, cosine 0.842 → 0.929, std 0.378 → 0.255, slot diversity 12.5 → 8.5, attention
  entropy 0.755 → 0.531 — while `L_recon_present` improved only 0.502 → 0.453. The optimizer traded
  away half the code's rank for a ~0.05 recon gain.

Two definitional facts matter for reading this correctly (verified in `diagnostics.py:76-100`):

- `c_effective_rank` **is centered** — so 10.5 is the true dimensionality of the code's variation,
  not an artifact of a big shared mean. The genuine information lives in a ~10-dimensional subspace
  of the 256 available.
- `c_cross_video_cosine` is **uncentered** — 0.929 largely measures that a shared mean direction
  dominates each video's small deviation (std 0.25 vs the 1.0 target). Combined with the positive
  gap, the accurate picture is: *every video's code = one big shared vector + a small, low-rank,
  but genuinely video-identifying deviation*, which the 512-dim/4-block decoder then decompresses.

Mechanistically this is what you'd expect with **zero anti-collapse forces**: cosine reconstruction
is satisfiable by a low-amplitude, low-rank code (the decoder can act as a codebook expander),
LayerNorm on `c_t` fixes overall scale but not spread, and weight decay (0.05, decoupled — the one
term AdamW's scale-invariance doesn't neutralize) continuously contracts every direction the loss
doesn't actively defend. The loss only defends ~10 directions, so the other ~246 decayed. The slot
metrics tell the same story from another angle: the 32 slots merged from 32 effective directions at
init to 8.5, with attention sharpening in lockstep.

## Run 052 vs run 053

| Metric | 052 (absolute target) | 053 (residual target) | Read |
|---|---|---|---|
| `L_recon_present` | 0.293 | 0.453 | **Not comparable** — different objectives; residual is strictly harder |
| `L_recon_video_gap` | (no probe; inferred ≈0) | **+0.433, growing** | The fix worked |
| `c_effective_rank` | 13.4 | 10.5 | Both collapsed; 053 slightly worse |
| `c_cross_video_cosine` | 0.906 | 0.929 | Same |
| `c_std_mean` | 0.295 | 0.255 | Same |

## What this run decides

Run 052 had two entangled hypotheses for the collapse: (H1) the template shortcut lets recon
succeed without using `c_t`; (H2) reconstruction pressure alone simply cannot maintain geometry.
Run 053 surgically removed H1 — and collapse persisted essentially unchanged. **So H1 and H2 are
separable, H1 is now solved, and H2 is confirmed: no content objective, however honest, will hold
rank/variance up by itself. An explicit anti-collapse force is a requirement, not a crutch.**
That's a real result, even though the run fails the GUIDE's geometry gates ("recon improves but
geometry remains collapsed" = its named fail/inconclusive bucket).

**Concrete next step (one lever):** rerun this exact recipe with the inv011-style geometry terms
restored — variance floor (`lambda_var`, target std 1.0, which directly attacks the 0.25 amplitude
problem) plus covariance penalty — *keeping* `--recon-residual-target`. That combination is now
unconfounded: we know the geometry regularizers produce rank 90–150 (runs 043–051) and we know the
residual target routes real content through `c_t` (this run). The hypothesis for the combined run
is "strong present representation," and `L_recon_video_gap` is the honest monitor that tells us
whether the geometry terms preserved or destroyed the content routing. A secondary, cheaper knob if
contraction persists: lower `weight_decay` on B (0.05 is acting unopposed here).
