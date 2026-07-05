# investigation_013 - Does subtracting the per-position feature mean (a residual reconstruction target) force reconstruction pressure to carry video-specific information through c_t, without any geometry regularizers?

**Status:** CLOSED
**Runs covered:** 053
**Theme:** residual reconstruction target as the run-052 template-collapse fix

## Question

Run 052 (investigation_012) proved that a sharpened-slot bottleneck under pure cosine
reconstruction, with every geometry regularizer off, does NOT produce a healthy `c_t`:
reconstruction got excellent (`L_recon_present` 0.293) while the representation collapsed
(rank 13.4, cross-video cosine 0.906, std 0.295, rising `L_cov`). The diagnosis was a
**template shortcut** — the decoder reconstructs a single video-independent "average
feature field" that decodes well for every clip, so reconstruction never has to route
video-specific information through `c_t` (an estimated ~85% of the reconstruction gain
was the shared template).

> If the reconstruction target is changed from the absolute frozen feature `e` to the
> **per-position residual** `e - mean` — where `mean` is an EMA per-tubelet-position mean
> of `e_t` tracked across training batches — the shared template earns exactly zero loss.
> Does that force all reconstruction pressure through `c_t` and, as a side effect, hold the
> representation geometry that run 052 lost?

## Why This Investigation Exists

Run 052 entangled two possible causes of the collapse: (H1) the template shortcut lets
reconstruction succeed without using `c_t` at all, and (H2) reconstruction pressure alone,
even when honest, simply cannot maintain rank/variance. This investigation surgically
removes H1 with the residual target, everything else held identical to run 052, so the two
hypotheses can be separated. It also adds the honesty diagnostics that run 052's post-mortem
recommended (`L_recon_shuffled_c`, `L_recon_video_gap`) so "is the decoder actually using
this video's code?" becomes a directly logged, RNG-free measurement rather than an inference.

## The mechanism under test

- **Residual target.** The reconstruction anchor scores `D(c_t)` against `e_t - mean`
  instead of the absolute `e_t`. `mean` is a parameter-free EMA buffer
  (`models.FeatureMeanTracker`, momentum 0.99) folded from the training batches only —
  the fixed diagnostic batch never leaks into it. Subtracting a video-independent mean
  makes the template worth zero, so every unit of reconstruction improvement must come
  from video-specific content carried by `c_t`.
- **Everything else = run 052.** Sharpened-slot bottleneck, cosine loss, decoder 512x4,
  `n_c=32`, `k=12`, present-recon-only, all four geometry lambdas at 0, seed 42. The only
  change from run 052 is `--recon-residual-target` (+ `--recon-mean-momentum 0.99`).
- **Honesty probe.** `L_recon_shuffled_c` decodes a deterministically rolled batch of
  latents against the unrotated targets; `L_recon_video_gap = L_recon_shuffled_c -
  L_recon_present`. A gap near zero means the decode barely depends on which video's latent
  it received (the template signature); a large, growing gap means the decoder genuinely
  depends on the correct code.

## Runs

| Run | Config vs run 052 | Role | W&B name (id) |
|---|---|---|---|
| 053 | + `--recon-residual-target` (residual target); all else identical | Isolate whether the residual target fixes the template shortcut and/or holds geometry | `ae_sharp_slots_residual_recon` (`7teohhwc`) |

## Result (2026-07-03)

The residual target **worked on honesty and failed on geometry** — a clean, decisive split.
Reconstruction became provably video-specific (`L_recon_video_gap` grew to 0.433 and was
still rising at the crash; a wrong video's latent decoded almost twice as badly; ~77% of the
decoder's improvement is conditioned on the correct code, versus ~15% in run 052). But the
representation still collapsed — rank fell to 10.5 (peak ~19.9 near step 4000), cross-video
cosine rose to 0.929, std sank to 0.255 — ending marginally worse than run 052 on every
geometry axis. H1 is therefore solved and H2 is confirmed: reconstruction pressure, however
honest, will not hold rank/variance on its own, so an explicit anti-collapse force is a
requirement of this architecture, not a crutch. Full read: [`run_053_ae_sharp_slots_residual_recon/ANALYSIS_053.md`](run_053_ae_sharp_slots_residual_recon/ANALYSIS_053.md).

## Parent / sibling context

- Template-collapse discovery: [investigation_012](../investigation_012/) (`ae_sharp_slots_recon_only`, `662hfy3c`).
- The rank-budget reframe that motivated the next lever (whitening): [investigation_014](../investigation_014/).
- The whitening + latent-stack follow-up that reuses this exact recipe: [investigation_015](../investigation_015/) (`ae_latent_stack_whiten_recon_only`, `lx1b6gw2`).
- Code: `cfg.train.recon_residual_target`, `--recon-residual-target`, `--recon-mean-momentum`,
  `models.FeatureMeanTracker`, the residual-target branches in `train.train_step` /
  `run_diagnostics` / `reconstruction_readouts`, and the honesty readouts in
  `diagnostics` / `train.reconstruction_readouts`.
