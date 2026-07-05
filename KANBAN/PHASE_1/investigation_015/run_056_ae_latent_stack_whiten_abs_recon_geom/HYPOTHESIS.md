# Run 056 — `ae_latent_stack_whiten_abs_recon_geom` (inv015)

## Why this run exists

Two separate threads of investigation_011–015 have each produced *half* of a healthy
present-only autoencoder, and no run has ever combined them:

- **Thread A — geometry (inv011 present-only sweep, runs 042–051).** On the OLD bottleneck
  and in RAW (unwhitened) recon space, adding explicit geometry regularizers
  (`lambda_sigreg`, `lambda_cov`, `lambda_var`) drove `c_effective_rank` from the historic
  ~13/256 ceiling all the way to ~150, while keeping videos distinct
  (`c_cross_video_cosine` ~0.08) and reconstruction strong (`L_recon_present` ~0.344). The
  peak point was **run 047 `po_geom_sig5_cov0p01`** (`az60m6mx`): `lambda_sigreg=5`,
  `lambda_cov=0.01`, `lambda_var=0.5`. Covariance was the dominant rank lever. But every one
  of these was a *present-side* win on the old architecture; none carried whitening or the
  latent-stack bottleneck, and none was ever transferred forward.

- **Thread B — honesty + architecture (inv015, runs 054/055).** The Perceiver latent-stack
  bottleneck plus fixed offline whitening fixed the honesty problem — run 054 reached ~92%
  video-conditioned reconstruction, run 055 (the clean/absolute-target arm) ~86% — and
  lifted the geometry *equilibrium* relative to inv013 (terminal `c_effective_rank` 10.5 →
  ~21.9). But with **zero geometry regularizers**, geometry still contracts: rank peaks
  ~34 near step 2.5k then falls to ~21, `c_std_mean` settles at ~0.40 (target 1.0), and
  centered slot diversity falls from ~31 to ~12. Run 055's own analysis names the fix
  explicitly — "an explicit anti-collapse force is required" (the settled H2 conclusion) —
  and its next-step item 3 pre-registers exactly this follow-up.

This run joins the two threads: **take the run 055 base (present-only, clean/absolute
whitened-cosine reconstruction, latent-stack bottleneck) and add back the inv011 winning
geometry regularizer bundle (sigreg 5, cov 0.01, var 0.5).**

## The hypothesis

**H:** The geometry contraction seen in run 055 is caused by unopposed weight decay grinding
down every direction the reconstruction loss does not actively defend, exactly as in
inv013's H2. Supplying the inv011 anti-collapse bundle will hold the geometry the recipe
otherwise loses — lifting terminal `c_effective_rank`, `c_std_mean`, and centered slot
diversity toward the sweep's high-rank regime — **while** the whitening + latent-stack +
clean-cosine machinery keeps reconstruction honest. The target outcome is the first
present-only run to show BOTH honest, video-specific reconstruction AND healthy
representation geometry in a single run, which neither thread achieved alone.

Concrete predictions, judged against run 055's terminal values (`c_effective_rank` 21.5,
`c_std_mean` 0.40, centered slot rank 12.1, `c_cross_video_cosine` 0.82, honesty share
~86%):

1. **Geometry lifts substantially.** `c_effective_rank` no longer contracts after its early
   peak; it holds or climbs well above 40 (values above ~40 cannot come from the 32 slots
   alone and must be defended feature directions). `c_std_mean` rises to ~1.0 (var floor +
   sigreg both target unit variance). `c_cross_video_cosine` falls well below run 055's 0.82,
   toward the sweep's ~0.08. Covariance is expected to be the dominant lever, matching inv011.
2. **Honesty is preserved or improved, not destroyed.** Pushing `c` to higher rank and
   isotropy makes it *more* informative, so the decoder should rely *less* on the surviving
   whitened per-position template. Prediction: `L_recon_shuffled_c` stays high (near run
   055's ~0.95 or better) and the video-conditioned share holds ≥ run 055's ~86%. The failure
   mode to watch is the opposite: a geometry term that shreds content routing and drops the
   video gap.
3. **Stability holds.** The sweep ran this exact bundle with AGC quiet and zero grad skips;
   the added latent-stack depth (run 055) was also stable. No instability expected.

## What is new vs. untested here

- **Combination, not mechanism.** Every ingredient has run before. What is untested is the
  *interaction* of the geometry regularizers (which act on `c_t`) with whitening (which acts
  on the recon target `e`) and the latent-stack bottleneck. Mechanically they are
  independent — the regularizers never see `e` and whitening never sees `c` — so the
  expectation is additive, but the equilibrium is genuinely new territory.
- **Whitened space may need a different `c` geometry than raw space.** In raw space the
  recon target concentrated energy in ~10 directions; whitening equalizes it. The geometry
  bundle pushes `c` toward isotropy regardless, so the two pressures should now agree rather
  than compete (raw-space recon pulled `c` low-rank; whitened recon pulls it broad, same
  direction the regularizers pull). This is a reason to expect the combination to work
  *better* than the raw-space sweep, not worse.

## The one deliberate divergence (chosen, not accidental)

The base is the **clean / absolute** reconstruction target (run 055), NOT the residual
target (run 054). Investigation_015's settled recipe decision keeps `--recon-residual-target`
because it buys ~6 points of honesty (86% → 92%) for zero geometric cost, and the
pre-registered next-step phrased the geometry follow-up on the *residual* base. This run
deliberately layers the geometry bundle on the *clean* base instead, to isolate whether the
anti-collapse regularizers alone can carry both geometry and honesty — and, per prediction 2,
whether a richer `c` partially substitutes for the residual target's honesty top-up. If this
run's honesty lands at or above run 055's ~86%, it argues the geometry terms are doing some
of the residual target's job; if it drops, it re-confirms the residual target is needed. The
residual-base variant is the natural sibling run and is noted in NEXT_STEPS.

## No code changes

All three regularizers are already implemented and flag-gated (see
[`DESCRIPTION.md`](DESCRIPTION.md)). This run is a pure config delta over run 055: three
regularizer weights turned on. Nothing in the pipeline changes.
