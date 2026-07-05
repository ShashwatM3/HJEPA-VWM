# Observations - investigation_013

## Cross-Run Synthesis

The residual reconstruction target did exactly what it was designed to do and nothing more.
It **fixed reconstruction honesty** — with the video-independent template worth zero loss,
the decoder became strongly dependent on each clip's own `c_t` — but it **did not hold
representation geometry**: with every anti-collapse regularizer off, rank, spread, and slot
diversity all contracted anyway, ending slightly below run 052 on each. The scientific value
is that it cleanly separated two failure modes that run 052 had conflated (the template
shortcut vs. reconstruction-cannot-hold-geometry), solving the first and confirming the
second. That confirmation — an explicit anti-collapse force is required, not optional — is
what the whole subsequent present-only arc (investigations 014-015) is built on.

## Run-by-Run Evidence

| # | Run | ID | State | Mode | Key config vs 052 | Verdict | Last key metrics (@ ~step 12000) |
|---:|---|---|---|---|---|---|---|
| 53 | [`ae_sharp_slots_residual_recon`](run_053_ae_sharp_slots_residual_recon/) | `7teohhwc` | crashed (external, ~12150/15000) | present-recon-only | + residual target | Low-rank decodable | rank 10.5; cross-video cosine 0.929; std 0.255; L_recon_present 0.453; L_recon_video_gap 0.433 (growing) |

## Pattern Across The Branch

This is a single-run investigation, so the "pattern" is the internal two-phase trajectory of
run 053: an early expansion (steps 0-~4000) out of the zero-gated init where the code spread
and the honesty gap opened, followed by a long monotone contraction (steps 4000-12150) where
every geometry metric degraded while reconstruction improved by only ~0.05. Contraction that
continues while reconstruction gains almost nothing is the signature of an objective that
defends only a few directions of `c` while the unopposed weight decay (0.05, decoupled)
grinds the rest away.

## What Changed The Research Direction

Because H2 (reconstruction alone cannot hold geometry) was now isolated and confirmed, the
next move was not "try yet another content objective" but "attack the feature space and the
architecture." The offline rank probe (investigation_014) measured that the frozen `e` cloud
concentrates its energy in a few hundred directions with a long weak tail, which explained
mechanistically why cosine reconstruction only defends ~10 directions of `c`. That directly
motivated whitening (equalize the target directions so reconstruction must defend many more)
plus a latent-stack bottleneck (slot competition so slots cannot cheaply merge), tested
together in investigation_015 run 054 on this exact residual-target recipe.

## Original Notes Preserved

The complete run-level analysis is in
[`run_053_ae_sharp_slots_residual_recon/ANALYSIS_053.md`](run_053_ae_sharp_slots_residual_recon/ANALYSIS_053.md).
Headline numbers, pulled live from W&B `7teohhwc` (`get_run_history`, full diagnostic record
every 500 steps):

- **Honesty (the win):** `L_recon_shuffled_c` stayed high (0.86 -> 0.887) while
  `L_recon_present` fell (1.012 -> 0.453), so the gap grew monotonically 0 -> 0.433 and never
  plateaued. Roughly 77% of everything the decoder learned is conditioned on receiving the
  correct video's latent (run 052 was ~15%). The template shortcut is closed.
- **Geometry (the failure):** rank rose to a peak ~19.9 near step 4000 then fell to 10.5;
  centered slot diversity fell 31 -> ~8; cross-video cosine bottomed at 0.842 (step ~3500)
  then rose to 0.929; std peaked at 0.378 then fell to 0.255. All four moved in the collapse
  direction after step ~4000, opposite to reconstruction.
- **Wiring confirmations:** `recon_mean_norm` held ~1624 (raw-feature space — this run was
  NOT whitened, unlike run 054's ~89.7), `recon_target_residual=1`, `present_recon_only=1`,
  `prediction_active=0`, `L_flow=0`, all four geometry lambdas 0. Stability was spotless
  (0 skips, 0 NaNs, grad_norm ~0.02-0.03, AGC never clipped); the crash at step 12150 was an
  external kill, not a training pathology, and the missing ~2850 steps do not change the verdict.
