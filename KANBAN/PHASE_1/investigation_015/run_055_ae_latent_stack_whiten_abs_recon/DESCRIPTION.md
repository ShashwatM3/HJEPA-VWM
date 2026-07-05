# Run 055 - `ae_latent_stack_whiten_abs_recon` (inv015)

## Status

PLANNED (not launched)

## Hypothesis

Run 054's exact recipe with the residual reconstruction target REMOVED: present-only
cosine reconstruction against the **absolute** whitened features, latent-stack
bottleneck, fixed offline whitening, zero geometry regularizers.

Question: **in whitened feature space, is the residual target still required to block
the run-052 template shortcut (H1)?** Whitening removes the single global feature mean,
but the per-tubelet-position mean structure survives it (run 054's whitened-space
tracker mean norm was ~89.7, not 0), so an absolute target leaves a partial
video-independent template available to the decoder. This run measures how much of the
reconstruction improvement routes through `c_t` when only whitening defends honesty.

This completes a 2x2 over {recon target: absolute/residual} x {feature space:
raw/whitened} on the honesty axis:

| | raw space | whitened space |
|---|---|---|
| absolute target | run 052 (~15% video-conditioned) | **run 055 (this run)** |
| residual target | run 053 (~77%) | run 054 (~92%) |

Caveat: 052/053 ran the pre-latent-stack bottleneck, so the 2x2 is matched on
objective/space but not on architecture. The honest comparisons are 055-vs-054
(residual target's marginal effect, everything else identical) and the shape of the
shuffled-c curve vs 052's template signature.

Note this is NOT the pre-registered ANALYSIS_054 §4 bottleneck-only control (which
drops the two `--whiten-*` flags and keeps the residual target). That control remains
open and load-bearing for the 053->054 attribution question.

## Config delta vs run 054 (`lx1b6gw2`)

Remove exactly two flags; everything else identical (seed 42, 15k steps, whitening
stats file reused):

```text
- --recon-residual-target
- --recon-mean-momentum 0.99
```

Launch procedure: [`GUIDE.md`](GUIDE.md). Parent framing:
[`../DESCRIPTION.md`](../DESCRIPTION.md), [`../ANALYSIS_054.md`](../ANALYSIS_054.md).
