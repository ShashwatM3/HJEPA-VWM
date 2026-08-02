# Observations — reconstruction-floor architecture audit

## 2026-07-16 — verdict

There is no hard-coded `0.7` clamp and no evidence of numerical failure. The repeated
band is best explained by a shared target/capacity regime:

1. both datasets are separately ZCA-whitened and then scored by per-token cosine, so
   their second-order channel geometry is intentionally standardized;
2. the nearly isotropic 1,024-channel target is immediately projected to 256 channels,
   then 1,024 detailed tokens are reduced to 32 slots — a 128:1 scalar bottleneck;
3. the absolute-target architecture can emit a position-specific template from an
   input-independent, nonzero slot code;
4. covariance/variance shape that code before reconstruction reaches full strength,
   while the cosine LR schedule has spent 99.1% of its cumulative multiplier by step
   12,500.

The first two are the leading explanation for the cross-dataset loss band. The template
channel, multi-objective gradient conflict, and schedule can raise or stabilize the
floor, but their exact shares are not currently measured.

## Correction to the run-058 diagnosis

Run 058's raw reconstruction plateau is real, but its global “template collapse” verdict
is not established by the recorded diagnostics. The EGO4D fixed validation batch is the
lexically first 16 chunks, and all 16 come from source recording
`01cab463-9a16-4817-84a4-a00ef5b7bf39`. Consequently:

- `c_cross_video_cosine` is a within-source, adjacent-chunk cosine, not a cross-source
  video cosine;
- `torch.roll(c, 1)` supplies another chunk from the same recording, not another
  video's code;
- the final `L_recon_video_gap = 0.018` proves weak discrimination among those adjacent
  chunks, but it cannot distinguish a global template from shared wearer/scene/source
  content.

The prior 5.4% “video-conditioned share” should therefore be read only as an
adjacent-chunk conditioned share. A source-diverse diagnostic batch is required before
assigning the global template-collapse label.

## Evidence that survives the correction

- Run 057 and run 058 are stable, completed runs with zero skipped steps.
- Their late random-training-batch means are genuinely close: `0.70695` on SSv2 and
  `0.69610` on EGO4D over steps 12,500-14,950.
- Their fixed-batch terminal losses are `0.71285` and `0.67825`; the EGO4D batch is
  easier and source-confounded, but still in the same whitened-target regime.
- The inspected EGO4D whitening artifact turns raw channel effective rank `230.48` into
  post-whitening effective rank `1023.00`; 1,023 of 1,024 directions are effectively
  unit variance.
- For an isotropic Gaussian target, an ideal rank-256 linear channel projection retains
  `25.027%` of variance, corresponding to the heuristic cosine-loss floor `~0.500`
  before the additional 1,024-to-32 token compression.
- An executable initialization probe confirms `B(zeros) == B(random)` exactly, yet
  `D(B(zeros))` is position-specific. The zero-latent decoder unit test does not close
  this constant-nonzero-code shortcut.
- No optimizer skip, NaN, or decoder AGC event explains the plateau. Global clipping is
  common during the first 1,500 steps and negligible later.

## What is not yet known

- The attainable loss of the best source-independent per-position cosine template in
  the whitened target space.
- The empirical rank-`r` reconstruction curve of the target, as opposed to the Gaussian
  heuristic.
- Whether a free 32x256 latent plus the current decoder can overfit a fixed feature
  batch near zero.
- Per-loss gradient magnitudes and alignment inside B, and decoder output-norm growth.
- Cross-source EGO4D geometry and shuffled-code gaps.

Full evidence and mechanism ranking: [`ANALYSIS.md`](ANALYSIS.md).
