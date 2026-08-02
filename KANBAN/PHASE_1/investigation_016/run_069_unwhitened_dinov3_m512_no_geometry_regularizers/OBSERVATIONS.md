# Observations — run 069

## 2026-07-21 — W&B history reconciled

The run finished cleanly with zero skipped, nonfinite, or warned updates. Over the last six
diagnostic points, median fixed-batch correct reconstruction was `0.11478`, rolled-code
reconstruction was `0.18083`, and their gap was `0.06612`. The gap proves exact-chunk code
dependence on the recorded batch, but it represents only about `7.8%` of the improvement from the
initial correct-code loss and the diagnostic batch contains adjacent chunks from one source UID.

Geometry contracted without explicit pressure: late median effective rank was `14.60`, centered
slot rank `10.51`, mean std `0.3329`, pair cosine `0.8699`, and dead-dimension fraction `0.0127`.
The correct verdict is **LOW-RANK DECODABLE; GLOBAL COLLAPSE INDETERMINATE**. The run validates the
DINO implementation and training path, not a healthy representation or a cross-source content
claim.

See [`ANALYSIS.md`](ANALYSIS.md) for the full present-only reading cycle and provenance boundary.
