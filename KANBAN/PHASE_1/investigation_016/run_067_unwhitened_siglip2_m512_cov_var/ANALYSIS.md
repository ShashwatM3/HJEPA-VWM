# Analysis — Run 67 · SigLIP2-B M=512 with covariance plus variance

## Scope and status

Run 67 is an intentionally stopped partial run, not a completed 15,000-step endpoint. It reached
step 11,000 without instability and retains a durable step-10,000 checkpoint. The complete run
contract, immutable identities, and stop reason are recorded in [`DESCRIPTION.md`](DESCRIPTION.md),
while the observed trajectory is recorded in [`OBSERVATIONS.md`](OBSERVATIONS.md).

## Partial interpretation

The geometry bundle clearly performed its intended function before the stop. Effective rank rose
from `33.23` at step 500 to `62.21` at step 11,000, cross-example cosine fell from `0.98148` to
`0.75727`, and mean code std reached `0.56499`, while centered slot rank remained high at `28.72`.
At the same time, fixed-batch reconstruction improved to `0.22494` and the shuffled-code penalty
grew to `0.05956`, so the representation was both increasingly diverse and increasingly used by
the decoder. Zero dead dimensions alone would be weak evidence, but here it agrees with rising
rank, rising std, falling cosine, and a widening dependence gap.

This partial run therefore rejects the narrow claim that variance/covariance regularization must
destroy SigLIP2 reconstruction. It does not answer whether those terms are necessary for the
architectural encoder-control question, because their gradients deliberately alter the bottleneck
geometry. Run 68 removes both weights while preserving the encoder, bottleneck, decoder, data,
seed, schedule, and reconstruction objective, which is the cleaner test the user intended.

## Conclusion

Run 67 was healthy and scientifically informative but answers a different question from the
intended unregularized standard-encoder control. Its partial evidence is retained as a regularized
reference; it must not be presented as a completed run or as the endpoint of the new comparison.
