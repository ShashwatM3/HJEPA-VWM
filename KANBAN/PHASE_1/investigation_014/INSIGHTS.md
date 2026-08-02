# Insights - V-JEPA Embedding Effective Rank

## Core Finding

On the fixed 64-video SSv2 validation probe set, frozen V-JEPA `e` embeddings have an entropy
effective rank of:

```text
e_effective_rank = 192.7 / 1024
```

This is the pooled-token measurement: every anchor-window token from every probe video is stacked
into one `(64 * 1024, 1024)` matrix, centered, converted to a covariance matrix, and ranked with the
same entropy-of-eigenvalues mechanism used for `c_effective_rank`.

The result says V-JEPA's embedding space is not using all 1024 feature dimensions equally. It behaves
more like a few hundred meaningful covariance directions plus a long weak tail.

## Full Readout

```text
e effective ranks over 64 probe videos (anchor windows, N_ctx=1024, D_e=1024):
  pooled tokens (c_effective_rank analog): 192.7 / 1024
  within-video tokens: mean 75.7 (min 58.8, max 94.1) / 1023
  cross-video pooled vectors: 23.8 / 63
  pooled rank@90%/99% energy: 333 / 785
```

Source artifact:

```text
logs/drift_probe/rank_results_ssv2_validation_n64_seed42.json
```

## Interpretation

The encoder representation is already compressed and anisotropic before our model touches it.
Although the raw V-JEPA feature dimension is `1024`, the effective dimensionality of the pooled token
cloud is about `193`.

The `rank@90%` and `rank@99%` values show why this should not be read as a hard cutoff. You need
`333` principal directions to capture 90% of the variance and `785` directions to capture 99%. That
means the spectrum has a long tail: lots of directions exist, but many are weak. Those weak
directions may include useful fine detail, but they may also include nuisance variation, redundancy,
or noise.

The within-video value, `75.7`, says each individual clip's 1024 tokens have real internal
spatial-temporal structure. The cross-video value, `23.8`, says whole-video mean embeddings vary in
a much lower-rank subspace than the full token field.

## Why This Changes How We Think About `e -> c`

The bottleneck is not compressing a clean full-rank `1024`-dimensional signal. It is compressing an
encoder representation that already has:

- concentrated high-energy directions;
- a moderate entropy rank around `193`;
- a broad low-energy tail;
- token-level structure that is richer than video-level mean structure.

So the abstract latent `c_t` is not just "smaller than V-JEPA." It is a second compression step on
top of an already uneven representation. Every training objective that maps `e -> c` must decide
which directions survive.

That makes the effective-rank drop from `e` to `c` scientifically meaningful. Some drop is expected
and may be desirable if the bottleneck removes noisy or irrelevant encoder directions. But too much
drop means `c_t` is throwing away information that V-JEPA made available.

## Practical Takeaway

Use the `e` rank readout as a rank budget, not as a target to copy blindly.

The goal is not to force `c_t` to reproduce every weak V-JEPA direction. The goal is to preserve the
subset of encoder-side dimensions that carry useful state for prediction while discarding nuisance
or noisy tail directions.

This makes future bottleneck design sharper:

- low `c_t` rank is not automatically acceptable just because `c_t` is abstract;
- high `c_t` rank is not automatically good if it spends capacity on weak/noisy `e` directions;
- reconstruction losses should be treated carefully, because reconstructing `e` may pressure `c_t`
  to preserve encoder-tail variance that may not help future prediction;
- rank diagnostics should be read as information-utilization evidence, not as standalone success.

## Open Hypothesis

The bottleneck should behave like a selective denoising compressor:

```text
frozen V-JEPA e: moderate-rank useful substrate + long weak tail
        ↓
trainable bottleneck B
        ↓
abstract c_t: lower-rank state that keeps predictive factors, not every encoder detail
```

The central research problem is now clearer: identify objectives that preserve predictive
encoder-side structure while not wasting the abstract latent on V-JEPA's low-energy tail.
