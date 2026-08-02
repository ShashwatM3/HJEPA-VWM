# Observations

## 2026-07-04 - First V-JEPA `e` Effective-Rank Measurement

The offline rank probe produced:

```text
e effective ranks over 64 probe videos (anchor windows, N_ctx=1024, D_e=1024):
  pooled tokens (c_effective_rank analog): 192.7 / 1024
  within-video tokens: mean 75.7 (min 58.8, max 94.1) / 1023
  cross-video pooled vectors: 23.8 / 63
  pooled rank@90%/99% energy: 333 / 785
```

### What The Numbers Say About `e`

The frozen V-JEPA token cloud is moderately high-rank, but very far from isotropic full-rank usage.
Across all probe videos and all anchor-window tokens, the entropy effective rank is about `192.7`
out of a possible `1024`.

The important point is not that V-JEPA uses exactly 193 dimensions. Effective rank is a spectrum
summary: it says the centered covariance behaves like roughly 193 equally weighted directions. The
actual spectrum is uneven.

The energy ranks make that unevenness visible:

- `rank@90% energy = 333`
- `rank@99% energy = 785`

So the representation has a long low-energy tail. Many dimensions carry some variance, but much of
that variance is weak. This is consistent with a representation that contains useful factors mixed
with redundancy, nuisance variation, and low-energy/noisy directions.

### Within-Video Structure

Within one video, the `1024` tubelet tokens span about `75.7` effective directions on average, with
the measured probe range from `58.8` to `94.1`.

This says an individual clip is not a single static vector. Its spatial-temporal token field has
real internal structure. But the internal structure is still much lower-dimensional than the raw
`1024 x 1024` token matrix might imply.

### Cross-Video Structure

After mean-pooling each video's tokens into one vector, the 64 video-level embeddings have effective
rank `23.8` out of a maximum `63`.

This says whole-clip identity/action variation is substantially lower-rank than the full pooled token
cloud. A lot of the pooled-token rank comes from within-video token structure, not just from
separating 64 different videos.

## Interpretation For The Bottleneck

The bottleneck `B` is not merely reducing a clean 1024-dimensional signal into `c_t`. It is forced
to choose what to keep from an encoder representation that is already compressed, anisotropic, and
long-tailed.

Under the current `c_effective_rank` diagnostic, `c_t` has a feature-rank ceiling of `D_c = 256`.
That ceiling is close to the measured entropy-rank scale of the frozen encoder token cloud
(`192.7`), but it is below the encoder's `rank@90%` energy count (`333`) and far below its
`rank@99%` count (`785`).

The practical implication is that `c_t` cannot preserve all covariance directions present in `e`
under this diagnostic. It must discard, merge, or reweight encoder-side directions. Some of that
discarding is desirable if it removes nuisance/noise directions. But if `c_t` lands far below the
encoder-side rank budget, then the bottleneck/training objective is likely destroying usable
information rather than merely denoising.

## Working Belief

The V-JEPA `e` embeddings contain a moderate-rank useful substrate plus a long weak tail. Phase 1
should treat the bottleneck as an information-selection problem, not just a dimension-reduction
problem:

- preserve enough rank to retain meaningful video/token factors;
- avoid wasting capacity reconstructing weak encoder-side tail directions;
- avoid collapse where `c_t` keeps only a small fraction of the encoder-side rank budget.

This measurement makes the encoder-side baseline concrete. Future bottleneck objectives should be
judged against the fact that the frozen input representation itself has an entropy rank around
`193`, with much broader low-energy variance behind it.
