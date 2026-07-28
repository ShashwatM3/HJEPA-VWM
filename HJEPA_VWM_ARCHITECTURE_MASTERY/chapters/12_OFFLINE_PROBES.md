# 12 — Offline probes

Training diagnostics answer “what is the live model doing?” Offline probes answer narrower questions
under fixed manifests, shared feature caches, and checkpoint-controlled comparisons.

## `rank_probe.py`

Purpose: characterize the frozen detailed-feature geometry before blaming the bottleneck.

Pipeline:

1. select a deterministic validation probe set;
2. encode anchor windows or load a verified cache;
3. pool detailed tokens;
4. form covariance eigenvalue spectra;
5. report entropy effective rank and energy-rank views;
6. optionally save a spectrum PNG and W&B artifact.

Supported encoders/data match the training surface. Default manifest offset coverage is aligned with
the drift probe so running rank first does not create an incompatible cache.

Important distinction:

- pooled token effective rank asks about feature-channel covariance over all tokens/videos;
- pre-concatenation frame views preserve frame-encoder structure;
- a V-JEPA tubelet lattice must not be mislabeled as eight independent frame features.

Rank is descriptive. It does not prove downstream compressibility, semantic factors, or temporal
forecastability.

## `drift_probe.py`

Purpose: measure how representations change as the same source is shifted in time.

The manifest binds anchor windows plus a ladder of offsets measured in original decoded-frame
indices. For each video/offset, the probe can compare:

- frozen detailed feature distance;
- online bottleneck latent distance from one or more checkpoints;
- EMA bottleneck latent distance with `--use-ema`.

Default Graph 2 aggregates non-overlapping offsets, defined as `k>14`, to avoid mistaking literal
shared frames for temporal persistence.

It reports curves and per-video relationships, including rank/correlation summaries. A useful
question is whether latent drift tracks frozen-feature drift or suppresses it into a static code.

## Shared feature cache

Rank and drift deliberately derive the same default cache path. The envelope binds:

- encoder fingerprint;
- dataset fingerprint;
- probe-manifest fingerprint and covered offsets;
- exact tensor shapes/dtype;
- cache payload hash.

FP16 reduces space. `--exact-latent-comparison` requires FP32 so small checkpoint differences are not
confounded by cache quantization.

Shape-matched caches from different encoders are rejected. All tensors must be finite.

## `whiten_stats.py`

Purpose: fit the fixed training-set feature coordinate transform.

Operational modes:

- fit from selected dataset/split/encoder with exact clip budget;
- inspect an existing artifact without training;
- optionally log artifact metadata to W&B.

The fit uses deterministic context-only samples, not context+target pairs. Clip count and total token
rows are audited separately:

```text
token rows = clips * N_e
```

The exact shipped expected budget is 12,800 clips when whitening is enabled. Streaming FP64
sum/outer-product accumulation avoids holding the whole corpus tensor in memory.

## `run_history.py`

Purpose: download unsampled W&B history using the public API's `scan_history`.

Capabilities:

- run ID or full entity/project/run path;
- metadata/config/summary only;
- full JSON history;
- `parse_logs`-compatible text;
- key filtering and step ranges;
- a Phase 1 acceptance report.

Unsampled history matters because late gradient spikes or brief skip events can disappear from
downsampled dashboard views.

## `parse_logs.py`

Purpose: turn console lines of the form:

```text
step=N {metric_dict}
```

into structured JSON. It supports local forensics when W&B is unavailable or when raw logs are the
only surviving evidence.

## Probe identities are joins

An offline result is valid only when the following join:

```text
encoder fingerprint
× dataset fingerprint
× probe manifest
× transform version
× checkpoint provenance
× whitening state
× probe configuration
```

“Same checkpoint filename” is insufficient because identical names can exist in different run
directories or recovered volumes.

## Online versus offline measurements

| Question | Best source |
|---|---|
| Did gradients explode at step 8,450? | unsampled W&B/log history |
| Is fixed-batch `c` rank healthy during training? | online diagnostics |
| Is frozen encoder feature space anisotropic? | rank probe |
| Does latent temporal drift follow encoder drift? | drift probe |
| Are whitening stats exact and compatible? | stats artifact inspection |
| Did the run beat copy over a stable late window? | online diagnostic history |

An offline probe should not overwrite the historical run verdict; it adds a controlled explanatory
measurement.

## Caching and paid-compute discipline

- Build the probe manifest once and reuse it across arms.
- Cache frozen features because encoder forward is the expensive shared operation.
- Verify cache identity before launching checkpoint sweeps.
- Use unique output tags/directories.
- Treat W&B artifact upload as a copy, not the only local result.
- Record exact command/config in the investigation record.

## Interpretation traps

- Effective rank of `e` and effective rank of `c` use different dimensions and populations.
- A high frozen-feature rank does not mean `B` can preserve it at 8,192 abstract scalars.
- Small drift at overlapping offsets is expected; inspect non-overlap offsets.
- EMA and online bottlenecks can differ materially; label which one was probed.
- FP16 feature caches are fine for broad curves but not exact checkpoint-delta claims.
- Correlation between encoder and latent drift is not causal proof.
- Probe-set source diversity matters as much as it does for online fixed-batch diagnostics.
