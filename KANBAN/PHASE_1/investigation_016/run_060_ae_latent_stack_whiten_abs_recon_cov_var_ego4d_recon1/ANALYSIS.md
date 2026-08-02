# Run 060 — Complete Post-Run Analysis

**Run folder:** `run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1`  
**Investigation:** `investigation_016`  
**Analysis date:** 2026-07-16  
**W&B run:** [`2423b84g`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2423b84g)  
**W&B display name:** `Investigation 16 · Whitened latent stack EGO4D · Reconstruction weight 1.00`  
**Run status:** Finished, 15,000/15,000 training steps

---

## 1. Executive verdict

Run 060 is an operationally clean, present-only feature-autoencoding run. It produces much
healthier late-run latent geometry than historical EGO4D run 058, but increasing
`lambda_recon` from `0.05` to `1.0` does not materially lower the reconstruction floor or make
the learned latent strongly sample-specific.

The strongest concise verdict is:

> **Stable representation geometry, weak demonstrated content conditioning, and no evidence
> about future prediction. The hypothesis that EGO4D failed mainly because reconstruction was
> underweighted is not supported by the observed result.**

At the final diagnostic point, Run 060 has:

- `c_effective_rank = 84.36`;
- `c_std_mean = 0.800`;
- `c_cross_video_cosine = 0.479`;
- `L_recon_present = 0.670912`;
- `L_recon_shuffled_c = 0.697356`;
- `L_recon_video_gap = 0.026444`;
- an exact-chunk conditioned share of only `7.66%`;
- no NaNs, skipped steps, or numerical instability.

The apparent improvement relative to Run 058 cannot be causally assigned to the reconstruction
weight. Run 058 and Run 060 used different repository revisions and materially different
initialization, data-order, preprocessing, encoder-contract, and provenance machinery. Run 060
is therefore an intended configuration-weight arm on a newer runtime, not a byte-identical
one-variable ablation.

There is also a measurement-contract defect: every example in the fixed EGO4D validation batch
comes from the same source UID. Consequently, the metrics named `cross_video_cosine` and
`L_recon_video_gap` measure discrimination between adjacent chunks from one source rather than
between genuinely unrelated videos. Run 060 cannot establish or refute global cross-video
conditioning.

This checkpoint should not be transferred into the predictive Phase 1 objective until the
source-aware validation contract is repaired and the reconstruction shortcut is characterized
with explicit template and overfit oracles.

---

## 2. Scope and evidence used

This analysis reconciles four evidence layers:

1. the current repository implementation and tests;
2. the architectural and behavioral contracts under `AGENT_FILES/` and `GUIDES/`;
3. the full KANBAN research progression through investigation 016;
4. the actual live W&B run record, including configuration, complete metric history, diagnostic
   history, logs, system metrics, artifacts, provenance, and checkpoint metadata.

The run analysis follows the project's eight-question reading cycle:

1. Was training stable?
2. Did the representation collapse?
3. Did effective rank remain healthy?
4. Was the latent static over time?
5. Did prediction beat copy and batch-mean baselines?
6. Did the decoder depend on the supplied sample latent?
7. Was reconstruction blind to sample identity?
8. What is the appropriate research verdict?

The current implementation test suite was run independently as a repository-integrity check:

```text
.venv/bin/pytest -q
145 passed, 3 expected legacy-checkpoint warnings
```

Those tests establish internal implementation consistency. They do not, by themselves, validate
the scientific meaning of the fixed validation manifest; that requires the source-aware audit
described below.

---

## 3. Correct run identity and documentation state

The latest actual run folder in the latest investigation is this Run 060 folder. The sibling
`reconstruction_floor_architecture_audit/` directory is an analysis/audit directory, not a run
folder.

The local Run 060 triad is stale at the time of this analysis:

- `DESCRIPTION.md` says the run is not launched;
- `OBSERVATIONS.md` and `NEXT_STEPS.md` remain launch-time placeholders;
- live W&B shows a completed 15,000-step execution.

For execution status and recorded results, the live W&B run is authoritative. The local planning
documents remain useful for recovering the intended hypothesis.

---

## 4. What the run was designed to test

### 4.1 Parent observation

Run 058 transferred the whitened, absolute-target, covariance-plus-variance present-only recipe
to EGO4D with `lambda_recon=0.05`. Its fixed-batch diagnostics showed weak latent geometry and a
small difference between decoding the correct code and a batch-rolled code.

The original Run 060 hypothesis was:

> The absolute-target EGO4D objective may be correct, but reconstruction may be too weak relative
> to the geometry terms. Raising `lambda_recon` twentyfold should force more video content through
> the bottleneck while preserving covariance/variance geometry.

### 4.2 Decision outcomes implied by the hypothesis

The intended decision table was:

| Observed result | Intended interpretation |
|---|---|
| Reconstruction improves, honesty gap opens, geometry stays healthy | Run 058 was primarily a relative-weight failure |
| Reconstruction improves, honesty remains weak | Stronger weight does not remove the decoder/template shortcut |
| Geometry collapses under the heavier reconstruction term | Reconstruction and geometry objectives are in conflict |
| Nothing materially changes | The floor is architectural, target-induced, data-induced, or optimization-limited rather than a scalar-weight problem |

The actual result most closely matches the second and fourth rows: geometry is healthier, but raw
reconstruction changes very little and conditioning remains weak.

---

## 5. End-to-end implementation path active in Run 060

### 5.1 Active path

Run 060 is not a full world-model training run. Its active computation is:

```text
EGO4D present clip x_t
    -> deterministic decode / crop / color jitter
    -> frozen V-JEPA2 encoder E
    -> detailed feature field e_t: (B, 1024, 1024)
    -> fixed offline ZCA whitener W
    -> whitened detailed field e'_t
    -> trainable bottleneck B
    -> abstract latent c_t: (B, 32, 256)
    -> trainable feature decoder D
    -> reconstructed detailed field e_hat_t: (B, 1024, 1024)
    -> cosine reconstruction against e'_t
```

The active trainable modules are the online bottleneck `B` and feature decoder `D`.

### 5.2 Inactive predictive path

Because `present_recon_only=true`:

- no future clip is encoded during the training step;
- the EMA target bottleneck does not supply a future target;
- the CoarseFlow prediction objective is skipped;
- `L_flow` is inactive;
- `prediction_active=0` is logged;
- horizon length and future sampling do not affect the learned objective;
- copy and batch-mean prediction gates are not scientifically applicable.

The CoarseFlow module still exists in the repository and optimizer/module infrastructure, but this
run provides no evidence about whether it can predict future latents.

### 5.3 Encoder contract

The frozen encoder is V-JEPA2 ViT-L/16 under a pinned revision and feature fingerprint. For an
eight-frame `256 x 256` clip, the current encoder boundary resolves a time-major
`4 x 16 x 16 = 1024` token field with feature dimension `1024`.

Important current contracts are:

- RGB normalization occurs exactly once inside the encoder seam;
- encoder parameters remain frozen and the module remains in evaluation mode;
- the encoder revision and feature fingerprint are part of the run identity;
- encoder output shape/order is represented by an explicit `EncoderSpec`;
- autocast behavior belongs to the encoder boundary rather than the dataset.

These contracts supersede older learning notes that describe normalization in the data loader or
an unpinned encoder interface.

### 5.4 Bottleneck contract

The bottleneck performs:

1. an immediate linear projection from encoder dimension `1024` to mixer dimension `256`;
2. per-temporal-slice spatial mixing with two ConvNeXt blocks;
3. learned memory-position tagging;
4. projection to key/value width `256`;
5. three Perceiver-style latent rounds over 32 orthogonal learned slot queries;
6. cross-attention read, slot self-attention/competition, and per-slot MLP refinement;
7. final layer normalization.

The cross-attention output, self-attention output, and MLP terminal projections are zero-initialized.
At initialization, the module therefore emits normalized slot identities that are identical for
every input. This explains the otherwise surprising step-zero state:

```text
c_std_mean              = 0
c_cross_video_cosine    = 1
c_effective_rank        ~= 31
```

The rank diagnostic pools all `batch x slot` vectors in the 256-dimensional feature space. The
32 fixed slot identities create approximately 31 centered rank dimensions even when there is no
sample-dependent information.

### 5.5 Decoder contract and template mechanism

The current `Decoder` is a feature-space reconstruction anchor, not the unimplemented Phase 3
pixel/VAE generator.

It uses fixed 3D sinusoidal lattice-position queries to cross-attend to the 32 latent slots and
expand them back to the detailed encoder field. There is no learned per-output-position content
table.

That restriction does **not** eliminate all template solutions. The 32 learned bottleneck slot
identities are distinct and nonzero even when identical across samples. Different fixed output
positions can attend to those slot values with different weights. The decoder can therefore
construct a position-dependent output template without receiving meaningful sample variation.

This mechanism is consistent with low reconstruction loss accompanied by a small
correct-code-versus-shuffled-code gap.

### 5.6 Whitening contract

Run 060 applies a fixed offline ZCA transform fitted from exactly 12,800 training clips with
`eps=1e-4`. The whitening payload fingerprint is recorded as part of run provenance.

Whitening prevents the decoder from winning merely by reconstructing dominant correlated
directions of the frozen encoder. It also makes the reconstruction target much harder: the target
energy is deliberately spread across nearly all encoder feature dimensions.

This interacts strongly with the immediate `1024 -> 256` bottleneck input projection and the
overall compression from:

```text
1024 detailed tokens x 1024 dimensions
to
32 latent slots x 256 dimensions
```

That is a 128:1 scalar compression before considering information redundancy. It does not prove
that the observed floor is a capacity limit, but it makes a capacity oracle necessary.

### 5.7 Active losses

The reconstruction loss is mean per-token cosine distance:

```text
L_recon = mean(1 - cosine(e_hat_t, e'_t))
```

It removes magnitude as an escape route and optimizes angular agreement with the frozen whitened
target.

The geometry terms are:

- a VICReg-style variance hinge on flattened sample latents;
- off-diagonal covariance suppression;
- no SIGReg term;
- no explicit slot-diversity loss.

After the reconstruction warmup is complete:

```text
L_total = 1.00 * L_recon + 0.50 * L_var + 0.01 * L_cov
```

The reconstruction multiplier ramps linearly for the first 2,000 steps. The global optimizer
warmup lasts 1,500 steps.

### 5.8 Optimization and update order

The training step performs:

1. deterministic per-step RNG setup;
2. present clip decode and frozen encoding;
3. whitening;
4. bottleneck and decoder forward passes;
5. active loss construction;
6. backward pass;
7. adaptive gradient clipping;
8. global norm clipping at `0.5`;
9. nonfinite/excessive-gradient skip check;
10. optimizer step when healthy;
11. EMA update of the target bottleneck only after a successful step.

No target encoder exists. The frozen encoder is shared, and only the bottleneck has an EMA copy.

---

## 6. Exact run configuration and provenance

### 6.1 Model and objective

| Field | Recorded value |
|---|---:|
| Dataset | Full EGO4D chunk dataset |
| Encoder | V-JEPA2 ViT-L/16 |
| Frozen encoder parameters | 325,971,328 |
| Detailed shape | `(B, 1024, 1024)` |
| Latent shape | `(B, 32, 256)` |
| Bottleneck ConvNeXt blocks | 2 |
| Bottleneck latent blocks | 3 |
| Decoder width | 512 |
| Decoder blocks | 4 |
| Decoder heads | 8 |
| Present-only | true |
| Residual reconstruction target | false |
| Feature whitening | true |
| Whitening epsilon | `1e-4` |
| Whitening sample count | 12,800 clips |
| Reconstruction mode | cosine |
| `lambda_recon` | `1.0` |
| `lambda_recon_pred` | `0.0` |
| `lambda_var` | `0.5` |
| `lambda_cov` | `0.01` |
| `lambda_sigreg` | `0.0` |
| `lambda_slot` | `0.0` |

### 6.2 Schedule and optimizer

| Field | Recorded value |
|---|---:|
| Steps | 15,000 |
| Batch size | 64 |
| Horizon | 12, operationally inactive |
| Temporal stride | 2 |
| Optimizer warmup | 1,500 steps |
| Reconstruction warmup | 2,000 steps |
| Bottleneck LR | `1e-4` |
| CoarseFlow LR | `1e-4`, inactive in present-only path |
| Decoder LR | `1e-4` |
| Bottleneck AGC | `0.2` |
| CoarseFlow AGC | `0.1` |
| Decoder AGC | `0.2` |
| Global gradient clip | `0.5` |
| Hard gradient skip threshold | `150` |
| Seed | 42 |

### 6.3 Data and identity

| Field | Recorded value |
|---|---|
| Train chunks | 151,426 |
| Validation chunks | 18,464 |
| Dataset fingerprint | `df36...` |
| Trainable initialization fingerprint | `e468...` |
| Whitening payload fingerprint | `fd00...` |
| Git commit | `a27cd84dd67783f7ee8e68bb68e7c1a38a309534` |
| Git worktree | dirty |
| Final checkpoint | `/workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d_recon1/phase1_step15000.pt` |
| Final checkpoint SHA-256 | `16dee2...` |

The recorded run used Python 3.11.10, PyTorch 2.4.1 with CUDA 12.4, Transformers 4.57.6, bf16
autocast, and scaled-dot-product attention.

---

## 7. Operational stability

### 7.1 Completion and timing

- W&B state: finished;
- runtime: 13,510.87 seconds, approximately 3 hours 45 minutes;
- completed training steps: 15,000;
- training-history rows: 300, logged every 50 steps;
- full diagnostic rows: 30, from step 0 through step 14,500.

### 7.2 Numerical health

- no NaN loss values;
- no nonfinite-gradient skips;
- no hard-threshold gradient skips;
- no logged AGC intervention events;
- maximum logged pre-clip gradient norm: approximately `5.024` at step 550;
- global clipping handled early large norms without destabilizing training.

### 7.3 Resource behavior

- device: NVIDIA A100-SXM4 80 GB;
- mean sampled GPU utilization: approximately `68.54%`;
- maximum sampled utilization: `100%`;
- allocated GPU memory: approximately `12.29 GB`;
- process RSS: approximately `2.47 GB` mean and `2.52 GB` maximum.

The run is therefore a valid optimization result rather than a crash, NaN, skipped-update, or
resource-failure artifact.

---

## 8. Metric trajectory

### 8.1 Selected diagnostic snapshots

| Step | Effective rank | Mean std | Cross-sample cosine | Present recon | Shuffled recon | Gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.997 | 0.0000 | 1.0000 | 1.01627 | 1.01627 | 0.00000 |
| 500 | 40.595 | 0.3527 | 0.8263 | 0.9930 | 0.9943 | 0.00123 |
| 1,000 | 43.247 | 0.2856 | 0.8950 | 0.8654 | 0.8729 | 0.00751 |
| 1,500 | 41.355 | 0.2535 | 0.9312 | 0.7901 | 0.8020 | 0.01192 |
| 2,000 | 43.124 | 0.2683 | 0.9279 | 0.7557 | 0.7704 | 0.01477 |
| 2,500 | 43.807 | 0.2983 | 0.9128 | 0.7273 | 0.7434 | 0.01607 |
| 3,000 | 81.546 | 0.6816 | 0.5856 | 0.7180 | 0.7359 | 0.01790 |
| 3,500 | 112.726 | 0.8465 | 0.4043 | 0.7024 | 0.7202 | 0.01784 |
| 5,000 | 83.258 | 0.7156 | 0.5982 | 0.6899 | 0.7112 | 0.02124 |
| 7,500 | 60.323 | 0.5765 | 0.7286 | 0.6774 | 0.7013 | 0.02384 |
| 10,000 | 110.361 | 0.9230 | 0.3450 | 0.6740 | 0.6995 | 0.02546 |
| 12,000 | 81.685 | 0.7787 | 0.5067 | 0.6715 | 0.6977 | 0.02617 |
| 14,500 | 84.363 | 0.8004 | 0.4790 | 0.6709 | 0.6974 | 0.02644 |

### 8.2 Late-run medians

For diagnostic points at or after step 12,000:

| Metric | Late median |
|---|---:|
| Effective rank | 85.055 |
| Mean std | 0.8059 |
| Cross-sample cosine | 0.4735 |
| Present reconstruction | 0.67117 |
| Shuffled-code reconstruction | 0.69749 |
| Reconstruction gap | 0.02633 |

The late medians agree with the final snapshot, so the final verdict is not based on one unusually
good or bad diagnostic point.

### 8.3 Geometry oscillation

The geometry has two major expansion waves:

1. near-collapse or scaffold-dominated geometry through step 2,500;
2. sharp opening around steps 3,000-3,500;
3. contraction through step 7,500;
4. a second opening near step 10,000;
5. settlement around rank 84-89 and mean std approximately 0.8.

Recorded extrema include:

- maximum effective rank: `112.73` at step 3,500;
- maximum mean std: `0.9230` at step 10,000;
- minimum cross-sample cosine: `0.3450` at step 10,000;
- minimum validation present reconstruction: approximately `0.67090` at step 14,000.

The geometry is not monotonically improving, but it does not undergo terminal collapse.

---

## 9. Final scalar objective composition

The final training-history row at step 14,950 records approximately:

```text
raw L_recon = 0.67418402
raw L_var   = 0.000341456
raw L_cov   = 0.27516437
total loss  = 0.67710638
```

After applying the configured weights:

```text
weighted reconstruction = 1.00 * 0.67418402 = 0.67418402
weighted variance       = 0.50 * 0.000341456 = 0.00017073
weighted covariance     = 0.01 * 0.27516437  = 0.00275164
```

Approximate scalar shares are:

| Objective | Weighted value | Share of total |
|---|---:|---:|
| Reconstruction | 0.674184 | 99.57% |
| Variance | 0.000171 | 0.03% |
| Covariance | 0.002752 | 0.41% |

Scalar loss shares are not gradient-norm shares, so they cannot prove which term dominates parameter
updates. They do prove that the reconstruction loss is not numerically hidden beneath the geometry
terms in the total objective.

---

## 10. Eight-question run reading

### 10.1 Was training stable?

**Verdict: yes.**

The run finished, remained finite, skipped no updates, and produced a valid checkpoint. There is no
operational failure that invalidates its scientific readout.

### 10.2 Did the representation collapse?

**Verdict: not in the classic dimensional sense at the end, but content collapse remains
possible.**

Evidence against terminal dimensional collapse:

- rank `84.36`, above the project's historical health threshold of 60;
- mean and median std approximately `0.8`;
- zero dead dimensions;
- cross-sample cosine below `0.5` at the last snapshot;
- centered slot rank approximately `30.81/32`.

Counter-evidence:

- part of pooled effective rank is supplied mechanically by fixed slot identities;
- covariance/variance pressure can create geometry without encoding sample content;
- the shuffled-code reconstruction gap remains small;
- the validation batch is not source-diverse.

The correct label is therefore **geometrically noncollapsed but weakly conditioned**, rather than a
simple pass/fail collapse label.

### 10.3 Did effective rank remain healthy?

**Verdict: conditionally yes.**

Late rank is healthy by the established threshold and well above Run 058. It remains below the
frozen V-JEPA representation's available rank budget measured in investigation 014. The current
rank diagnostic also mixes within-video slot rank with between-sample variation, so it must be read
alongside batch-level standard deviation, cosine, and the reconstruction honesty test.

### 10.4 Was the latent static over time?

**Verdict: not measurable in this run.**

Only present clips are encoded for the objective. There is no paired future latent trajectory on
which to evaluate temporal change or static-`c` behavior.

### 10.5 Did prediction beat copy and batch-mean baselines?

**Verdict: not applicable.**

CoarseFlow is inactive. `prediction_active=0`. No claim about prediction quality, copy beating,
batch-mean beating, or temporal world modeling is licensed by Run 060.

### 10.6 Did reconstruction depend on the supplied sample latent?

**Verdict: weakly, under the current within-source probe.**

At the final diagnostic:

```text
L_recon_present    = 0.6709120
L_recon_shuffled_c = 0.6973557
gap                = 0.0264437
```

The correct code helps, but only slightly.

### 10.7 Was reconstruction blind to sample identity?

**Verdict: mostly blind to exact chunk identity; global video identity remains unmeasured.**

The initial untrained diagnostic loss was `1.016265`. The final correct-code loss was `0.670912`:

```text
learned improvement = 1.016265 - 0.670912
                    = 0.345353
```

The exact-code advantage is `0.026444`, yielding:

```text
conditioned share = 0.026444 / 0.345353
                  = 0.07657
                  = 7.66%
```

Equivalently, approximately 92.34% of the learned validation improvement remains when the code is
rolled to another chunk from the fixed batch.

Because every batch element is from the same source video, this is an exact-chunk conditioned share,
not a global cross-video conditioned share.

### 10.8 Overall verdict

**Stable present representation with healthy geometry but weak exact-chunk conditioning.**

It is useful as evidence about the reconstruction-weight extreme. It is not a successful predictive
checkpoint and not a clean causal weight ablation.

---

## 11. Reconstruction interpretation

### 11.1 What a loss near 0.67 means

For cosine reconstruction:

```text
L_recon = 1 - mean cosine similarity
```

Therefore the final diagnostic values correspond approximately to:

```text
correct-code mean cosine similarity  = 1 - 0.670912 = 0.329088
rolled-code mean cosine similarity   = 1 - 0.697356 = 0.302644
exact-code similarity advantage      = 0.026444
```

The decoder has learned a substantial generic alignment with the target field, but the exact sample
latent contributes relatively little additional angular alignment.

### 11.2 Reconstruction floor

The validation loss approaches approximately `0.671` and changes little after roughly step 10,000.
The random-training-batch reconstruction loss also plateaus near `0.67`, so the floor is not solely
an artifact of the fixed one-source validation batch.

Possible nonexclusive causes are:

1. the full-rank difficulty introduced by ZCA whitening;
2. the immediate `1024 -> 256` projection before latent attention;
3. the `1024 x 1024 -> 32 x 256` compression ratio;
4. a decoder/template shortcut that captures position-dependent average structure;
5. the angular objective's insensitivity to recoverable magnitude information;
6. optimization or schedule limitations;
7. insufficient decoder or bottleneck capacity;
8. EGO4D-specific diversity relative to prior SSv2 experiments.

Run 060 distinguishes none of these cleanly. A cached-feature overfit ladder and reconstruction
oracles are required.

---

## 12. Source-manifest defect and its consequences

### 12.1 Recorded validation examples

The fixed validation batch consists of the lexically first 16 validation chunks:

```text
validation/01cab463-9a16-4817-84a4-a00ef5b7bf39_00000.mp4
...
validation/01cab463-9a16-4817-84a4-a00ef5b7bf39_00015.mp4
```

Every chunk has source UID:

```text
01cab463-9a16-4817-84a4-a00ef5b7bf39
```

These are adjacent non-overlapping chunks of one original EGO4D recording.

### 12.2 Why `torch.roll` is insufficient here

The honesty diagnostic rolls `abstract` by one batch position while preserving each target. With a
source-diverse manifest, that produces a different-video code/target pairing. In this manifest, it
produces a neighboring-chunk pairing from the same source.

Consequences:

- `L_recon_video_gap` is misnamed for this run;
- `c_cross_video_cosine` is also misnamed for this fixed batch;
- the current gap may be low because neighboring chunks share scene, wearer, environment, and
  persistent visual statistics;
- a global source template could perform better or worse than the observed rolled-code result;
- the previous categorical phrase “global template collapse” is not established by this batch.

### 12.3 What remains valid

The following conclusions survive the manifest defect:

- training is stable;
- random training-batch reconstruction plateaus near the same level;
- within-source exact-chunk conditioning is weak;
- the representation geometry on these 16 chunks is much healthier than Run 058;
- no future-prediction conclusion is available.

### 12.4 Required repair

The fixed validation batch should be rebuilt as follows:

1. parse the source UID from each EGO4D chunk path;
2. choose one deterministic validation chunk per unique source UID;
3. assert that all selected source UIDs are unique;
4. construct a derangement where every code/target pair has different source UID;
5. store both chunk paths and parsed UIDs in provenance;
6. fail fast if source uniqueness is violated.

---

## 13. Comparison with Run 058

### 13.1 Live run identities

| Run | W&B ID | Reconstruction weight | Status |
|---|---|---:|---|
| Run 058 | [`mvbx96nv`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/mvbx96nv) | 0.05 | Finished |
| Run 060 | [`2423b84g`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2423b84g) | 1.00 | Finished |

### 13.2 Final diagnostic comparison

| Metric | Run 058 | Run 060 | Absolute change |
|---|---:|---:|---:|
| Present reconstruction | 0.678246 | 0.670912 | -0.007334 |
| Shuffled reconstruction | 0.696221 | 0.697356 | +0.001135 |
| Reconstruction gap | 0.017975 | 0.026444 | +0.008469 |
| Exact-chunk conditioned share | 5.36% | 7.66% | +2.30 percentage points |
| Effective rank | 52.910 | 84.363 | +31.453 |
| Mean std | 0.419 | 0.800 | +0.381 |
| Cross-sample cosine | 0.863 | 0.479 | -0.384 |
| Raw slot rank | 31.074 | 29.966 | -1.108 |
| Centered slot rank | 30.669 | 30.814 | +0.145 |
| Mean attention entropy | 0.549 | 0.844 | +0.295 |
| Minimum attention entropy | 0.0027 | 0.2830 | +0.2803 |

### 13.3 What improved

- pooled latent rank increased by approximately 59%;
- mean standard deviation nearly doubled;
- flattened sample cosine fell from near-identical `0.863` to borderline-healthy `0.479`;
- the exact-chunk reconstruction gap increased by about 47% relative;
- the representation no longer ends in the same obvious geometry contraction seen in Run 058.

### 13.4 What did not improve enough

- present reconstruction improved only `0.00733`, approximately 1.08% relative;
- the shuffled reconstruction loss became slightly worse rather than exhibiting a large separation;
- conditioned share increased only from 5.36% to 7.66%;
- more than 92% of the learned improvement remains under a rolled within-source code;
- no future prediction was activated or evaluated.

### 13.5 Objective-share comparison

At the final Run 058 training row:

```text
0.05 * L_recon ~= 0.034845
0.50 * L_var   ~= 0.000270
0.01 * L_cov   ~= 0.002190
total          ~= 0.037304
```

The reconstruction term already supplied approximately 93.4% of Run 058's scalar loss. Run 060
increases this to approximately 99.6%, yet raw reconstruction changes only slightly.

This is evidence against a simple scalar-loss-underweighting account. It does not rule out gradient
competition, because scalar values do not measure per-module gradient magnitude or direction.

---

## 14. Why the Run 058 comparison is causally confounded

The original local plan labels Run 060 a byte-identical control except for `lambda_recon`. The
recorded executions do not satisfy that claim.

Run revisions:

```text
Run 058: 21d2aa8...
Run 060: a27cd84dd67783f7ee8e68bb68e7c1a38a309534
```

There are six intervening commits with changes across 103 files. Scientifically relevant runtime
changes include:

### 14.1 Initialization

Current code isolates trainable-module initialization after encoder loading and uses a dedicated
seeded initialization boundary. Run 058 did not record the same strict trainable-init identity.

### 14.2 Data order and augmentation randomness

Current code uses:

- an explicit seeded `torch.randperm` for epoch order;
- sample/epoch/transform-version-hashed RNG for crop and color jitter;
- exact sampler position in checkpoints.

Historical Run 058 relied on older shuffle and worker/global RNG behavior.

### 14.3 Encoder boundary

Current code adds:

- a pinned V-JEPA revision;
- feature fingerprint verification;
- normalization exactly once inside the encoder seam;
- explicit bf16/autocast ownership;
- strict output geometry and ordering contracts.

### 14.4 Present-only execution

Current code skips target-side work in present-only mode and derives per-step RNG explicitly. Even
if high-level inputs are intended to match, the runtime execution is not historical byte identity.

### 14.5 Provenance and whitening identity

Current runs bind dataset fingerprints, initialization fingerprints, validation manifests, encoder
fingerprints, and whitening payloads into provenance and resume validation. Run 058 predates part of
this machinery.

### 14.6 Causal conclusion

The Run 060 versus Run 058 difference is an observational comparison between two intended recipes,
not a controlled one-variable experiment. To isolate the reconstruction weight, a same-commit
`0.05` companion must reuse:

- the exact trainable initialization fingerprint;
- the exact training order fingerprint;
- the exact whitening payload;
- the exact source-aware validation manifest;
- the same encoder fingerprint and preprocessing version;
- the same runtime and optimizer implementation.

---

## 15. Relationship to the research progression

Run 060 sits at the end of a long Phase 1 sequence:

1. Early investigations established basic execution and revealed low-rank collapse.
2. Strong variance pressure produced slot spread without reliable information content.
3. AGC and clipping repaired late-run numerical failures.
4. Feature reconstruction stabilized the bottleneck but developed a persistent reconstruction
   floor and weak predictive relevance.
5. Larger decoder/reconstruction changes showed that some reconstruction configurations were inert
   or template-dominated.
6. SIGReg raised rank but often reduced temporal usefulness.
7. Residual flow targets made the latent more dynamic without beating copy baselines.
8. Cosine reconstruction lowered SSv2 reconstruction loss, but present-only experiments exposed
   the distinction between geometric health and decoder conditioning.
9. Runs 052-053 demonstrated that absolute reconstruction can exploit position templates and that
   residual targets improve honesty without automatically preserving geometry.
10. Frozen-encoder rank probes established that V-JEPA itself has much more rank than the learned
    bottleneck uses.
11. Runs 054-057 combined whitening, deeper latent processing, covariance, variance, and optional
    SIGReg. Run 057 was the strongest present-only SSv2 checkpoint, with high rank and a much larger
    conditioned share.
12. Run 058 transferred the recipe to EGO4D and appeared to fail, but its global-collapse label was
    later weakened by discovery of the single-source validation batch.
13. Run 060 tested the extreme reconstruction-weight arm on the newer runtime.

The current research state is therefore not “the representation problem is solved.” It is:

> The code can train stable, high-rank present latents, but high rank does not guarantee sample
> information. The next bottleneck is a combination of measurement validity, reconstruction
> shortcut characterization, and a capacity/optimization oracle—not another uncalibrated scalar
> weight sweep.

---

## 16. Hypothesis accounting

| Hypothesis or claim | Run 060 status | Reason |
|---|---|---|
| Run 058 failed mainly because reconstruction weight was too small | **Not supported** | 20x weight yields only ~1.08% relative raw reconstruction improvement and conditioned share stays <8% |
| Stronger reconstruction can coexist with covariance/variance geometry | **Supported observationally** | Late rank ~85, std ~0.8, no dead dimensions |
| Stronger reconstruction opens sample conditioning | **Weakly supported at most** | Gap rises from 0.018 to 0.026, still small and within-source only |
| The learned latent is globally video-specific | **Unanswered** | Fixed validation batch contains one source UID |
| Run 060 beats copy or batch-mean future prediction | **Not tested** | Present-only mode disables prediction |
| The ~0.67 floor is a hard architectural capacity limit | **Unanswered** | No cached-feature overfit/capacity oracle was run |
| Absolute-target decoding uses a position template | **Plausible, not proven** | Small code gap and architecture permit it; explicit zero/mean/B-zero baselines are missing |
| Residual targets are preferable on EGO4D | **Unanswered by Run 060** | Residual mode is disabled; the queued arm remains necessary after measurement repair |

---

## 17. Required next experiments

The order matters. Measurement repair should precede another expensive architecture run.

### 17.1 Repair the source-aware validation contract

Implement and enforce:

- one deterministic chunk per source UID;
- unique-source assertion;
- cross-source derangement for shuffled-code reconstruction;
- provenance recording of source UIDs;
- source-aware names for the resulting metrics.

Re-evaluate Run 060's checkpoint on this manifest if the checkpoint and matching whitening/runtime
environment are available. This can answer the global-conditioning question without retraining.

### 17.2 Add template baselines

Measure reconstruction from:

1. the correct `c_t`;
2. a cross-source shuffled `c_t`;
3. a zero latent;
4. a dataset-mean latent;
5. a bottleneck output for zeroed/dummy detailed input;
6. a constant per-position predictor;
7. a low-rank/PCA predictor.

These baselines separate:

- generic per-position encoder structure;
- source-level persistent information;
- exact-clip information;
- true bottleneck-conditioned reconstruction.

### 17.3 Run a cached-feature fixed-batch overfit ladder

Use a small deterministic feature cache and attempt to overfit progressively:

1. decoder only on fixed true latents;
2. bottleneck plus decoder on a tiny batch;
3. wider bottleneck input projection;
4. more latent slots;
5. wider latent dimension;
6. deeper/wider decoder.

Interpretation:

- if the current architecture cannot overfit a tiny fixed batch, the floor is architectural or
  optimization-related;
- if it can overfit but full training plateaus, the problem is schedule, generalization, target
  difficulty, or shortcut competition;
- if only wider input/latent capacity overfits, compression is the binding constraint.

### 17.4 Re-run the actual reconstruction-weight control

On one current commit, launch paired `lambda_recon=0.05` and `1.0` runs with:

- identical trainable initialization;
- identical sample ordering;
- identical augmentation identities;
- identical whitening payload;
- identical source-diverse validation batch;
- identical diagnostics and logging cadence.

Only this pair can estimate the causal effect of reconstruction weight.

### 17.5 Choose the next architectural arm from oracle results

Candidate arms, in evidence-driven order:

1. residual reconstruction target, after source-aware metrics exist;
2. bottleneck capacity increase if the overfit ladder implicates compression;
3. whitening-strength/eigenvalue-floor ablation if target conditioning is the issue;
4. optimizer or reconstruction schedule changes if the architecture can overfit but training cannot;
5. SigLIP2 comparison only with a same-commit V-JEPA control.

### 17.6 Do not activate prediction yet

The checkpoint should not be used as the basis for judging CoarseFlow or transferred automatically
into future prediction. First require:

- source-aware conditioning evidence;
- a materially larger correct-versus-cross-source reconstruction gap;
- stable geometry under that measurement;
- explicit prediction baselines once prediction is activated.

---

## 18. Final decision

### 18.1 What Run 060 succeeded at

- completed a long EGO4D run stably;
- demonstrated that a weight-1 reconstruction objective can coexist with covariance/variance
  geometry;
- ended above the historical rank threshold;
- eliminated terminal dead dimensions;
- improved geometry substantially relative to historical Run 058;
- produced strict provenance and a valid final checkpoint;
- supplied evidence that scalar reconstruction underweighting alone is not the main explanation.

### 18.2 What Run 060 did not achieve

- did not materially lower the whitened reconstruction floor;
- did not demonstrate strong exact-chunk conditioning;
- did not measure cross-source/global conditioning;
- did not constitute a controlled comparison with Run 058;
- did not train or evaluate future prediction;
- did not establish that its rank corresponds to useful temporal or semantic information;
- did not identify whether capacity, whitening, target structure, template shortcuts, or optimization
  is the dominant limiting factor.

### 18.3 Final label

> **Run 060: stable, geometrically healthy present-only autoencoder; weak-conditioned
> reconstruction; source-confounded diagnostics; not prediction-ready.**

The appropriate project action is to repair measurement and execute the oracle ladder before
committing further compute to another broad architectural or loss-weight sweep.

---

## 19. Key implementation references

- Present-only training branch and active losses: `train.py`
- Fixed-batch present-only diagnostics and `torch.roll` honesty probe: `train.py`
- Bottleneck, fixed slot initialization, decoder, and whitener: `models.py`
- Cosine reconstruction, variance, covariance, SIGReg, and slot losses: `losses.py`
- Rank, variance, cosine, slot-rank, attention, and prediction baselines: `diagnostics.py`
- Deterministic transforms and epoch ordering: `data.py`
- Encoder revision/fingerprint and resolved feature geometry: `encoders.py`
- Dataset, initialization, whitening, validation-manifest, and checkpoint identity: `provenance.py`
- Experiment reading protocol: `GUIDES/READING_EXPERIMENTS.md`
- Metrics/problem map: `GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`
- Source-contract correction: `../NEXT_STEPS.md`
- Reconstruction-floor audit: `../reconstruction_floor_architecture_audit/ANALYSIS.md`

---

## 20. Data-source note

The numerical results above were read from the actual W&B backend rather than copied from the stale
Run 060 KANBAN placeholders. The inspected live surfaces included run metadata, resolved config,
summary, complete history, diagnostic history, logs, system samples, artifact inventory, provenance,
dataset identity, validation paths, and final checkpoint metadata.

