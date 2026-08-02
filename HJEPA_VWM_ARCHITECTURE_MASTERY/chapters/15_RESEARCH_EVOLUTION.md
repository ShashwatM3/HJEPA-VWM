# 15 — Research evolution

This chapter explains why the current architecture contains apparently unusual guardrails. It is a
causal history, not a claim that every historical run used today's code.

## Phase 1's recurring separation

The program discovered three independent questions:

1. Can `B` produce a stable, broad, video-specific representation?
2. Does that representation actually change with time?
3. Can `F_c` predict the change better than copy and batch-mean baselines?

An intervention can improve one and harm another. Most of the research arc is learning not to
collapse them into “the loss went down.”

## Early data throughput

The initial data path decoded more video than required. Restricting Decord to the union of requested
context/target frames reduced an observed early step time from about 1.66 to 1.41 seconds. This
established a principle carried into the current loader: temporal sampling is not only statistical
semantics; it is also a throughput contract.

## Late training explosions

Early runs could look healthy for thousands of steps and then suffer a gradient spike near peak LR.
Responses accumulated:

- reduce Phase 1 from an overlong tiny-subset schedule to 15,000 steps;
- reduce warmup to 1,500 and lower peak learning rates;
- global clip at 0.5;
- tail skip threshold;
- explicit instability warning;
- module-specific AGC;
- exclude geometry and zero-init gates from AGC/weight decay;
- serialize exact state for forensics/resume.

Stability is now guarded rather than assumed, but a run must still pass the Q1 history check.

## Collapse and harder temporal windows

The shipped default `k=4` shares six of eight decoded frames. Experiments with `k=12` reduced exact
overlap to two frames. Raising the variance-floor weight helped prevent constant representations.

Lesson: a good prediction number on a heavily overlapping window can be a task shortcut. Temporal
offset and representation regularization must be read together.

## Slot collapse

Early learned queries began at tiny random scale, making cross-attention logits nearly uniform and
different slots read almost the same token average. The architecture adopted:

- orthogonal unit-row query initialization;
- learned sharpened cosine attention;
- explicit slot-rank and per-head entropy diagnostics;
- zero-initialized residual bridges.

A direct slot-diversity penalty improved its own target but did not reliably fix overall
representation/prediction. It became a flag-gated ablation, not the default solution. This is a
classic Goodhart lesson: slot metric improvement is not a full-system verdict.

## The rank-13 ceiling

Runs could achieve acceptable per-dimension standard deviation while using only roughly 10–15
effective feature directions. The one-sided variance floor prevents constant coordinates but does
not prevent correlations.

Responses explored:

- covariance decorrelation;
- SIGReg isotropic-Gaussian pressure;
- feature reconstruction;
- decoder width/depth;
- sharper slots;
- residual feature targets;
- a deeper Perceiver-style latent stack;
- frozen-feature whitening;
- internal bottleneck width;
- external slot-count/width grids.

The lesson is structural: variance, covariance rank, slot diversity, and content recoverability are
different axes.

## Reconstruction anchor and blindness

Present feature reconstruction stabilized some runs but did not by itself break the rank ceiling.
Worse, predicted `c_hat` and true `c⁺` could have nearly equal reconstruction loss even while the
copy gate failed. The decoder had learned much of a video-independent feature template.

Responses:

- fixed sinusoidal output positions rather than learned content queries;
- shuffled-video reconstruction readout;
- `L_recon_video_gap`;
- prediction-side reconstruction option;
- per-position feature-mean residual target.

Prediction-side reconstruction did not automatically repair copy ratio. This showed the problem was
not just missing gradient from `D` into `F_c`.

## SIGReg: rank without prediction

SIGReg strongly raised effective rank by pushing pooled `c` toward isotropic Gaussian geometry.
However, stronger geometry could worsen the predictor:

- online `B` moved rapidly;
- `B_EMA` lagged;
- much of the extra information could be static appearance;
- copy remained competitive.

The current code therefore:

- ramps SIGReg over 2,000 steps;
- logs both online and target rank/std;
- keeps it off by shipped default;
- never accepts rank as prediction success.

## Temporal residual prediction

Predicting:

```text
Δ = B_EMA(e_future)-B_EMA(e_present)
```

made temporal change explicit and scaled flow noise to its standard deviation. Copy became zero
residual. This could make `c` more dynamic, but the predictor often still tied zero residual.

Run 037 is the canonical clean negative:

```text
c_effective_rank ≈ 61
c_std_mean       ≈ 1.0
cross-video cos  ≈ 0.16
copy ratio       ≈ 1.06
```

Representation health passed; prediction failed. This single result motivates the separate
prediction gates better than any abstract explanation.

## Present-only autoencoder arc

Present-only runs removed `F_c` and target encoding to isolate whether the bottleneck/decoder could
carry content.

### Run 052

Absolute reconstruction became excellent while representation collapsed:

```text
L_recon ≈ 0.293
rank ≈ 13.4
cross-video cosine ≈ 0.906
std ≈ 0.295
```

The decoder exploited a video-independent template. Sharp attention was insufficient.

### Run 053

Subtracting the per-position feature mean improved honesty:

```text
video gap ≈ +0.433
~77% video-conditioned by the investigation's accounting
rank ≈ 10.5
```

It fixed the template shortcut but not geometry. Honesty and anti-collapse were independently
necessary.

### Runs 054–057

The deeper latent stack plus whitening improved video-conditioned reconstruction and geometry.
Whitening alone contributed much of the honesty; the residual target added further honesty without
geometric cost. Explicit covariance/variance produced strong present representations on SSv2
recipes, but this still did not constitute prediction.

## EGO transfer and diagnostic confounding

Run 058 transferred a strong SSv2-style recipe to EGO and looked stable but collapsed/aligned.
Run 060 with stronger reconstruction showed healthy recorded-batch geometry:

```text
rank 84.36
std 0.800
recorded cosine 0.479
video gap 0.02644
```

However, all 16 diagnostic chunks shared one source UID. Therefore global “cross-video” and
conditioning conclusions were not justified. The important lesson is epistemic: a metric
implementation can be correct while its sampled population makes its label misleading.

## Slot capacity sweep

Same-commit EGO runs compared `N_c=32,64,128`:

```text
late L_recon: 0.67703 / 0.67396 / 0.66298
late std:     0.806   / 0.326   / 0.649
late cosine:  0.473   / 0.914   / 0.677
```

The 32→128 reconstruction gain, 0.01405 or 2.08%, missed preregistered support thresholds, and larger
slot counts damaged recorded-batch geometry. The 256-slot extension was not launched.

Lesson: more abstract tokens are not automatically more usable capacity.

## Internal-width sweep

Raw/unwhitened EGO, present-only runs compared `M=512` with `M=1024`, holding `N_c=32,D_c=256`:

```text
late active recon: 0.260785 / 0.259879
fixed correct-code: 0.305063 / 0.303584
shuffled gap:       0.161563 / 0.163592
rank:              10.75 / 15.38
```

The much larger `M=1024` bottleneck improved loss by only 0.000906 and gap by 0.002029, below the
preregistered thresholds. `M=512` was selected as the practical width.

Parameter cost explains why this matters:

```text
B_VJEPA(M=512)  = 18.32M
B_VJEPA(M=1024) = 70.73M
ratio           = 3.86×
```

## Raw encoder controls

By 2026-07-21, repository records showed:

- raw V-JEPA and SigLIP codes stay more open with covariance plus variance;
- SigLIP2 and DINOv3 no-geometry controls learned positive correct-versus-shuffled gaps but
  contracted to low-rank, highly aligned codes;
- DINOv3 completed the full strict EGO path and validated its `(B,2048,768)` contract;
- explicit geometry pressure remained necessary across substrates.

These are repository-recorded experiment conclusions, not universal claims about the pretrained
encoders.

## Investigation 017 snapshot

As recorded on 2026-07-21, three encoder-specific sweeps were planned:

```text
N_c ∈ {16,32,64}
D_c ∈ {128,256,512}
M=512
three encoders
3*3*3 = 27 planned runs
```

Recipe: raw/unwhitened EGO, present-only absolute cosine reconstruction, three latent blocks,
decoder 512×4, `lambda_recon=1`, `lambda_var=0.5`, `lambda_cov=0.01`, no SIGReg/slot loss, seed 42,
batch 64, 15k steps.

The audited repository status called all 27 planned/unlaunched. This is a dated local-record
snapshot, not a live W&B assertion.

## Current consolidated knowledge

- Training stability is much improved and guarded.
- Frozen-feature representations are anisotropic and encoder-dependent.
- Bottleneck content dependence can be achieved.
- Explicit geometry pressure is needed across tested encoders.
- `M=512` is the practical selected internal width for current sweeps.
- More slots did not buy healthy capacity in the tested EGO sweep.
- Reconstruction can be honest yet low-rank.
- High-rank/video-specific `c` can still fail prediction.
- No repository-recorded full-prediction run has passed both copy and batch-mean gates.
- Source-diverse diagnostics are necessary for firm EGO conclusions.

The main unresolved Phase 1 question remains: can `c_t` be simultaneously broad, dynamic, and
forecastable enough that `F_c` beats copy by a decisive stable margin?
