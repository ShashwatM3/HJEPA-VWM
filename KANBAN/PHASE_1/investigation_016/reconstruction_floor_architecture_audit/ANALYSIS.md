# End-to-end reconstruction-floor architecture audit

> Audit completed 2026-07-16. Run histories were read with the present-only branch of
> the repository's Reading Cycle B. Code was traced at the historical run commits and
> at current HEAD. Facts separated from inference live in
> [`METRIC_READOUT.md`](METRIC_READOUT.md).

## Run map — what every numbered run in this analysis was testing

### Run 041 — fixed-position present reconstruction (`hcr2qx19`)

An earlier SSv2 present-only, raw-feature experiment that introduced the fixed-position
decoder while keeping active geometry pressure. It reached reconstruction loss `~0.345`
with rank `~49.6`, so it was decodable but still low-rank; the run stopped externally
near the end and is used here only as historical loss-scale context.

### Run 052 — sharpened-slot absolute reconstruction (`662hfy3c`)

An SSv2 present-only test asking whether the sharpened-slot architecture could remain
healthy using raw absolute reconstruction alone, with all geometry regularizers off.
It reconstructed extremely well (`~0.293`) but collapsed to rank `~13.4` and mostly
exploited shared/template structure, showing that low reconstruction is not sufficient.

### Run 053 — sharpened-slot residual reconstruction (`7teohhwc`)

The direct follow-up to run 052 replaced the raw absolute target with a per-position
residual target so a shared template no longer earned the easy reconstruction gain.
Reconstruction became genuinely code-conditioned, but rank still contracted to `~10.5`;
the externally killed run established that honest reconstruction alone does not hold geometry.

### Run 054 — latent stack, whitening, and residual reconstruction (`lx1b6gw2`)

This SSv2 present-only run combined the three-block latent-stack bottleneck with full
feature whitening and the residual target, while leaving geometry regularizers off.
It achieved highly code-conditioned reconstruction (`~92%` by the old shuffled-code
calculation) but still settled at low rank `~21.9` and loss `~0.652`.

### Run 055 — whitened absolute-target control (`nzz64pl6`)

Run 055 was run 054 with only the residual-target subtraction removed, restoring the
absolute whitened feature target while keeping geometry regularizers off. Its geometry
was almost unchanged (rank `~21.5`), loss improved to `~0.642`, and a small
position-template channel returned, so the residual target remained useful for honesty.

### Run 056 — full whitened geometry bundle (`tl5dh73c`)

This SSv2 present-only run added variance, covariance, and SIGReg to run 055's whitened
absolute-target autoencoder. The bundle held strong geometry (rank `~201.6`, std `~1.03`)
but raised reconstruction to `~0.731`, demonstrating the decodability cost of forcing a
broad, decorrelated representation.

### Run 057 — covariance plus variance, without SIGReg (`cdvp6hou`)

Run 057 removed only SIGReg from run 056 while retaining whitening, the absolute target,
`lambda_var=0.5`, `lambda_cov=0.01`, and `lambda_recon=0.05`. It became the settled SSv2
control: rank `~208.2`, std `~1.12`, reconstruction `~0.713`, and a recorded shuffled-code
gap of `~0.227`.

### Run 058 — EGO4D transfer of run 057 (`mvbx96nv`)

Run 058 moved the run-057 present-only recipe from SSv2 to EGO4D and used the matching
EGO4D whitening statistics; its architecture and loss implementation were unchanged.
It finished cleanly near reconstruction `~0.678`, but its apparent collapse/template
verdict is now unresolved because all 16 diagnostic clips came from one source recording.

### Run 060 — EGO4D reconstruction-weight-1 arm (`2423b84g`)

Run 060 was intended to test whether run 058's `lambda_recon=0.05` underweighted
reconstruction by increasing it to `1.0` on the EGO4D recipe. At this audit's live
step-5,450 snapshot it had reconstruction near `0.69`, but no final verdict is assigned,
and it runs on a newer strict/deterministic pipeline rather than run 058's exact code state.

## Executive answer

The recurring `0.6-0.7` value is not a magic constant in the code, and it is not a
signature of one broken gradient. It is the stable rate-distortion regime produced by
the combination of:

1. **target standardization:** SSv2 and EGO4D are separately ZCA-whitened, making their
   pooled channel covariances nearly identity;
2. **an early irreversible squeeze:** every 1,024-D detailed token is projected to 256
   channels before any spatial/temporal aggregation;
3. **a severe global information bottleneck:** 1,024 x 1,024 detailed scalars become
   32 x 256 latent scalars, a 128:1 ratio;
4. **an angular objective:** the logged number is `1 - cosine`, not MSE, variance
   explained, or pixel error;
5. **a multi-objective, staged optimization:** variance/covariance act at full strength
   while reconstruction ramps from zero, global clipping dominates the early warmup,
   and the LR later decays essentially to zero.

The first three explain why changing the semantic dataset does not necessarily move the
numeric floor. Whitening deliberately removes most second-order dataset-specific
channel geometry before the same narrow autoencoder sees either corpus.

A separate, equally important finding changes the interpretation of run 058 without
changing the existence of its reconstruction plateau: the fixed EGO4D diagnostic batch
contains 16 adjacent chunks from one source recording. Its so-called cross-video and
wrong-video metrics are therefore not cross-source metrics. Run 058 may have a genuine
template/content problem, but the recorded `0.018` gap cannot prove a global template
collapse.

These are two different questions and must stay separate:

| Question | Answer from this audit |
|---|---|
| Why does raw `L_recon` land in the same broad band? | Whitened target geometry plus a fixed, narrow information channel is the leading mechanism. |
| Why did EGO4D look globally collapsed while SSv2 looked healthy? | The comparison is confounded by a single-source EGO4D validation batch; global collapse remains unmeasured. |

## 1. Causal boundary of the comparison

Run 057 (`cdvp6hou`) and run 058 (`mvbx96nv`) were intended as dataset twins. Their W&B
configs match on the complete present-only recipe except for dataset and whitening
file. The architecture and loss source files are identical across the two commits:
`models.py` and `losses.py` did not change. The 71 insertions / 22 deletions in the
relevant runtime files added EGO4D file support, dataset selection, and a Stage-0 EMA
assertion repair.

That makes the B/D/loss comparison strong, but not perfectly byte-identical:

- run 057 commit: `5ea4421ff7aa034894dc07ab6270ae766a00d83a`;
- run 058 commit: `21d2aa8a97e8f60536a681bc5a4324d1a2c28838`;
- the old encoder loader did not pin a Hugging Face revision or record a feature
  fingerprint, and the runs executed on different hosts;
- each dataset necessarily used a different offline whitening tensor.

The live weight-1 run 060 is less causally clean against run 058. Its intended config
delta is only `lambda_recon=0.05 -> 1.0`, but its executable pipeline also gained:

- an immutable V-JEPA revision and feature fingerprint;
- isolated, deterministic initialization of the trainable modules;
- sample/epoch-scoped transform RNG and explicit epoch samplers;
- raw-RGB data with encoder-owned normalization;
- a strict whitening envelope bound to encoder, data, transform, and clip count;
- materially expanded provenance/resume machinery.

Those are good reproducibility changes, but they mean run 060 is not evidence from a
literal single-code-delta replay of run 058. Its live metrics are useful context, not a
finished causal verdict.

## 2. The exact computation being optimized

The present-only path is:

```text
EGO4D/SSv2 file
    -> temporal sample + crop + color jitter
    -> frozen encoder E (no gradient)
    -> detailed tokens e: [B, 1024, 1024]
    -> fixed ZCA whitener W (no gradient)
    -> whitened target e_w: [B, 1024, 1024]
    -> trainable bottleneck B
    -> c: [B, 32, 256]
    -> trainable decoder D
    -> e_hat_w: [B, 1024, 1024]
    -> mean over batch and token positions of 1 - cosine(e_hat_w, stopgrad(e_w))
```

In symbols, the run-058 reconstruction term is:

```text
L_recon = mean_(b,p) [1 - <normalize(D(B(e_w)))[b,p], normalize(e_w)[b,p]>]
```

The total optimized objective is:

```text
L_total = 0.5 L_var(c) + 0.01 L_cov(c)
        + lambda_recon * recon_ramp(step) * L_recon
```

with `lambda_recon=0.05` in runs 057/058 and `1.0` in run 060. `L_flow`, F_c,
future targets, prediction-side reconstruction, and the EMA bottleneck target are not
part of the present-only task. B_EMA may still be mechanically updated, but it does not
produce an optimized target in this mode.

### What `0.7` means

For this loss:

- perfect angular alignment gives `0`;
- unrelated/orthogonal directions give about `1`;
- opposite directions give `2`;
- loss `0.70` means average token cosine about `0.30`;
- loss `0.64` means average token cosine about `0.36`.

It does **not** mean 70% pixel error, 70% unexplained variance, or an MSE of 0.7. The
target and prediction magnitudes are discarded from the value. Comparisons to the old
relative-MSE runs are numerically invalid.

The target is detached correctly. Gradients from `L_recon` reach D and B, and nowhere
else. The run histories show the path is active: loss falls from about 1.0 to the
0.68-0.71 band. The question is therefore attainable distortion and optimization, not
whether the reconstruction branch is disconnected.

## 3. Data production and sampling audit

### Source construction

[`select_ego4d_uids.py`](../../../../select_ego4d_uids.py) splits EGO4D at the source-
video UID level, which correctly prevents train/validation leakage across chunks.
[`chunk_ego4d.py`](../../../../chunk_ego4d.py) converts each long recording into
non-overlapping 4-second, 12-fps H.264 clips named:

```text
<source-video-uid>_<chunk-index:05d>.mp4
```

The repository already documents the relevant statistical property in
[`make_ego4d_subset.py`](../../../../make_ego4d_subset.py): chunks from one long source
are near-duplicates, so its tiny-subset builder caps chunks per source. The full
training and validation loader has no analogous source-balancing rule; it treats every
chunk file as a sample.

Effect on the floor:

- **training:** chunk-level random permutation can overweight long source recordings
  and does not guarantee source uniqueness inside a batch; this reduces effective
  source diversity, but random training batches are not the lexically contiguous
  validation batch;
- **diagnostics:** lexical sorting makes adjacent chunks from the same UID contiguous,
  creating the major measurement confound detailed later;
- **whitening:** the 12,800-clip statistics sample is also clip-weighted rather than
  source-balanced, so long recordings can contribute disproportionately to the target
  geometry.

These can affect generalization and measurement. They do not create a literal loss
clamp.

### Temporal window

[`data.py`](../../../../data.py) reads 8 frames at stride 2. All three target runs carry
`horizon_k=12` even though prediction is disabled. `_context_indices` still reserves
that future horizon so present-only and prediction modes choose the same context.

For a normal 48-frame EGO4D chunk:

```text
span = (8 - 1) * 2 + 12 = 26 frames
max_start = 48 - 1 - 26 = 21
```

Training starts range from 0 to 21; validation uses the center start 10. Thus an
inactive future-task hyperparameter still constrains the present-only data distribution
and keeps the final portion of a chunk out of the context window. This is a design leak
worth cleaning up or explicitly retaining, but it is shared by runs 057/058 and is not
a good explanation for their similar floors.

### Spatial/photometric transform

Training applies random crop plus one clip-consistent brightness, contrast, and
saturation factor, each sampled in `[0.6, 1.4]`. Validation uses center crop and no
color jitter.

The autoencoder is asked to preserve the frozen representation of the augmented clip,
including whatever augmentation-specific information V-JEPA retains. Strong jitter can
raise the rate needed for reconstruction, and using the same augmentation recipe on
both datasets is another common pressure toward a common floor. It remains a secondary
hypothesis until the fixed-batch/no-augmentation probe is run.

The old run-058 pipeline decoded both context and target windows even in present-only
mode, then used only the context branch. Current code avoids decoding the unused target.
The context indexing contract remains equivalent.

## 4. Frozen encoder audit

The encoder produces 1,024 tokens of width 1,024 for every clip: four temporal tubelet
positions by a 16x16 spatial grid. It is frozen, held in eval mode, and run under
no-grad. Current code owns ImageNet normalization inside the encoder and runs V-JEPA in
bf16. Run 058 normalized in `data.py` and also ran bf16.

Consequences:

- the reconstruction task is a feature autoencoder, not a video/pixel autoencoder;
- encoder features can contain strong position, appearance, and nuisance structure;
- no reconstruction signal can adapt E to make the target more compressible;
- target drift from encoder training is ruled out;
- old runs did not record an immutable encoder revision, so exact feature identity
  across old pods is unproven even though the configured repository was the same.

There is no evidence that the dataset switch silently failed. W&B reports EGO4D, the
correct data counts, the EGO4D whitening path, and an EGO-specific fixed-batch identity.

## 5. Whitening is the main reason dataset semantics do not set the loss scale

[`models.FeatureWhitener`](../../../../models.py) applies global channel ZCA:

```text
e_w = (e - mu) U diag((lambda + eps)^-1/2) U^T
```

The mean and covariance are pooled over clips **and token positions**. This removes one
global 1,024-D mean but does not subtract the mean separately at each of the 1,024
lattice positions. A position-specific template can therefore remain.

### Exact current EGO4D spectrum

The strict EGO4D artifact used by run 060 contains 13,107,200 token rows from 12,800
training clips. Its raw channel covariance has:

| Quantity | Value |
|---|---:|
| effective rank | 230.4794 / 1,024 |
| eigenvalue sum | 6,895.9735 |
| min | `2.63e-13` |
| 1st percentile | 0.1783 |
| median | 1.6567 |
| 90th percentile | 12.1041 |
| 99th percentile | 83.0430 |
| max | 569.9024 |

After whitening with `eps=1e-4`, each covariance eigenvalue becomes approximately:

```text
lambda_w = lambda / (lambda + eps)
```

The resulting trace is 1,022.8863 and effective rank is 1,022.99999. Only one raw
eigenvalue lies below `eps`; 1,023 directions are essentially unit variance. The
transform's gain ranges from 0.0419 to 100.

This is not a mild normalization. It changes a rank-230 energy distribution into an
almost fully isotropic 1,023-dimensional target. Directions V-JEPA originally assigned
little energy receive almost the same loss importance as dominant directions.

### The 1,024-to-256 projection calculation

The very first bottleneck operation is a learned linear projection from 1,024 channels
to 256. For the inspected post-whitening spectrum:

```text
sum(top 256 lambda_w) / sum(all lambda_w) = 0.25026968
```

For an isotropic Gaussian target, the best rank-256 linear observation retains that
fraction of variance. Its expected angular alignment heuristic is:

```text
cosine ~= sqrt(0.25026968) = 0.50026961
loss   ~= 1 - cosine       = 0.49973039
```

This is a diagnostic oracle, not a theorem about the nonlinear trained model. Natural
whitened features can have nonlinear dependencies even when their covariance is
identity, so B can infer some discarded components from retained ones. Conversely, the
model must then compress 1,024 projected tokens into only 32 slots, which makes the
real task harder than the channel-only oracle.

The calculation nevertheless establishes scale: a loss around 0.5 is already the
channel-only expectation for an independent isotropic target. Adding 32x token
compression, limited decoder form, regularizer constraints, and finite optimization
makes an observed `0.64-0.71` entirely plausible. It is not absurd that EGO4D and SSv2
land near one another after each is mapped to this same standardized geometry.

### Why the dataset swap is not a strong test of a capacity floor

The intended hypothesis was: EGO4D has different video content, so the reconstruction
floor should change if the dataset controls the difficulty. But the actual comparison
is:

```text
ZCA(SSv2 V-JEPA features) -> same B/D/cosine system
ZCA(EGO4D V-JEPA features) -> same B/D/cosine system
```

ZCA deliberately makes both pooled channel covariances approximately identity. The
semantic distributions still differ in higher-order and tokenwise structure, which
explains the nonzero difference, but much of the obvious dataset-dependent energy
structure has been removed. Similar loss is therefore evidence that the shared
whitened channel/bandwidth problem dominates, not evidence that the dataset loader is
broken.

### The old artifact boundary

Run 058's legacy whitening file is not bound by a recorded payload fingerprint. The
exact spectrum above belongs to the current strict EGO4D artifact, not provably to the
old bytes. The method, dataset, feature width, clip count, seed, and `eps` match the
intended recipe, so the architectural inference is strong; the exact decimal values
must not be back-projected as historical measurements.

## 6. Bottleneck capacity and initialization audit

The production run-058 bottleneck has 4,837,123 trainable parameters, but information
bandwidth is controlled by activations, not parameter count:

| Stage | Shape per clip | Scalars | Compression from previous |
|---|---:|---:|---:|
| whitened detailed target | `1024 x 1024` | 1,048,576 | 1x |
| after `in_proj` | `1024 x 256` | 262,144 | 4x |
| abstract latent | `32 x 256` | 8,192 | 32x |
| total | | | **128x** |

### Early channel destruction

`Bottleneck.in_proj` is applied independently to each detailed token before ConvNeXt
mixing or cross-token attention. Any component in its 768-dimensional per-token null
space can only be reconstructed if it is statistically predictable from the retained
256 components elsewhere in the clip. Full ZCA makes that inference deliberately hard
at the linear second-order level.

This is more important than simply saying “c is small.” The architecture throws away
three quarters of each token's channel coordinates **before** it gets a chance to pool
space/time intelligently. Increasing `n_c` alone does not repair that first squeeze;
increasing mixer width alone does not repair the later slot squeeze. They are separate
axes.

### Fixed learned content at initialization

The bottleneck adds a trainable 1,024-position embedding to its token memory and begins
from 32 learned orthogonal query vectors. Every cross-attention, self-attention, and MLP
residual output projection is zero-initialized. Therefore, at initialization:

```text
B(any detailed input) = LayerNorm(learned queries)
```

for every sample. This is intentional and covered by
[`tests/test_bottleneck_attention.py`](../../../../tests/test_bottleneck_attention.py),
but it has two implications:

1. the code starts with rich **slot identity** and zero sample information;
2. the input/content path opens in stages because zero output bridges initially block
   gradients to their upstream q/k/v and memory projections.

At the first optimizer step, reconstruction's ramp is exactly zero. Variance and
covariance are already at full configured weight. The initial representation is thus
organized by geometry objectives before reconstruction can exert any force.

The staged opening is not sufficient to explain a permanent floor — the path clearly
learns — but it creates a strong early inductive bias toward a fixed slot code and
interacts with the template mechanism below.

## 7. Decoder audit: the old shortcut was narrowed, not eliminated

The decoder has 14,318,080 trainable parameters in these runs. It projects the 32
latent slots to width 512. Fixed 3-D position codes query the same 32-slot memory in an
initial cross-attention and four cross-attention/MLP blocks, then a 512-to-1,024 head
emits every detailed token.

Good properties:

- no learned per-output-token content table;
- fixed position is used as an attention query, not directly added to output content;
- zero latent values cannot generate position-specific output;
- gradients reach c and D.

Remaining limitation:

- all 1,024 output tokens are functions of the same 32 memories;
- there is no output-token self-attention, so output locations do not exchange newly
  decoded local content;
- fixed positions can select different mixtures of **distinct constant slots**.

### Executable shortcut probe

Using seed 42 and the actual run dimensions (`32x256` latent, decoder `512x4`) on current
code produced:

```text
B input sensitivity at identity init, max_abs = 0.00000000
B output batch identity, max_abs             = 0.00000000
B output slot std                            = 0.97542262

D(zero latent):
  mean feature std across output positions   = 0.00000000

D(B(zero detailed)):
  mean feature std across output positions   = 0.06995050
  mean std across identical batch samples    = 0.00000000

D(B(random detailed)):
  mean feature std across output positions   = 0.06995050
  mean std across identical batch samples    = 0.00000000
```

So the existing decoder test proves only that **zero, identical slots** cannot create a
position template. B does not emit zero, identical slots. It emits 32 distinct,
input-independent identities. Fixed position queries can select different mixtures of
those identities and produce a position-specific, sample-independent field.

This is the key architectural loophole:

```text
fixed nonzero slot identities + fixed output positions -> position-specific template
```

The learned B position embedding can reinforce it after the zero bridges open. Global
whitening removes the overall feature mean but leaves per-position mean deviations, so
the absolute target offers exactly such a template. The existing residual-target flag
subtracts a running per-position mean and is the right mechanism to remove this
particular reward.

Important restraint: the probe proves the shortcut **exists**, not how much run 058
used it. The source-confounded shuffled diagnostic cannot measure its global share. The
required measurement is `L(D(B(zeros)), e)` or a train-derived per-position constant
predictor on a source-diverse validation batch.

## 8. Loss-function and output-norm audit

[`losses.reconstruction_loss`](../../../../losses.py) computes in fp32 after normalizing
prediction and target along the 1,024-D channel axis. It then averages equally over all
batch elements and all 1,024 positions.

Consequences:

- every token position has equal weight regardless of target norm or raw covariance;
- magnitudes are unconstrained and unreported;
- a position/template direction can earn loss reduction even if it lacks clip-specific
  detail;
- the same numerical range is natural across datasets because cosine is scale-free;
- W&B's train `L_recon` is one random batch every 50 steps, not a full-dataset mean;
- `L_recon_present` is one fixed 16-sample batch, not a validation-set mean.

Although loss value is invariant to prediction norm, its gradient is not. For a target
unit direction `t_hat` and prediction `p`:

```text
d(1 - cos(p,t))/dp = -(t_hat - cos(p,t) p_hat) / ||p||
```

If D's output norms grow, gradients into its head and c shrink. `out_norm` normalizes
the hidden state before `out_proj`, but `out_proj` weights/bias can still change output
norm. No current metric logs prediction-norm percentiles, target norms, or their ratio.
This is a credible optimizer amplifier, but it is unmeasured and ranks below the known
capacity constraint.

The whitener computes its dense product in fp32, then returns bf16 to match the frozen
features. With 1,023 well-supported directions and only one eigenvalue below `eps`,
bf16 and the 100x maximum gain could add tail noise, but there is no NaN/skip evidence
and no reason for a precise `0.7` attractor. Numerical precision is a lower-ranked
hypothesis.

## 9. Geometry regularizers are not neutral autoencoder auxiliaries

These runs are “present-only” but not a pure autoencoder. B simultaneously optimizes:

- `L_var`: flatten `32x256` per sample and push every coordinate's across-batch std to
  at least 1;
- `L_cov`: pool batch and slots into `B*32` rows over 256 feature dimensions, then
  decorrelate those dimensions.

### Scalar objective balance in run 058

| Step | `0.5 L_var` | `0.01 L_cov` | `0.05*ramp*L_recon` | Total |
|---:|---:|---:|---:|---:|
| 0 | 0.50000 | 0.06726 | 0.00000 | 0.56726 |
| 500 | 0.10125 | 0.05295 | 0.01215 | 0.16635 |
| 1,000 | 0.02190 | 0.02685 | 0.02219 | 0.07094 |
| 1,500 | 0.01225 | 0.02083 | 0.03141 | 0.06449 |
| 2,000 | 0.00419 | 0.01368 | 0.04010 | 0.05797 |
| 5,000 | 0.00101 | 0.00415 | 0.03636 | 0.04153 |
| 10,000 | 0.00048 | 0.00272 | 0.03520 | 0.03840 |
| 14,500 | 0.00023 | 0.00213 | 0.03444 | 0.03679 |

Reconstruction dominates the scalar objective after about step 1,500-2,000, but scalar
contributions do not reveal gradient contributions or alignment. The repository does
not log `||grad_B L_recon||`, `||grad_B L_var||`, `||grad_B L_cov||`, or their cosines.

### Metric/loss Goodhart surfaces

`L_cov` and `c_effective_rank` both pool slots as observations. Because the 32 slot
identities are distinct even when every video receives the same code:

- `c_effective_rank` begins near 31 while cross-sample std is effectively zero;
- `L_cov` can be improved through slot geometry without proving video information;
- terminal rank can mix slot identity and sample content.

`L_var` is the cleaner cross-sample anti-collapse signal because it compares the same
flattened coordinate across batch items. On the EGO4D fixed batch, however, it measures
variation among chunks of one source, not global source variation.

This does not make covariance useless: run 057 shows it can sustain a high-rank,
sample-varying code. It means the objective and diagnostic need a decomposition:

- between-source variation of flattened c;
- within-source variation;
- within-video slot diversity;
- per-slot across-source feature covariance.

Without it, slot identity can make geometry look healthier than content routing is.

## 10. Gradient routing and `lambda_recon`

| Module | Reconstruction gradient | Var/cov gradient | Trained in present-only? |
|---|---|---|---|
| frozen encoder E | no | no | no |
| whitener | no | no | no |
| bottleneck B | yes | yes | yes |
| decoder D | yes | no | yes |
| coarse flow F_c | no | no | no effective update (`grad=None`) |
| EMA bottleneck | no direct gradient | no | copied from B, otherwise inert |
| feature-mean tracker | no parameters | no | inactive for absolute target |

The common statement “`lambda_recon=0.05` makes D learn 20x too slowly” is incomplete.
For D, every gradient comes from reconstruction. If all D gradients are multiplied by a
positive constant `a`, Adam's first moment scales by `a`, its second moment by `a^2`,
and `m/sqrt(v)` is approximately invariant once moments are established and `eps` is
small. Raising lambda from 0.05 to 1.0 therefore does not imply a 20x D parameter step.

Scale invariance is not exact because:

- the reconstruction ramp changes over time;
- Adam `eps` and weight decay do not scale the same way;
- global clipping rescales the combined parameter vector;
- early moment estimates depend on the transient;
- B receives a **mixture** of reconstruction, variance, and covariance gradients.

Thus the weight-1 arm is primarily a test of B's objective direction and early clipping
regime, not a clean 20x decoder-speed test. If the scientific question is decoder
undertraining, sweep `lr_decoder`, measure D update/weight ratios, or temporarily
optimize D with a free/frozen latent.

No current diagnostic can say whether B's reconstruction gradient fights covariance or
variance. Per-loss gradient norms and cosines are required before calling objective
conflict a major culprit.

## 11. Optimizer, clipping, and schedule audit

Both B and D use AdamW at peak LR `1e-4`, betas `(0.9, 0.95)`. Weight decay applies to
matrix weights, while biases, norms, learned queries/embeddings, and related parameters
are in no-decay groups. AGC runs before a global norm clip of 0.5.

### Stability is not the problem

Runs 057 and 058 complete with:

- zero skipped steps;
- zero NaN-gradient rows;
- no decoder AGC activations in the logged history;
- no run-058 B AGC activations in the logged history;
- finite, declining gradient norms.

This rules down catastrophic gradients, skip logic, and AGC as floor causes.

### Early global clipping matters, but does not explain the late floor alone

In run 058 the pre-global-clip norm exceeds 0.5 on:

- 90% of logged rows before step 1,500;
- 75% before step 2,000;
- 30% before step 5,000;
- 10% over the full run.

During exactly the period when reconstruction ramps up and the zero bridges open, the
combined update is frequently normalized to length 0.5. This couples objective weights:
larger geometry or reconstruction gradients reduce the applied scale of every module in
the shared clip. It can alter early representation formation. Once clipping stops, it
cannot explain why a capacity-limited decoder converges to the same angular regime.

### The late schedule is effectively closed

The LR warms for 1,500 steps and cosine-decays to zero at 15,000. The fraction of total
discrete LR-multiplier area already consumed is:

| Step | Consumed | Remaining |
|---:|---:|---:|
| 5,000 | 54.17% | 45.83% |
| 7,500 | 78.21% | 21.79% |
| 10,000 | 92.97% | 7.03% |
| 12,500 | 99.08% | 0.92% |
| 14,000 | 99.94% | 0.06% |

Run 058's fixed-batch loss drops only about 0.004 from step 10,000 to the end. That is
consistent with both a real rate-distortion plateau and a schedule with little update
budget left. Extending training with the same schedule would do almost nothing; a
constant-LR tail or restart is the discriminating test.

The plateau starts earlier than the final LR closure: run 058 is already 0.709 at step
5,000, with 45.8% of LR area remaining, and only gains 0.031 afterward. Schedule is
therefore a **secondary stabilizer**, not the primary origin.

## 12. The EGO4D diagnostic contract is broken for source-level claims

Current provenance records the validation batch exactly:

```text
01cab463-9a16-4817-84a4-a00ef5b7bf39_00000.mp4
01cab463-9a16-4817-84a4-a00ef5b7bf39_00001.mp4
...
01cab463-9a16-4817-84a4-a00ef5b7bf39_00015.mp4
```

All are non-overlapping four-second chunks from source UID
`01cab463-9a16-4817-84a4-a00ef5b7bf39`. The causal chain is explicit:

1. EGO4D names chunks `<uid>_<index>.mp4`;
2. `data.py` lexically sorts paths;
3. validation uses sequential order;
4. `run_training` takes the first batch of 16 once and reuses it;
5. diagnostics flatten that batch and name the pairwise result
   `c_cross_video_cosine`;
6. diagnostics use `torch.roll(..., dims=0)` and describe it as another video's c.

The only runtime guard checks `batch_size > 1`. Tests check that shuffled metrics exist,
not that rolled pairs have different source IDs.

Historical run-058 code used the same sorted-path, no-shuffle, first-batch design. The
strict provenance did not yet exist to print the IDs, but the file naming and loader
logic establish the same ordering consequence.

### What the recorded metrics actually say

Run 058 terminal values:

```text
present       = 0.678246
rolled chunk  = 0.696221
gap           = 0.017975
pair cosine   = 0.863109
```

They support:

- adjacent chunks from the same long recording produce similar codes;
- swapping codes among those chunks changes reconstruction little;
- clip-level discrimination within that source is weak on this fixed batch.

They do **not** distinguish:

- an input-independent global template;
- source/wearer/scene identity shared across chunks;
- genuinely static content over adjacent time;
- a representation that separates different EGO4D source videos well but not chunks
  within one source.

Therefore the prior statement “only 5.4% of improvement is video-conditioned” is too
strong. It is at most “5.4% is conditioned on the exact adjacent chunk rather than the
shared source context,” under the old normalization of improvement.

### Why the raw plateau still survives this correction

`L_recon` is logged on random training batches, not only on the fixed validation batch.
Its late mean is `0.69610` on EGO4D versus `0.70695` on SSv2. The common-band finding
therefore does not disappear when the validation semantics are corrected. What
disappears is confidence in the global template-collapse and cross-dataset geometry
comparison.

### Correct diagnostic design

Use sample/source IDs already carried in current `ClipBatch`:

1. parse EGO source UID with the final `_<chunk-index>` removed;
2. construct a fixed seeded batch with one clip per source;
3. assert at least 16 unique source IDs;
4. build a derangement whose every target-code pair has different source UID;
5. log separate same-source and different-source controls;
6. rename unverified metrics from `cross_video` to `cross_sample`;
7. add zero-code, mean-code, B-zero-input, and per-position-mean baselines.

That converts the current ambiguity into useful science: same-source gap measures
temporal/source invariance, while cross-source gap measures clip/source-conditioned
content.

## 13. Historical reconstruction values rule out a universal architectural constant

The architecture family has produced substantially different cosine reconstruction
values before the current whitened recipe:

| Run/regime | Target space / constraints | Recorded present loss |
|---|---|---:|
| 052 | raw absolute, no geometry; template-collapsed | ~0.293 |
| 053 | raw residual, no geometry; crashed near 12k | ~0.453 |
| 041 / inv011 geometry family | raw cosine feature reconstruction | ~0.345 |
| 054 | whitened residual, no geometry | ~0.652 |
| 055 | whitened absolute, no geometry | ~0.642 |
| 056 | whitened absolute + var/cov/SIGReg | ~0.731 |
| 057 | whitened absolute + var/cov | ~0.713 |
| 058 | EGO4D whitened absolute + var/cov | ~0.678 |

These are not a controlled ablation table: target definitions, geometry, and some code
changed. They should not be used to claim exact causal deltas. They are sufficient to
reject a hard-coded `0.6-0.7` architectural clamp. The loss moved below 0.5 in raw
feature regimes and entered the 0.64-0.73 band with full whitening and/or high-rank
geometry pressure.

The controlled comparison inside investigation 015 is especially informative:

- run 055, no geometry: rank 21.5, loss 0.642;
- run 057, covariance+variance: rank 208.2, loss 0.713.

Keeping a broad, decorrelated code imposes a decodability cost. A lower reconstruction
number can be purchased by a cramped or template-dominated representation, so raw loss
alone is not the goal.

## 14. Ranked mechanisms

### For the recurring whitened reconstruction band

| Rank | Mechanism | Confidence | Why |
|---:|---|---|---|
| 1 | Full ZCA makes the target nearly isotropic while `in_proj` immediately maps 1,024 -> 256 | high | Direct artifact measurement plus exact architecture; linear oracle already suggests loss ~0.50 before token compression. |
| 2 | 1,024 detailed tokens -> 32 latent slots and a 32-memory decoder | high | Exact 128:1 information ratio; every output shares the same 32 memories. |
| 3 | Geometry objectives constrain B's code away from the easiest reconstruction manifold | medium-high | Controlled run 055 -> 057: rank 21.5 -> 208 while loss worsens 0.642 -> 0.713. Gradient conflict is not yet directly logged. |
| 4 | Absolute-target constant-code/per-position template shortcut | mechanism high, run share unknown | Executable probe proves existence; global EGO use cannot be inferred from the single-source roll. |
| 5 | Early zero-gated content path + reconstruction ramp + global clipping | medium | Geometry controls the first phase; clip active on 90% of early logged rows. Path later opens normally. |
| 6 | Cosine LR decay closes convergence | medium | Only 0.92% LR area remains after 12.5k; plateau begins earlier, so it amplifies rather than originates the floor. |
| 7 | Decoder output-norm gradient attenuation | plausible/unmeasured | Loss gradient scales as `1/||prediction||`; norms are not logged. |
| 8 | Strong augmentation and clip/source weighting | plausible secondary | Adds target entropy and reduces effective source balance; fixed-batch probe can isolate it. |
| 9 | bf16 whitening tail / AGC / skip logic | low | No skips/NaNs/AGC activity; only one direction is below whitening eps. |

### For the apparent EGO4D “template collapse” versus SSv2

| Rank | Mechanism | Confidence | Why |
|---:|---|---|---|
| 1 | Single-source fixed diagnostic batch | certain | All 16 recorded IDs share one UID; metric labels and roll assumption are false. |
| 2 | Real shared source/scene content across adjacent chunks | high plausibility | Non-overlapping chunks from one egocentric recording naturally share wearer and scene. |
| 3 | Actual absolute-target template shortcut | architecturally possible, contribution unknown | Constant nonzero slots can emit position-specific output; needs zero/mean/source-diverse probes. |
| 4 | True EGO4D representation collapse | unresolved | Random-train loss and fixed-batch within-source geometry are insufficient to establish cross-source collapse. |

## 15. Hypotheses ruled down or removed from scope

- **A hard-coded loss floor:** no clamp, threshold, or constant exists in the loss path;
  historical values vary widely.
- **F_c, flow cooldown, EMA target, or future horizon loss:** F_c/future objectives are
  inactive in present-only mode. Horizon still affects context sampling but not a future
  loss.
- **Silent dataset-selection failure:** config, data identity, file names, counts, and
  whitening metadata all report EGO4D.
- **Encoder learning or target drift:** E is frozen and the target is detached.
- **NaNs/skipped updates:** zero in both completed runs.
- **AGC pinning the decoder:** D AGC never activates in the sampled histories.
- **Simply multiplying lambda to train D 20x faster:** Adam largely cancels a constant
  scalar on D; the meaningful change is B's multi-loss mixture and transient clipping.

## 16. Discriminating experiments, not another blind sweep

### Stage A — no paid training run

On source-diverse cached frozen features, compute:

1. **constant cosine oracle:** for each detailed position, the unit direction minimizing
   expected cosine loss; evaluate train and validation;
2. **global-mean and per-position-mean baselines:** quantify the template channel;
3. **rank-r target oracle:** PCA/linear reconstruction at `r=64,128,256,512,1024` in raw
   and whitened space;
4. **source decomposition:** within-source versus cross-source target cosine and c
   cosine;
5. **target norm and per-position statistics:** expose positions that dominate
   angular predictability.

The constant predictor has an exact cosine interpretation. If unit targets at a
position are `t_i`, the best constant unit direction is along `mean(t_i)`, and its
expected cosine is `||mean(t_i)||`. This directly measures how much loss can be earned
without c.

### Stage B — fixed-batch overfit ladder

Use one deterministic source-diverse feature batch and a non-decaying LR:

1. free trainable `c_i` for each sample + current D;
2. B+D, no var/cov, no augmentation variation;
3. B+D plus var only;
4. B+D plus cov only;
5. B+D plus both;
6. production warmup/clipping/schedule.

Interpretation:

```text
free c + D fails        -> latent bandwidth and/or decoder form
free c + D passes,
but B + D fails         -> bottleneck channel/content path
fixed-batch B + D passes,
but full corpus fails   -> rate/generalization, augmentation, regularizer, or schedule
only production schedule fails -> optimizer/schedule
```

This ladder identifies a module boundary. A 15,000-step lambda sweep does not.

### Stage C — capacity factorial

After the probes, vary independent axes:

- input channel width: mixer 256 -> 1,024 while holding c fixed;
- slot count: 32 -> 128 while holding feature width fixed;
- latent width: 256 -> 512;
- decoder memory/depth separately from c bandwidth;
- whitening strength: none, partial/power whitening, full ZCA;
- raw versus residual target after source-aware honesty is valid.

The most informative result is not merely lower loss. It is a curve of loss versus
bandwidth, together with cross-source shuffled gap and representation geometry.

### Stage D — gradient and update instrumentation

At diagnostic cadence, log:

- `||grad_B L_recon||`, `||grad_B L_var||`, `||grad_B L_cov||` before AGC;
- pairwise gradient cosines;
- per-module pre/post-clip gradient norm;
- per-module update norm / weight norm;
- decoder output norm mean/p50/p95 and target norm mean/p50/p95;
- `||B(e)-B(0)|| / ||B(e)||`;
- `L(D(B(0)), e)`, `L(D(mean_c), e)`, and source-diverse shuffled loss.

Those measurements decide whether weight, conflict, output norms, or a constant code is
responsible. Without them, assigning a gradient culprit is speculation.

## Final scientific verdict

The user's core observation is correct: switching from SSv2 to EGO4D did not materially
change the late reconstruction band. The interpretation changes after tracing the
actual task. This was not a clean swap between two raw video reconstruction problems.
It was a swap between two separately whitened frozen-feature distributions, each made
almost isotropic before passing through the same early 1,024-to-256 projection, the
same 32-slot code, the same decoder, and the same angular loss. Under that formulation,
a shared architectural rate-distortion floor is the expected leading outcome.

The strongest current architectural suspect is therefore **not a mysterious cooldown
or dead gradient**. It is the combination of full whitening and information bandwidth,
with regularizer/schedule effects layered on top. At the same time, the strongest
diagnostic suspect is definite: the EGO4D fixed batch violates the source-diversity
assumption behind the metric names and prior template-collapse verdict.

Fix that measurement contract, compute the cached-feature capacity oracles, and run the
fixed-batch overfit ladder before spending another full run. Those three actions will
turn the present list of plausible mechanisms into a localized failure boundary.
