# PR #6 — `Codex/task3 dino run066`: end-to-end trade-off analysis

**PR:** [ShashwatM3/HJEPA-VWM#6](https://github.com/ShashwatM3/HJEPA-VWM/pull/6)

**Base:** `phase1-v0.2-frozen-encoder` at `771cbba077d9f846bdf7a7dd48e12dbf29d54b49`

**Head:** `codex/task3-dino-run066` at `867141b4440ccef874516cb3df0051f34f9c453e`

**Merge base:** `4c402556e0b85684499f5a111630b7c8bc20ca89`

**Exact review diff:** `git diff 771cbba...867...` using three-dot merge-base semantics

**Audit date:** 2026-07-26

## Bottom line

**Do not merge this PR as-is.**

The important distinction is:

- **The DINO implementation is basically viable.** It is narrow, fail-fast, covered by useful
  unit tests, passes the full repository test suite in the hypothetical merged tree, and has
  already completed real A100 training runs.
- **The documentation and experiment record are not merge-ready.** The PR says DINO has no
  pinned default even though the code pins one; its CLI help says an explicit revision is
  required when it is not; its run numbering has been superseded by the current research
  record; and its planned DINO whitening run has already finished in W&B.

The cleanest decision is to separate the two concerns:

1. Keep the DINO adapter, immutable pin, and tests after fixing the missing default-alias test
   and documentation contradictions.
2. Rewrite/reconcile the KANBAN portion against the current workspace and live W&B before it
   lands. Do not merge the stale run folders unchanged.

“Able to merge” means only that GitHub can construct a merge commit without textual conflicts.
It does **not** mean the scientific record is consistent, the operator guide is runnable, or the
test suite ran. CodeRabbit explicitly skipped review; GitGuardian is the only substantive green
check shown by GitHub.

## What this project actually implements

The intended HJEPA-VWM system is:

```text
context video
  -> frozen detailed encoder E
  -> detailed latent e_t
  -> bottleneck B
  -> abstract latent c_t
  -> coarse flow F_c predicts future abstract state
  -> future FineFlow predicts detailed future state
  -> frame generator renders pixels
```

Only the first coarse phase is implemented. The current repository has:

```text
raw context[/future] clips
  -> frozen encoder E
  -> optional fixed feature whitener
  -> online B and EMA target B_EMA
  -> optional coarse rectified flow F_c
  -> feature decoder D
  -> losses, diagnostics, W&B, atomic checkpoints
```

It does **not** have FineFlow, Stages 2/3, a VAE/pixel generator, an inference rollout sampler,
multi-horizon conditioning, or an end-to-end video generation evaluation.

That matters for this PR: adding DINO does not complete the world model. It adds another frozen
feature substrate to the implemented Phase-1 system.

## The implemented pipeline, temporally and step by step

### 1. Video sampling

For each sample, the loader chooses eight context frames:

```text
context[i] = start + i * frame_stride, i = 0..7
```

The default `frame_stride` is 2. A full-prediction target is another eight-frame window whose
last frame is `horizon_k` original frames later. Present-only mode reserves the same legal context
range but never decodes the future frames.

Training uses a deterministic random start/crop/jitter tied to seed, epoch, sample identity, and
transform version. Validation uses a deterministic centered window. Context and future clips share
the same spatial/color transform in full-prediction mode.

The loader returns raw float RGB in `[0,1]`, shape `(B,8,3,256,256)`. Encoder normalization is not
allowed in the data layer.

### 2. Frozen detailed encoding

`FrozenEncoder` validates the raw clip, normalizes once in fp32, owns inference precision and frame
microbatching, keeps the backend in `eval()` with zero trainable parameters, and returns dense
tokens `(B,N_e,D_e)`.

The selectable contracts are:

| Encoder | Temporal behavior inside `E` | Detailed output |
|---|---|---:|
| V-JEPA2 ViT-L/16 | Four learned two-frame tubelet positions | `(B,1024,1024)` |
| SigLIP 2 ViT-B/16 | Eight independent images | `(B,2048,768)` |
| DINOv3 ViT-B/16 from this PR | Eight independent images; CLS + four registers removed | `(B,2048,768)` |

### 3. Optional fixed whitening

When enabled, all context and future features are transformed with one offline training-set ZCA
mapping:

```text
e_white = (e - mean) @ U @ diag((eigenvalue + eps)^-1/2) @ U.T
```

This is not batch whitening. The statistics are fixed, encoder/dataset/revision/preprocessing
bound, applied in fp32, and validated before training. The bottleneck, EMA target, flow target,
decoder target, and diagnostics all then live in the whitened coordinate system.

### 4. Online abstract bottleneck

`B` maps detailed features to a stable abstract interface `(B,N_c,D_c)`, normally `(B,32,256)`:

1. Project `D_e` to internal width `M`.
2. Reshape using the resolved encoder layout.
3. Apply shared ConvNeXt mixing separately at each temporal slot.
4. Add learned detailed-token position tags.
5. Let 32 learned orthogonal slots repeatedly read from the detailed memory.
6. In each latent block, apply cross-attention, slot self-attention, and an MLP.
7. Project once from internal `M` to external `D_c`, then LayerNorm.

The current encoder experiments use `M=512`, but the external coarse interface remains `32x256`.

### 5. EMA target branch

There is no second frozen encoder. `TargetBottleneck` is a frozen EMA copy of `B`.

For a full-prediction sample:

```text
c_t       = B(e_t)              # online; gradients allowed
c_plus    = B_EMA(e_future)     # target; detached
```

The EMA is updated only after a successful optimizer step.

### 6. Coarse rectified flow

In normal mode, the target is `c_plus`. In residual mode, the target is:

```text
delta = B_EMA(e_future) - B_EMA(e_present)
```

For either target `y`:

```text
eps ~ Normal(0, I)                       # scaled by std(delta) in residual mode
tau ~ Uniform(0, 1)
z_tau = (1 - tau) * eps + tau * y
target velocity u = y - eps
u_hat = F_c(z_tau, tau, c_t)
L_flow = mean((u_hat - u)^2)
```

`F_c` concatenates noised-target and conditioning streams, stamps slot/type embeddings, and uses
six adaLN-Zero transformer blocks. Ten percent condition dropout is the shipped default. There is
still no horizon embedding.

### 7. Feature reconstruction decoder

`D` expands any `(B,32,256)` abstract latent back to the selected detailed lattice. Fixed
time/y/x position codes tell the decoder where to write; only values read from `c` tell it what to
write. There is no learned per-output-token content template.

This decoder reconstructs frozen **features**, not pixels.

### 8. Loss assembly

The current cosine reconstruction loss is:

```text
L_recon = mean over tokens(1 - cosine(D(c), stopgrad(e)))
```

The legacy `relative_mse` mode is:

```text
mean((D(c) - e)^2) / Var(e)
```

The geometry terms are:

```text
L_var  = mean_j max(0, 1 - Std_batch(flatten(c)_j))

L_cov  = sum of squared off-diagonal entries of
         Cov(reshape(c, B*N_c, D_c)) / D_c

L_slot = mean squared off-diagonal cosine between per-video,
         slot-mean-centered slot vectors
```

`L_sigreg` projects pooled `(B*N_c,D_c)` rows onto random unit directions and applies the BHEP
normality statistic against `Normal(0,1)`. In plain terms: it asks every random view of the code to
look like a unit Gaussian, which encourages isotropic/high-rank use of `D_c`.

For full prediction:

```text
L_total =
    L_flow
  + lambda_var * L_var
  + lambda_cov * L_cov
  + lambda_slot * L_slot
  + lambda_sigreg * sigreg_ramp * L_sigreg
  + lambda_recon * recon_ramp * L_recon_present
  + lambda_recon_pred * recon_ramp * L_recon_predicted_future
```

Zero-weight covariance, slot, and SIGReg terms may still be calculated for logging but do not
affect gradients.

For present-only mode, `L_flow=0`, future frames and `F_c` are inactive, and the loss starts from
the geometry/reconstruction terms only. Therefore present-only success says nothing about
forecasting.

### 9. Backward, clipping, optimizer, and EMA

Each step:

1. Seed training randomness from `seed * 1_000_003 + step`.
2. Zero gradients.
3. Run the relevant present or prediction branch under configured autocast.
4. Backpropagate the total loss.
5. Apply module-specific AGC to `B`, `F_c`, and `D`.
6. Apply global norm clipping at `0.5`.
7. Skip the update if the post-AGC pre-global-clip norm is nonfinite or above `150`.
8. Otherwise run AdamW and update `B_EMA`.

### 10. Schedule

Shipped defaults:

| Item | Value |
|---|---:|
| Global batch | 64 |
| Phase-1 steps | 15,000 |
| LR warmup | 1,500 steps |
| LR after warmup | cosine decay to zero over Phase 1 |
| Peak LR, `B` | `1e-4` |
| Peak LR, `F_c` | `2e-4` |
| Peak LR, `D` | `1e-4` |
| AdamW betas | `(0.9,0.95)` |
| Weight decay | `0.05`, only on genuine linear/conv weights |
| EMA momentum | cosine from `0.996` toward `0.9999` over 105,000 latent steps |
| Reconstruction/SIGReg warmup | 2,000 steps |
| Normal logging | every 50 steps |
| Diagnostics | every 500 steps |
| Checkpoint | every 2,500 steps |

The recent encoder-substrate runs override `horizon_k=12`, use `M=512`, use present-only
reconstruction at weight 1, and set all three learning rates to `1e-4` (`F_c` is inactive).

### 11. Diagnostic meaning

The full-prediction decision cycle is:

1. Stability: skips, NaNs, warning spikes, gradient trajectory.
2. Collapse: code std, dead dimensions, cross-example cosine.
3. Rank: effective rank, with `>60` as the working healthy target.
4. Static-code test: does present/future latent distance vanish?
5. Copy gate: `coarse_model_loss / coarse_copy_loss <= 0.70`.
6. Conditioning gate: `coarse_model_loss / coarse_batch_mean_loss <= 0.50`.
7. Reconstruction honesty: present/true-future/predicted-future readouts.
8. Verdict.

The copy baseline literally predicts “no temporal change.” The batch-mean baseline ignores this
video and predicts the batch-average target. A model must beat both.

Present-only runs omit the prediction gates. They ask whether reconstruction improves while the
code stays high-rank, spread, and example-specific.

### 12. Checkpoint and MLOps contract

Checkpoints atomically save trainable modules, optimizer, `next_step`, completed updates, schedule,
EMA/mean/whitener state, config, exact `EncoderSpec`, dataset/run provenance, RNG/sampler state,
trainable-init hash, and W&B ID. Frozen encoder weights are intentionally excluded.

That keeps checkpoints smaller, but it means loading a DINO checkpoint later still requires the
same gated Hub weights or an already-populated cache.

## What the experiments have established so far

I read the KANBAN history through Investigation 17 and re-read representative histories from the
live W&B project rather than trusting only the prose.

Live W&B currently contains **74 entries**:

```text
finished: 33
crashed:  29
killed:   11
failed:    1
```

“Crashed” and “killed” are not automatically optimizer failures; many were externally stopped
while numerically healthy.

### Research timeline

| Investigation | What was tested | What was learned |
|---:|---|---|
| 1 | Original long baseline and retuned baseline | The original schedule became unstable near peak LR; the shorter/lower-LR schedule became the operational base. |
| 2 | Decode/throughput smokes | Selective frame decoding improved step time; data plumbing was not the central scientific failure. |
| 3 | Slot penalty, longer horizon, stronger variance | Variance fixed std/cosine, but rank stayed around 13 and copy was not beaten. Slot loss could Goodhart its own metric. |
| 4 | Covariance idea | Not isolated cleanly; paused rather than overclaimed. |
| 5 | Long acceptance/stability runs and AGC | AGC prevented some skip spirals, but a late representation/predictor cliff still occurred. Stability alone was not success. |
| 6 | Present and prediction-side reconstruction | Reconstruction stabilized training, but the decoder could not distinguish true from predicted future latents well enough to supervise forecasting. |
| 7 | Reconstruction weight, decoder size, latent capacity | Weight and decoder size did not move the roughly `0.585` legacy reconstruction floor. The code remained low-rank. |
| 8 | SIGReg sweep | Strong SIGReg raised rank up to roughly 73, but prediction became worse; geometry was not the predictor fix. |
| 9 | Residual prediction | The latent became more temporally dynamic and healthier, but `F_c` mostly tied the zero-residual/copy baseline. |
| 10 | Clean residual run | Canonical negative result: rank about 61, std about 1.006, cosine about 0.16, but copy ratio about 1.07 and batch-mean ratio about 1.15. Healthy representation, no predictor. |
| 11 | Cosine recon, fixed-position decoder, present geometry sweep | Cosine loss lowered reconstruction to about `0.346`; covariance pushed present rank above 100. The wins were present-side only. |
| 12 | Sharp slots + absolute reconstruction | Reconstruction reached about `0.293` while the code collapsed into a mostly video-independent template. |
| 13 | Residual reconstruction target | Closed the template loophole: shuffled-code gap became large, but geometry still contracted without an explicit anti-collapse force. |
| 14 | Offline V-JEPA rank probe | Raw `e` is anisotropic: pooled effective rank about 193/1024 with a long weak tail. This motivated whitening. |
| 15 | Latent-stack bottleneck, whitening, covariance/variance | Whitening plus covariance/variance produced rank around 208 on SSv2; covariance was the operative rank lever, while SIGReg imposed a small content/attention tax. |
| 16 | EGO4D transfer, reconstruction weight, slots, internal width, encoder substrates | SSv2 success did not transfer directly. More slots and `M=1024` did not materially fix the reconstruction floor. Raw no-geometry V-JEPA/SigLIP/DINO codes remained decodable but geometrically contracted. |
| 17 | Three-encoder latent-shape sweep | Incomplete. A duplicate V-JEPA center was stopped; the `16x512` arm later crashed at step 7,100. The written KANBAN is behind live W&B. |

### The decisive full-prediction result

Run `2vbo6pbm` is still the clean scientific boundary:

```text
late c_effective_rank          61.02
late c_std_mean                 1.006
late c_cross_video_cosine       0.161
late coarse_vs_copy_ratio       1.070
late coarse_vs_batch_mean       1.149
```

The representation was healthy. The predictor still did not beat “copy the present” or “use the
batch mean.” This is why a new encoder cannot be called a world-model improvement from
reconstruction alone.

### The latest live DINO result, absent from the PR’s observations

W&B run [`qqozribu`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/qqozribu) finished
15,000 steps on the exact pinned DINO commit `083cf8a...`. It is the experiment the PR still labels
“PLANNED — not yet launched.”

Late six-diagnostic medians:

| Metric | DINO unwhitened/no geometry `it7sq8nz` | DINO whitened + cov/var `qqozribu` |
|---|---:|---:|
| `c_std_mean` | 0.3329 | **0.8095** |
| `c_cross_video_cosine` | 0.8699 | **0.4926** |
| `c_effective_rank` | 14.60 | **67.61** |
| centered slot rank | 10.51 | **21.25** |
| correct-code reconstruction | 0.1148 | 0.3487 |
| shuffled-code reconstruction | 0.1808 | 0.4570 |
| correct-vs-shuffled gap | 0.0661 | **0.1083** |
| skipped/NaN/warned updates | 0/0/0 | **0/0/0** |

Do not compare the two raw reconstruction values: whitening changes the target coordinate system.
The defensible conclusion is that the whitening + covariance + variance **bundle** moved DINO from
low-rank contracted geometry to a borderline/healthy present representation with a real
correct-code gap.

The fixed EGO4D validation batch consists of adjacent chunks from one source UID. Therefore the
gap and cosine establish exact-chunk/within-source separation, not global cross-source video
specificity.

This run still has `prediction_active=0`. There is no DINO copy ratio, no batch-mean ratio, and no
forecasting result.

### Live artifact verification

I inspected the actual W&B payloads, not only their artifact names.

The latest run’s provenance records:

- DINO revision `5931719e67bbdb9737e363e781fb0c67687896bc`;
- `(T,H,W,D)=(8,16,16,768)`;
- 85,660,416 frozen parameters;
- full EGO4D dataset fingerprint `df36af5d...`;
- A100-SXM4-80GB, CUDA 12.4, PyTorch 2.4.1, Transformers 4.57.6;
- clean training commit `083cf8a...`;
- whitening payload fingerprint `72030fb7...`.

The downloaded whitening artifact is schema `hjepa-whitening-v2`, about 2.37 MB, with finite
`mean`, `768` eigenvalues, and `768x768` eigenvectors. Its metadata binds it to 12,800 deterministic
training clips and 26,214,400 DINO token rows. It cannot be reused for V-JEPA, SigLIP, another
revision, another dataset identity, or another identity-bearing runtime configuration.

## What PR #6 changes

The PR is 23 files, `+1662/-23`, across five commits. Most of the size is experiment
documentation. The meaningful changes are below; the `.gitignore` addition is negligible and is
intentionally omitted.

## 1. `encoders.py`: add `_DINOAdapter`

### What it does

- Loads `transformers.DINOv3ViTModel`.
- Uses the vision backbone’s `last_hidden_state`.
- Requires `config.num_register_tokens == 4`.
- Requires exactly 261 tokens per frame: one CLS, four registers, 256 patches.
- Requires width 768.
- Removes the first five special tokens.
- Relies on the common `FrozenEncoder` to encode frames in microbatches and restore time-major
  `(B,2048,768)` order.

### Benefit

This is the right architectural seam. DINO-specific repository/class/token rules stay inside
`encoders.py`; data, bottleneck, decoder, whitening, probes, checkpoints, and W&B continue to use
one generic `EncoderSpec`.

The strict assertions are valuable. A model/config change fails loudly instead of silently shifting
the patch lattice by five tokens.

### Cost and trade-off

The adapter is intentionally tied to one exact model contract. Another DINO size, patch size,
register count, or Transformers output format requires code and test changes. That is maintenance
coupling, but it is safer than pretending arbitrary DINO checkpoints are compatible.

The literal `768`, `256`, and four-register checks duplicate registry/spec facts. Repository
standards prefer resolved configuration over magic numbers. A better implementation would pass
the expected private contract into the adapter or validate against one private registration object.

### Risk

**Low for the pinned checkpoint; medium for future DINO variants.**

The real `it7sq8nz` and `qqozribu` runs prove the present checkpoint works under Transformers
4.57.6 on CUDA. They do not prove forward compatibility.

## 2. Registry: make DINO selectable and pin an immutable default

### What it does

The registry changes DINO from a reserved/unimplemented alias to:

```text
alias: dinov3_vitb16
repo: facebook/dinov3-vitb16-pretrain-lvd1689m
default revision: 5931719e67bb9737e363e781fb0c67687896bc
layout: 8x16x16 frame tokens
feature width: 768
normalization: ImageNet
```

The factory logic is also split into two checks:

1. Is there an adapter implementation?
2. Is there either an explicit revision or an immutable default?

### Benefit

`--encoder dinov3_vitb16` becomes reproducible without following mutable Hub `main`. Existing
V-JEPA and SigLIP aliases are unchanged, and the global default remains V-JEPA.

The two-stage factory check supported the intended workflow: first test DINO with an explicit
candidate revision, then pin it only after real validation.

### Cost and trade-off

Selecting DINO is now easy enough that an operator may assume it is a drop-in equivalent to
V-JEPA. It is not:

- DINO is image-native; V-JEPA is video/tubelet-native.
- DINO doubles detailed-token count.
- DINO changes feature semantics, covariance, width, checkpoint shapes, decoder output shape, and
  whitening identity.
- Cross-encoder resume is prohibited.

This is expected substrate switching, not an accidental break, but experiment comparisons must
hold those facts explicitly.

### Risk

**Low to existing runs, because V-JEPA remains the default. Medium to experiment interpretation.**

## 3. Architecture trade-off when DINO is selected

### Lighter backbone, larger downstream lattice

Per clip:

```text
V-JEPA detailed scalars = 1024 * 1024 = 1,048,576
DINO detailed scalars   = 2048 *  768 = 1,572,864
```

DINO carries **50% more detailed scalar activations** even though its frozen backbone has about
86M parameters rather than V-JEPA’s 326M.

Token-dependent bottleneck and decoder work also grows:

- eight ConvNeXt grids instead of four;
- twice as many memory positions for bottleneck cross-attention;
- twice as many decoder output queries;
- a 50%-larger first/last feature projection workload by the simple `N*D` count.

The encoder itself is much smaller and shallower, so runtime is not determined by token count
alone.

Observed W&B system data on A100-SXM4-80GB, batch 64, unwhitened `M=512` present-only runs:

| Run | Encoder | Median W&B allocated GPU memory | Runtime |
|---|---|---:|---:|
| `4biwq87o` | V-JEPA2 | 12.71 GiB | 3.19 h |
| `it7sq8nz` | DINOv3 | 20.28 GiB | 3.09 h |
| `qqozribu` | DINOv3 + whitening/cov/var | 20.41 GiB | 3.22 h |

These are observational runs on different commits, not a controlled resource-preflight pair.
Still, they show the practical shape of the trade: **DINO saves frozen-model parameters and can be
roughly as fast, but the 2048-token downstream path can consume more activation memory.**

### Temporal semantics

DINO independently encodes each frame. It cannot use frame order or motion inside the frozen
backbone. Time ordering is added only downstream through the resolved lattice position codes.

That may be fine for present reconstruction. It is a real risk for forecasting:

- V-JEPA enters `B` with two-frame tubelet features that already contain temporal interaction.
- DINO enters `B` with spatial features from isolated frames.
- `B`’s ConvNeXt mixing is per temporal grid; temporal integration happens mainly through the
  global Perceiver read, not a dedicated temporal encoder.

No DINO full-prediction run exists, so this trade-off remains experimentally unresolved.

### Feature-space comparability

A lower DINO reconstruction loss does not mean a better world model. DINO, SigLIP, and V-JEPA
define different targets. Compare:

- within-run improvement;
- correct-vs-shuffled separation;
- geometry;
- stability;
- memory/throughput;
- and, eventually, copy/batch-mean prediction gates.

Do not rank encoders by raw reconstruction loss.

## 4. MLOps trade-offs

### Gated dependency and license

DINO’s checkpoint is gated and uses the DINOv3 license rather than the V-JEPA checkpoint’s existing
license path. A fresh pod or a new collaborator needs:

- accepted checkpoint access;
- a read-capable Hugging Face credential;
- Transformers 4.57.6;
- the exact immutable revision;
- and awareness of redistribution/use restrictions.

Because checkpoints exclude frozen encoder weights, a trained HJEPA checkpoint alone is not
self-contained. Reproduction and inference require gated DINO access or a valid local cache.

### Separate artifacts

DINO requires its own:

- Hub cache object;
- whitening statistics;
- feature/rank/drift caches;
- checkpoints;
- provenance and W&B run identity.

Equal tensor shape does not make DINO and SigLIP artifacts interchangeable.

### Configuration-reading trap

The production path correctly uses `resolved_provenance.encoder_spec.feature_dim=768`, but the
legacy `ModelConfig.d_e=1024` field still appears in W&B’s top-level model config for compatibility.
A naive dashboard/config diff can therefore report the wrong detailed width. Operational analysis
must read `resolved_provenance.encoder_spec`, not the legacy `model.d_e`.

This is an existing compatibility seam that becomes easier to misread once DINO is enabled.

### Dependency pin

The PR does not add a new package pin; this repository already requires
`transformers==4.57.6`. That is good for reproducibility but intentionally prevents effortless
upgrades. A Transformers upgrade requires repeating all real-adapter contract checks.

## 5. Tests added by the PR

### What is covered well

The new fake-model tests check:

- repository/revision/cache/attention arguments;
- ImageNet normalization exactly once through the public factory;
- `(B,2048,768)` output;
- time-major frame ordering;
- special tokens absent from output;
- zero encoder trainables;
- rejection of bad register count, sequence length, width, and missing `last_hidden_state`;
- presence of the pinned registry revision.

### What remains partial

- The default alias test inspects the private registry string but does not actually call
  `build_frozen_encoder(EncoderConfig(alias="dinov3_vitb16"))` and prove requested/resolved
  revision equality.
- The DINO-specific additions do not explicitly test repeated-frame equality or frame-permutation
  equivariance, both required by the encoder dossier.
- Real weight loading is not a normal CI test, which is reasonable for a gated model. The W&B
  runs supply external integration evidence, but GitHub does not expose that evidence as a required
  PR check.
- Several added functions/tests do not meet this repository’s strict docstring convention.

### Actual verification performed for this review

I constructed the hypothetical three-way merged tree without touching the working tree and ran it
with the repository’s pinned environment:

```text
tests/test_encoders.py: 35 passed
full pytest suite:       168 passed, 3 expected legacy-checkpoint warnings
Black, changed Python:   passed
git diff --check:        passed
Python compileall:       passed
```

Ruff was not installed in the local project environment, so I did not claim a Ruff result.

The first test attempt under the unrelated system Python failed only because that environment had
Transformers 5.5.3; the repository environment correctly has the required 4.57.6 and passes.

## 6. `train.py` and living documentation

The only `train.py` behavior change is help text. It says:

```text
DINO requires --encoder-revision until its default checkpoint is pinned.
```

But this same PR pins the default. `AGENT_FILES/AGENTS.md`,
`AGENT_FILES/KNOWLEDGE/encoders/README.md`, and `GUIDES/CODEBASE_STRUCTURE.md` repeat the same false
state.

This is not cosmetic. An operator may:

- believe the stable alias is still gated in code;
- keep copying explicit revisions everywhere;
- or distrust the registry/default behavior that the PR actually ships.

It also violates the repository rule that implementation/default/CLI docs stay synchronized in
the same PR.

**Required fix:** consistently document DINO as implemented and pinned, and add one default-alias
factory test. Explicit revisions can remain in historical run commands for provenance, but the
general CLI/help must describe the current behavior.

## 7. KANBAN run 066: planned SigLIP unwhitened arm

The PR adds a 519-line run package for a present-only EGO4D SigLIP run with:

```text
M=512, N_c=32, D_c=256
lambda_recon=1
no whitening
lambda_var=lambda_cov=lambda_sigreg=lambda_slot=0
```

### Benefit

It is a clean encoder-only substrate comparison against the historical V-JEPA `M=512` arm.

### Cost and current problem

This configuration has already been executed in live W&B as `j7a3tzj5` and is recorded in the
newer workspace KANBAN as run 68, not run 66. Its late geometry contracted:

```text
std 0.255, cosine 0.912, effective rank 20.08
correct/shuffled reconstruction 0.210/0.289, gap 0.079
```

Merging the PR’s placeholder record now invites a duplicate GPU run and creates a second numbering
scheme for the same experiment.

On the current GitHub base this does not produce a textual Git conflict because the newer KANBAN
work is still uncommitted. Against the actual workspace/research state, it is a semantic conflict
that must be reconciled before those changes are published.

## 8. KANBAN run 067: completed DINO unwhitened arm

This folder records the real `it7sq8nz` run and correctly concludes:

```text
LOW-RANK DECODABLE; GLOBAL COLLAPSE INDETERMINATE
```

### Benefit

It provides real integration evidence for:

- the pinned DINO adapter;
- exact feature shape/revision;
- full EGO4D training;
- checkpoints/provenance;
- numerical stability.

### Cost and current problem

- The newer workspace record calls this run 69.
- The PR leaves preregistered late-window metrics “pending” even though the W&B history is
  available and the current record already has the medians.
- The parent investigation observations/next steps and top-level Phase-1 run index are not updated.
- No `METRIC_READOUT.md` or `ANALYSIS.md` completes the KANBAN post-run protocol.

The data are useful. The folder should be reconciled, not merged twice under competing run numbers.

## 9. KANBAN run 068: DINO whitening + covariance + variance

### Scientific design

Relative to unwhitened DINO, it changes three scientific things together:

```text
whitening:  off -> on
lambda_var: 0   -> 0.5
lambda_cov: 0   -> 0.01
```

### Benefit

This is a practical “healthy geometry bundle” test. Live W&B shows that the bundle works much
better than raw no-geometry DINO on the recorded validation batch.

### Trade-off

It is not a causal whitening ablation. Because whitening, variance pressure, and covariance
pressure change together, the result cannot tell us which component caused the improvement.

That may be acceptable if the question is “does the complete bundle work?” It is not acceptable
if the result is described as “whitening alone fixed DINO.”

### Current operational problems

- The PR says **PLANNED — not yet launched**. The matching W&B run `qqozribu` is finished.
- Its guide pins `EXPECTED_GIT_SHA=083cf8a...`, then switches to branch
  `codex/task3-dino-run066` and pulls. That branch now points to `867141b...`, so following the
  guide literally fails its own SHA check.
- The guide uses the same W&B display name and checkpoint/stat paths as the completed run.
  Its safety checks may stop some overwrites, but a stale “launch this” document is still an
  expensive operational hazard.
- The 437-line guide dominates the PR’s size and maintenance burden even though it is now a
  historical procedure.

This folder must be converted from planned to completed, populated from live W&B/artifacts, and
renumbered/reconciled with the current experiment ledger.

## 10. PR structure and reviewability

The PR bundles:

- a new production encoder;
- a registry-state transition from reserved to explicit-revision to pinned;
- unit tests;
- general living-doc changes;
- a SigLIP experiment plan;
- a completed DINO experiment record;
- and a second DINO experiment plan.

That makes review and rollback harder than necessary. The title says “dino run066,” but run 066 in
the PR is SigLIP, DINO unwhitened is run 067, and DINO whitened is run 068.

This is a real architectural/MLOps concern because code readiness and experiment-record readiness
have different answers. A smaller code PR plus a separately reconciled KANBAN PR would make the
decision obvious.

## Independent Standards review

This axis is kept separate from scientific/spec correctness.

### Documented-standard violations

1. `AGENT_FILES/AGENTS.md`, the encoder knowledge README, `CODEBASE_STRUCTURE.md`, and the CLI help
   claim DINO needs an explicit revision or lacks a default while `encoders.py` pins one. This
   violates the repository’s implementation/doc synchronization rule.
2. The new DINO adapter methods do not fully follow the mandatory three-part tensor-path docstring
   convention. The literal width/token checks also conflict with the preference for configuration
   over magic numbers.
3. Several newly added test and nested helper functions lack the repository-required docstrings.
4. The three run directories use implementation-style slugs even though W&B display names are
   already known, contrary to `KANBAN/PROTOCOL.md`.
5. The completed run 067 record does not complete the post-run KANBAN updates: analysis/readout,
   parent synthesis/next steps, and Phase-1 index row.

### Code-smell judgement calls

1. Possible duplicated code between SigLIP and DINO `last_hidden_state` selection/type checks.
2. Possible speculative generality in optional adapter-factory/default-revision branches once all
   currently implemented registrations are complete.
3. Possible shotgun surgery because availability state is repeated across registry, CLI help, and
   several docs; that duplication directly produced contradictory state.

**Standards total:** five documented violations and three judgement calls. The worst standards
issue is the false pinned/default state repeated in mandatory living documentation.

## Independent Spec review

This axis uses the preserved encoder-pluggability/DINO dossiers as the originating design. That
design is explicitly a pre-join record, so later human-approved KANBAN experiments may legitimately
change the science; the deviations still need to be named.

1. **High — original paired experiment is missing/replaced.** The preserved plan specifies full
   SSv2, whitening, `lambda_recon=0.05`, variance/covariance geometry, and shape-matched
   DINO/SigLIP arms. The added run 066/067 records instead use EGO4D, no whitening,
   `lambda_recon=1`, zero geometry weights, and `M=512`. That is a substantial experiment-scope
   change, not the originally specified pair.
2. **High — pinned-default behavior is documented incorrectly.** The spec explicitly asks for the
   tested SHA to become the immutable alias default and for omission of `--revision` to resolve to
   it. The code does this; the PR’s living docs say the opposite.
3. **Medium — stable-alias proof is partial.** The test only inspects the registry value; it does
   not exercise default resolution through the public factory.
4. **Medium — DINO temporal preflight is partial.** The dossier asks for repeated-frame equality
   and frame-permutation equivariance. The new monotonic-frame fake test checks ordering but not
   those two explicit cases.

**Spec total:** four findings. The worst spec issues are the unacknowledged experiment change and
the false documentation of the pinned alias.

## Decision matrix

| Area | Verdict | Why |
|---|---|---|
| DINO adapter forward contract | **Accept after small cleanup** | Correct shape/token behavior, fail-fast checks, real CUDA evidence. |
| Immutable DINO pin | **Accept** | Necessary for reproducibility; real run resolved the same SHA. |
| Existing V-JEPA/SigLIP behavior | **Low risk** | Default remains V-JEPA; generic seam is already present. |
| Training schedule/loss/gradient routing | **No PR change** | This PR does not modify the scientific training math. |
| DINO forecasting claim | **Not established** | Every DINO run is present-only. |
| Tests | **Good but incomplete** | Full suite passes; default-alias and two temporal-contract cases should be added. |
| Architecture cost | **Accept knowingly** | Smaller encoder, twice the tokens, more downstream activation memory, no frozen temporal mixing. |
| MLOps | **Manageable but real** | Gated weights/license, separate stats/caches/checkpoints, exact version/revision requirements. |
| Living docs | **Reject as written** | They contradict the code’s pinned-default state. |
| KANBAN run records | **Reject/reconcile** | Superseded numbering, stale statuses, incomplete post-run updates, duplicate-run risk. |
| PR as one unit | **Do not merge as-is** | Code and docs have different readiness. |

## Exact changes I would require before merge

1. Update `train.py` help, `AGENT_FILES/AGENTS.md`, the encoder knowledge README, and
   `GUIDES/CODEBASE_STRUCTURE.md` to say DINO is implemented and pinned to `593171...`.
2. Add a public-factory test that omits the revision and proves both requested and resolved
   revisions equal the pinned SHA.
3. Add repeated-frame and frame-permutation DINO contract tests.
4. Bring new adapter/test docstrings into repository convention; remove or centralize duplicated
   private token-contract literals where practical.
5. Reconcile run numbering with the current workspace ledger before committing either history.
   Preserve corrections as dated history; do not delete valid observations.
6. Mark `qqozribu` completed, attach its W&B ID, late-window reading, provenance, and whitening
   fingerprint, and update parent/top-level KANBAN indexes.
7. Complete the `it7sq8nz` late-window record rather than leaving metrics pending.
8. Mark the SigLIP no-geometry configuration as already executed (`j7a3tzj5`) so nobody relaunches
   it.
9. Convert the stale run-068 launch guide into an explicitly historical, exact-commit procedure or
   replace it with a completed-run record. Do not leave “PLANNED” commands that cannot pass their
   own branch/SHA check.
10. Prefer splitting production adapter/tests from experiment-history reconciliation. If the PR
    must remain one unit, all of the above should be corrected before merge.

## Final research-state statement

The project has learned how to build a stable, decodable, reasonably high-rank **present**
representation under the right whitening/covariance/variance pressure. The latest DINO bundle is
another positive present-side result.

The project has **not** yet learned a coarse predictor that beats copy and batch-mean baselines.
No DINO result tests that question at all. The actual next scientific boundary is a matched
full-prediction comparison—ideally including V-JEPA, SigLIP, and DINO under one clean
dataset/provenance/resource recipe—or completion of the registered latent-shape sweep before that
comparison.

Merging the DINO adapter helps reach that experiment. Merging the PR’s current prose without
reconciliation makes it harder to know which experiment has already happened and what its result
was.
