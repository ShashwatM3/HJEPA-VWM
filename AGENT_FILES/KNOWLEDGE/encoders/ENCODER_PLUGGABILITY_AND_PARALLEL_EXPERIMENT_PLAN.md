# Encoder Pluggability and the First Paired Experiment

> Status: implementation and pre-join design, audited against the repository on 2026-07-14.
> The common raw-data/model/training, determinism/checkpoint, whitening/probe/artifact,
> provenance/preflight stack and pinned V-JEPA2/SigLIP2 adapters are implemented. SigLIP
> passed a real Mac/MPS smoke. CUDA/real-dataset evidence remains RunPod-only. DINO is the
> only deliberately unresolved alias until its gated lane is implemented and validated.
> The current operational state and copy/paste RunPod hand-off live at the very top of
> [`GUIDE_encoders.md`](GUIDE_encoders.md).

## Do these human-only steps first

These are the only steps a coding agent cannot safely do for you. Do not edit Python,
manifests, or model files yourself.

### Standard ViT now: no dashboard work

SigLIP 2 is public and fully configured in code. You do not need to request access, create
a Hugging Face token, or change a RunPod template. On the pod, pull the verified commit and
select it with `--encoder siglip2_vitb16`; select V-JEPA again with
`--encoder vjepa2_vitl16`. Follow the numbered **Start here — brain-dead Standard ViT
hand-off** at the top of [`GUIDE_encoders.md`](GUIDE_encoders.md) for the remaining CUDA,
real-data, whitening, resource, and W&B evidence. Do not reuse stats across aliases.

### Right now: finish and freeze the completed EGO4D build

1. The four download/chunk batches, Stage 4D, and Stage 5 are complete. Do not rerun them,
   regenerate the manifests, or rebuild the tiny subset with different selection settings.
2. In the existing `ego4d_smoke` tmux session, finish
   [`../ego4d/GUIDE.md`](../ego4d/GUIDE.md) **Stage 6**. If Paste 3 reports the historical
   `B_EMA did not update` false negative, pull commit `0af0dc7` and rerun Paste 3; no dataset
   or checkpoint cleanup is required.
3. When Stage 6 prints `Stage 6 switchability smoke passed`, stop before the EGO4D guide's
   Stage 7 real experiment. The first encoder comparison stays on SSv2 so that dataset and
   encoder are not changed in the same experiment.
4. Do not change the EGO4D tiny-subset seed or counts. If you ever do, delete and rebuild
   the tiny directory first; the current subset builder does not remove stale symlinks
   created by a different selection.
5. Keep `/workspace/ego4d_raw/manifests/selection_manifest.json` and
   `/workspace/data/ego4d/chunk_manifest.json`. Raw `video_540ss` files are transient, but
   both manifests are required to identify and validate the processed corpus.

### One time in the browser: obtain DINOv3 access

1. Sign in to Hugging Face in the browser.
2. Open
   [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
3. Read the DINOv3 license and data-sharing terms. Click the button that agrees to the
   conditions and requests access. Access is personal to the Hugging Face account and the
   request itself can only be made in a browser.
4. Open [Hugging Face token settings](https://huggingface.co/settings/tokens).
5. Click **Create new token**. Name it `hjepa-dinov3-read`, choose **fine-grained**, grant
   read access only to the DINOv3 repository, create it, and copy the value into your
   password manager. Never paste it into this repository, a prompt, a shell command that
   will be logged, or W&B.
6. In the RunPod console, open **Secrets**, click **Create Secret**, name it
   `hf_dinov3_read`, paste the token as its value, and save. In a future pod template,
   map environment variable `HF_TOKEN` to
   `{{ RUNPOD_SECRET_hf_dinov3_read }}`. A template secret does not magically appear in
   an already-running pod, and editing/restarting the current pod could disrupt EGO4D.
7. After the EGO4D build is safe, authenticate the current pod interactively if it will be
   reused:

   ```bash
   cd /workspace/hierarchal-jepa-flow-world-model
   hf auth login
   hf auth whoami
   ```

   Paste the token only into the hidden interactive prompt. A coding agent can then run the
   non-secret download/preflight commands.

Hugging Face documents both the browser-only gated-access request and `hf auth login`
[here](https://huggingface.co/docs/hub/models-gated), and recommends a fine-grained
repository-scoped token for gated production access
[here](https://huggingface.co/docs/hub/security-tokens). RunPod's secret syntax is
documented [here](https://docs.runpod.io/pods/templates/secrets).

### One time: verify W&B authentication

1. In W&B, open your profile menu, then **User Settings**.
2. If the pod is not already authenticated, create a personal API key named
   `hjepa-runpod` and store it in your password manager.
3. Prefer a RunPod secret named `wandb_hjepa` mapped to `WANDB_API_KEY` for future pods.
4. On the current pod, verify without printing the key:

   ```bash
   wandb status
   wandb login --verify
   ```

5. The paired launch must show entity `smahalanobis-uc-davis` and project `hjepa-vwm`.
   The new strict launch mode must abort rather than silently train if W&B initialization
   fails. W&B's current key and verification steps are
   [documented here](https://docs.wandb.ai/models/ref/cli/wandb-login).

### Later, at launch time: provide one stable GPU or two equivalent GPUs

1. Choose either concurrent execution on one pod with two identical GPUs/two identical
   one-GPU pods, or sequential execution on one unchanged GPU. Sequential execution is
   scientifically valid because the arms do not communicate; concurrency only saves
   wall-clock time. Two A100-class GPUs are the historical concurrent reference; smaller
   GPUs are allowed only after a common-batch preflight.
2. Pre-download DINOv3 and SigLIP 2 sequentially before launching both processes so two
   jobs do not race in the Hugging Face cache.
3. Give each arm its own W&B name, log path, and checkpoint directory. Concurrent arms
   also receive separate GPUs; sequential arms reuse the same GPU without changing the
   pod, code commit, dependency lock, CUDA stack, or resource envelope.
4. Use the same common batch size and frame microbatch in both arms. If either arm cannot
   fit, lower the physical batch for **both**. Do not quietly use naive gradient
   accumulation: the variance and covariance objectives depend on the logical batch.

Everything else in this document is coding-agent work.

## Decision

The first new-encoder pair is:

| Role | Encoder | Frozen vision parameters | Detailed feature contract at 8x256x256 |
|---|---|---:|---|
| Selected DINO | DINOv3 ViT-B/16, `facebook/dinov3-vitb16-pretrain-lvd1689m` | 85.7M | strip CLS + four registers; eight frame-major 16x16 grids -> `(B, 2048, 768)` |
| Independent control | SigLIP 2 Base/16 FixRes 256, `google/siglip2-base-patch16-256` | 85,843,200 retained patch-tower parameters | eight frame-major 16x16 patch grids -> `(B, 2048, 768)` |
| Regression control | V-JEPA2 ViT-L/16, `facebook/vjepa2-vitl-fpc64-256` | 325,971,328 | four temporal-major 16x16 tubelet grids -> `(B, 1024, 1024)` |

DINOv3-B is the correct DINO choice because it is far lighter than the present V-JEPA2-L,
keeps a 768-wide dense patch representation, and is the first DINOv3 tier with a material
quality jump over the small tiers. Meta reports 86M parameters, patch size 16, four
registers, 768 dimensions, and strong dense/video evaluations on the
[official model card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).

SigLIP 2-B is the correct non-DINO control because it is independently pretrained, strong
on semantic and dense tasks, Apache-2.0, and—critically—has the same 8x16x16x768 output
geometry. The exact background is in [`SigLIP_2_ViT-B-16.md`](SigLIP_2_ViT-B-16.md);
the DINO rationale is in [`DINOv3_ViT-B-16.md`](DINOv3_ViT-B-16.md).

All eight frames are retained. Pooling pairs of frames merely to imitate V-JEPA's four
tubelet slots would throw away temporal resolution before the learned bottleneck sees it
and would make the two frame encoders less faithful to their pretrained representations.

## Scientific scope

The first 15,000-step pair is a **present-only, whitened feature-autoencoder substrate
test on full SSv2**, using the accepted run-057 downstream recipe. It asks:

- can B and D compress/reconstruct the frozen feature field honestly;
- does the abstract code stay high-rank, spread, slot-diverse, and video-specific;
- what are the resource and stability costs of each feature space?

It does **not** prove temporal prediction quality. Both new encoders are frame-native and
receive no temporal attention inside the frozen backbone. A lower reconstruction loss is
also not automatically a better encoder: feature spaces have different semantics and
compressibility even after whitening. The first pair is a validity and substrate screen.

The current implemented "full pipeline" means Phase 1:

```text
video -> frozen E -> optional fixed whitener -> B / B_EMA
      -> present decoder D
      -> optional coarse rectified flow F_c -> predicted latent -> D
      -> losses, diagnostics, checkpoint, probes
```

Fine flow, pixel VAE, frame flow, and a frame generator are described as intended
architecture but do not exist in the repository. This plan guarantees pluggability through
the entire implemented Phase 1 and records the feature contract those future modules must
consume; it cannot truthfully certify unimplemented stages.

## Architecture: one deep encoder seam

The earlier draft proposed a root `encoders/` package. That conflicts with the
human-owned flat-layout rule in `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` and the flat
scaffolding contract in `tests/test_phase1_contract.py`. Implement one new root file,
`encoders.py`. Do not create a package tree.

The public surface is deliberately small:

```python
@dataclass(frozen=True)
class FeatureLayout:
    temporal: int
    height: int
    width: int
    order: Literal["time_y_x"]
    temporal_unit: Literal["frame", "tubelet"]
    temporal_stride_frames: int
    temporal_support_frames: int

    @property
    def n_tokens(self) -> int: ...

@dataclass(frozen=True)
class EncoderSpec:
    family: str
    repo_id: str
    requested_revision: str
    resolved_revision: str
    input_frames: int
    input_height: int
    input_width: int
    feature_dim: int
    layout: FeatureLayout
    normalization_id: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    preprocess_version: str
    inference_precision: str
    frame_microbatch: int
    attention_implementation: str
    parameter_count: int
    fingerprint: str

class FrozenEncoder(nn.Module):
    spec: EncoderSpec
    def forward(self, raw_clips: Tensor) -> Tensor: ...

def build_frozen_encoder(cfg: EncoderConfig) -> FrozenEncoder: ...
```

`FrozenEncoder.forward` accepts only canonical raw clips in `[0, 1]` with shape
`(B, T, 3, H, W)` and returns only dense detailed tokens `(B, N_e, D_e)`. It does not
return a spec object or a hypothetical token mask every batch. Every currently supported
backend is a fixed dense lattice; a mask should be added only if a real future adapter
requires it.

Private backend implementations—`_VJEPA2Adapter`, `_DINOv3Adapter`, and
`_SigLIP2Adapter`—stay inside `encoders.py`. Hugging Face output classes, special tokens,
processor quirks, and frame microbatching must never leak through the public seam.

### Responsibilities owned by the wrapper

- load the requested repository/revision into `cfg.hf_cache_dir`;
- capture the resolved immutable commit SHA and parameter count;
- keep the backend frozen and in eval mode even if a parent calls `train()`;
- register normalization mean/std as buffers and normalize in fp32 exactly once;
- own encoder inference autocast/precision for training, stats, and every probe;
- flatten frames for image encoders, microbatch them, restore time-major order, and
  concatenate only dense patch tokens;
- strip DINO CLS/register tokens exactly and assert the expected count;
- load only SigLIP's vision tower, not the 0.4B text tower;
- validate input range/shape and output `N_e`/`D_e`/finite values;
- expose one immutable resolved `EncoderSpec` and fingerprint.

For V-JEPA2 and DINOv3 the canonical normalization is ImageNet mean/std. SigLIP uses its
documented 0.5/0.5 mapping to [-1, 1]. The data layer must not know either fact.

### Resolved geometry, not duplicate configuration

`D_e`, `N_e`, temporal slots, and grid height/width are resolved properties of the
encoder. They stop being independently configurable truths in `ModelConfig`. The
bottleneck, decoder, mean tracker, whitener, stats payloads, checkpoint loader, and probes
are constructed from the same resolved spec.

Tests that must not download a model receive an explicit synthetic `EncoderSpec`. Do not
load a remote model merely to derive test geometry.

Version 1 intentionally supports square 256px inputs because all three chosen checkpoints
and the current crop path use 256x256. It must reject `H != W` or a size that is not
supported rather than pretending arbitrary rectangular preprocessing works. Internally,
`FeatureLayout.height` and `width` remain separate so downstream code no longer uses
`sqrt(N)` and can later support a real rectangular adapter.

## Configuration and provenance

Add `EncoderConfig` to the top-level config. Its user-facing selector should be stable
aliases, not raw class names:

| Alias | Family | Default repository |
|---|---|---|
| `vjepa2_vitl16` | `vjepa2` | `facebook/vjepa2-vitl-fpc64-256` |
| `dinov3_vitb16` | `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| `siglip2_vitb16` | `siglip2` | `google/siglip2-base-patch16-256` |

Required fields:

- alias/family, repository, and **requested immutable revision**;
- input frames/height/width (8/256/256 for this project);
- inference precision (`bf16` on supported CUDA, `fp32` on CPU);
- frame microbatch, positive or automatic;
- attention implementation as an adapter-owned preference;
- cache directory.

Every registry alias must ship with the exact tested 40-character Hub commit SHA as its
default revision. Omitting `--encoder-revision` in a documented command therefore means
"use that committed immutable default", never `main`. The implementation may accept `main`
only behind an explicit exploratory-smoke opt-in that real training, stats, probes, and
checkpoints reject. W&B and checkpoints record requested and resolved revisions and require
them to match.

The runtime provenance block must also include:

- git commit and dirty-tree flag;
- Python, PyTorch, Transformers, CUDA, cuDNN, GPU model;
- feature fingerprint and preprocessing version;
- dataset fingerprint;
- data/model/training/diagnostic seeds;
- trainable initialization hash;
- whitening-stat fingerprint when active;
- physical batch, frame microbatch, and number of feature rows used for statistics.

**Implemented audit note (2026-07-14):** runtime provenance now records the CUDA/cuDNN/GPU
identity and every named seed stream. Fresh whitening consumption enforces the exact run seed,
12,800-clip budget (configurable only through the explicit `--whiten-expected-clips` recipe
field), and eigensolver settings. Validation requires B>1; rank and drift share one default
cache path; drift verifies the embedded whitener hash. Explicit dataset transfer removes only
dataset-owned provenance fields and records both dataset fingerprints and the checkpoint hash.

The dependency is now pinned to `transformers==4.57.6`. Its installed source exposes
V-JEPA2, DINOv3 ViT, and SigLIP2 vision model classes, and the real pinned V-JEPA2 adapter
passed on this version. The DINOv3 and SigLIP2 lanes must validate this same pin against
their real weights. Do not change it for one lane: a dependency change invalidates every
earlier real-adapter result and requires all three preflights to be rerun.

## Determinism and fair initialization

The existing `set_seed(42)` is not enough. `build_phase1_modules` currently loads the
Hugging Face encoder before B/F/D; different backend constructors can consume different
amounts of global randomness and silently change the trainable initialization.

Implement named, independent RNG streams:

- `data_seed`: sampler order, worker seeds, crop/window/jitter;
- `model_init_seed`: B, B_EMA, F_c, and D construction inside
  `torch.random.fork_rng`, after which B_EMA is copied from B;
- `training_seed`: flow noise, tau, and condition dropout;
- `diagnostic_seed`: validation flow samples and shuffled-code permutations;
- existing SIGReg per-step generator stays isolated.

The two shape-matched new arms must have byte-identical trainable state dicts immediately
before optimization. Hash the ordered parameter names, dtypes, shapes, and bytes and store
`trainable_init_hash`. The launch preflight compares those hashes and refuses the pair if
they differ. V-JEPA has a different downstream input shape and cannot share that hash.

Build dataloaders with an explicit generator and deterministic worker initialization. Log a
hash of the first fixed sample-ID sequence and validation batch identity. Diagnostics must
not advance the training RNG; changing `diag_every` may change observation cost but must
not change training samples/noise.

## End-to-end pipeline audit and required corrections

The table is the final re-walk from source file to research verdict. "Gate" is the test or
assertion that prevents the same inconsistency from reappearing.

| Step | Current behavior | Gap or coupling | Required correction and gate |
|---:|---|---|---|
| 1 | `DataConfig.dataset_root()` selects `ssv2`, `ssv2_tiny`, `ego4d`, or `ego4d_tiny`. | Dataset name is logged, content identity is not. | Compute a deterministic dataset fingerprint before model construction. |
| 2 | The dataset sorts `.webm` and `.mp4` paths. | A path list can change while keeping the same dataset name. | Hash split, sorted relative paths, sizes/frame metadata, and the relevant manifests. |
| 3 | SSv2 tiny has `manifest.json`; EGO4D has selection, chunk, and tiny manifests. | EGO4D `chunk_manifest.json` alone does not bind the selection file, and a partial corpus looks valid by name. | Resolve the retained selection manifest and authoritative `video_540ss_manifest.csv` through explicit configurable provenance paths (defaults under `/workspace/ego4d_raw`), never by searching or guessing. EGO full fingerprint includes both source hashes, selection + final chunk hashes, and inventory; tiny includes tiny + parent hashes. Refuse final stats/probes if Stage 4D completeness gates fail. |
| 4 | `SSV2Dataset._window_indices` samples one context/future pair with stride 2 and horizon k. | Present-only mode still computes a target window it never uses. | Add a clear batch mode: present-only returns/decodes context only; full mode returns the paired target. Test both short-video clamping and EGO's 48-frame no-clamp case. |
| 5 | Decord opens one reader and decodes only requested indices. | Correct and backend-independent. | Preserve `num_threads=1` VP9 behavior and test duplicate/clamped indices. |
| 6 | Context and target are stacked, resized, cropped, and jittered together. | Correct for full prediction; docs call outputs encoder-normalized. | Preserve shared transforms; rename/document output as canonical raw [0,1]. Version the transform recipe. |
| 7 | Resize/crop use `cfg.model.h` and ignore `w`. | False rectangular generality. | For v1 assert square 256; retain separate layout H/W downstream. |
| 8 | `data._normalize_encoder` applies ImageNet statistics on CPU. | Wrong for SigLIP and privately imported by drift probe; could be applied twice. | Remove normalization from data. Wrapper normalizes exactly once in fp32. Raw loader smoke asserts [0,1]. |
| 9 | DataLoader shuffles with implicit global state. | Same seed is not a complete order contract; exact resume cannot restore progress. | Explicit generator, worker seeding, sample IDs, and resumable sampler/epoch/offset state. |
| 10 | Training moves clips to device under a single outer bf16 autocast. | Stats/probes currently encode in fp32, so training and offline feature distributions differ. | Wrapper owns `encoder_inference_precision`; every caller uses it. |
| 11 | V-JEPA returns 1024x1024 tubelet tokens. | FrozenEncoder is V-JEPA-specific and ignores `hf_cache_dir`. | V-JEPA private adapter, resolved spec, explicit cache/revision, exact legacy-output regression. |
| 12 | No DINO path exists. | CLS/register tokens would corrupt N/order if passed through. | Flatten BxT frames, DINO forward, strip 1 CLS + 4 registers, assert 256 patches/frame, reshape to 8x16x16x768 time-major. |
| 13 | No SigLIP path exists. | Loading the composite model wastes memory on the text tower; normalization differs. | For the selected FixRes checkpoint, load `SiglipVisionModel` (its published config is `model_type: "siglip"`), not the composite model or the NaFlex `Siglip2VisionModel`; assert 256 patch tokens/frame and no unexpected special token, use 0.5 normalization. |
| 14 | `ModelConfig` derives tokens from V-JEPA patch/tubelet constants. | Three sources of truth can disagree. | Resolve one `EncoderSpec` first; construct every downstream module from it. |
| 15 | Optional ZCA whitener is sized by `cfg.model.d_e`. | Stats contain only tensors; same D could still be the wrong encoder/dataset/precision. | Stats schema includes full feature/dataset fingerprints and fails before state loading on any mismatch. |
| 16 | The whitening CLI decodes future clips it discards and can encode a different precision. | Wasted I/O and distribution mismatch. | Context-only loader, common wrapper precision, deterministic transform order; record clips and token rows. |
| 17 | `FeatureMeanTracker(N,D)` tracks residual-target means. | Shape/provenance are implicit. | Build from spec, save fingerprint, and never share tracker state across feature spaces. |
| 18 | B projects D_e, reshapes using V-JEPA's four square grids, applies ConvNeXt and Perceiver latent blocks. | Hardcoded temporal/grid semantics and V-JEPA docstrings. | Use `layout.temporal/height/width/order`; assert N product; preserve output `(B, N_c=32, D_c=256)` and gradient contracts. |
| 19 | B_EMA is a frozen deepcopy initialized after module build. | Different arm construction RNG can make online weights differ. | Isolate init, copy B -> B_EMA, then compare trainable hashes across DINO/SigLIP. |
| 20 | Present-only calls E once; full mode calls E on context and target. | Training accepts only a two-tensor batch even when target is absent. | Type and test the two batch modes; never fabricate or decode a future clip in present-only. |
| 21 | Full mode constructs residual/full flow targets, noise, tau, condition dropout, and F_c output. | Uses global training RNG. | Dedicated training generator/state; exact resume restores it. Abstract shape remains encoder-independent. |
| 22 | D builds fixed 3-D tubelet codes from V-JEPA config and emits N_e x D_e. | Position helper and terminology are V-JEPA-specific. | Build deterministic codes from generic layout; time/y/x order; no learned content queries; assert exact output shape. |
| 23 | Cosine or relative-MSE reconstruction compares D output with frozen features. | "per-tubelet" language is wrong for frame patches. | Rename to detailed tokens; formulas and stop-gradient boundary stay unchanged. |
| 24 | Flow, variance, covariance, slot, SIGReg, and reconstruction losses combine with warmups. | Naive microbatch accumulation is not equivalent for batch-statistic losses. | Keep one physical logical batch for the first pair. If accumulation is later implemented, prove gradient equivalence or compute global stats across microbatches. |
| 25 | AdamW has B/F/D decay/no-decay groups; CLI lacks decoder LR and batch-size overrides. | Resource parity and exact recipe cannot be fully expressed. | Add validated `--batch-size` and `--lr-decoder`; log group names/base LRs. |
| 26 | AGC, global clip/skip, optimizer step, EMA update, and LR/EMA schedules run per step. | Most behavior is encoder-independent. | Preserve formulas and the exact dtype-rounded Stage-0 `_assert_ema_transition` check added in `0af0dc7`; add tests showing every adapter's Stage 0 invokes it and the same abstract inputs yield the same update regardless of encoder alias. |
| 27 | Diagnostics run on a fixed validation batch every 500 steps. | Diagnostic RNG is global; metrics are lost if `diag_every` is not also a log step. | Dedicated diagnostic RNG and immediate diagnostic logging, or validate divisibility. Use actual key `coarse_vs_batch_mean_ratio`. |
| 28 | Reconstruction honesty uses shuffled c and video gap. | Shuffle requires B>1 and raw loss is not cross-feature-space quality. | Assert validation batch >1; compare within-run improvement, gap/share, rank, stability, and compute—not raw recon alone. |
| 29 | W&B init exceptions silently disable tracking. | An expensive pair can run untracked. | Add strict `--require-wandb`; record resolved config after encoder construction; distinct run/group/tags. |
| 30 | Checkpoints save trainable modules, config, optimizer, mean, and whitener, excluding E. | No schema/fingerprint validation; optimizer incompatibility silently resets; intermediate step repeats on resume. | Version schema, store `next_step` and provenance, validate before loading, require explicit `--reset-optimizer`. |
| 31 | Resume builds the whitener from the external stats path before loading embedded state. | A "self-describing" checkpoint cannot resume if the external file is gone. | Inspect checkpoint first; build whitener from embedded metadata/state on resume; external stats required only for fresh runs. |
| 32 | Resume does not save Python/NumPy/Torch/CUDA RNG, sampler position, or W&B run ID. | Interrupted runs diverge and create new W&B runs. | Save/restore all RNG/sampler state and W&B ID; test interrupted vs uninterrupted equivalence. |
| 33 | `whiten_stats.py` saves non-atomic tensor stats with minimal metadata. | Partial/wrong stats can be consumed. | Atomic temp+rename payload with schema, fingerprints, counts, transforms, precision, revision, and eigensolver settings. |
| 34 | `rank_probe.py` constructs V-JEPA directly and reports `N_ctx/D_e`. | Cannot select/label/validate another feature space. | Use factory/spec, generic names, dataset fingerprint, family/revision in JSON and filenames. |
| 35 | `drift_probe.py` imports private data normalization and caches a bare feature dict. | Cache can be silently reused for the wrong encoder, manifest, transform, or dtype. | Raw clip -> factory; cache envelope with all fingerprints, storage dtype, offsets, and atomic write; reject mismatches even for an explicit path. |
| 36 | Drift checkpoint rebuild uses only legacy `ModelConfig`. | New B shape/layout cannot be reconstructed reliably. | Save/rebuild from checkpoint `EncoderSpec`; provide explicit legacy V-JEPA migration only. |
| 37 | Output is metrics/checkpoints/probe JSON/plots, not pixels. | Documentation can imply the intended frame generator exists. | Label current output honestly; future fine/pixel modules must accept the same spec or encoder-independent e/c tensors. |

After applying every correction above, re-walk the exact sequence again in integration
tests:

```text
dataset fingerprint -> sample ID -> raw context[/target] -> adapter normalization
-> dense features + resolved spec -> optional matching whitener -> B/B_EMA
-> present/full branches -> D/F_c -> every loss -> backward/AGC/optimizer/EMA
-> diagnostics -> W&B -> atomic checkpoint -> exact resume -> rank/drift probes
```

No stage may infer feature geometry independently.

### Final pre-RunPod audit closure (2026-07-14)

The implementation now closes the last four cross-cutting gaps found during the second
complete re-walk:

- dataset identity schema v2 hashes each clip's sorted relative path, resolved byte size,
  and decoded frame count, and records split frame totals/ranges;
- current checkpoints validate the saved sampler state against their own next step,
  dataset count, and physical batch before state mutation, then feed that exact saved
  epoch/offset into the first resumed dataloader;
- explicit resource preflight uses synchronized CUDA events, measures the training step
  separately from diagnostics, and reports encoder-only/total peak memory plus examples,
  input frames, and detailed tokens per second;
- the generic rank probe reconstructs only `FeatureLayout`; frame layouts report
  per-frame token norms and effective rank before frame rows are temporally concatenated.

Offline regressions cover every closure, including an actual resumed sampler suffix and a
same-path/same-size inventory whose frame count changes. These changes alter provenance,
so whitening stats, feature caches, checkpoints, and resource reports made before this
schema must not be reused for the final pair.

## Whitening and artifact identity

Whitening is one artifact per:

```text
encoder repo + resolved revision + token selection + preprocess version
+ input geometry + encoder precision + frame microbatch + attention implementation
+ dataset + split + dataset fingerprint
```

The stats filename should include a short fingerprint, but correctness must depend on
metadata validation, not the filename. DINO and SigLIP have the same shape and must still
reject each other's stats. Old V-JEPA whitening statistics are not valid for a refactored
run if encoder inference precision or preprocessing changes.

On SSv2 full, generate a deterministic dataset identity from `labels.json` plus the sorted
resolved clip inventory because there is no full-subset manifest. On SSv2 tiny include
`manifest.json`. On EGO4D full include both the selection and final chunk manifests plus
the file inventory. On EGO4D tiny include its manifest and the parent EGO identity.

Statistics are fitted on the training split only. Validation/test always reuse training
statistics. Save the stats payload as a W&B artifact and record its artifact ID in the run.

## Checkpoint contract

New checkpoints require a schema version and these top-level conceptual blocks:

- `next_step`, schedule horizon, and completed-update count;
- trainable state dicts and optimizer state;
- resolved encoder spec/fingerprint, with no frozen weights;
- dataset and preprocessing fingerprint;
- whitening and feature-mean state/fingerprint;
- all RNG states and sampler epoch/offset;
- W&B run ID and runtime provenance;
- trainable initialization hash.

Compatibility is checked before any tensor state is loaded. A feature-space mismatch,
layout mismatch, architecture mismatch, or preprocessing mismatch is a hard error. Dataset
transfer, optimizer reset, or legacy migration requires an explicit flag and is logged as
such; it is never inferred from an exception.

The first paid pair starts from scratch and does not resume. Resume correctness must still
be implemented and tested before pluggability is declared complete because it is part of
the current execution flow.

## Diagnostics and interpretation

Keep the existing representation-health and honesty signals:

- `c_effective_rank`, `c_std_mean`, `c_dead_dim_frac`;
- `c_slot_diversity_rank_centered`;
- `c_cross_video_cosine`;
- `c_attn_entropy` and `c_attn_entropy_min`;
- `L_recon_present`, `L_recon_shuffled_c`, `L_recon_video_gap`;
- stability, AGC, gradient, and skipped-step metrics;
- for full mode, `coarse_vs_copy_ratio` and `coarse_vs_batch_mean_ratio`.

Add resolved encoder information to W&B config/summary, not repeated scalar history.
Measure feature norms/rank and adjacent-versus-distant temporal drift offline or at Stage 0.
For frame encoders, also report per-frame token statistics before time concatenation.

Timing belongs in a dedicated preflight or low-cadence summary using CUDA events. Do not
synchronize every training step merely to log `encoder_seconds`; that would change the
run being measured. Report encoder-only peak memory, total training peak memory, examples/s,
frames/s, and tokens/s.

Cross-encoder reading rules:

- raw `L_recon` values are not an absolute quality ranking;
- use loss improvement from initialization, shuffled-code gap, conditioned share, code
  geometry, stability, and compute together;
- raw feature effective rank has a ceiling of 768 for the new pair and 1024 for V-JEPA;
  normalize or report both absolute and fraction-of-ceiling;
- temporal drift differs semantically between frame patches and tubelets; compare curve
  shape within an encoder, not only magnitude across encoders;
- present-only success advances an encoder to full prediction; it does not declare a
  temporal winner.

## Test matrix

### Offline unit tests

- FeatureLayout products/order and rectangular internal representation.
- EncoderSpec canonical serialization/fingerprint changes for every identity-bearing field.
- Synthetic backends for V-JEPA-shaped and frame-shaped outputs.
- Sticky eval/freeze and zero trainable encoder parameters.
- Raw [0,1] range validation and exactly-once normalization.
- DINO special-token stripping and time-major reshape.
- SigLIP no-special-token reshape and vision-only load seam.
- Frame microbatch equivalence to unmicrobatched evaluation.
- B and D shapes for `(4,16,16,1024)` and `(8,16,16,768)`.
- Generic fixed-position buffer remains non-trainable and cannot emit position-specific
  content from identical latent values.
- Present-only decodes/encodes exactly one context and never accesses target.
- Full mode encodes context and target exactly once each.
- Every adapter's Stage 0 preserves and invokes the exact dtype-rounded `B_EMA` transition
  assertion for both successful and skipped optimizer steps.
- All existing gradient boundaries and loss tests.
- Stats/cache/checkpoint mismatch rejection.
- Optimizer reset requires an explicit flag.
- Save after step N resumes at N+1.
- Interrupted/resumed deterministic test restores RNG and data position.
- DINO/SigLIP shape-matched trainable state hashes are identical.
- Diagnostics do not perturb training RNG and are logged at any legal cadence.

### Real-adapter integration tests on an authenticated GPU

- One fixed raw clip through every adapter.
- Expected finite shape, dtype, order, parameter count, resolved commit, zero trainables.
- DINO exact CLS/register strip.
- SigLIP exact vision-only memory footprint and patch count.
- V-JEPA new wrapper vs legacy wrapper within documented numerical tolerance.
- One batch of `ssv2_tiny` and one batch of `ego4d_tiny` through every adapter.
- Whitening fit/load/forward for each encoder and both tiny datasets, including at least
  one full-prediction forward/backward/checkpoint/probe round-trip with whitening active.
- 100-step present-only smoke and 100-step full-prediction smoke for each adapter.
- Save/resume/probe the produced checkpoints.
- CPU synthetic Stage 0 uses `torch.rand` raw pixels, not normalized `torch.randn` pixels.

### Scientific validity gates

Before 15,000 steps:

1. clean committed code and recorded commit;
2. final dataset fingerprint and EGO4D completeness if EGO is selected;
3. immutable encoder revisions and successful authenticated downloads;
4. two new stats artifacts, each fingerprint-matched;
5. all offline and real-adapter tests pass;
6. common native physical batch fits both arms;
7. trainable initialization hashes match;
8. first sample-order/validation hashes match;
9. W&B strict init succeeds;
10. resolved-config diff is restricted to encoder identity, encoder stats/fingerprint,
    run name, checkpoint/log path, and GPU assignment.

## Implementation sequence

1. Add failing contract tests and `EncoderConfig`/`FeatureLayout`/`EncoderSpec`.
2. Implement the single deep `encoders.py` module, two-layout fake fixtures, and V-JEPA
   regression adapter first; freeze this public seam before either new real adapter.
3. Move normalization to the wrapper and make the data layer emit canonical raw clips;
   update the EGO4D Stage 6 dataloader smoke in the same commit so it asserts `[0,1]`
   instead of the pre-refactor ImageNet-normalized range.
4. Make present-only loading genuinely context-only.
5. Resolve spec before constructing B/D/tracker/whitener; remove downstream geometry
   inference.
6. Add DINO and SigLIP as independent private-adapter lanes. They may run concurrently only
   in separate worktrees/branches based on the frozen common seam; neither lane depends on
   the other, and each owns its exact preprocessing/token rules and real-adapter preflight.
7. Isolate RNG streams and add trainable/data hashes.
8. Version stats, checkpoints, caches, and dataset identity; fix strict resume.
9. Migrate train, Stage 0, diagnostics, rank, drift, CLI, W&B, and docs.
10. Run the common full test matrix and refactored V-JEPA regression without requiring
    either new encoder's weights.
11. Integrate both adapter lanes, rerun the entire three-real-adapter matrix, and invalidate
    any earlier evidence affected by shared-code or dependency changes.
12. Fit separate DINO/SigLIP SSv2 whitening artifacts, find one common resource envelope,
    compare provenance, and run paired smokes.
13. Launch the paid pair concurrently or sequentially only after all validity gates pass.

When implementation lands, update the human architecture documents at the same time:

- replace the ImageNet-only normalization invariant with "adapter-specific canonical
  normalization applied exactly once";
- document that `D_e` and detailed-token geometry are resolved encoder properties while
  `N_c`/`D_c` and gradient-routing invariants remain locked;
- add `encoders.py` to the flat file map;
- distinguish current implemented Phase 1 from intended future stages.

The user's request for encoder switching authorizes those encoder-dimension/layout
changes. It does not authorize changing abstract latent geometry, loss formulas, gradient
routing, or the accepted run-057 recipe.

## Exact first paired experiment

Dataset: full `ssv2`.

Mode: present-only reconstruction.

Steps: 15,000 from scratch.

Seed family: 42 with isolated named streams.

Detailed input: all eight 256px frames.

Common downstream shape: `N_e=2048`, `D_e=768`, `N_c=32`, `D_c=256`.

Shared accepted run-057 recipe:

```text
horizon_k=12                 # recorded but target is not decoded in present-only
lambda_recon=0.05
lambda_recon_pred=0.0
recon_loss_mode=cosine
recon_residual_target=false
whiten_features=true         # separate matched stats per encoder
lambda_var=0.5
lambda_cov=0.01
lambda_sigreg=0.0
lambda_slot=0.0
recon_warmup_steps=2000
bottleneck_latent_blocks=3
decoder_dim=512
decoder_blocks=4
n_c=32
lr_bottleneck=1e-4
lr_coarse_flow=1e-4          # inactive but held
lr_decoder=1e-4
log_every=50
diag_every=500
```

Arm A changes only the encoder/stat/run identity to DINOv3-B. Arm B changes only those
fields to SigLIP 2-B. The common batch and frame microbatch are chosen by the resource
preflight and recorded. Do not pool frames or change decoder width to rescue one arm.

Use run 057 (`cdvp6hou`) as historical context, not as a direct numeric baseline: the
refactor standardizes encoder precision and provenance, and V-JEPA has a different feature
shape. The proper V-JEPA control is a fresh refactored rerun with new matching stats.

### Paired verdict

Validity fails if either arm has a mismatched dataset/sample/init hash, wrong token shape,
wrong stats fingerprint, non-strict W&B, sustained instability, or asymmetric resource
change.

An arm passes the substrate screen if late training is stable, reconstruction improves,
`L_recon_video_gap` remains materially positive, the shuffled-code loss remains worse,
and the abstract geometry does not collapse. Compare the two arms on:

1. video-conditioned reconstruction improvement/share;
2. normalized effective rank, centered slot rank, spread, and cross-video cosine;
3. stability and convergence trajectory;
4. peak memory and throughput;
5. raw encoder rank/norm and temporal drift probes;
6. consistency on `ego4d_tiny` smoke/probe data.

If these disagree, there is no single scalar winner; advance both to the full-prediction
smoke and use the temporal copy gate as the deciding evidence.

## EGO4D and full-prediction follow-up

EGO4D is highly relevant, but it must not be entangled with the first encoder-only pair.
Its egocentric motion was selected to challenge the SSv2 copy-ratio failure. The staged
research sequence is:

1. complete EGO4D Stages 4D-6 without encoder refactor;
2. implement pluggability and pass all three encoders on both tiny datasets;
3. run the SSv2 DINO/SigLIP present-only pair;
4. fit separate EGO4D whitening stats only after the full corpus fingerprint is final;
5. run full-prediction wiring smokes on `ego4d_tiny`;
6. run the eventual full EGO4D prediction comparison with a refactored V-JEPA control.

A DINO-vs-SigLIP full experiment changes frozen temporal semantics: both are framewise,
whereas V-JEPA is tubelet/video-aware. Therefore a two-arm new-encoder result cannot
isolate whether a failure is common to frame encoders. The proper follow-up is three arms
if budget permits, or DINO/SigLIP in parallel plus V-JEPA sequentially on the same commit,
dataset fingerprint, recipe, and common resource policy.

The last canonical full-prediction result, run 037, produced a healthy representation but
did not beat copy. Its recipe is a historical starting point, not automatically the final
EGO recipe. The full-run launch should be approved as a new KANBAN investigation after the
substrate pair; the code must already support it without another encoder-specific change.
Success still requires both:

```text
coarse_vs_copy_ratio <= 0.70
coarse_vs_batch_mean_ratio <= 0.50
```

at stable late diagnostic points, alongside healthy c geometry and reconstruction honesty.

## Completion criterion

Encoder pluggability is complete only when:

- one flat encoder module supports all three real adapters through the same raw-clip and
  resolved-spec interface;
- data, whitening, B, D, diagnostics, checkpoints, resume, rank, drift, and W&B contain no
  V-JEPA-only geometry or preprocessing assumption;
- stats/checkpoints/caches reject wrong feature or dataset identities before use;
- present-only and full Phase 1 work on both SSv2 and EGO4D tiny;
- the refactored V-JEPA path passes numerical non-regression;
- DINO/SigLIP trainable initialization and data-order parity are proven, not assumed;
- the entire offline and real-adapter matrix passes;
- all human architecture/index documents describe the shipped implementation accurately;
- the paired launch command differs only in the explicitly allowed arm fields.

At that point a new encoder requires one private adapter plus its contract tests and
registry entry—not edits scattered through data, models, training, stats, probes, and
checkpoints.
