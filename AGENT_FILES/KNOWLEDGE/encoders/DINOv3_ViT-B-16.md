# DINOv3 ViT-B/16 (LVD-1689M)

**Research date:** 2026-07-13

**Recommended checkpoint:** `facebook/dinov3-vitb16-pretrain-lvd1689m`

**Decision:** use this as the DINO arm of the first encoder experiment.

## Executive decision

DINOv3 ViT-B/16 is the best quality/weight trade-off for this repository. It is the
latest official Meta DINO generation found as of the research date, has 85.7M parameters,
and provides strong frozen dense features without inheriting the 300M-parameter cost of
the current V-JEPA2 ViT-L or DINOv3 ViT-L. The model is not a video transformer: it must
encode each frame independently and the HJEPA pipeline must carry the frame lattice and
temporal ordering explicitly.

The first experiment should keep all eight input frames. At 256x256, each frame produces
a 16x16 patch grid, so the canonical DINO feature lattice is:

```text
(B, T=8, H_p=16, W_p=16, D=768)
             -> time-major flatten -> (B, 2048, 768)
```

Only patch tokens belong in the reconstruction target. The class token and four register
tokens must be removed.

## Why ViT-B/16, not another DINOv3 size

The official DINOv3 family spans ViT-S/16 (21M), S+/16 (29M), B/16 (86M), L/16
(300M), H+/16 (840M), and 7B/16 (6.716B). All use patch size 16. Meta reports the
following frozen-backbone results for the web-data models; higher is better except NYU
depth.

| Model | Params | Width | Layers | ADE20k | NYU depth | DAVIS | NAVI | SPair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv3 ViT-S/16 | 21M | 384 | 12 | 49.5 | 0.403 | 72.7 | 56.3 | 50.4 |
| **DINOv3 ViT-B/16** | **86M** | **768** | **12** | **58.5** | **0.373** | **77.2** | **58.8** | **57.2** |
| DINOv3 ViT-L/16 | 300M | 1024 | 24 | 63.1 | 0.352 | 79.9 | 62.3 | 61.3 |

Sources: the [official DINOv3 model card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
and [reference repository](https://github.com/facebookresearch/dinov3).

ViT-B is the knee of this curve for HJEPA:

- It is about 71% smaller by parameter count than the current 0.3B V-JEPA2 ViT-L.
- It is large enough to retain strong dense, correspondence, and video-tracking features;
  ViT-S gives up a material amount of quality on every relevant dense benchmark.
- DINOv3 ViT-L is only a modest dense-benchmark improvement over B but is 3.5x larger.
  Because it must run on eight frames, choosing L would preserve the old width while
  missing the user's requirement to avoid an extremely heavy encoder.
- The other experimental arm, SigLIP 2 ViT-B/16, also emits 768-dimensional 16x16 patch
  features. The two new runs therefore have identical native tensor shapes and comparable
  encoder scale.

There is relevant world-model precedent: Meta's JEPA-WMs repository uses DINOv3
ViT-L/16 at 256x256 for DROID and RoboCasa world models. That validates the *family* for
JEPA-style physical prediction, although it does not prove that the smaller B model or
this repository's objective will win. See the [official JEPA-WMs model table](https://github.com/facebookresearch/jepa-wms#pretrained-models).

## Exact architecture and feature contract

| Property | Value |
|---|---|
| Architecture | Vision Transformer Base |
| Checkpoint | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Parameters | 85.7M/approximately 86M |
| Patch size | 16x16 |
| Hidden width | 768 |
| Transformer depth | 12 |
| Attention heads | 12 |
| FFN | MLP, ratio 4 (intermediate width 3072) |
| Position encoding | 2-D RoPE |
| Special tokens | 1 class token + 4 register/storage tokens |
| Patch tokens at 256x256 | 256 per frame |
| Full sequence at 256x256 | 261 per frame before special-token removal |
| Native video awareness | None; frames are independent |
| Recommended precision | frozen inference under bf16 autocast, with feature whitening in fp32 |
| Input normalization | ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |
| License/access | gated checkpoint under the DINOv3 License |

The architecture values are defined in Meta's
[reference ViT implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py),
and the special-token contract is documented in the
[official model card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
The model supports images whose height and width are multiples of 16; 256 is therefore a
native, non-cropped size.

With Hugging Face Transformers, use the bare backbone output and slice patch tokens by
metadata, not by an unexplained magic constant:

```python
outputs = model(pixel_values=frames)
n_special = 1 + model.config.num_register_tokens
patch_tokens = outputs.last_hidden_state[:, n_special:, :]
```

The implementation should assert `num_register_tokens == 4`, `patch_tokens.shape[1] ==
16 * 16`, and `patch_tokens.shape[-1] == 768` for this pinned checkpoint. Do not use
`pooler_output`: a global class embedding cannot support the current per-position feature
decoder.

## What DINOv3 learned

DINOv3 ViT-B is distilled from the 7B teacher trained on LVD-1689M, a curated set of
1.689B web images. The official training recipe combines:

- DINO self-distillation with multi-crop views;
- iBOT masked-image modeling;
- KoLeo regularization on class tokens;
- Gram anchoring, introduced to prevent dense feature maps from degrading during long
  training;
- post-training distillation from the frozen ViT-7B teacher.

The [DINOv3 paper](https://arxiv.org/abs/2508.10104) positions the model specifically as
a general frozen backbone with strong dense features. That is a good match for HJEPA's
feature reconstruction target. It is not direct evidence of motion modeling: all
pretraining inputs to this checkpoint are still images.

## Difference from the current V-JEPA2 encoder

| Contract | Current V-JEPA2 ViT-L | DINOv3 ViT-B/16 |
|---|---|---|
| Modality | video/tubelets | individual frames |
| Parameters | 0.3B | 85.7M |
| Width/depth | 1024 / 24 | 768 / 12 |
| Temporal interaction | inside the frozen encoder | none inside the encoder |
| Input clip | `(B,8,3,256,256)` | reshape to `(B*8,3,256,256)` |
| Temporal lattice | 4 tubelet slots | 8 frame slots |
| Spatial grid | 16x16 | 16x16 |
| Patch tokens/clip | 1024 | 2048 |
| Raw feature shape | `(B,1024,1024)` | `(B,2048,768)` |
| Normalization | ImageNet | ImageNet |
| Special-token handling | `get_vision_features` already returns patch/tubelet tokens | remove class + 4 registers |

The current V-JEPA2 checkpoint is documented as a 0.3B, 24-layer, width-1024 model in
its [official Hugging Face card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)
and [configuration](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/raw/main/config.json).

Two implications are load-bearing:

1. DINO's lower parameter count does **not** mean the downstream activation footprint is
   automatically smaller. It produces twice as many patch tokens for the eight-frame
   clip. The bottleneck and decoder must be profiled at native 2048-token length.
2. DINO has no tubelet motion features. Any claim that it replaces V-JEPA2 for full
   prediction must be tested on future prediction and not inferred from autoencoder loss.

## Recommended temporal treatment

For the first paired experiment, preserve all eight frames and use no learned temporal
adapter. Flatten the lattice time-major and attach explicit layout metadata. This is the
cleanest comparison because DINOv3-B and SigLIP2-B then have exactly the same
`(T,H,W,D)=(8,16,16,768)` contract.

Do **not** silently force DINO into V-JEPA2's 1024-token contract. The available shortcuts
all change the scientific question:

| Shortcut | Benefit | Scientific cost |
|---|---|---|
| Keep frames 1,3,5,7 only | 1024 tokens | discards half the observations |
| Mean adjacent frame pairs | 1024 tokens | erases within-pair change and blurs motion |
| Learned stride-2 temporal adapter | 1024 tokens and can retain change | adds trainable capacity before B; no longer an encoder-only replacement |
| Keep all 8 frames | no discarded observations; matched DINO/SigLIP arms | doubles N relative to V-JEPA2 and raises downstream memory |

The all-frame choice should be revisited only if the Stage-0 memory benchmark fails.
Then the fallback should be a *shared, explicitly named* temporal adapter used in both
frame-encoder arms, with an additional no-adapter control later.

## Expected strengths for this project

- Strong spatially dense patch features are aligned with the fixed-position feature
  decoder and the reconstruction objective.
- Gram anchoring and large-scale distillation make dense maps a first-class property,
  not an accidental intermediate representation.
- Frozen DINOv3 features are officially evaluated on DAVIS tracking and dense
  correspondence tasks, which are closer to video continuity than classification alone.
- ViT-B leaves substantially more memory for the bottleneck, decoder, and future
  FineFlow work than a framewise ViT-L.
- ImageNet normalization matches the current data path, so V-JEPA2 and DINO can be made
  numerically equivalent at preprocessing while normalization ownership is moved behind
  the encoder interface.

## Expected weaknesses and failure modes

- No temporal attention, tubelet embedding, or motion-aware pretraining. Two frame orders
  with the same frame set are indistinguishable unless the downstream model receives and
  uses time positions.
- Twice the patch-token count of the current eight-frame V-JEPA2 path.
- DINO and V-JEPA2 feature spaces have different width, covariance spectrum, norm, and
  semantics. V-JEPA whitening statistics and reconstruction baselines cannot be reused.
- Register/class tokens can accidentally contaminate the spatial lattice if slicing is
  wrong. Shape-only tests will not catch a five-token offset unless token provenance is
  tested.
- Raw reconstruction losses are not a quality comparison across encoders, even when both
  use cosine loss. Each encoder defines a different target space.
- The model is gated and uses a custom license. A coding agent cannot accept those terms
  for the user.

## Required access and reproducibility actions

The user must:

1. Open the [official checkpoint page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
   review the terms, and request/accept access while logged into the Hugging Face account
   used on the training pod.
2. Review the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
   It is not the MIT license used by the current V-JEPA2 checkpoint. It includes
   attribution/redistribution obligations and restricted-use terms.
3. Put a read-capable Hugging Face token on the pod through the normal secret mechanism;
   do not commit it or paste it into a launch command captured by W&B.
4. Pin both the model revision/commit SHA and the Transformers version. The official
   DINOv3 Hugging Face integration requires Transformers 4.56.0 or newer.
5. Record the resolved revision, model config, preprocessing config, and license identifier
   in the W&B run config.

## Preflight acceptance checks

Before an expensive run, the implementation must prove:

- frozen/eval status and zero encoder gradients;
- exact output `(B,2048,768)` for an eight-frame 256x256 clip;
- class/register tokens are absent;
- time-major token order is deterministic and reconstructible from metadata;
- the same crop/jitter parameters are applied to every frame in a clip;
- encoder-specific normalization is applied exactly once;
- independent whitening statistics have dimension 768 and carry the encoder revision;
- repeated input frames produce repeated per-frame DINO patch features before temporal
  positions are added downstream;
- permuting frames permutes feature time slots rather than changing per-frame content;
- peak memory and examples/second are logged on the target A100.

## Bottom line

Use DINOv3 ViT-B/16, keep all eight frame grids, strip all five special tokens, compute a
new 768-dimensional whitening transform, and treat the result as a frame-feature lattice
rather than pretending it is a tubelet encoder. The first present-only experiment can
establish whether HJEPA can encode and honestly reconstruct this feature space. Only a
later matched full-prediction run can establish whether the loss of frozen temporal
modeling is acceptable.
