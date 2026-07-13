# SigLIP 2 ViT-B/16 at 256px

**Research date:** 2026-07-13

**Recommended checkpoint:** `google/siglip2-base-patch16-256`

**Decision:** use this as the non-DINO/non-V-JEPA control arm.

## Executive decision

SigLIP 2 ViT-B/16 is the best independent standard-ViT control for the first encoder
experiment. Its vision tower is an approximately 86M-parameter, 12-layer, width-768,
patch-16 Transformer. At 256x256 it emits 256 unpooled patch tokens per frame, exactly
matching DINOv3 ViT-B/16's patch lattice and width after DINO's special tokens are
removed.

That creates an unusually clean pair:

```text
DINOv3-B:  (B, 8, 16, 16, 768) -> (B, 2048, 768)
SigLIP2-B: (B, 8, 16, 16, 768) -> (B, 2048, 768)
```

The backbone size, spatial resolution, frame count, token count, hidden width, and
downstream HJEPA architecture can all remain fixed. The main experimental difference is
the representation learned by image-only self-supervision/distillation versus
vision-language, localization, captioning, self-distillation, and masked-patch training.

## Why this encoder

The user asked for a good standard encoder that is neither DINO nor V-JEPA and avoids
both a weak tiny model and a heavy giant model. The practical candidates were:

| Candidate | Approx. vision params | Native 256px spatial output | Strength | Reason not selected |
|---|---:|---|---|---|
| Vanilla supervised ViT-B/16 | 86M | 16x16, width 768 | simplest control | much older and weaker dense-feature pretraining |
| MAE ViT-B/16 | 86M | 16x16, width 768 | image-only reconstruction prior | older; less semantic/localization strength out of the box |
| ConvNeXt V2 Base | 89M | hierarchical, final stride 32 | strong non-transformer control | final map is 8x8 at 256px; choosing an intermediate stage changes width/semantics |
| SigLIP 2 ViT-L/16 | 303M | 16x16, width 1024 | very strong | framewise cost is in the same heavy class as current V-JEPA2-L |
| **SigLIP 2 ViT-B/16** | **86M** | **16x16, width 768** | modern semantic and dense features | **selected** |

SigLIP 2 specifically improved the weak point of older CLIP-style encoders: the paper
adds self-distillation, masked prediction, and location/caption decoder losses and reports
large gains on localization and dense prediction. It also emphasizes strong B-sized
models. See the [SigLIP 2 paper](https://arxiv.org/abs/2502.14786) and the
[official checkpoint](https://huggingface.co/google/siglip2-base-patch16-256).

ConvNeXt V2 remains the best later *architectural stress test* because it would prove the
interface supports hierarchical CNN feature maps, but it is a less controlled first
parallel run. DINOv3-B and SigLIP2-B being shape-matched removes a large source of
confounding.

## Exact vision-tower contract

| Property | Value |
|---|---|
| Architecture | fixed-resolution SigLIP 2 ViT-Base |
| Checkpoint | `google/siglip2-base-patch16-256` |
| Vision parameters | approximately 86M |
| Patch size | 16x16 |
| Hidden width | 768 |
| Transformer depth | 12 |
| Attention heads | 12 |
| FFN width | 3072 |
| Activation | `gelu_pytorch_tanh` |
| Position encoding | learned patch positions for the fixed-resolution tower |
| Special tokens in unpooled output | none; patch sequence only |
| Patch tokens at 256x256 | 256 per frame |
| Native video awareness | none; frames are independent |
| Input normalization | mean `(0.5,0.5,0.5)`, std `(0.5,0.5,0.5)` |
| License | Apache-2.0 |

The Base architecture defaults are defined in the official
[Transformers SigLIP configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/configuration_siglip.py).
The checkpoint's published preprocessing config fixes 256x256 input and 0.5/0.5
normalization. The [Transformers SigLIP 2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2)
supports loading the vision tower without a text forward pass:

```python
from transformers import AutoProcessor, SiglipVisionModel

processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-256")
model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-256")
outputs = model(**processor(images=frames, return_tensors="pt"))
patch_tokens = outputs.last_hidden_state
```

The selected 256px FixRes repository is intentionally backward-compatible with the
Transformers SigLIP implementation: its published config uses `model_type: "siglip"` and
its processor uses `SiglipImageProcessor`. Therefore this exact checkpoint must use
`SiglipVisionModel`. `Siglip2VisionModel` is the NaFlex-style interface that requires patch
masks and spatial-shape inputs; using it here would mismatch the selected checkpoint's
published config. The authenticated adapter preflight must verify this again against the
exact pinned Transformers release and model revision.

For this checkpoint, the adapter must assert 256 patch tokens and width 768. Do not use
`pooler_output`: it is designed to make a global image embedding and destroys the spatial
lattice required by HJEPA's feature reconstruction.

### Storage versus runtime parameters

The Hugging Face repository packages both the text and vision towers in a roughly 1.5GB
checkpoint file. `SiglipVisionModel` instantiates and runs only the vision tower, but the
pod may still need to download/cache the combined safetensors file. Distinguish these in
resource reports:

- **runtime frozen vision parameters:** approximately 86M;
- **Hub download/cache footprint:** the full combined checkpoint, approximately 1.5GB in
  fp32 safetensors.

Do not load `AutoModel` for the experiment: that constructs the text tower even though the
pipeline never supplies text.

## What SigLIP 2 learned

SigLIP 2 starts from the sigmoid image-text objective and adds several signals during
pretraining:

- multilingual image-text matching;
- captioning and location-aware decoder objectives;
- image-only self-distillation;
- masked patch prediction;
- online/active data curation and distillation for the smaller B models.

The authors report that these changes improve zero-shot classification, retrieval,
localization, and dense prediction over SigLIP at matching scales. Frozen dense probes in
the paper show gains on segmentation, depth, and surface normals. This makes SigLIP 2 a
much stronger spatial-token control than vanilla CLIP while keeping a clearly different
pretraining prior from DINOv3.

## Difference from DINOv3-B and V-JEPA2-L

| Contract | V-JEPA2 ViT-L | DINOv3 ViT-B/16 | SigLIP 2 ViT-B/16 |
|---|---|---|---|
| Pretraining modality | video | still images | image-text + image-only auxiliaries |
| Parameters | 0.3B | 85.7M | ~86M vision tower |
| Width/layers | 1024/24 | 768/12 | 768/12 |
| Temporal interaction | tubelet transformer | none | none |
| Tokens for 8x256px clip | 1024 | 2048 patch tokens | 2048 patch tokens |
| Special tokens to remove | none in current feature API | class + 4 registers | none from `last_hidden_state` |
| Input normalization | ImageNet | ImageNet | 0.5 mean / 0.5 std |
| Representation emphasis | motion/video semantics | image structure and dense correspondence | language-aligned semantics + localization/dense features |
| License | MIT | custom gated DINOv3 | Apache-2.0 |

The normalization difference is the most immediate engineering trap. If the existing
dataset continues to emit ImageNet-normalized tensors, SigLIP 2 will receive the wrong
distribution. Shared augmentation and encoder-specific normalization must be separated.

## Recommended feature extraction for HJEPA

1. Decode and apply the same random crop/color-jitter parameters across every frame in a
   clip.
2. Keep the augmented clip in a canonical raw float range `[0,1]`.
3. Inside the SigLIP adapter, normalize with mean/std 0.5.
4. Reshape `(B,T,C,H,W)` to `(B*T,C,H,W)` and encode in one batch or deterministic
   microbatches.
5. Use `last_hidden_state`, reshape to `(B,T,16,16,768)`, and flatten time-major.
6. Return patch tokens only; expose layout and preprocessing metadata through the wrapper's
   immutable `EncoderSpec` rather than changing the per-forward return type.
7. Apply fixed offline whitening computed specifically for this encoder/revision.

Use all eight frames and no temporal adapter in the first run, for the same reason as the
DINO arm: this preserves the observations and holds both frame-encoder arms to the same
contract.

## Expected strengths for this project

- Perfect shape match with DINOv3-B makes it a scientifically clean paired control.
- Strong object semantics may help the abstract bottleneck retain action-relevant entities
  in Something-Something V2 clips.
- Localization and masked-patch objectives make the unpooled tokens more plausible
  reconstruction targets than older language-only contrastive encoders.
- The 86M vision tower leaves room for native 2048-token downstream processing.
- Apache-2.0 and an ungated checkpoint simplify automation and reproducibility.

## Expected weaknesses and failure modes

- Like DINO, it has no temporal attention or motion pretraining. It cannot replace
  V-JEPA2's tubelet dynamics by itself.
- Language alignment may suppress texture or low-level changes that are useful to a
  feature autoencoder/world model; good semantic probes do not guarantee low HJEPA loss.
- The attention pooling output is attractive but wrong for this use case. Using it would
  turn the experiment into global-image reconstruction.
- Its preprocessing is not the current ImageNet normalization. Double normalization or
  accidental reuse of DINO preprocessing will invalidate the run.
- The combined Hub file can make disk/cache measurements look much larger than the actual
  vision tower.
- No class token means a generic adapter must not assume every ViT begins with one.
- As with DINO, raw reconstruction loss is encoder-relative. It cannot by itself say which
  backbone is a better world-model representation.

## Preflight acceptance checks

Before launch, prove:

- `SiglipVisionModel`, not the full text+vision `AutoModel`, is instantiated for this
  FixRes checkpoint;
- encoder parameters are frozen, the module is in eval mode, and gradients stay absent;
- one 256px frame returns `(B,256,768)` patch features;
- an eight-frame clip returns canonical `(B,2048,768)` features;
- no pooled/global token is present in the reconstruction target;
- the 0.5/0.5 normalization is applied once and only once;
- frame permutation only permutes time slots before downstream position encoding;
- whitening stats are newly computed and fingerprinted as SigLIP2-B/16-256;
- peak GPU memory, encoder time, total step time, and cache footprint are logged separately;
- output feature norms/rank are profiled before training and compared with the DINO and
  V-JEPA probes.

## What this control can and cannot decide

The paired present-only run can decide whether the current HJEPA bottleneck/decoder can
form a high-rank, video-specific, honestly decoded code from SigLIP 2 patch features, and
compare that behavior with a shape-matched DINO substrate.

It cannot decide whether SigLIP 2 predicts the future better. The current research history
already shows that excellent present representation geometry does not imply a predictor
that beats copy. A later matched full-prediction run and encoder-quality probes are
required.

## Bottom line

Use `google/siglip2-base-patch16-256` through `SiglipVisionModel`, preserve all 2048 patch
tokens, own its distinct normalization inside the adapter, and compute separate whitening
statistics. It is a stronger and cleaner first control than vanilla ViT/MAE or ConvNeXt
because it is modern, high-quality, lightweight, and exactly shape-matched to the chosen
DINOv3-B arm.
