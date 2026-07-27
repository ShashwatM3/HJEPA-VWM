# 17 — Exact-number atlas

Use this chapter for memorization. Unless marked otherwise, values are shipped defaults.

## Input and temporal numbers

| Quantity | Value |
|---|---:|
| batch | 64 |
| frames/window | 8 |
| channels | 3 |
| input resolution | 256×256 |
| within-window stride | 2 decoded frames |
| context index span | 0–14 relative |
| default horizon `k` | 4 decoded frames |
| target index span | 4–18 relative |
| exact shared frames at `k=4` | 6/8 |
| first even non-overlap horizon | 16 |
| transform version | `raw-rgb-resize-crop-jitter-v2` |
| train jitter factor range | 0.6–1.4 |

## Encoder atlas

| | V-JEPA2 | SigLIP2 | DINOv3 |
|---|---:|---:|---:|
| input frames | 8 | 8 | 8 |
| temporal planes | 4 | 8 | 8 |
| spatial grid/plane | 16×16 | 16×16 | 16×16 |
| detailed tokens `N_e` | 1,024 | 2,048 | 2,048 |
| width `D_e` | 1,024 | 768 | 768 |
| scalars/video | 1,048,576 | 1,572,864 | 1,572,864 |
| BF16/video | 2 MiB | 3 MiB | 3 MiB |
| parameters | 325,971,328 | 85,843,200 | 85,660,416 |
| normalization | ImageNet | 0.5/0.5 | ImageNet |

Pinned revisions:

```text
V-JEPA2 b3c1679b7c34d3255ef3547f27c7b226aefab26f
SigLIP2 3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab
DINOv3  5931719e67bbdb9737e363e781fb0c67687896bc
Transformers 4.57.6
```

## Abstract latent

```text
N_c = 32
D_c = 256
scalars/video = 8,192
BF16/video = 16 KiB
```

Compression:

| Encoder | token count compression | scalar count compression |
|---|---:|---:|
| V-JEPA2 | 1024/32 = 32:1 | 1,048,576/8,192 = 128:1 |
| frame encoders | 2048/32 = 64:1 | 1,572,864/8,192 = 192:1 |

## Bottleneck defaults

| Quantity | Value |
|---|---:|
| internal width `M` | 256 |
| ConvNeXt blocks | 2 |
| depthwise kernel | 7×7 |
| ConvNeXt/latent MLP ratio | 4 |
| cross-attention heads | 8 |
| latent blocks | 3 |
| query rows | 32 orthogonal unit rows |
| position-table init std | 0.5 truncated normal |
| cosine temperature init | 0.07 |
| logit scale cap | 100 |

## Coarse flow defaults

| Quantity | Value |
|---|---:|
| input sequence | 64 tokens |
| width | 256 |
| blocks | 6 |
| heads | 8 |
| MLP ratio | 4 |
| condition dropout | 0.10/example |
| slot-position init std | 0.02 |
| type embeddings | zero |
| AdaLN modulation | zero |
| horizon embedding | none |

## Decoder defaults

| Quantity | Value |
|---|---:|
| width | 256 |
| blocks | 2 |
| heads | 8 |
| MLP ratio | 4 |
| position code | fixed 3-D sinusoidal |
| axis split | 85 time / 85 row / 86 column |
| output | `N_e×D_e` feature tokens |

## Exact parameter counts at `M=256,N_c=32,D_c=256`

| Component | V-JEPA2 | SigLIP2/DINOv3 |
|---|---:|---:|
| online `B` | 4,837,123 | 5,033,731 |
| `B_EMA` | 4,837,123 | 5,033,731 |
| `F_c` | 7,643,904 | 7,643,904 |
| `D` | 2,172,160 | 2,106,368 |
| trainable `B+F_c+D` | 14,653,187 | 14,784,003 |
| model bundle `B+B_EMA+F_c+D` | 19,490,310 | 19,817,734 |

Frozen encoder is excluded from these totals.

## Width scaling at `N_c=32,D_c=256`

| Encoder | `M` | `B` | trainable total | bundle with EMA |
|---|---:|---:|---:|---:|
| V-JEPA2 | 256 | 4,837,123 | 14,653,187 | 19,490,310 |
| V-JEPA2 | 512 | 18,324,739 | 28,140,803 | 46,465,542 |
| V-JEPA2 | 1024 | 70,727,427 | 80,543,491 | 151,270,918 |
| frame | 256 | 5,033,731 | 14,784,003 | 19,817,734 |
| frame | 512 | 18,717,955 | 28,468,227 | 47,186,182 |
| frame | 1024 | 71,513,859 | 81,264,131 | 152,777,990 |

## Investigation 017 V-JEPA trainable grid at `M=512`

| `N_c \ D_c` | 128 | 256 | 512 |
|---:|---:|---:|---:|
| 16 | 22,307,331 | 28,124,419 | 50,899,203 |
| 32 | 22,319,619 | 28,140,803 | 50,923,779 |
| 64 | 22,344,195 | 28,173,571 | 50,972,931 |

Increasing `D_c` is expensive because it widens all six flow blocks. Increasing `N_c` is mostly
linear position/query cost.

## Attention tensor ledger, batch 64

| Tensor | Elements | BF16 |
|---|---:|---:|
| bottleneck cross logits, V-JEPA `64×8×32×1024` | 16,777,216 | 32 MiB |
| bottleneck cross logits, frame `64×8×32×2048` | 33,554,432 | 64 MiB |
| one flow block logits `64×8×64×64` | 2,097,152 | 4 MiB |

These are logical tensors; fused SDPA may not materialize them identically.

## Raw tensor ledger

```text
one FP32 RGB clip = 8*3*256*256*4 = 6 MiB
batch 64, one clip = 384 MiB
batch 64, context+target = 768 MiB
```

## Tracker/whitener storage

| State | V-JEPA2 | frame encoders |
|---|---:|---:|
| feature mean | 4 MiB | 6 MiB |
| ZCA mean + two matrices | ~8.004 MiB | ~4.503 MiB |

All are FP32 buffers.

## Training defaults

| Quantity | Value |
|---|---:|
| steps | 15,000 |
| warmup | 1,500 |
| LR `B` | 1e-4 |
| LR `F_c` | 2e-4 |
| LR `D` | 1e-4 |
| Adam betas | 0.9, 0.95 |
| weight decay | 0.05 |
| global clip | 0.5 |
| skip threshold | 150 |
| warning norm/loss | 30 / 1.0 |
| AGC `B/F_c/D` | 0.20 / 0.10 / 0.20 |
| AGC epsilon | 0.001 |
| EMA start/end | 0.996 / 0.9999 |
| EMA denominator | 105,000 |
| recon/SIG ramp | 2,000 |

## Loss defaults

| Weight/setting | Value |
|---|---:|
| `lambda_var` | 0.10 |
| std target | 1.0 |
| `lambda_cov` | 0 |
| `lambda_slot` | 0 |
| `lambda_sigreg` | 0 |
| `lambda_recon` | 0 |
| `lambda_recon_pred` | 0 |
| recon mode | cosine |
| SIG projections | 128 |
| SIG row cap | 512 |
| SIG beta | 1 |
| mean momentum | 0.99 |
| whiten epsilon | 1e-4 |
| whitening expected clips | 12,800 |

## DataLoader defaults

| Quantity | Value |
|---|---:|
| data | `ssv2_tiny` |
| workers | 8 |
| pin memory | true |
| train drop-last | true |
| seed | 42 |
| frame encoder microbatch | 8 |
| precision | BF16 |
| attention implementation | SDPA |

## Dataset counts and epochs

| Dataset | Train | Validation | Batches/epoch at 64 | Tail dropped |
|---|---:|---:|---:|---:|
| SSv2 tiny | 4,002 | 348 | 62 | 34 |
| SSv2 full | 168,913 | 24,777 | 2,639 | 17 |
| EGO tiny | 4,000 | 350 | 62 | 32 |

15,000 steps:

```text
examples = 960,000
SSv2 tiny = 241 full epochs + 58 batches
SSv2 full ≈ 5.684 drop-last epochs
EGO tiny = 241 full epochs + 58 batches
```

## Cadence totals

| Event | Cadence | Count in 15k |
|---|---:|---:|
| train log | 50 | 300 |
| diagnostics | 500 | 30 |
| scheduled checkpoint | 2,500 | 6 |

Final scheduled path is step 15,000 and final save atomically replaces that same path.

## Scientific gates

```text
c_std_mean target range       0.8–1.2
c_dead_dim_frac               near 0
c_cross_video_cosine          <0.5
c_effective_rank              >60 of 256
coarse_vs_copy_ratio          <=0.70
coarse_vs_batch_mean_ratio    <=0.50
```

These are project decision gates/heuristics, not mathematical bounds.

## Numbers that are deliberately not fixed

- full EGO chunk count: derive from manifest;
- current live W&B run count/state: query live system;
- GPU step time/memory: measure recipe/hardware;
- checkpoint byte size: inspect the actual file;
- future fine-flow/generator shapes: planned design;
- source FPS for arbitrary SSv2 files: read container metadata.
