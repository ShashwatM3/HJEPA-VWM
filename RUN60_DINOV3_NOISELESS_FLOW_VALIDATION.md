# Run 60 DINOv3 bottleneck validation for noiseless present-to-future flow

Date: 2026-08-06  
Status: partial validation; blocked at the real-data forward gate.  
Restrictions honored: no training, no production-code changes, no S3 writes, no checkpoint mutation, and only the explicitly selected checkpoint was downloaded.

## Selected artifact

- W&B: `60yaqw6d`, finished.
- Local path: `artifacts/checkpoints/inv019_covvar_dinov3_n64_d512_m512_phase1_step15000.pt`.
- S3 source: `s3://hjepa-volume/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt`.
- Size: 602,344,090 bytes.
- SHA-256: `931c27b47df3331a1968b7d33afc74556e5779c02a1997989bf27324cf9270b1`, exactly matching W&B and checkpoint provenance.
- Schema/global step: `hjepa-phase1-checkpoint-v2`, 15,000 completed updates.

## Exact reconstructed configuration

The checkpoint embeds the full resolved configuration and immutable encoder contract:

- dataset: EGO4D, fingerprint `df36af5d...`;
- encoder: `dinov3_vitb16`, repository `facebook/dinov3-vitb16-pretrain-lvd1689m`;
- requested/resolved revision: `5931719e67bbdb9737e363e781fb0c67687896bc`;
- encoder feature layout: `(T,H,W)=(8,16,16)`, time-major, giving `(B,2048,768)`;
- encoder feature fingerprint: `963cf988...`, reproduced exactly from current `EncoderSpec`;
- bottleneck: `N_c=64`, `D_c=512`, memory width 512, 2 ConvNeXt mixers, 3 latent blocks, 8 cross-attention heads;
- decoder: width 512, 4 blocks, 8 heads, absolute cosine reconstruction;
- whitening disabled; `lambda_recon=1`, `lambda_var=0.5`, `lambda_cov=0.01`;
- run mode: present-reconstruction-only, so `F_c` was not trained in this run.

The current repository strictly loaded the saved online bottleneck, EMA target bottleneck, CoarseFlow, and decoder with zero missing and zero unexpected keys. The strict state counts were 93, 93, 70, and 59 entries respectively.

## Shape and CoarseFlow contract

An inference-only synthetic contract check using the saved DINO `EncoderSpec` produced:

```text
synthetic detailed features: (1, 2048, 768)
online bottleneck:           (1, 64, 512)
EMA bottleneck:              (1, 64, 512)
CoarseFlow output:           (1, 64, 512)
decoder output:              (1, 2048, 768)
```

Therefore the selected bottleneck output is exactly `(B,64,512)`, and current `CoarseFlow` instantiates and forwards at `N_c=64,D_c=512` without architectural changes.

Parameter and weight-memory implications:

| Module | Parameters | FP32 weights | BF16 weights |
|---|---:|---:|---:|
| one bottleneck copy | 18,603,523 | 70.97 MiB | 35.48 MiB |
| CoarseFlow | 30,524,928 | 116.44 MiB | 58.22 MiB |
| decoder | 14,317,824 | 54.62 MiB | 27.31 MiB |

If CoarseFlow is later trained in FP32 with Adam, weights + gradients + two FP32 moment buffers alone are about 465.8 MiB, excluding activations, allocator overhead, and any mixed-precision master-copy policy. Its attention sequence is `2*N_c=128`; at batch 64, one FP32 eight-head attention-score tensor is about 32 MiB per block. Six retained blocks can therefore consume roughly 192 MiB for scores alone, before Q/K/V, MLP, residual, and autograd storage. A real GPU preflight remains mandatory.

## Run-level reconstruction and representation evidence

The final W&B values at step 14,950 describe the **online bottleneck**, because this present-only objective decodes and diagnoses the online path:

| Metric | Final value |
|---|---:|
| `L_recon_present` | 0.124998 |
| train-step `L_recon` | 0.101048 |
| effective rank | 363.907 / 512 |
| cross-video cosine | 0.110492 |
| standard-deviation mean | 1.103191 |
| dead-dimension fraction | 0 |
| slot-diversity rank | 54.487 / 64 |
| centered slot-diversity rank | 58.804 / 64 |
| shuffled-c reconstruction | 0.537993 |
| video reconstruction gap | 0.412995 |

The run was stable (`grad_skipped=0`, `grad_has_nan=0`, `instability_warn=0`). Under the repository's present-reconstruction reading cycle this is a strong, decodable, high-rank, video-specific representation. It is not evidence that its saved `F_c` predicts future states, because `prediction_active=0` and `L_flow=0` by design.

## Online versus EMA comparison

The state dictionaries are extremely close but not identical:

- relative parameter L2 distance: `1.4968e-5`;
- parameter-delta RMS: `1.8233e-6`;
- maximum absolute parameter delta: `3.1233e-5`.

On one deterministic-shape synthetic DINO-feature tensor, output RMS distance was `0.002487` and flattened cosine was `0.999998`. This proves coordinate proximity and output-shape parity, but it is **not** a substitute for the requested fixed EGO4D batch.

### Provisional freeze recommendation

Freeze the **saved online bottleneck** as the single coordinate system, then use that exact frozen module for both present and future. Reasons:

1. the decoder was jointly trained against online outputs;
2. all proven reconstruction and representation-health metrics above measure the online copy;
3. the EMA copy is only a tiny lagged perturbation at the final checkpoint;
4. the later temporal-target provenance used Run 60's online bottleneck as the warm-start source and initialized the new target copy exactly from it.

This recommendation remains provisional until online and EMA are evaluated side-by-side on the same real validation features. Under no circumstance should online encode one endpoint while EMA encodes the other.

## EGO4D sampler overlap audit

Run 60 used `T=8`, `frame_stride=2`, and `horizon_k=12`, where horizon is measured in decoded/raw frame indices. For any unclamped start `s`:

```text
context = [s, s+2, s+4, s+6, s+8, s+10, s+12, s+14]
future  = [s+12, s+14, s+16, s+18, s+20, s+22, s+24, s+26]
intersection = {s+12, s+14}
```

Thus the intended Run 60 horizon has two exact shared frames: 25% of each sampled clip. EGO4D provenance reports 48 frames for every validation chunk and 42–48 for training chunks, so `k=12` does not invoke last-frame clamping (`15+k=27` frames are sufficient). The overlap is genuine sampling overlap, not padding.

Chunks were prepared at 12 FPS. Consequently `k=12` shifts corresponding positions by 1.0 second, while the sampled clip spans 14 raw-frame intervals (about 1.167 seconds). The two shared indices are also identical timestamps/decoded frames. Strict non-overlap starts at integer `k=15`; the stride-aligned choice is `k=16` (about 1.333 seconds).

## Blocked empirical requirements

The machine contains neither:

- the immutable DINOv3 snapshot at revision `5931719e...`; nor
- any local EGO4D validation `.mp4` files or a source-unique 32-example manifest.

The checkpoint excludes the frozen encoder weights and video bytes. The checkpoint provenance's recorded 16-item validation batch is also unsuitable for requirement 9: all 16 chunks share source UID `01cab463-9a16-4817-84a4-a00ef5b7bf39`, so it is not source-unique.

Because the instruction authorized downloading only the checkpoint—and earlier dataset restrictions did not authorize EGO4D objects—I did not download an encoder snapshot, manifests, or videos. Therefore the following remain unmeasured:

- real-batch online-versus-EMA reconstruction loss;
- EMA effective rank, cross-video cosine, and slot-diversity ranks;
- real-batch online-versus-EMA output distance;
- the temporal fixed-coordinate/Hungarian diagnostic on at least 32 source-unique EGO4D examples.

## Verdict

**NO-GO for implementing present-to-future flow yet.**

This is an evidence-completeness NO-GO, not a rejection of Run 60. Checkpoint identity, strict compatibility, geometry, representation health, and CoarseFlow construction all pass. The exact blockers are the absent immutable DINOv3 weights and absent 32-source-unique EGO4D validation batch, which prevent requirements 3 and 9 from being performed.

Smallest valid remedy: separately authorize retrieval of the exact DINOv3 revision and exactly 32 named, source-unique EGO4D validation chunks (plus only the minimal manifest metadata needed to bind their source UIDs). Then run one inference-only pass that compares online and EMA on identical cached encoder features, freezes the winning single copy, and computes the full temporal slot-alignment diagnostic at `k=12` and the proposed non-overlap `k=16`.
