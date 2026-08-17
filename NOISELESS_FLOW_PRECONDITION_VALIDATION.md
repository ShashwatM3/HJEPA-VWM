# Noiseless present-to-future flow: precondition validation

Date: 2026-08-05  
Scope: investigation only; no training, source edits, checkpoint mutation, S3 writes, or checkpoint downloads were performed.

## Executive conclusion

The frame sampler is understood exactly, and its ordinary-length behavior supports a clean non-overlap setting. The two model-side preconditions are not yet strong enough for an unconditional Task 3 launch. On the reconstructed Run 040 system, keeping slot index fixed costs little relative to Hungarian matching on average, but nearest-neighbor identity and assignment consistency are not robust at the harder horizon. Separately, S3 contains strong reconstruction-pretrained checkpoints for the newer EGO4D/DINOv3 or EGO4D/V-JEPA2 `64 x 512` geometry, but no locally evaluated checkpoint simultaneously matches the current/default SSV2 setup, its intended encoder/geometry, and the current architecture.

## Check 1 — context/target overlap

### Confirmed code behavior

The authoritative implementation is `SSV2Dataset._context_indices()` and `_window_indices()` in `data.py:303-348`; defaults are `input_frames=8` (`config.py:32`), `horizon_k=4`, and `frame_stride=2` (`config.py:293-294`). Let `T=8`, `s=2`, `k=horizon_k`, `L=num_frames`, and `last=L-1`.

```text
span      = (T - 1) * s + k
max_start = L - 1 - span
start     = 0                         if max_start < 0
          = randint(0, max_start)     for train
          = max_start // 2            for validation/test

context[i] = min(last, start + i*s)
tgt_end    = start + (T - 1)*s + k
target[i]  = min(last, tgt_end - (T - 1 - i)*s)
```

Therefore `horizon_k` is measured in **original/raw decoded frame indices** between corresponding context and target positions (and between their endpoints). It is not measured in sampled-frame steps and not in clip windows. The CLI help independently states “ORIGINAL frames” (`train.py:2035-2040`).

Repository and recorded W&B configurations show two horizons actually used in relevant historical SSV2 prediction runs: `k=4` (early/default runs) and `k=12` (later plateau runs, including Run 040). `k=16` appears as a proposed non-overlap experiment, not as a confirmed completed relevant historical run.

### Representative calculations without clamping

The following uses representative `start=0`; any legal unclamped start translates both lists by the same constant and leaves overlap counts unchanged.

| Horizon | Context indices | Target indices | Intersection | Shared frames | Percent of each 8-frame clip |
|---|---|---|---|---:|---:|
| `k=4` | `[0,2,4,6,8,10,12,14]` | `[4,6,8,10,12,14,16,18]` | `{4,6,8,10,12,14}` | 6 | 75% |
| `k=12` | `[0,2,4,6,8,10,12,14]` | `[12,14,16,18,20,22,24,26]` | `{12,14}` | 2 | 25% |
| `k=15` | `[0,2,4,6,8,10,12,14]` | `[15,17,19,21,23,25,27,29]` | empty | 0 | 0% |
| `k=16` | `[0,2,4,6,8,10,12,14]` | `[16,18,20,22,24,26,28,30]` | empty | 0 | 0% |

For even `k < 16`, shared count is `8-k/2`. Odd horizons have no exact index intersection in an unclamped stride-2 pair, but that parity accident should not be used as the scientific non-overlap rule. Algebraically, context ends at `start+14` and target begins at `start+k`, so the minimum integer raw-frame horizon for strict separation is **`k=15`**. If horizons are restricted to stride-aligned/even values, the minimum is **`k=16`**, matching the CLI's conservative wording.

### Short-video clamp/repeat behavior

When `L-1 < 14+k`, the code forces `start=0` and independently clamps every overrun index to `last`. This repeats the final decoded frame within one or both clips and can create cross-clip overlap even when `k>=15`.

Example: `L=10`, `k=12`:

```text
context = [0,2,4,6,8,9,9,9]
target  = [9,9,9,9,9,9,9,9]
set intersection = {9}
```

The unique shared-frame count is 1, but all eight target positions contain a frame also present in context; three context positions are repeats of it. Consequently no finite `k` guarantees non-overlap for every possible short video under the current clamp policy. For a specific sample to be unclamped, it needs `L >= 15+k`; at `k=16`, that is at least 31 decoded frames. The smallest valid remedy is to reject/filter samples below `15+k` frames (or implement a documented non-repeating short-video policy) before claiming non-overlap.

## Check 2 — temporal bottleneck-slot alignment

### Frozen checkpoint and fixed batch

The only local SSV2 artifact was used:

- checkpoint: `artifacts/checkpoints/inv011_fixed_position_decoder_phase1_step15000.pt`;
- size: 301,116,256 bytes; SHA-256 `969d197854320c71e774961de1039db69ab2eed682696d3b5297e96a43de7927`;
- Run 040 / W&B `io74f32b`, saved `global_step=15000`;
- exact detached source: `87ee2e1a9a8d1d084a6d7ca128374bfe108a8cc5`;
- encoder: V-JEPA2 ViT-L, local snapshot `b3c1679b7c34d3255ef3547f27c7b226aefab26f`;
- latent geometry: `(B,32,256)`, memory width 256, whitening unavailable/off in this historical code;
- fixed lexicographic validation subset: videos `100007`, `100011`, `100021`, `100026` from the previously reconstructed approved 16-video batch.

Strict loading of `bottleneck`, `target_bottleneck`, `coarse_flow`, and `decoder` had zero missing and zero unexpected keys in that detached historical worktree. The encoder and all modules were in evaluation mode; the diagnostic used `torch.inference_mode()`, constructed no optimizer, and performed no backward pass.

For both present and future, the diagnostic applied the **same frozen encoder instance** followed by the **same loaded `target_bottleneck` instance**. Thus both sides use exactly the same weights, learned query order, slot positions, and coordinate system. No target-dependent matching changed either latent: Hungarian assignment was computed only after the `(32 x 32)` cosine matrix was produced.

For every example and horizon, the complete matrix was computed as

```text
M[b,i,j] = cosine(c_present[b,i,:], c_future[b,j,:])
```

with `c_present,c_future` each shaped `(1,32,256)`. “Unrestricted best” is the mean row maximum and can reuse future slots. “Hungarian” is the maximum-weight one-to-one assignment. Motion magnitude is latent RMS displacement under identity pairing, `sqrt(mean((c_future-c_present)^2))`; it is a diagnostic stratifier, not optical flow.

### Aggregate results

| Horizon | Same-index cosine | Unrestricted best | Hungarian 1:1 | Identity-to-Hungarian gap | Row-best is identity | Motion RMS |
|---|---:|---:|---:|---:|---:|---:|
| `k=4` | 0.6673 | 0.6875 | 0.6684 | 0.0011 | 36.72% | 0.9217 |
| `k=12` | 0.4367 | 0.4787 | 0.4493 | 0.0125 | 28.13% | 1.2186 |

The small identity-to-Hungarian gaps are evidence that identity is nearly as good as the best one-to-one permutation in mean cosine, particularly at `k=4`. They do **not** mean that each slot has an unambiguous identity correspondence: unrestricted row maxima choose the same index for only 36.7% (`k=4`) and 28.1% (`k=12`) of slots.

### Per-example results

| Horizon | Video | Same index | Best | Hungarian | Gap | Best-is-identity | Motion RMS |
|---|---|---:|---:|---:|---:|---:|---:|
| 4 | 100007 | 0.6946 | 0.7093 | 0.6963 | 0.0017 | 37.50% | 0.9019 |
| 4 | 100011 | 0.5065 | 0.5641 | 0.5092 | 0.0027 | 18.75% | 1.1481 |
| 4 | 100021 | 0.6148 | 0.6191 | 0.6148 | 0.0000 | 43.75% | 1.0127 |
| 4 | 100026 | 0.8535 | 0.8576 | 0.8535 | 0.0000 | 46.88% | 0.6241 |
| 12 | 100007 | 0.6171 | 0.6289 | 0.6179 | 0.0008 | 40.63% | 1.0113 |
| 12 | 100011 | 0.3042 | 0.4206 | 0.3481 | 0.0439 | 6.25% | 1.3650 |
| 12 | 100021 | 0.4030 | 0.4278 | 0.4079 | 0.0050 | 34.38% | 1.2573 |
| 12 | 100026 | 0.4228 | 0.4373 | 0.4233 | 0.0005 | 31.25% | 1.2408 |

### Assignment consistency and motion stratification

| Horizon | Mean pairwise assignment agreement | Mean per-slot modal agreement | Slots identical across all 4 videos |
|---|---:|---:|---:|
| `k=4` | 92.71% | 95.31% | 87.50% |
| `k=12` | 36.46% | 61.72% | 15.63% |

At `k=4`, low-motion versus high-motion halves had same-index/Hungarian cosine `0.7740/0.7749` versus `0.5606/0.5620`; gaps remained tiny (`0.0008` versus `0.0013`). At `k=12`, low-motion was `0.5199/0.5206` with gap `0.0006`, whereas high-motion was `0.3536/0.3780` with gap `0.0244`. This is only a four-video exploratory split, but it localizes the risk: harder, higher-motion pairs show the strongest evidence of permutation ambiguity.

The experiment therefore supports fixed slot coordinates only conditionally. It is compatible evidence for historical Run 040, not for the newer current architecture: current `Bottleneck` has a three-block latent stack and different state-dict structure, while Run 040 lacks that stack and uses `(32,256,M=256)`.

## Check 3 — reconstruction-pretrained checkpoint selection

S3 inspection used capped `list-objects-v2` calls only under `ckpt/` and `checkpoints/`, always with `--region ap-south-1 --no-cli-pager`. Tiny adjacent `run_provenance.json` objects were read for serious candidates. No checkpoint payload was downloaded. `checkpoints/` contains only legacy anonymous step files (largest final `phase1_step15000.pt`, 153,113,745 bytes) with no adjacent provenance, so it is not a serious selection candidate.

### Serious candidates

| Candidate | Final checkpoint and size | Run / data / encoder | Geometry | Objective and final reconstruction | Representation health | Intended Task 3 compatibility |
|---|---|---|---|---|---|---|
| Run 040 local/S3 | `s3://hjepa-volume/ckpt/inv011_fixed_position_decoder/phase1_step15000.pt`, 301,116,256 B | `io74f32b`; SSV2; V-JEPA2 ViT-L, historical revision not recorded (reconstruction used candidate snapshot `b3c1679...`) | `N_c=32`, `D_c=256`, `M=256`, whitening off/unavailable | full prediction; cosine reconstruction `lambda_recon=0.05`, predicted recon `0.05`; final `L_recon_present=0.3464` | rank 50.76, cross-video cosine 0.1652, std 1.0053; low-rank; copy ratio 0.9707 | Dataset/encoder/default latent shape match, but historical bottleneck architecture is not current-checkpoint compatible; not a valid current warm start |
| Run 069 | `s3://hjepa-volume/ckpt/inv016_dinov3_unwhitened_memory_m512_cov_var/phase1_step15000.pt`, 511,080,866 B | `fiactcw6`; EGO4D; DINOv3 ViT-B/16 revision `5931719e...` | `32 x 256`, `M=512`, whitening off | present-only cosine reconstruction, `lambda_recon=1`; final `L_recon_present=0.1334`, `L_recon=0.1177` | rank 69.07, std 0.6215, cross-video cosine 0.6864, slot rank 26.75; representation-specificity gate fails | Current architecture and default `32 x 256`, but wrong dataset/encoder for SSV2/V-JEPA intent and weak representation health |
| Investigation 19 DINO | `s3://hjepa-volume/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt`, 602,344,090 B | `60yaqw6d`; EGO4D; DINOv3 ViT-B/16 revision `5931719e...` | `64 x 512`, `M=512`, whitening off | present-only cosine reconstruction, `lambda_recon=1`; final `L_recon_present=0.1250`, `L_recon=0.1010` | rank 363.91, std 1.1032, cross-video cosine 0.1105, slot rank 54.49, centered 58.80; stable | **Exact warm-start source for the later DINO/EGO4D temporal-target configuration**, but not current/default SSV2/V-JEPA geometry |
| Investigation 19 V-JEPA2 | `s3://hjepa-volume/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_vjepa2_n64_d512_m512/phase1_step15000.pt`, 593,434,202 B | `93ildhmk`; EGO4D; V-JEPA2 ViT-L revision `b3c1679...` | `64 x 512`, `M=512`, whitening off | present-only cosine reconstruction, `lambda_recon=1`; final `L_recon_present=0.2208`, `L_recon=0.2128` | rank 394.58, std 1.1367, cross-video cosine 0.1006, slot rank 56.78, centered 60.40; stable | Current architecture and V-JEPA2, but trained on EGO4D and not default `32 x 256` geometry |

The S3 inventory also contains incomplete/crashed or less relevant families (smokes, earlier `N=16/32/64/128` sweeps, whitened candidates, and SigLIP2 runs). They were not elevated because they fail completion, dataset/encoder, geometry, or representation-health criteria.

### Selection decision

No checkpoint is recommended unconditionally for an SSV2 Task 3 defined by the current/default V-JEPA2 `N_c=32,D_c=256` configuration. Run 040 matches SSV2 and the old default geometry but is structurally incompatible and low-rank. Run 069 matches `32 x 256` in the current architecture but changes both encoder and dataset and fails the cross-video-specificity gate. The strong Investigation 19 pair changes dataset and latent geometry.

If Task 3 is explicitly redefined as the already-proven **DINOv3/EGO4D `64 x 512, M=512`, unwhitened** configuration, then `60yaqw6d` is the single strongest checkpoint and has already been used as the exact warm start in Investigation 20 provenance. That conditional recommendation must not be transferred to SSV2 without a matched reconstruction-pretraining run.

For the SSV2 experiment as requested here, a reconstruction-pretraining run is required before Task 3. The smallest scientifically valid run holds the intended SSV2 split, exact frozen encoder revision, current bottleneck architecture, `N_c/D_c/M`, whitening policy, and decoder objective fixed; it must demonstrate falling reconstruction loss plus healthy rank/std/cross-video/slot-diversity metrics before freezing `B` and evaluating temporal slot alignment on the same geometry.

## Explicit verdicts

- **Frame overlap: CONDITIONAL.** Blocker: `k=4` shares 75% and `k=12` shares 25% of ordinary clips; clamping can reintroduce repeated overlap at any horizon. Smallest valid remedy: use `k>=15` (prefer stride-aligned `k=16`) and reject/filter videos with fewer than `15+k` decoded frames, then log/assert empty intersections.
- **Slot alignment: CONDITIONAL.** Blocker: the only measured checkpoint is historical Run 040, and at `k=12` row-best identity is only 28.13%, pairwise Hungarian assignment agreement is 36.46%, with the largest gaps in high-motion samples. Smallest valid remedy: repeat this exact diagnostic on the selected current-architecture frozen checkpoint and a larger fixed, source-unique batch stratified by horizon and independently measured motion; require a preregistered identity/Hungarian-gap and assignment-consistency threshold before training.
- **Frozen bottleneck: CONDITIONAL.** Blocker: no proven reconstruction checkpoint simultaneously matches SSV2, the intended encoder revision, current bottleneck architecture, and intended latent geometry. Smallest valid remedy: either explicitly adopt the DINOv3/EGO4D `60yaqw6d` configuration unchanged, or run matched SSV2 reconstruction pretraining and validate it before Task 3.
