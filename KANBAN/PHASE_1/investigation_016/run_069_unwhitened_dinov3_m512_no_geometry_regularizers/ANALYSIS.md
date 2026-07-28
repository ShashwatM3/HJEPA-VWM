# Analysis — run 069 DINOv3 unwhitened M=512 without geometry regularizers

## Evidence contract

This read uses the complete unsampled W&B history for `it7sq8nz` (300 logged rows, `_step`
0–14,950), the final run summary, and the resolved provenance stored in W&B. Late values below are
medians over the final six diagnostic rows unless stated otherwise.

The run finished and recorded final checkpoint
`/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/phase1_step15000.pt` with SHA-256
`f4513c9ce01e18f4db6cc6c3434c5a5f53749019f36992ed332fc14e67dd33db`.

Resolved encoder evidence:

| Field | Value |
|---|---|
| family | `dinov3` |
| repository | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| requested/resolved revision | `5931719e67bbdb9737e363e781fb0c67687896bc` |
| output layout | `8 × 16 × 16`, time-major, frame units |
| feature width / shape | `768` / `(B,2048,768)` |
| retained parameters | `85,660,416`, all frozen |
| feature fingerprint | `963cf988fb16b1e3971f952ade8afe90b29cef7dfd7103e4f312dada5c0eb415` |
| trainable initialization hash | `e2e66ac8482cb852caeb59c6a60cc8e9b4c05b1eca62280f103aa15354563cfa` |

The legacy `config.model.d_e`, `encoder_repo`, patch, and tubelet values remain V-JEPA-shaped for
historical checkpoint compatibility. Production construction resolved the DINO `EncoderSpec`
first and built B/D from it; the provenance `encoder_spec` and observed output shape are therefore
the authoritative runtime contract.

## Present-only Reading Cycle B

### Q1 — operational stability

PASS. W&B state is `finished`; every logged `grad_skipped`, `grad_has_nan`, and
`instability_warn` value is zero. Late gradients are small and finite, and the final checkpoint
hash is present. There is no optimizer cliff or partial-run ambiguity.

### Q2 — collapse geometry

| Metric | Late median |
|---|---:|
| `c_std_mean` | 0.33290 |
| `c_cross_video_cosine` | 0.86993 |
| `c_dead_dim_frac` | 0.01270 |
| `c_effective_rank` | 14.5969 |
| `c_slot_diversity_rank_centered` | 10.5056 |

The code is finite and not literally constant, but it occupies a narrow, highly aligned
subspace. This fails the healthy-geometry target. Because the fixed EGO4D batch contains 16
adjacent chunks from one source UID, the pair cosine is a within-source statistic and cannot by
itself establish global video-independent collapse.

### Q3 — reconstruction

| Metric | Late median |
|---|---:|
| training `L_recon` | 0.09758 |
| final-50-row training `L_recon` | 0.09863 |
| fixed correct-code `L_recon_present` | 0.11478 |
| fixed rolled-code `L_recon_shuffled_c` | 0.18083 |
| `L_recon_video_gap` | 0.06612 |

The decoder clearly uses the correct code: rolling codes makes reconstruction worse. The initial
fixed correct-code loss was `0.96206`; the late gap is about `7.8%` of the improvement from that
initial level to the late correct loss. Most of the learned reconstruction remains shared within
this recorded source context.

### Q4 — static-code and source controls

The positive rolled-code gap rejects a fully ignored code on the fixed batch. It does not reject a
global template, source/wearer/scene code, or other source-level shortcut because every diagnostic
example comes from the same recording and `torch.roll` swaps only adjacent chunks. No
source-diverse derangement was logged for this run.

### Q5 — relationship between reconstruction and geometry

Reconstruction became strong while rank, spread, and pair separation remained weak. This repeats
the no-geometry equilibrium seen with the other raw-feature substrates: reconstruction alone
selects a small decodable subspace and does not defend the rest of `D_c` against contraction.

### Q6 — verdict

**LOW-RANK DECODABLE; GLOBAL COLLAPSE INDETERMINATE.** Operationally, the run is a complete success
for the DINO adapter and common pipeline. Scientifically, it is a no-geometry baseline, not a
healthy present representation and not prediction evidence (`F_c` was inactive).

## Cross-encoder boundary

DINOv3 and SigLIP 2 have the same native `(2048,768)` detailed shape, and the corresponding
`N_c=32`, `D_c=256`, `M=512` builds share the same trainable initialization hash. That makes
structural/provenance comparisons useful. Their raw cosine reconstruction values and raw feature
rank budgets are still encoder-specific and must not be ranked against each other as if the target
space were common. Investigation 017 therefore performs each shape comparison within an encoder
lane and reserves cross-encoder synthesis for normalized geometry, code dependence, stability,
and compute.
