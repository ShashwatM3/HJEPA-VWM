# 21 — Failure modes and quiz traps

## Architecture confusions

| Trap | Correct answer |
|---|---|
| “The target encoder is EMA.” | The encoder is one shared frozen instance; only `B` has EMA. |
| “The decoder generates frames.” | `D` reconstructs frozen detailed features. |
| “Fine flow exists but is disabled.” | `F_e` is absent from current code. |
| “`M=512` is the model default.” | Default is 256; 512 is a selected experiment width. |
| “`f_c_dim` controls flow width.” | Current flow uses external `D_c`; `f_c_dim` is retained legacy config. |
| “Slots are permutation-invariant objects.” | They are ordered learned query identities. |
| “Zero-init means output velocity is zero.” | AdaLN blocks are identity, then flow returns normalized marked `z` tokens. |

## Shape confusions

| Trap | Correct answer |
|---|---|
| V-JEPA has eight temporal token planes | Tubelet-2 gives four |
| SigLIP/DINO have 1024 detailed tokens | They have 8×256=2048 |
| DINO passes 261 tokens/frame | It strips CLS+4 registers and passes 256 |
| `M` equals `D_c` architecturally | Only by shipped default; they are distinct interfaces |
| Raw scalar count equals information | It is only a dimensional ledger |
| `k` is measured in encoder tubelets | It is decoded-frame indices |

## Temporal confusions

- Default windows are not disjoint: six of eight frames overlap.
- Span non-overlap requires `k>14`; normal even ladder begins at 16.
- Odd `k` can have no exact index matches while spans still overlap.
- EGO at 12 FPS gives `k=4` ≈0.333 s, but SSv2 real time depends on container timing.
- A four-second EGO chunk is not the model's eight-frame temporal span; the model samples only part
  of it per window.

## Loss confusions

| Trap | Correction |
|---|---|
| variance floor ensures high rank | it does not decorrelate dimensions |
| covariance ensures video specificity | it pools batch×slots and attacks feature correlations |
| slot loss ensures semantic objects | it only decorrelates centered slot directions |
| SIGReg success means prediction success | it can add static/high-rank content |
| copy loss is optimized | it is a read-only fixed baseline |
| lower copy loss is good | it often means `c_t≈c⁺`, making copy unbeatable |
| reconstruction loss proves content | shuffled-video gap is needed to expose template shortcuts |
| predicted recon affects only `F_c` | it also affects `B` and `D` |

## Precision confusions

- Outer BF16 autocast does not mean every computation is BF16.
- Reconstruction normalization and SIGReg operate in FP32.
- Whitening eigensystem/statistics use FP64; runtime matrices use FP32.
- BF16 encoder mode is rejected off CUDA.
- The local test failure on Transformers 5.5.3 is a pin violation, not evidence the code supports it.

## Optimizer confusions

- Optimizer membership does not guarantee a parameter has a gradient.
- Decoder parameters with `None` grads do not decay/move.
- AGC precedes global clipping.
- `grad_norm` is post-AGC but pre-global-rescale.
- `grad_global_norm_postclip` near 0.5 is expected when clipping fires.
- A skipped step also skips EMA.
- Bias/norm/geometry/zero-init bridges receive neither weight decay nor AGC.

## Schedule confusions

- Step numbering is zero-based.
- LR at step zero is `1/1500`, not zero.
- Recon/SIG ramp at step zero is zero.
- Step 1,500 is exactly peak LR.
- `--steps` changes loop length, not the 15k LR schedule.
- Default 15k Phase 1 does not reach EMA end momentum.
- `phase1_step2500.pt` means step 2500 is next.

## Data/provenance confusions

| Trap | Correction |
|---|---|
| same directory path means same data | identity includes paths, sizes, frame counts, manifests |
| same tensor shape means same encoder | fingerprints reject semantic mismatch |
| EGO chunks may be randomly split | source UID is the anti-leakage unit |
| manifest count proves completeness | filesystem/frame inventory and source sets are also checked |
| resuming model weights is exact resume | optimizer, EMA, buffers, sampler, RNG, provenance all matter |
| dirty Git flag captures changes | it says dirty, but does not serialize the diff |

## Diagnostic confusions

- Effective rank max is 256 because covariance is over feature width, not 8,192 flattened values.
- Variance floor flattens slots/features per video; effective rank pools slots and uses feature
  dimension. They measure different populations.
- Raw slot rank can rise mechanically from slot identities; centered rank is cleaner.
- Attention entropy one means uniform, but low entropy is not automatically useful.
- Batch-mean ratio can pass while copy ratio fails.
- A same-source EGO fixed batch invalidates global cross-video language.
- One lucky diagnostic point does not pass a late-window gate.

## MLOps confusions

- W&B stores path/checksum, not checkpoint bytes by default.
- A tmux session is not proof the child process is healthy.
- GPU utilization is not proof metrics/checkpoints are correct.
- Proxy SSH may not support SCP/SFTP.
- A paid run launch requires authorization and exact destination isolation.
- “Resume” must continue W&B run identity or explicitly record a new lineage.
- Pod deletion and volume/data deletion are separate destructive permissions.

## Historical-versus-current confusions

- Original brief numbers are design ancestry, not executable shapes.
- KANBAN entries may retain a “planned/running” statement superseded by later dated updates.
- `YOUR_FILES/` is stale and non-authoritative.
- Full EGO counts are manifest-derived, not a permanent 170k/19k constant.
- Live W&B state is temporally unstable; repository-recorded snapshots must be dated.

## Failure-mode decision table

| Observation | Likely class | Next evidence |
|---|---|---|
| NaNs/skips/late spike | optimization invalid | unsampled history, last good checkpoint |
| std low, cosine high | global collapse | variance/cov/SIG scales, diverse fixed batch |
| std okay, rank 10–15 | correlated low-rank code | covariance spectrum, geometry terms |
| raw slot rank high, centered low | shared slot mean | centered slot rank, entropy |
| recon low, shuffled gap ~0 | template decoder | residual feature target/fixed query audit |
| rank high, copy loss falling | static appearance code | drift probe, horizon sweep |
| copy ratio ~1, copy loss rising | dynamic code/no predictor | condition use, flow optimization |
| batch ratio good, copy ratio bad | per-video but lazy forecast | focus on temporal delta |
| online rank high, EMA rank low | target lag | EMA schedule/SIG ramp |
| EGO cosine high on first 16 | possible same-source confound | source-diverse validation batch |

## Oral-exam “why?” chain

If asked why a component exists:

- pinned frozen encoder: stable rich coordinate system and controlled encoder comparisons;
- learned-query bottleneck: fixed-bandwidth abstract compression;
- local ConvNeXt: spatial token mixing before global read;
- orthogonal queries/sharpened cosine: prevent uniform same-slot readout;
- zero residual bridges: stable identity initialization;
- EMA bottleneck: slowly moving future target;
- slot/type flow codes: expose ordered stream identities;
- flow matching: continuous latent dynamics objective;
- variance/covariance/SIG: distinct anti-collapse axes;
- fixed-position decoder: position without learned content template;
- shuffled reconstruction: prove video-specific dependence;
- strict provenance: prevent shape-compatible semantic contamination;
- fixed diagnostics: compare the model rather than random validation samples;
- copy/batch mean: prove temporal and conditional value.

## Final self-check

You are ready for a technical-lead quiz when you can, without notes:

1. draw both V-JEPA and DINO shape paths;
2. derive default overlap and first disjoint horizon;
3. derive every module's parameter formula;
4. route gradients for all seven objectives;
5. reproduce the exact training-step order;
6. distinguish all three random streams and resume state;
7. explain why run 037 failed despite rank 61;
8. explain why run 052's low recon was dishonest;
9. state both prediction gates and when they do not apply;
10. name every planned-but-absent stage.
