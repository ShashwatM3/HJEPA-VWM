# Analysis — run 070 DINOv3 whitened M=512 covariance plus variance

## Evidence contract

This read uses W&B `qqozribu`, its resolved config/provenance, 30 fixed diagnostic rows through
step 14,500, and the final summary/checkpoint hash. Late values are medians over the final six
diagnostic rows. W&B `lwx0mu34` is a passed 100-step gate, not a second scientific result.

Authoritative encoder provenance is DINOv3 ViT-B/16, feature width 768, frame lattice
`8×16×16`, revision `5931719e67bbdb9737e363e781fb0c67687896bc`, and feature fingerprint
`963cf988fb16b1e3971f952ade8afe90b29cef7dfd7103e4f312dada5c0eb415`.
The ordinary `model.d_e=1024` value is legacy compatibility metadata.

## Present-only Reading Cycle B

### Q1 — intended mode and stability

PASS. State is `finished`; all 15,000 updates completed. `present_recon_only=1`,
`prediction_active=0`, `L_flow=0`, and `L_recon_pred=0`. Skipped, nonfinite, warned, and decoder
AGC-clipped values are zero throughout the diagnostic history.

### Q2 — collapse geometry

PASS on the recorded batch, with a source-scope caveat. Late std is `0.80948`, dead fraction is
zero, and pair cosine is `0.49264`. These meet the project thresholds, but the pair is
within-source because the fixed batch contains one source UID.

### Q3 — rank and slot diversity

PASS for pooled feature rank. Effective rank is `67.61`, above the threshold of 60. Centered slot
rank is `21.25/31`: clearly non-degenerate, though not near the slot ceiling.

### Q4 — present reconstruction

PASS. Fixed correct reconstruction improves from `0.98019` at step 0 to a late median `0.34871`.
Rolling codes raises loss to `0.45704`, for an exact-chunk gap of `0.10831`. `recon_scale` reaches
1 and decoder AGC does not clip.

### Q5 — joint geometry/content read

PASS on the recorded batch. Reconstruction improves while rank, spread, and pair separation reach
the formal healthy bands. The positive rolled-code gap shows that the supplied chunk code matters.
It does not establish cross-source identity because derangement never leaves the one source.

### Q6 — verdict

**STRONG PRESENT REPRESENTATION ON THE RECORDED WITHIN-SOURCE BATCH; GLOBAL CROSS-SOURCE
SPECIFICITY UNMEASURED.**

This is present-reconstruction evidence only. The run does not encode a future target, optimize
`F_c`, exercise DINO prediction memory, or evaluate copy/batch-mean gates.

## Comparison boundary

Run 071 removes whitening but retains the same encoder, external shape, covariance/variance
weights, schedule, seed, and trainable initialization hash. Whitening also changes the target
distribution, so the raw cosine losses are not commensurate. The meaningful comparison is the
geometry/dependence equilibrium, with the single-source limitation retained.
