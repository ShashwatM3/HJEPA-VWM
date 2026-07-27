# Analysis — run 071 DINOv3 unwhitened M=512 covariance plus variance

## Evidence contract

This read uses W&B `fiactcw6`, its resolved config/provenance, 30 fixed diagnostics through step
14,500, and the final summary/checkpoint hash. Late values are medians over the final six
diagnostic rows.

Resolved provenance—not legacy `model` compatibility fields—records DINOv3 feature width 768,
frame lattice `8×16×16`, revision `5931719e67bbdb9737e363e781fb0c67687896bc`, and feature
fingerprint `963cf988fb16b1e3971f952ade8afe90b29cef7dfd7103e4f312dada5c0eb415`.

## Present-only Reading Cycle B

### Q1 — intended mode and stability

PASS. The run is `finished`, with zero skipped, nonfinite, or warned updates.
`present_recon_only=1`, `prediction_active=0`, `L_flow=0`, and `L_recon_pred=0`; reconstruction
warmup reaches full scale.

### Q2 — collapse geometry

FAIL on the recorded batch. Late std is `0.62217`, below the healthy 0.8–1.2 band, and pair cosine
is `0.68489`, above the 0.5 ceiling. Dead fraction is zero, so the code is not literally constant.
The one-source batch prevents promotion of this failure to a global cross-source statement.

### Q3 — rank and slot diversity

PASS in isolation. Effective rank is `69.00`, above 60, and centered slot rank is `29.03/31`.
These metrics show many active directions and distinct slots, but they do not override Q2: pooled
rank can coexist with weak sample separation.

### Q4 — present reconstruction

PASS. Fixed correct reconstruction falls from `0.96249` at step 0 to late median `0.13382`.
Rolling codes raises it to `0.17961`, producing a positive exact-chunk gap `0.04584`. Decoder AGC
does not clip.

### Q5 — joint geometry/content read

FAIL. Reconstruction and rank improve, but sample spread/alignment remain outside the healthy
bands. This is a decodable, rank-rich representation that is still too source-chunk invariant on
the recorded batch.

### Q6 — verdict

**COLLAPSED REP ON THE RECORDED WITHIN-SOURCE BATCH; GLOBAL COLLAPSE INDETERMINATE.**

The run is operationally complete and supplies exact DINO center evidence. It is not a strong
present representation under the project thresholds and is not prediction evidence.

## Relationship to Runs 069 and 070

Against the no-geometry raw DINO Run 069, restoring covariance plus variance raises effective rank
from `14.60` to `69.00`, centered slot rank from `10.51` to `29.03`, and std from `0.3329` to
`0.6222`, while reducing pair cosine from `0.8699` to `0.6849`. The geometry bundle is therefore a
real lever, but it is insufficient by itself.

Run 070 also adds fixed feature whitening and reaches the recorded-batch thresholds. Because
whitening changes the target distribution, raw reconstruction magnitudes are not directly
comparable; geometry and dependence are the supported comparison.
