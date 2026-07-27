# Whiteboard drills

Time each drill. Draw first; use prose only after the diagram is complete.

## 1. Full shape graph — 8 minutes

Draw shipped V-JEPA full prediction from raw batch to loss. Label every tensor shape, both stop-grad
branches, and the optimizer/EMA transition.

Rubric: raw pair, `E` shared/frozen, `1024×1024`, online/EMA bottlenecks, `32×256`, flow equations,
loss routes, `D` inactive by default, AGC/clip/skip/Adam/EMA.

## 2. Encoder swap — 5 minutes

Erase only the V-JEPA-specific parts and replace with DINOv3. Preserve every invariant downstream.

Rubric: flatten frames, 261→256 token strip, `2048×768`, changed B/D counts and position length,
unchanged `c/F_c`.

## 3. Temporal lattice — 5 minutes

For arbitrary `k`, draw both eight-index grids. Solve exact overlap for even `k`, span non-overlap,
and calculate `k=4,12,14,16`.

Rubric: distinguishes exact index overlap from interval overlap.

## 4. Bottleneck exploded view — 10 minutes

Draw one detailed temporal plane through ConvNeXt and all slots through a latent block. Annotate what
mixes space, time, and slots.

Rubric: time folded into batch for ConvNeXt; global time access through cross-attention; slot exchange
through self-attention; zero bridges.

## 5. Parameter derivation — 12 minutes

Derive `P_B`, `P_Fc`, and `P_D` from primitives. Calculate default V-JEPA totals without looking.

Rubric: includes biases, norms, position/query tables, optional `M→D_c`, and no buffer count.

## 6. Gradient map — 8 minutes

Rows: flow, var, cov, slot, SIG, present recon, predicted recon. Columns: `E,B,B_EMA,F_c,D,mean,W`.
Mark gradient, EMA mutation, buffer mutation, or none.

Rubric: predicted residual add-back reaches B; trackers are never optimizer parameters.

## 7. Schedule strip — 7 minutes

Draw a common step axis 0→105k with LR, recon/SIG ramp, and EMA momentum. Mark 0,1000,1499,1500,
2000,7500,14999,15000,105000.

Rubric: distinct denominators and step-zero behavior.

## 8. Diagnostic decision tree — 10 minutes

Start at stability and end in every full-run verdict. Include the static-c branch and simultaneous
copy/batch-mean gates.

Rubric: no rank-only success; reconstruction honesty comes after prediction gates.

## 9. Exact resume state machine — 10 minutes

Draw preparation, pre-mutation checks, load, sampler reconstruction, RNG restore, W&B continuation,
next update, and final atomic save.

Rubric: dataset transfer and optimizer reset are explicit side branches.

## 10. Dataset factory — 10 minutes

Draw raw SSv2 and EGO inputs to the common runtime directory contract.

Rubric: SSv2 symlinks; EGO source filtering/split/download batches/redaction/chunk files; exact
manifest/runtime inventory.

## 11. Current versus planned — 6 minutes

Use solid boxes for current and dashed boxes for planned. Show coarse, fine, and frame stages plus
teacher-forcing gradient stops.

Rubric: current `D` is not future generator; no horizon/integrator.

## 12. Run 037/052/053 diagnosis — 8 minutes

Place each run in a 2×2 space of geometry health and content/prediction honesty. Explain why each
intervention followed.

Rubric: 037 healthy geometry/no predictor;052 dishonest template collapse;053 honest but low-rank.

## 13. Memory ledger — 7 minutes

Calculate raw pair batch, detailed batch, abstract batch, bottleneck attention logits, and tracker
storage for V-JEPA and a frame encoder.

Rubric: state dtype and distinguish logical tensors from peak CUDA allocation.

## 14. Remote launch proof — 6 minutes

Draw local Git/test flow to remote exact SHA, preflight, tmux, W&B, volume checkpoint, monitoring,
and completion proof.

Rubric: explicit paid authorization; W&B not checkpoint backup.

## 15. Adversarial quiz — 15 minutes

Have a partner point at an arbitrary arrow or parameter in any prior drawing. For each, answer:

```text
owner
shape/value
initialization
dtype
gradient/mutation path
checkpoint/provenance status
failure metric
truth layer
```

If any field is unknown, return to the owning chapter and redraw only that seam.
