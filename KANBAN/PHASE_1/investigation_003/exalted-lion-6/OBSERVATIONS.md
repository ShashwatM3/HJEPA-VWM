# Observations — exalted-lion-6

## Outcome

**Diagnostic success.** Training healthy; **rank collapse reproduced** at small scale.
Informed the **slot-loss arc** that followed (Runs 3–5).

## Key numbers (step 0 → 400)

| Metric | step 0 | step 400 | Reading |
|---|---|---|---|
| `L_flow` | 2.87 | 1.40 | learning |
| `c_effective_rank` | 9.0 | ~10.3 | **bouncing ~8, not climbing** |
| `c_slot_diversity_rank` | 16.2 | 16.6 | slots partially differentiate |
| `c_cross_video_cosine` | 0.72 | 0.25 | videos distinguishable |
| `c_attn_entropy` | ~1.0 | ~1.0 | reads uniform (head-averaged metric) |
| `grad_skipped` | 0 | 0 | clean |

## Key finding

**Attention saturation is not the bottleneck.** `c_attn_entropy` reads ~uniform,
but `c_slot_diversity_rank ≈ 16/32` shows slots are not fully collapsed — the
head-averaged entropy metric masked per-head selectivity. The dominant symptoms
are **slot redundancy** (not using full 32-slot capacity) and **feature
correlation** (low `c_effective_rank` ~8 despite distinguishable videos).

This ruled out "fix attention first" as the primary lever and pointed the team
toward slot-diversity and decorrelation regularizers — the slot-loss arc in
[`serene-cloud-8`](../serene-cloud-8/) through [`copper-sky-12`](../copper-sky-12/).

## Interpretation

- Collapse is **persistent objective failure**, not an init transient
- Init fixes alone do not lift rank past ~8–10 on tiny data
- Per-head `c_attn_entropy_min` fix (`a96c0d6`) followed; full-SSv2 baseline next
  ([`sleek-leaf-7`](../sleek-leaf-7/))
