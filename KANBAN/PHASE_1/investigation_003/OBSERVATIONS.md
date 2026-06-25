# Observations — Investigation 003 (collapse)

## Hypothesis timeline

### H1: Init is the root cause (orthogonal queries + zero-init `out_mlp`)

**Evidence:** External review + code audit. Near-uniform attention from tiny random queries.

**Result:** **Partially true but insufficient.** Init fixes (baked into model, commit `62b94dd`)
improved starting rank (~9 vs ~5) and attention entropy. Rank still stalled without stronger
variance pressure. Init is **hygiene**, not the cure ([`ANALYSIS_AND_DECISIONS.md`](../../AGENT_FILES/KANBAN/04-FIX-DIMENSIONAL-COLLAPSE/ANALYSIS_AND_DECISIONS.md)).

### H2: Missing decorrelation term (VICReg-C / SIGReg)

**Evidence:** `L_var` only prevents constant dims, not correlated subspaces. Effective rank ~5
with healthy dead-dim fraction.

**Result:** VICReg-C implemented (flag-gated) but **deprioritized** after H4 won. See [investigation_004](investigation_004/). SIGReg held as escalation, not tried.

### H3: Slot-diversity loss fixes redundant slots

**Evidence:** `c_slot_diversity_rank` collapsed to **1.62/32** on full SSv2
(`sleek-leaf-7`, per-head metric) with near-uniform attention. External research
(Perceiver / Slot-Attention "prototype/slot collapse"; the "Trap of Mediocrity"
uniform-attention work) identified a slot-diversity/orthogonality penalty as the standard
remedy. Built as `slot_diversity_loss`, flag-gated `--lambda-slot`.

**Result:** **Rejected as a training objective**, across four runs and a code fix:
- `serene-cloud-8` (k=4, **raw** loss, λ=0.25): latent drifted to video-independence
  (cosine ~0.74) while staying 5–15× worse than copy — early Goodhart on a partly-broken
  loss.
- `skilled-waterfall-10` (k=4, raw loss, λ=0.05): `L_slot` glued ~1.0 — **inert**.
  Exposed the **loss/metric mismatch** (loss used raw slot cosine, diagnostic used
  centered slots). Fixed in `ffc33ed`.
- `olive-terrain-11` then `copper-sky-12` (k=12, **centered** loss, λ=0.05): the penalty
  now bites (slot diversity pinned ~20–26/32) yet `c_effective_rank` **collapses to ~4.8**
  and `c_cross_video_cosine` climbs to **0.84**, with 10⁴-class grad spikes — the clean,
  decisive Goodhart. Optimizing the slot metric ≠ improving the representation.

The centering fix was worth making (it turned a confounded result into a trustworthy
rejection), but the objective itself is wrong. *Correction:* the earlier KANBAN
attributed "~0.84 cosine / rank worsened" to `serene-cloud-8`; per W&B those belong to
`copper-sky-12` (serene's rank actually *rose* to ~14). Fixed in the run files.

### H4: Variance floor too weak (`lambda_var=0.1`)

**Evidence:** Every slot-arc run showed `c_std_mean ~0.4–0.5` (well under the 1.0 floor)
with `L_flow` out-pulling `L_var`; cosine drifting to ~0.7–0.84. The binding constraint
was the *dose* of the anti-collapse term already present, not a missing term.

**Result:** **Confirmed — this is the cure.** `cerulean-snow-13` with `lambda_var=0.5`
(single-variable, slot+cov off): `c_std_mean` climbed 0.5 → 1.04 and pinned at target,
`c_cross_video_cosine ~0.20–0.26` and stayed there, `c_effective_rank` **reversed** and
rose to ~13.7 (still climbing at 6900), `coarse_vs_copy_ratio < 1`, **zero grad skips**.
The grad spikes from the slot arc were collapse-coupled, not LR — so no LR change was
needed. Slot diversity stayed healthy (~19) *with no slot penalty* → the slots were a
**symptom**, not the lever.

### H5: Task too easy (`horizon_k=4`)

**Evidence:** Copy baseline strong at k=4 (context/target overlap ~75%, `copy_loss`
~0.15); the latent barely changes, so there's little pressure to encode rich dynamics.

**Result:** **Adopted as a necessary adjunct, k=12 (not k=16).** The tech lead proposed
k=16; the team rejected it as over-aggressive for the diagnostics and settled on k=12.
Note the launch reality: k=12 was *intended* from `skilled-waterfall-10` but only
actually applied from `olive-terrain-11` onward (waterfall ran k=4 due to a launch
drift). k=12 + `lambda_var=0.5` is the winning combination.

## Cross-run synthesis (verified vs W&B)

Metrics are end-of-run / best-window values; see each run's OBSERVATIONS for trajectories.

| Run | k | λ_slot | λ_var | Rank (end) | Cross-video cos | Copy ratio | Verdict |
|---|---|---|---|---|---|---|---|
| peachy-terrain-5 (ref) | 4 | — | 0.1 | ~3–5 | healthy | improved→regressed | collapse + crash |
| exalted-lion-6 | 4 | 0 | 0.1 | ~6–10 | 0.25 | falling | init non-lever; collapse persists |
| sleek-leaf-7 (Run 2 @3500) | 4 | 0 | 0.1 | **8.7** | 0.25 | **7.42** | architectural, not data-limited; slot rank 1.62 |
| serene-cloud-8 (Run 3) | 4 | 0.25 raw | 0.1 | ~14 (rose) | ~0.74 | 5–15× | reject aggressive slot (raw loss) |
| skilled-waterfall-10 (Run 4) | **4** | 0.05 raw | 0.1 | ~7 | ~0.6 | inert L_slot | loss/metric bug → centering fix |
| olive-terrain-11 (Run 5a) | 12 | 0.05 ctr | 0.1 | ~9 | ~0.65 | ~2.8 | Goodhart begins (slot↑ rank↓) |
| copper-sky-12 (Run 5b) | 12 | 0.05 ctr | 0.1 | **4.8** | **0.84** | grad 4×10⁴ | reject slot loss (decisive) |
| **cerulean-snow-13** (Run 6) | 12 | 0 | **0.5** | **13.7↑** | **0.22** | **< 1** | **winning config** |
| jolly-forest-14 (Run 6b) | 12 | 0 | 0.5 | ~9 @3900 | 0.23 | beats batch-mean | reproduced cerulean, crashed |

## Conclusion

**Primary collapse lever: raise `lambda_var` to 0.5.** Init fixes stay baked in
(`62b94dd`). **Do not train with slot-diversity loss** — it Goodharts. **Use
`horizon_k=12`.** VICReg-C remains optional if rank plateaus low after a stable long run
([investigation_004](../investigation_004/)).

Wrong turns retained in record: VICReg-C-first (H2) and slot loss (H3) both looked
right on isolated metrics; the data showed the lever was an under-dosed term we already
had.

## Links

- Brief run table: [`AGENT_FILES/KNOWLEDGE/BRIEF_V0_3.md`](../../AGENT_FILES/KNOWLEDGE/BRIEF_V0_3.md) §13
- Local log: [`logs/cerulean-snow-13/output.log`](../../../logs/cerulean-snow-13/output.log)
