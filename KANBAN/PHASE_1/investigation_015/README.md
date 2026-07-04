# Reading run 054: attributing the bottleneck fix and whitening independently

One run, two deltas relative to run 053 (`7teohhwc`, inv013 — same recipe otherwise):

- **Delta A — latent-stack bottleneck** (`BottleneckLatentBlock` x3). Treated as an
  architecture FIX (slot competition + repeated refinement), not a hypothesis.
- **Delta B — fixed offline whitening** (`FeatureWhitener`). The hypothesis under test
  (from investigation_014: V-JEPA `e` concentrates energy in few directions, so
  unwhitened cosine recon only "defends" those few directions of `c`).

A single run cannot give a factorial ablation. It CAN give a defensible per-change read,
because the two deltas act through different mechanisms on **different metric axes**,
and we have run 052/053 as matched references for the no-delta trajectory.

## 1. Why the deltas separate

**Delta A acts on the SLOT axis.** Slot self-attention is the only new machinery that
lets slot i see slot j and de-duplicate. Whitening is a fixed linear reparametrization
of the input/target space — it has no mechanism to force 32 slots apart (053's slots
merged despite anisotropic targets; rotating/rescaling those targets does not create
inter-slot repulsion). So slot-axis metrics move primarily with Delta A.

**Delta B acts on the FEATURE-DIMENSION axis of the recon gradient.** The 053 analysis
showed cosine recon in raw space is satisfiable by reproducing V-JEPA's ~10 dominant
directions, so weight decay contracted the other ~246 dims of `c`. Whitening equalizes
target directions: reducing whitened-cosine loss requires matching MANY equally weighted
directions, so recon pressure defends many more feature dimensions of `c`. So
feature-dim rank behavior during the contraction window moves primarily with Delta B.

The axes are logged separately:

- Slot axis: `c_slot_diversity_rank_centered` (within-video rank over the 32 slots),
  supported by `c_attn_entropy` / `c_attn_entropy_min`.
- Feature axis: `c_effective_rank` (pooled covariance rank over the 256 dims),
  supported by `c_std_mean` and `c_dead_dim_frac`.

They are not perfectly orthogonal (differentiated slots also lift pooled rank a
little — 32 slots cap that contribution at ~32 of 256), so read magnitudes, not just
directions: pooled rank well above ~32-40 cannot come from slot differentiation alone
and must be defended feature directions (Delta B).

## 2. Metric -> change mapping

| Metric (diag cadence) | Reads on | How to read it |
|---|---|---|
| `c_slot_diversity_rank_centered` | **A** | 052/053 declined monotonically (053: 12.5 -> 8.5). Holding or rising through steps 4k-12k = slot competition works. Raw `c_slot_diversity_rank` ~32 is mechanical; ignore. |
| `c_attn_entropy`, `c_attn_entropy_min` | **A** | Final-block read sharpness. Healthy = mean falling below ~0.65 with min clearly below mean (some head specializes). Entropy collapsing toward 0 together with centered slot rank falling = slots sharp but redundant (A failing). |
| `c_effective_rank` | **B** (above ~32-40) | The headline. 053 halved in the 4k-12k window (19.9 -> 10.5). Holding or growing while `L_recon_present` falls = recon now defends many directions = whitening mechanism. Values > ~40 are unreachable via slot effects alone. |
| `c_std_mean`, `c_dead_dim_frac` | **B** | 053's amplitude contraction (std 0.378 -> 0.255) was the loss-defends-nothing signature. Whitening predicts a higher, stabler equilibrium std; dead dims should stay 0. |
| `c_cross_video_cosine` | joint | Health gate, not attribution: both deltas should push it below the 052/053 ~0.91 regime. If it stays high while rank rises, suspect a shared whitened-space mean direction — check with the honesty gap. |
| `L_recon_video_gap`, `L_recon_shuffled_c` | carried from 053 (+B flavor) | Honesty monitor, attribution-neutral. In whitened space `L_recon_shuffled_c` should pin near ~1.0 (wrong-video latents decode near-orthogonally); a materially lower plateau means residual shared structure survived whitening — a Delta B warning. |
| `L_recon_present` | neither, alone | Progress indicator only. NOT comparable to 052 (0.293) or 053 (0.453) — different target space. Use its SHAPE against the geometry curves (the Q5 cooperation read). |
| `recon_mean_norm`, `whiten_active` | B sanity | `whiten_active=1` throughout; `recon_mean_norm` far below 053's ~1624 confirms the tracker lives in whitened space. Wiring checks, not results. |
| `grad_*`, `agc_*`, `instability_warn` | joint | Run validity gate. The latent stack adds depth; if AGC clips B persistently or grads skip, tag the run invalid before attributing anything. |

**The one decisive picture:** overlay `L_recon_present`, `c_effective_rank`, and
`c_slot_diversity_rank_centered` for steps 4,000-12,000 and compare against 053's run
(`7teohhwc`), where recon improved 0.502 -> 0.453 while rank halved and slots merged.
Attribution comes from which of the two geometry curves breaks 053's pattern.

## 3. The 2x2 outcome matrix

| | `c_effective_rank` holds (>= ~40 and not contracting post-4k) | `c_effective_rank` contracts (053 pattern) |
|---|---|---|
| **`c_slot_diversity_rank_centered` holds** (no monotone decline) | Both deltas working. Next: transfer to full prediction. | A works, B insufficient: slots differentiated but few feature directions defended. Next: sweep `--whiten-eps` (floor too high under-equalizes; too low amplifies encoder tail noise) before re-adding geometry terms. |
| **`c_slot_diversity_rank_centered` declines** | B works, A insufficient: recon defends directions but slots still merge onto them. Next: keep whitening; strengthen slot competition (more latent blocks) or small `lambda_slot`. | Neither sufficient. Extends 053's H2: even equalized recon pressure needs an explicit anti-collapse force. Next: whitening + residual target + inv011 geometry terms (`lambda_var`, small `lambda_cov`). |

Pre-registered follow-ups for each cell: [`NEXT_STEPS.md`](NEXT_STEPS.md).

## 4. What this run canNOT tell you (be honest in the writeup)

- **Interaction effects.** If both axes hold, we cannot say whether either delta alone
  would have sufficed — only that the pair does. Claim the pair.
- **Absolute recon quality vs 052/053.** Different target space; never quote
  `L_recon_present` across whitened/raw runs as a comparison.
- **Whitening's effect on PREDICTION.** This is an AE-only run; nothing here passes or
  fails the Phase 1 copy/batch-mean gates.
- **Perfect axis separation.** The mapping above is "primarily attributable", not
  causal proof. If a cell's read is load-bearing for a big decision and feels
  ambiguous, the tiebreaker is cheap: one bottleneck-only control (drop the two
  `--whiten-*` flags, same seed, ~6 h) turns this into a real ablation.

## 5. Reference values (for the comparison plots)

| Metric | 052 (`662hfy3c`) final | 053 (`7teohhwc`) @12k | Healthy target here |
|---|---|---|---|
| `c_effective_rank` | 13.4 | 10.5 (peak 19.9 @4k) | >= ~40, not contracting post-4k |
| `c_slot_diversity_rank_centered` | (declining) | 8.5 (from 12.5) | holding / rising |
| `c_cross_video_cosine` | 0.906 | 0.929 (best 0.842 @4k) | << 0.9, ideally < 0.5 |
| `c_std_mean` | 0.295 | 0.255 (peak 0.378 @4k) | stable, well above ~0.3 |
| `L_recon_video_gap` | ~0 (inferred) | +0.433, growing | clearly positive, growing |
| `L_recon_shuffled_c` | n/a | 0.886 | ~1.0 plateau (whitened space) |
