# Observations - run 053 `ae_sharp_slots_residual_recon`

**W&B run id:** `7teohhwc`
**State:** `crashed` (external kill at step ~12150; no training pathology)
**Mode:** present-only run; `F_c` is inactive and the copy / batch-mean gates do not apply
**Verdict:** **Low-rank decodable** (video-specific and honest, but geometrically contracted —
and, importantly, NOT template collapse this time)

## Reading Cycle (Cycle B, present-recon-only)

| Q | Question | Result | Evidence (@ ~step 12000 unless noted) |
|---|---|---|---|
| Q1 | Training alive, right mode? | PASS | 0 skips, 0 NaNs, grad_norm ~0.02-0.03 flat, AGC never clipped; `present_recon_only=1`, `prediction_active=0`, `L_flow=0`, `L_recon_pred=0`, `recon_target_residual=1`; crash at 12150 was external. |
| Q2 | `c_t` alive / video-specific? | SPLIT | Geometrically weak (`c_std_mean=0.255` vs ~1.0 target; cross-video cosine 0.929) but informationally strongly video-specific — see Q4's honesty gap. |
| Q3 | Rich latent? | FAIL | `c_effective_rank=10.5` (target >60; run 052 ended 13.4); centered slot diversity 8.0/32. The genuine information lives in a ~10-dimensional subspace of the 256 available. |
| Q4 | Reconstruction learning video content? | PASS (strongly) | `L_recon_present` 1.012 -> 0.453 monotone; `L_recon_shuffled_c` stayed high ~0.874-0.887; `L_recon_video_gap` 0 -> 0.433, still growing at the crash. |
| Q5 | Geometry and content cooperating? | FAIL | After step ~4000 they moved in opposite directions: reconstruction -0.05 while rank halved, cosine +0.09, std -0.12. |
| Q6 | Verdict | Low-rank decodable | Video-specific and honestly decoded, but the code contracted geometrically with no anti-collapse force present. |

## Metric trajectory (landmark diagnostic steps, W&B `7teohhwc`)

| step | c_eff_rank | slot_rank_centered | cross_video_cosine | c_std_mean | L_recon_present | L_recon_shuffled_c | video_gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.99 | 30.99 | 1.000 | 0.000 | 1.012 | 1.012 | 0.000 |
| 1000 | 32.21 | 30.74 | 0.989 | 0.098 | 0.618 | 0.860 | 0.242 |
| 2000 | 26.74 | 19.11 | 0.897 | 0.300 | 0.550 | 0.866 | 0.316 |
| 3500 | 21.92 | 13.04 | 0.843 (best) | 0.375 (peak) | 0.511 | 0.874 | 0.363 |
| 4500 | 18.28 | 10.67 | 0.844 | 0.376 | 0.495 | 0.880 | 0.385 |
| 7500 | 14.08 | 8.92 | 0.875 | 0.337 | 0.472 | 0.883 | 0.412 |
| 10000 | 11.85 | 8.31 | 0.909 | 0.289 | 0.460 | 0.886 | 0.426 |
| 11500 | 10.76 | 7.95 | 0.925 | 0.261 | 0.454 | 0.887 | 0.432 |

## Last Key Metrics

`c_effective_rank`=10.76 @ 11500; `c_cross_video_cosine`=0.925 @ 11500; `c_std_mean`=0.261 @
11500; `L_recon_present`=0.454 @ 11500; `L_recon_shuffled_c`=0.887 @ 11500;
`L_recon_video_gap`=0.432 @ 11500. (Extrapolating the decaying slopes to step 15000 gives
rank ~9, cosine ~0.94.)

## Interpretation

**Finding 1 — the residual target worked; the honesty probe proves it.** Decoding a video's
own `c_t` reaches 0.453; decoding a wrong (shuffled) video's `c_t` only reaches ~0.887. From
the untrained 1.012, the decoder improved by 0.559 with the correct latent but only ~0.126
with a wrong one — so roughly **77% of everything the decoder learned is conditioned on the
specific video's latent** (run 052 was ~15%, i.e. mostly template). `L_recon_shuffled_c`
actually rose slightly over training while the gap widened at every diagnostic step, never
plateauing: the decoder became progressively MORE dependent on the correct code, exactly the
incentive the residual target builds. Run 052's specific failure — reconstruction satisfiable
with no information flowing through `c_t` — is closed.

**Finding 2 — geometry collapsed anyway, in two clean phases.** Steps 0-~4000 were an
expansion out of the zero-gated init: cosine fell to its best 0.843, std rose to its peak
0.376, the gap grew fastest. Steps 4000-12150 were a monotone contraction: rank 19.9 -> 10.5,
cosine 0.843 -> 0.925, std 0.376 -> 0.261, centered slot diversity 13 -> 8, attention entropy
0.76 -> 0.53 — while `L_recon_present` improved only ~0.05. The optimizer traded away half the
code's rank for a tiny reconstruction gain.

**Reading the two cosine-vs-rank facts correctly.** `c_effective_rank` is centered, so 10.5 is
the true dimensionality of the code's variation, not a big-shared-mean artifact.
`c_cross_video_cosine` is uncentered, so 0.929 mostly measures that a shared mean direction
dominates each video's small (std 0.26) deviation. Combined with the positive honesty gap, the
accurate picture is: every video's code = one big shared vector + a small, low-rank, but
genuinely video-identifying deviation that the 512x4 decoder then decompresses. With zero
anti-collapse forces, cosine reconstruction is satisfiable by a low-amplitude, low-rank code;
LayerNorm on `c_t` fixes overall scale but not spread; and decoupled weight decay (0.05)
continuously contracts every direction the loss does not actively defend — and the loss only
defends ~10 directions, so the other ~246 decayed.

## What This Run Changed

It surgically removed run 052's H1 (template shortcut) and showed collapse persists
essentially unchanged, so H1 and H2 are separable, H1 is solved, and H2 is confirmed:
**no content objective, however honest, will hold rank/variance by itself — an explicit
anti-collapse force is a requirement.** That is the program's most robust negative result on
the present-only side, and it is the premise the whitening/latent-stack work (investigations
014-015) builds on.

## Run 052 vs run 053

| Metric | 052 (absolute target) | 053 (residual target) | Read |
|---|---|---|---|
| `L_recon_present` | 0.293 | 0.453 | Not comparable — residual is a strictly harder objective |
| `L_recon_video_gap` | ~0 (no probe; inferred) | +0.433, growing | The fix worked |
| `c_effective_rank` | 13.4 | 10.5 | Both collapsed; 053 slightly worse |
| `c_cross_video_cosine` | 0.906 | 0.929 | Same regime |
| `c_std_mean` | 0.295 | 0.255 | Same regime |
| Video-conditioned share of decoder gain | ~15% | ~77% | The honesty axis, fixed |

## Reading Constraints

- Present-only run: the copy / batch-mean gates do NOT apply; `coarse_*` panels are absent.
- `L_recon_present` here is on the residual target, so it is NOT numerically comparable to run
  052's absolute-target 0.293 or run 054's whitened-space 0.652. Compare trends and the
  honesty gap, not absolute values.
- Raw `c_slot_diversity_rank` (~32) is mechanical from the fixed slot identities; judge slots
  only on `c_slot_diversity_rank_centered`.
