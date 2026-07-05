# Observations - run 054 `ae_latent_stack_whiten_recon_only`

**W&B run id:** `lx1b6gw2`
**State:** `finished` (full 15000 steps; last diagnostic step 14500)
**Mode:** present-only run; `F_c` is inactive and the copy / batch-mean gates do not apply
**Verdict:** **Low-rank decodable** - video-specific and honest at the strongest level the
program has produced, but geometrically contracted, settling at a much higher equilibrium than
run 053.

## Reading Cycle (Cycle B, present-recon-only)

| Q | Question | Result | Evidence |
|---|---|---|---|
| Q1 | Trained alive, right mode? | PASS | 0 skips, 0 NaNs, no instability warnings, AGC never clipped (max ratio ~0.0006 vs threshold 1.0), grad_norm 0.005-0.032; `present_recon_only=1`, `prediction_active=0`, `L_flow=0`, `L_recon_pred=0`, `recon_target_residual=1`, `whiten_active=1` every step; ran the full schedule. |
| Q2 | `c_t` alive / video-specific? | SPLIT (better than 053) | `c_std_mean` 0.42 peak (vs ~1.0 target), cross-video cosine 0.796 best (target < 0.5) — both clearly better than run 053 (0.255, 0.929); informationally strongly video-specific per Q4. |
| Q3 | Rich latent? | FAIL | rank peaked 34.7 then fell to 21.9; never exceeded the ~32-slot mechanical span; centered slot diversity 31 -> 12.1 (decline flattened hard near the end). |
| Q4 | Reconstruction learning video content? | PASS (strongest yet) | `L_recon_present` 1.0025 -> 0.652 monotone; `L_recon_shuffled_c` pinned 0.969-0.975 all run; ~92% of decoder improvement is video-conditioned (run 053 ~77%, run 052 ~15%). |
| Q5 | Geometry and content cooperating? | PARTIAL FAIL | after step ~7500 all four geometry measures drift in the collapse direction while reconstruction keeps improving — same signature as run 053 but ~1/4 the rate and decelerating toward a plateau. |
| Q6 | Verdict | Low-rank decodable | video-specific and honest, but geometrically contracted at a much higher equilibrium than 053. |

## Metric trajectory (landmark diagnostic steps, W&B `lx1b6gw2`)

| step | c_eff_rank | slot_rank_centered | cross_video_cosine | c_std_mean | L_recon_present | L_recon_shuffled_c | video_gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.99 | 30.99 | 1.000 | 0.000 | 1.0025 | 1.0025 | 0.000 |
| 2500 | 34.68 (peak) | 28.62 | 0.928 | 0.239 | 0.740 | 0.969 | 0.229 |
| 4000 | 30.37 | 20.58 | 0.878 | 0.312 | 0.703 | 0.970 | 0.267 |
| 7500 | 26.78 | 13.43 | 0.783 (best) | 0.436 (peak) | 0.666 | 0.974 | 0.308 |
| 10000 | 24.43 | 12.68 | 0.796 | 0.422 | 0.657 | 0.974 | 0.318 |
| 14500 | 21.93 | 12.13 | 0.820 | 0.397 | 0.652 | 0.975 | 0.322 |

## Last Key Metrics

`c_effective_rank`=21.93 @ 14500; `c_cross_video_cosine`=0.820 @ 14500; `c_std_mean`=0.397 @
14500; `L_recon_present`=0.652 @ 14500 (flat from step 13000); `L_recon_shuffled_c`=0.975
(pinned all run); `L_recon_video_gap`=0.322; video-conditioned share of decoder improvement 92.0%.

## Interpretation

**Three phases, not run 053's two.** Phase 1 (0-2500) was expansion out of the zero-gated init:
rank rose to its 34.7 peak, std grew from zero, cosine fell from 1.0, attention entropy peaked
~0.807. Phase 2 (2500-7500) was a mixed regime run 053 never showed — feature rank and slot
diversity contracted WHILE cross-video cosine kept improving to its best 0.783 and std kept
rising to its peak 0.436. That decoupling is the first clean demonstration in the program that
rank and video-separation are not the same failure axis. Phase 3 (7500-end) was a broad, mild
contraction: every geometry measure moved in the collapse direction together, but at ~1/4 run
053's rate and decelerating toward a plateau (rank losing ~0.9/1k steps by step 10000 vs run
053 halving over the same window).

**Honesty is the clean win and it fully converged.** `L_recon_shuffled_c` sat 0.969-0.975 the
entire run — a wrong video's latent decodes near-orthogonally in whitened space, so no shared
cross-video structure survived the whitening for the decoder to exploit. With the correct
latent the decoder improved 1.0025 -> 0.657 (gain 0.346); with a wrong one only -> ~0.974 (gain
0.028). So ~92% of what the decoder learned depends on the correct code, held at 92.0% at the
final measurement. Run 052's template shortcut is not merely closed (as run 053 showed) — in
whitened space it is essentially eliminated.

**Geometry still contracts; the equilibrium moved a long way.** vs run 053: terminal rank 21.9
vs 10.5 (~2.3x), std 0.397 vs 0.255 (~65% higher), cosine 0.820 vs 0.929. So the two deltas
raised the number of directions the loss defends and/or made slot merging more expensive, but
did not change the SIGN of the drift. The sharpest evidence for H2 is the final 1500 steps:
reconstruction gained essentially nothing (0.6524 -> 0.6521, even ticking up at the last point)
while rank fell another 0.34, cosine rose another 0.003, and std slipped another 0.004 —
contraction continuing with zero content benefit isolates the unopposed decoupled weight decay
(0.05) as the grinding force, exactly run 053's diagnosis.

**Attribution caveat (the one honest limitation).** At step 0, with std exactly zero (the code
carried no video information), pooled rank already read 31.0 — entirely from the 32 fixed slot
identity vectors. Since rank never exceeded ~35, there is no point in the trajectory where
feature rank demanded an explanation beyond slot structure, so this run gives no POSITIVE
evidence for the whitening mechanism at `whiten_eps=1e-4`, and it cannot separate which of the
two deltas bought the improved equilibrium. That is why the bottleneck-only control (drop the
whitening flags, keep the residual target) is now load-bearing.

## What This Run Changed

It re-confirmed H2 at a third, higher level of content-objective quality (052 absolute, 053
residual, 054 whitened residual — all fail to hold geometry alone), settling that an explicit
anti-collapse term is a requirement of this architecture family. And it established the
whitened-residual recipe as the honest CONTENT CHANNEL to build on: a 92% video-conditioned
decoder with the wrong-code loss pinned at 0.97 is exactly the honest-content property Phase 1
needs underneath any future geometry regularization.

## Reading Constraints

- Present-only run: no copy/batch-mean gates; `coarse_*` panels are absent.
- `L_recon_present` is in whitened target space — NOT comparable to run 052 (0.293) or 053
  (0.453). Compare trends and the honesty gap. The absolute video-gap size (0.322) is also not
  cross-comparable to run 053's 0.433 (different target spaces); the pinned shuffled-code loss
  ~0.97 is the cleaner cross-run honesty signal.
- Raw `c_slot_diversity_rank` (~32) is mechanical; judge slots on the centered value.
- Rank up to ~32-40 is reachable from slot structure alone; only rank well above ~40 would be
  positive evidence for whitening.
