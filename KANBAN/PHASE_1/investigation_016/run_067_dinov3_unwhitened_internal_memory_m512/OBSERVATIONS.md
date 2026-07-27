# Observations — Run 67, DINOv3 unwhitened internal-memory M=512

> **Local-label correction (2026-07-27):** W&B `it7sq8nz` is canonical local
> [Run 069](../run_069_unwhitened_dinov3_m512_no_geometry_regularizers/). The canonical analysis
> supersedes the pending late-window placeholders retained below.

## Result

Run 67 completed all 15,000 steps. W&B run
[`it7sq8nz`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/it7sq8nz) has state `finished`,
the committed provenance artifact is `it7sq8nz-provenance:v0`, and the verified final checkpoint is:

```text
/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/phase1_step15000.pt
sha256 f4513c9ce01e18f4db6cc6c3434c5a5f53749019f36992ed332fc14e67dd33db
```

The runtime identity stayed correct: clean commit `083cf8a6e87168702efe46ac6bfe485756dcb439`, full
EGO4D, DINOv3 revision `5931719e67bbdb9737e363e781fb0c67687896bc`, frame microbatch 32,
`present_recon_only=1`, `whiten_active=0`, `M=512`, three latent blocks, `N_c=32`, external
`D_c=256`, 512-by-4 decoder, absolute cosine reconstruction at weight 1, and zero
variance/covariance/SIGReg/slot weights.

## Completion verification

| Check | Verified result |
|---|---|
| W&B state | `finished` |
| Training steps | `15000/15000` |
| Final checkpoint | present at the intended persistent-volume path |
| Final checkpoint SHA-256 | `f4513c9ce01e18f4db6cc6c3434c5a5f53749019f36992ed332fc14e67dd33db` |
| Provenance file | `/workspace/ckpt/inv016_dinov3_unwhitened_memory_m512/run_provenance.json` |
| W&B provenance artifact | `it7sq8nz-provenance:v0`; `run-provenance`; `COMMITTED` |
| Checkpoint cadence | `2500, 5000, 7500, 10000, 12500, 15000` |
| Checkpoint upload policy | final model checkpoint retained only on the RunPod persistent volume |

## Present-only reading cycle — final verified snapshot

| Q | Question | Verdict | Evidence |
|---|---|---|---|
| Q1 | Did it train in the intended mode? | PASS | 15,000 steps; W&B `finished`; `present_recon_only=1`; `whiten_active=0`; `grad_skipped=0`; final `grad_norm=0.053812`. |
| Q2 | Is the code alive and video-specific? | FAIL ON RECORDED BATCH, GLOBAL STATUS UNRESOLVED | Final std `0.332081` and cross-example cosine `0.870476` show contraction on the fixed within-source batch. They do not prove global cross-source collapse. |
| Q3 | Is the code rich rather than low-rank? | FAIL ON RECORDED BATCH | Final effective rank `14.521889`, far below the rank-60 representation target. |
| Q4 | Did reconstruction learn useful content? | PASS | Final active `L_recon=0.096526`; fixed correct-code `L_recon_present=0.114407`. |
| Q5 | Did the decoder use the supplied clip code without preserving healthy geometry? | MIXED | Final shuffled-code loss `0.180776` exceeds correct-code loss by `0.066369`, while std/rank/cosine remain unhealthy on the recorded batch. |
| Q6 | Present-only verdict | **LOW-RANK DECODABLE; GLOBAL COLLAPSE INDETERMINATE** | The decoder uses the exact-chunk code, but the recorded latent geometry fails the health gates. No forecasting claim is possible. |

## Final verified metrics

| Metric | Final value |
|---|---:|
| Active training `loss` / `L_recon` | 0.096526 |
| Fixed correct-code `L_recon_present` | 0.114407 |
| Fixed shuffled-code `L_recon_shuffled_c` | 0.180776 |
| Correct-vs-shuffled `L_recon_video_gap` | 0.066369 |
| Mean code std | 0.332081 |
| Effective rank | 14.521889 |
| Cross-example cosine | 0.870476 |
| Gradient norm | 0.053812 |
| Skipped updates | 0 |
| Whitening active | 0 |
| Present reconstruction only | 1 |

The final-snapshot conditioned share is approximately `7.49%`, computed as
`L_recon_video_gap / (1 - L_recon_present)`. It is an exact-chunk/within-source quantity, not a
global video-conditioned share.

## Preregistered late window

The run-066 template requests medians over diagnostic points at steps 12,000–14,500. Those medians
have not been supplied in this completion handoff and are intentionally left pending rather than
being reconstructed from the final snapshot.

| Metric | DINOv3 late median |
|---|---:|
| Active training `L_recon` | pending W&B late-window extraction |
| Fixed correct-code `L_recon_present` | pending W&B late-window extraction |
| Fixed shuffled-code `L_recon_shuffled_c` | pending W&B late-window extraction |
| Correct-vs-shuffled gap | pending W&B late-window extraction |
| Exact-chunk conditioned share | pending W&B late-window extraction |
| Mean code std | pending W&B late-window extraction |
| Cross-example cosine | pending W&B late-window extraction |
| Effective rank | pending W&B late-window extraction |
| Slot-diversity rank | pending W&B late-window extraction |

## Comparison against the V-JEPA M=512 control

| Metric | Run 064 V-JEPA late median | Run 67 DINOv3 |
|---|---:|---:|
| Active training `L_recon` | 0.260785 | pending late-window extraction |
| Fixed correct-code `L_recon_present` | 0.305063 | pending late-window extraction |
| Fixed shuffled-code `L_recon_shuffled_c` | 0.466626 | pending late-window extraction |
| Correct-vs-shuffled gap | 0.161563 | pending late-window extraction |
| Exact-chunk conditioned share | 23.2486% | pending late-window extraction |
| Mean code std | 0.204902 | pending late-window extraction |
| Cross-example cosine | 0.953795 | pending late-window extraction |
| Effective rank | 10.754587 | pending late-window extraction |
| Slot-diversity rank | 9.917890 | pending late-window extraction |

Do not compare the raw reconstruction columns as though DINOv3 and V-JEPA share a target space.
Use each run's correct-vs-shuffled separation, conditioned share, geometry trajectory, and stability.
