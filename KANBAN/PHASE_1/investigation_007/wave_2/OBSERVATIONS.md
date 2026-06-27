# Observations — Wave 2

**This wave produced no training data.** What follows is the forensic record of the failure (so a
future agent doesn't mistake the step-0 values for results) plus what remains predicted-but-untested.
Full version: [`../END_OF_WAVE_2.md`](../END_OF_WAVE_2.md). Numbers pulled from W&B.

---

## 2026-06-27 — Forensic record of the failed wave

### The failure signature (identical across all 5)

| Run | runtime | last step | last heartbeat | logged history |
|---|---|---|---|---|
| [pious-mountain-28](pious-mountain-28/) | 350 s | 200 | 03:47:43Z | step 0 only |
| [classic-yogurt-29](classic-yogurt-29/) | 381 s | 200 | 03:47:43Z | step 0 only |
| [earnest-dragon-25](earnest-dragon-25/) | 352 s | 200 | 03:47:43Z | step 0 only |
| [helpful-snow-25](helpful-snow-25/) | 346 s | 200 | 03:47:43Z | step 0 only |
| [quiet-firebrand-25](quiet-firebrand-25/) | 347 s | 200 | 03:47:43Z | step 0 only |

**Smoking gun:** five independent processes stopped at the *same second* → the pod / tmux session
/ process group went down together, not a per-run crash.

### What it rules out (so nobody re-debugs the code)

- **Not divergence.** Step-0 init healthy: `loss` ≈ 3.1–3.2, `grad_norm` ≈ 1.5–1.7,
  `grad_has_nan = 0`, `grad_skipped = 0`, `instability_warn = 0`.
- **Not an n_c shape bug.** `helpful-snow-25` (n_c=32, identical architecture to all of Wave 1)
  died at the same step as the n_c=256 run. A latent-shape bug would have spared n_c=32.
- **Not throughput.** ~1.75 s/step vs Wave 1's ~1.55 s/step (delta = first-200-step startup).

### Most likely cause

A whole-pod event ~6 min after launch: pod stop/reclaim, tmux session killed (SSH closed without
`Ctrl-B D` detach?), network-volume dropout across all 5 dataloaders, or a disk/OOM cascade. W&B
can't disambiguate — the evidence is in `logs/<tag>.log` on the pod (if still alive). Diagnose with
`tail -n 50 logs/*.log` (traceback vs. bare `Killed`/SIGTERM), RunPod pod events, `dmesg | grep -i oom`.

### Why the step-0 numbers are NOT results

Diagnostics log every 500 steps, so each run has only its step-0 row. Those values
(`L_recon_present` ≈ 1.03 = "as bad as predicting the mean"; `c_cross_video_cosine` ≈ 0.69–0.74 =
collapsed random init; `copy_ratio` ≈ 47–67) are **pure initialization**, identical in character to
every Wave-1 run at step 0. They say nothing about n_c. See each run's `OBSERVATIONS.md`.

## Still predicted, still untested

The [Wave-1 pre-registration](../WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md) stands unchanged: ~60% the
floor stays flat (utilization confirmed → pivot); ~30% it drops cosmetically but `copy_ratio` stays
>1 (→ pivot); ~10% floor down ∧ `copy_ratio`→<1 ∧ rank up (latent capacity is the lever → Stage 2).
**A re-run (reduced to `n_c=64` + `n_c=256`) is required to settle it** — [NEXT_STEPS.md](NEXT_STEPS.md).
