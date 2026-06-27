# Observations — helpful-snow-25 (λ_recon=1.0)

## 2026-06-27 — FAILED, no usable data

Died at **step 200** after **346 s**, last heartbeat **03:47:43Z** — same instant as all siblings
(synchronized whole-pod death; forensics in [Wave 2 OBSERVATIONS](../OBSERVATIONS.md)). Only **step 0
(initialization)** was logged.

### Step-0 values = initialization, NOT results

| Metric @ step 0 | Value | Note |
|---|---|---|
| `loss` | 3.087 | healthy init |
| `grad_norm` | 1.478 | healthy, no blow-up |
| `grad_has_nan` / `grad_skipped` | 0 / 0 | clean |
| `L_recon_present` | 1.025 | untrained |
| `c_effective_rank` | 9.47 | pre-training |

### This run's forensic value

Because it is **architecturally identical to Wave 1** (n_c=32, decoder 256×2) and still died at the
same step 200 as the n_c=256 run, it is the control that **rules out an `n_c` shape bug** as the cause
of the Wave-2 failure. The healthy step-0 `grad_norm`/`loss`/no-NaN also rule out divergence. The
failure was external (whole-pod / session / volume) — see [Wave 2 OBSERVATIONS](../OBSERVATIONS.md).

## What we still don't know

Whether λ=1.0 confirms the weight axis stays flat at the extreme (predicted ~0.581, no break) and
whether `L_flow` degrades. Untested — but **predictable**, hence low re-run priority
([NEXT_STEPS.md](NEXT_STEPS.md)).
