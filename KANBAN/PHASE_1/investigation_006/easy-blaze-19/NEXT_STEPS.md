# Next steps — easy-blaze-19

## Status

Run **finished** (full 15k). Clean A/B vs `fanciful-lake-18`. **Option 3 is a negative
result.** Investigation 006 stays **OPEN** — both options now characterized; the open
question has shifted from "which anchor" to "is reconstruction the right lever at all,
given the capacity floor."

## → This spawned investigation_007 (the capacity-floor sweep)

The open question above is now its own investigation:
**[investigation_007](../../investigation_007/)** — *what binds the ~0.60 reconstruction
floor?* An **8-run OFAT sweep** over three axes — `lambda_recon` (weight-bound?), decoder size
(decoder-bound?), and `n_c` (latent-capacity-bound? — the prior bet) — run **8-wide in parallel
on a 6–8 GPU pod** (≈ one run's wall-clock). `lambda_recon_pred=0` throughout to isolate the
floor (option 3 was inert at it). Primary readout: does `L_recon_present` drop below ~0.55?
If yes, which axis moved it; if no axis does → reconstruction is the wrong lever, pivot to the
horizon/task. Full design in [investigation_007/SWEEP_PLAN](../../investigation_007/SWEEP_PLAN_decoder_capacity.md),
execution in [investigation_007/GUIDE](../../investigation_007/GUIDE.md). CLI flags
(`--decoder-dim/-blocks/--n-c/--checkpoint-dir`) are implemented and committed.

## What this run settled

- **The predicted-latent anchor (option 3) does not improve prediction.** Copy gate
  unchanged, `L_recon_chat` did not drop. Do not re-run option 3 at this latent capacity.
- **Cheap-vs-clean `c_hat` is moot.** The ~0.60 floor makes both equivalent — don't spend
  time building the clean from-noise variant expecting it to fix this.
- **The reconstruction channel is capacity-saturated (~0.60).** This is the gating fact for
  any further reconstruction-based idea.
- **Stability is a solved property of recon-into-`B`** (no cliff in either run) — carry forward.

## The decision fork (pick one before the next launch)

**(A) Break the capacity floor, then re-test option 3.** Make reconstruction unsaturated so it
*can* carry prediction error. Levers, in order of directness:
- Bigger latent: raise `n_c` (32) or `d_c` (256) — widens the actual information channel.
  (Decoder size is NOT the limit; `D`≈2.17M params already.) Requires shape plumbing + a fresh
  EMA/checkpoint (not resume-compatible).
- Or higher `lambda_recon` (e.g. 0.2–0.5) to force more of `c`'s existing 256 dims into use —
  cheap to try first; tests whether the floor is weight-bound or truly capacity-bound.
- Success test: does `L_recon_present` drop below ~0.55? If yes, the floor moved and option 3
  is worth re-running; if it stays ~0.60, the floor is structural at this `c` size.

**(B) Abandon reconstruction for the prediction problem; attack the task instead.** The copy
baseline is strong because `c` barely moves over horizon-12 (`‖Δc‖/‖c‖` ~0.38 and falling;
`coarse_copy_loss` 0.23 → 0.13). That points at a **task/horizon** issue, not an `F_c` failure:
- Probe a longer horizon (more `c` movement to predict) or a horizon sweep.
- Reconsider whether `coarse_vs_copy_ratio ≤ 0.70` is even achievable when the target is near-static.

**(C) Revisit the rank gate itself.** `c` is healthy by every measure except rank (~13/256).
Re-examine whether `>60` is the right target at 128:1 compression before treating rank as the
headline failure (carried from `fanciful-lake-18`).

**Recommendation:** try **(A)-cheap first** — a `lambda_recon=0.2` ablation (keep
`lambda_recon_pred=0`) is one run that decisively tells us whether the 0.60 floor is weight- or
capacity-bound. That answer gates everything else (bigger `c` is expensive; the cheap ablation
de-risks it). If the floor doesn't move, pivot to **(B)**.

## Suggested next run (option A-cheap, capacity-floor probe)

```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 \
  --lr-coarse-flow 1e-4 --lambda-recon 0.2 --recon-warmup-steps 2000 \
  --log-every 50 --diag-every 500
```
Watch **`L_recon_present`**: drops below ~0.55 ⇒ floor is weight-bound (raise λ / proceed to
bigger `c` + re-test option 3); stays ~0.60 ⇒ floor is structural ⇒ pivot to (B).

If running several λ values, do them **8-wide in parallel** (one per GPU, `CUDA_VISIBLE_DEVICES`,
per-run `--checkpoint-dir`) — a sweep of ≤8 configs costs one run's wall-clock (~3–4h).

## Abort rules (carry from 005)

- `L_flow > 1.5` for 200 consecutive steps
- `c_effective_rank` drops > 3 points in 500 steps
- `agc_Fc_max_ratio` median > 200 over any 500-step window
