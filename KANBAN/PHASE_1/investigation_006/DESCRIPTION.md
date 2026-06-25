# Investigation 006 — Does a reconstruction anchor break the rank ceiling?

**Status:** OPEN
**Opened:** 2026-06-25 (after `royal-cherry-17` rank collapse, investigation 005)
**Closed:** —

## Question

Does adding a **reconstruction anchor** — a small decoder `D` that rebuilds the
frozen detailed features `e_t` from the abstract latent `c_t`, trained with an MSE
whose gradient flows into **B only** (option 1, NOT through `F_c`) — lift
`c_effective_rank` past its ~13/256 ceiling, and does that richer `c` also reduce
the dimensional-collapse failure, without destabilizing training?

## Why it matters

Every healthy run to date (`cerulean-13`, `elated-15`, `drawn-16`, `royal-17`) caps
at `c_effective_rank ≈ 13.6–13.9` out of 256, far below the >60 spec gate
(`PHASE_1.md` §9.2). `L_flow + L_var` give `c` no reason to use more dimensions —
the copy baseline (~0.14) shows `c` is nearly static over the horizon. A
reconstruction MSE is a direct **information-richness** floor on `c`. Secondary
hypothesis: a higher-rank `c` is less "sharp" in the flow-matching landscape and so
less prone to the `royal-cherry-17` step-8600 cliff (see [005](../investigation_005/royal-cherry-17/OBSERVATIONS.md)).

## Scope (what this is and is not)

- **Option 1 only.** Decode the ONLINE `c_t`; gradient reaches `D` and `B`, never
  `F_c` (we decode `c_t`, not `c_hat`) and never the EMA targets. Built
  option-3-ready: the through-`F_c` "future anchor" is a later flag value, not a rewrite.
- **Variance floor stays ON.** `copper-sky-12` / `olive-terrain-11` are the natural
  experiment that the variance floor is load-bearing against representational
  collapse; it is not removed in the same step.
- Default `lambda_recon=0.0` → byte-identical baseline; the decoder is built/saved
  but not run in the train step until the flag is nonzero. Split recon readouts
  (`L_recon_present` / `_cplus` / `_chat`) are logged at diag cadence to calibrate
  `lambda_recon` and to scope whether option 3 is later warranted.

## Design options & decisions (the full reconstruction-loss discussion)

This investigation was scoped over a long design discussion; the durable decisions
are recorded here so they are not lost to chat history.

**The three options considered.** "Reconstruction" is not one mechanism — it is a
choice of *what* you decode and *where* the gradient goes:

| Option | Decode | Gradient reaches | Attacks | Verdict |
|---|---|---|---|---|
| **1 — present anchor** | online `c_t` → `e_t` | `D`, `B` (never `F_c`) | representation richness / stability | **Built first** (de-risk). Done: `fanciful-lake-18`. |
| **2 — the brief's version** | `c_hat` → `e_t`, with brief's stop-grad rule #2 (`c_hat` detached before fine head) | `D` only (recon does not shape `c`/`F_c`) | — | **Rejected.** The detach makes recon a passive readout, not a force on `c`; "brief is a pre-experiment prior, not ground truth." |
| **3 — predicted-latent anchor** | predicted `c_hat` → `e_{t+k}` | `D`, `F_c`, **`B` via conditioning** | prediction quality (copy gate) | **Next run.** Implemented as `--lambda-recon-pred`. |

**Why option 1 first, then option 3 (not straight to 3).** The path was deliberately
*phased to de-risk*, not because the endpoint differs from the joint objective. Option 1
isolates whether a reconstruction gradient into `B` helps at all and whether it
destabilizes the (empirically fragile) `F_c` site before we add gradient there. That gate
has now passed — option 1 was a clean stabilizer — so option 3 turns on the prediction
branch *alongside* the present anchor. The end state (both objectives, shared decoder) is
exactly the tech-lead's (Arbab) VITA-based proposal; the phasing was the only divergence
and it has served its purpose. See [`fanciful-lake-18/OBSERVATIONS.md`](fanciful-lake-18/OBSERVATIONS.md).

**Prior art.** VITA (also un-phased) found the load-bearing trick is to train
reconstruction on the **predicted** latent, not the actual latent — i.e. option 3, not
option 1. `fanciful-lake-18` independently produced the evidence for this: present-anchor
recon (`L_recon_chat ≈ L_recon_cplus`) is *blind to prediction error* while the copy gate
stays failed.

**Two orthogonal failure modes / levers** (don't conflate):
- **Prediction gap** — `F_c` loses to copy-forward → attacked by **option 3**.
- **Rank ceiling** (~13/256) — looks capacity/weight-bound, NOT a collapse → a *separate*
  lever (bigger `D`, higher `lambda_recon`, or revisiting whether >60 is the right target
  at 128:1 compression). Option 3 is not expected to fix this on its own.

**Finalized engineering decisions.**
- **Scale-free relative MSE** (`reconstruction_loss` = `mse / target.var()`, baseline ≈1.0)
  so `lambda_recon` needs no per-run calibration — `0.05` is a definitive weight.
- **Warmup ramp** (`recon_warmup_steps=2000`) eases the anchor in to protect the fragile
  early phase; shared by both anchors.
- **Option 3 conditioning is NOT detached** ("let it flow"): the prediction-recon gradient
  reaches `B` through the `F_c` conditioning on `c_t`, so `c` is shaped to be *predictable*,
  not merely reconstructable. This is the VITA joint-training intent and Arbab's
  expectation; reversible via a one-line `.detach()` if it misbehaves.

## Parent context

- Collapse evidence + the two failure modes: [investigation_005](../investigation_005/) (`royal-cherry-17`)
- Variance floor is load-bearing: [investigation_003](../investigation_003/) (`copper-sky-12`, `olive-terrain-11`)
- Spec gates (rank > 60, copy ratio < 0.70): [`AGENT_FILES/PHASES/PHASE_1.md`](../../AGENT_FILES/PHASES/PHASE_1.md) §9, §12
- Code: `models.Decoder`, `losses.reconstruction_loss`, `train.reconstruction_readouts`;
  flags `--lambda-recon` (option 1) and `--lambda-recon-pred` (option 3).

## Runs

| Run | Role | Outcome |
|---|---|---|
| [`fanciful-lake-18`](fanciful-lake-18/) | First active run: royal-cherry regime + `lambda_recon=0.05` | Mode-B cliff removed; rank ceiling held (~13.3); copy gate failed (2.59) → motivates option 3 |
| _(next)_ | Option 3: add predicted-latent anchor through `F_c` (joint objective) | pending launch |
