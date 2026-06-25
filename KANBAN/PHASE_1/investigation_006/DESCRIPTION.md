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

## Parent context

- Collapse evidence + the two failure modes: [investigation_005](../investigation_005/) (`royal-cherry-17`)
- Variance floor is load-bearing: [investigation_003](../investigation_003/) (`copper-sky-12`, `olive-terrain-11`)
- Spec gates (rank > 60, copy ratio < 0.70): [`AGENT_FILES/PHASES/PHASE_1.md`](../../AGENT_FILES/PHASES/PHASE_1.md) §9, §12
- Code: `models.Decoder`, `losses.reconstruction_loss`, `train.reconstruction_readouts`, flag `--lambda-recon`.

## Runs

| Run | Role |
|---|---|
| _(pending launch)_ | Calibration: `lambda_recon=0`, read baseline `L_recon_present` magnitude |
| _(pending launch)_ | First active run: `royal-cherry-17` regime + calibrated `--lambda-recon` |
