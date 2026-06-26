# Investigation 007 — What binds the reconstruction capacity floor?

**Status:** OPEN
**Opened:** 2026-06-27 (after `easy-blaze-19` / investigation_006 capacity-floor finding)
**Closed:** —

## Question

Every reconstruction readout in `fanciful-lake-18` (option 1) and `easy-blaze-19` (option 3)
is pinned at **`L_recon_* ≈ 0.60`** (relative MSE). That floor is *why* the reconstruction
anchor (and option 3 in particular) failed to improve prediction — a good and a bad prediction
reconstruct identically, so the objective is blind to prediction quality. **What binds the
floor — the loss weight, the decoder size, or the latent `c` size?**

## How we answer it

An **8-run, one-factor-at-a-time (OFAT) sweep** over three axes, run 8-wide in parallel on a
6–8 GPU pod (≈ one run's wall-clock):
- `lambda_recon` ∈ {0.1, 0.2, 0.5} — is the floor **weight-bound**?
- decoder size (`decoder_dim × decoder_blocks`) ∈ {256×2, 512×2, 512×4} — **decoder-bound**?
- latent size `n_c` ∈ {32, 64, 128} — **capacity-bound**? (my prior bet)
- + 1 combined "all bigger" run. `lambda_recon_pred = 0` throughout (isolate the floor; option 3
  was inert at it).

**Primary readout:** does `L_recon_present` drop below ~0.55? **Secondary:** if it drops, does
`coarse_vs_copy_ratio` fall toward <1 (is prediction actually reconstruction-bound)?

## Full design, values, safety analysis, execution

- **Design + reasoning + interpretation matrix:** [`SWEEP_PLAN_decoder_capacity.md`](SWEEP_PLAN_decoder_capacity.md)
- **End-to-end execution (pod → parallel launch → monitor):** [`GUIDE.md`](GUIDE.md)
- **Per-run records:** each completed run gets an `investigation_007/<wandb-name>/` triad here.

## Parent context

- The capacity-floor finding + option-3 negative result: [investigation_006](../investigation_006/) (`easy-blaze-19`)
- The reconstruction-anchor design taxonomy (option 1/2/3): [investigation_006/DESCRIPTION.md](../investigation_006/DESCRIPTION.md)
- Code: flags `--lambda-recon`, `--decoder-dim`, `--decoder-blocks`, `--n-c`, `--checkpoint-dir`
  (all on `phase1-v0.2-frozen-encoder`); `models.Decoder`, `losses.reconstruction_loss`.

## Runs

| Run | Config (λ_recon · decoder · n_c) | Outcome |
|---|---|---|
| _(pending launch — Stage 1 wave of 8)_ | see `SWEEP_PLAN` §3 | — |
