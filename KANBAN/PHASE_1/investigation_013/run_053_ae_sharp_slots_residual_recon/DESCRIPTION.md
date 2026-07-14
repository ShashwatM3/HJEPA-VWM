# Run 053 - `ae_sharp_slots_residual_recon`

**Current W&B run name:** `Investigation 13 · Sharp slots · Residual reconstruction`

**Investigation:** [investigation_013](../)
**W&B:** `7teohhwc` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/7teohhwc
**State:** `crashed` (external kill at step ~12150 / 15000; no training pathology)
**Created:** 2026-07-03
**Last history step:** 12150
**Mode:** present-reconstruction-only. `F_c`, the target-clip prediction, and the copy /
batch-mean gates are inactive for this run.
**Reading-cycle verdict:** **Low-rank decodable** - the decoder can reconstruct from `c_t`
and the code is genuinely video-specific, but the bottleneck uses far too few effective
directions/slots (and this time it is NOT template collapse).

## Research Role

The direct successor to run 052. It tests the run-052 fix in isolation: does subtracting the
per-position feature mean (residual reconstruction target) force reconstruction pressure to
carry video-specific information through `c_t`, and does that alone hold the geometry that
run 052 lost?

## Hypothesis Being Tested

Run 052 collapsed for two possible reasons, entangled: (H1) the decoder learned a
video-independent template that satisfied reconstruction without using `c_t`, and (H2)
reconstruction pressure by itself cannot maintain rank/variance. Reconstructing the residual
`e - mean` (where `mean` is an EMA per-tubelet-position mean) makes the template worth zero
loss, surgically removing H1. If geometry then holds, H2 was never the problem; if geometry
still collapses, H1 is solved and H2 is confirmed. The added honesty probe
(`L_recon_video_gap`) makes "is the decoder using this video's code?" a direct measurement.

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2 |
| `steps` | 15000 (crashed ~12150) |
| `horizon_k` | 12 |
| `present_recon_only` | true |
| `recon_loss_mode` | cosine |
| `recon_residual_target` | **true** (the change vs run 052) |
| `recon_mean_momentum` | 0.99 |
| `recon_warmup_steps` | 2000 |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `lambda_var` / `lambda_sigreg` / `lambda_cov` / `lambda_slot` | 0 / 0 / 0 / 0 |
| `whiten_features` | false (raw feature space; contrast run 054) |
| `n_c` | 32 |
| `decoder_dim` / `decoder_blocks` | 512 / 4 |
| `lr_bottleneck` / `lr_coarse_flow` | 1e-4 / 1e-4 |
| `weight_decay` | 0.05 |
| `seed` | 42 |

## Command

```bash
python train.py \
  --data ssv2 --steps 15000 --seed 42 --horizon-k 12 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 \
  --lambda-var 0.0 --lambda-sigreg 0.0 --lambda-cov 0.0 --lambda-slot 0.0 \
  --lambda-recon 0.05 --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine --recon-residual-target --recon-mean-momentum 0.99 \
  --recon-warmup-steps 2000 --present-recon-only \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv013_residual_recon_only \
  --log-every 50 --diag-every 500
```

Full launch procedure (pod state, smoke gate, tripwires): [`GUIDE.md`](GUIDE.md).

## Difference From Previous W&B Run (run 052 `662hfy3c`)

Exactly one flag added: `--recon-residual-target` (plus `--recon-mean-momentum 0.99`, its
companion). Everything else — sharp-slot bottleneck, cosine loss, present-recon-only, decoder
512x4, `n_c=32`, `k=12`, all geometry lambdas 0, seed 42 — is identical to run 052. This makes
053-vs-052 a clean single-variable comparison on the reconstruction TARGET (absolute vs
residual), matched on architecture and everything else.

## Chronological Linkage

- Previous W&B run: Run 052 [`ae_sharp_slots_recon_only`](../../investigation_012/run_052_ae_sharp_slots_recon_only/) (the template-collapse baseline this run fixes).
- Next in the research chain: [investigation_014](../../investigation_014/) (offline rank probe, no training run) then Run 054 `ae_latent_stack_whiten_recon_only` (`lx1b6gw2`, [investigation_015](../../investigation_015/)), which reruns this exact recipe with whitening + the latent-stack bottleneck.
- Parent investigation conclusion: the residual target fixes reconstruction honesty but not geometry; H1 solved, H2 confirmed.

## How To Read This Run

Use the present-reconstruction-only cycle (Cycle B) in `GUIDES/READING_EXPERIMENTS.md`.
Do NOT apply the copy / batch-mean prediction gates — `F_c` is inactive. Success for a
present-only run is a `c_t` that is decodable, spread (`c_std_mean` ~1), video-specific
(`c_cross_video_cosine` < 0.5), and high-rank (`c_effective_rank` well above the ~32-slot
mechanical span), with a clearly positive, growing `L_recon_video_gap`. Absolute
`L_recon_present` values are NOT comparable to run 052's 0.293 — the residual target is a
strictly different (harder) objective.

## W&B Evidence

- Run id `7teohhwc`, group `inv013_residual_recon_only`, state `crashed` (external), last
  diagnostic step 12000, last history step 12150.
- Full analysis reading Cycle B against all 25 diagnostic points (no downsampling):
  [`ANALYSIS_053.md`](ANALYSIS_053.md).
- Trajectory pulled live via `get_run_history` on 2026-07-03; numbers reconciled again for
  this record on the current session.
