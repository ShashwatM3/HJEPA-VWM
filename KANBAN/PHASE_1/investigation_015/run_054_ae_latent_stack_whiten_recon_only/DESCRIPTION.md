# Run 054 - `ae_latent_stack_whiten_recon_only`

**Current W&B run name:** `Investigation 15 · Whitened latent stack · Residual reconstruction`

**Investigation:** [investigation_015](../)
**W&B:** `lx1b6gw2` - https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/lx1b6gw2
**State:** `finished` (full 15000-step schedule; last diagnostic step 14500)
**Created:** 2026-07-04
**Mode:** present-reconstruction-only. `F_c`, the target-clip prediction, and the copy /
batch-mean gates are inactive for this run.
**Reading-cycle verdict:** **Low-rank decodable** - video-specific and honestly decoded (the
strongest honesty result of the program), but geometrically contracted, settling at a much
higher equilibrium than run 053.

## Research Role

The current frontier of the present-only autoencoder arc. It reruns run 053's exact recipe
(present-recon-only, cosine loss, residual target, zero geometry regularizers) with two deltas,
to test whether they can hold the geometry that pure reconstruction keeps losing.

## Hypothesis Being Tested

Run 053 confirmed that reconstruction alone cannot hold geometry (H2), and diagnosed the
mechanism as feature-space anisotropy: cosine reconstruction in raw space is satisfiable by
reproducing V-JEPA's few dominant directions, so decoupled weight decay contracts the rest.
Investigation_014 measured that anisotropy directly (pooled `e` entropy rank ~193/1024 with a
long low-energy tail). This run acts on both the space and the architecture:

- **Delta B - fixed offline whitening** (the hypothesis under test). Map every frozen feature
  through a fixed ZCA whitening built once offline (`whiten_stats.py`). Equalizing the target
  directions means reducing whitened-cosine loss requires matching MANY equally weighted
  directions, so reconstruction pressure should defend many more feature dimensions of `c`.
- **Delta A - the Perceiver latent-stack bottleneck** (an architecture fix, not a hypothesis).
  Three `BottleneckLatentBlock`s add slot self-attention (slot competition) so the 32 slots
  cannot cheaply merge onto the same readout.

Because pod time forced both changes into one run, the read is per-axis: whitening reads on the
feature-rank axis (`c_effective_rank` above the ~32-slot mechanical span), the latent stack
reads on the slot axis (`c_slot_diversity_rank_centered`). Attribution framework:
[`../README.md`](../README.md).

## Config Highlights

| Config key | Value |
|---|---:|
| `dataset` | ssv2 |
| `steps` | 15000 (finished) |
| `horizon_k` | 12 |
| `present_recon_only` | true |
| `recon_loss_mode` | cosine |
| `recon_residual_target` | true |
| `recon_mean_momentum` | 0.99 |
| `whiten_features` | **true** (delta B) |
| `whiten_stats_path` | logs/whiten/whiten_stats_ssv2_train_seed42.pt |
| `whiten_eps` | 1e-4 |
| `bottleneck_latent_blocks` | 3 (delta A, new default) |
| `lambda_recon` | 0.05 |
| `lambda_recon_pred` | 0 |
| `lambda_var` / `lambda_sigreg` / `lambda_cov` / `lambda_slot` | 0 / 0 / 0 / 0 |
| `n_c` | 32 |
| `decoder_dim` / `decoder_blocks` | 512 / 4 |
| `lr_bottleneck` / `lr_coarse_flow` | 1e-4 / 1e-4 |
| `weight_decay` | 0.05 |
| `seed` | 42 |

## Command

```bash
# one-time prerequisite (per dataset):
python whiten_stats.py --data ssv2 --split train --max-batches 200 --seed 42

python train.py \
  --data ssv2 --steps 15000 --seed 42 --horizon-k 12 \
  --lr-bottleneck 1e-4 --lr-coarse-flow 1e-4 \
  --lambda-var 0.0 --lambda-sigreg 0.0 --lambda-cov 0.0 --lambda-slot 0.0 \
  --lambda-recon 0.05 --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine --recon-residual-target --recon-mean-momentum 0.99 \
  --recon-warmup-steps 2000 --present-recon-only \
  --whiten-features --whiten-stats-path logs/whiten/whiten_stats_ssv2_train_seed42.pt \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv015_ae_latent_stack_whiten \
  --log-every 50 --diag-every 500
```

Full launch procedure incl. the one-time whitening prerequisite: [`../GUIDE.md`](../GUIDE.md).

## Difference From Previous W&B Run (run 053 `7teohhwc`)

Two deltas vs run 053: `--whiten-features` (+ its stats path / eps) and the latent-stack
bottleneck (`bottleneck_latent_blocks=3`, now the default). Everything else — present-recon-only,
cosine loss, residual target, decoder 512x4, `n_c=32`, `k=12`, all geometry lambdas 0, seed 42 —
matches run 053. One visible wiring consequence: `recon_mean_norm` is ~89.7 here vs run 053's
~1624, because the mean tracker now lives in whitened space (a confirmation whitening is live).

## Chronological Linkage

- Previous W&B run: Run 053 [`ae_sharp_slots_residual_recon`](../../investigation_013/run_053_ae_sharp_slots_residual_recon/) (the raw-space residual-target baseline this run whitens).
- Next: Run 055 [`ae_latent_stack_whiten_abs_recon`](../run_055_ae_latent_stack_whiten_abs_recon/)
  (completed as W&B `nzz64pl6`) removed the residual target to fill the {absolute, residual} x
  {raw, whitened} honesty 2x2. It left geometry unchanged and reduced conditioned improvement
  from about 92% to 86%, so the residual target stayed in the recipe. The bottleneck-only control
  (drop the two whitening flags, keep the residual target) remains open only for causal
  architecture-versus-whitening attribution.
- Parent investigation conclusion: "neither delta sufficient" — geometry still contracts, but at a much better equilibrium; H2 re-confirmed a third time.

## How To Read This Run

Present-recon-only cycle (Cycle B). The copy / batch-mean gates do NOT apply. Success would be
`c_effective_rank` NOT contracting through the 4k-12k window (053 halved there), centered slot
diversity NOT declining monotonically, cross-video cosine well below the 052/053 ~0.91 regime,
and a growing honesty gap. Absolute `L_recon_present` is in whitened target space and is NOT
comparable to run 052 (0.293) or run 053 (0.453). Read shapes, trends, and the honesty gap.
Rank up to ~32-40 is reachable from slot structure alone (32 fixed slot identities), so only
rank well above ~40 is positive evidence for the whitening mechanism.

## W&B Evidence

- Run id `lx1b6gw2`, group `inv015_ae_latent_stack_whiten`, state `finished`, last diagnostic
  step 14500. All 30 diagnostic checkpoints read individually (no downsampling).
- Full analysis: [`../ANALYSIS_054.md`](../ANALYSIS_054.md) (main body written live at step
  10000, with a final-numbers addendum after the run finished).
- Trajectory reconciled live via `get_run_history` for this record.
