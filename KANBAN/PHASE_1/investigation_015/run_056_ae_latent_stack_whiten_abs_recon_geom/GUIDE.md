# GUIDE — run 056 `ae_latent_stack_whiten_abs_recon_geom` (geometry-regularized run 055)

This launches investigation_015 run 056 on a 1x A100 pod:

```text
present-only reconstruction
cosine loss
ABSOLUTE (clean) whitened-feature target (residual target OFF, same as run 055)
FIXED offline feature whitening (reuses run 054/055 stats file)
Perceiver latent-stack bottleneck (default architecture, 3 blocks)
+ geometry regularizers: lambda_var=0.5, lambda_sigreg=5, lambda_cov=0.01 (inv011 run 047 bundle)
no prediction, no slot penalty
```

Identical to [`../run_055_ae_latent_stack_whiten_abs_recon/GUIDE.md`](../run_055_ae_latent_stack_whiten_abs_recon/GUIDE.md)
except three regularizer weights are turned on (and sigreg gets a warmup). Why this run:
[`HYPOTHESIS.md`](HYPOTHESIS.md); implementation detail: [`DESCRIPTION.md`](DESCRIPTION.md);
how to read the result: [`OBSERVATIONS.md`](OBSERVATIONS.md).

**Entry point:** [`AGENT_FILES/SETUPS/NEW_POD.md`](../../../../AGENT_FILES/SETUPS/NEW_POD.md)
sections 0-6 completed (SSH, apt, `HF_HOME`, repo pull, pip install, `wandb login`, data
verified). Do NOT run NEW_POD's section 7 Stage 0 — use section 3 below.

Repo path on RunPod:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

## 0. What This Run Is Testing

Run 055 (clean whitened AE) held reconstruction honest (~86% video-conditioned) but let
geometry contract with zero anti-collapse terms (terminal `c_effective_rank` 21.5,
`c_std_mean` 0.40, centered slot rank 12.1). This run adds the inv011 winning geometry bundle
on top of the exact same base. The question is whether the explicit anti-collapse force lifts
geometry toward the sweep's high-rank regime WITHOUT destroying the honesty the whitening +
latent stack established.

```text
lambda_recon      = 0.05          lambda_var    = 0.5    (ON — was 0.0)
lambda_recon_pred = 0.0           lambda_sigreg = 5.0    (ON — was 0.0)
present_recon_only = true         lambda_cov    = 0.01   (ON — was 0.0)
recon_residual_target = FALSE     lambda_slot   = 0.0
whiten_features   = true          sigreg_warmup_steps = 2000
bottleneck_latent_blocks = 3 (default)
```

Keep `recon_residual_target=FALSE` (clean base — the run-055 arm, deliberate) and
`lambda_slot=0` (slot loss was rejected in inv003; do not resurrect it).

## 1. Verify Code Version

Same commit family as runs 054/055 (latent stack + whitening + the three regularizers must
be present). No code change is needed for this run.

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git log --oneline -3
python3 -c "from models import BottleneckLatentBlock, FeatureWhitener; print('modules OK')"
python3 -c "from losses import variance_floor, sigreg_loss, covariance_floor; print('regularizers OK')"
pytest -q
python3 -c "from models import smoke_test_models; smoke_test_models()"
python3 -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

Stop on any failure.

## 2. Whitening Statistics (REUSE run 054/055's file)

The stats file is one-time per dataset and MUST be the same file runs 054/055 used — never
recompute it mid-experiment:

```bash
ls -la /workspace/hierarchal-jepa-flow-world-model/logs/whiten/whiten_stats_ssv2_train_seed42.pt
```

If (and only if) the file is missing — e.g. a fresh pod volume — recompute it exactly as run
054's guide section 2 did, in tmux:

```bash
tmux new -s inv015_whiten_stats
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache
python3 whiten_stats.py --data ssv2 --split train --max-batches 200 --seed 42
```

## 3. Stage 0 With The Exact Recipe

```bash
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache

python3 train.py \
  --stage0-only \
  --data ssv2 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 5.0 \
  --sigreg-warmup-steps 2000 \
  --lambda-cov 0.01 \
  --lambda-slot 0.0 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ssv2_train_seed42.pt \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32
```

Stop if:

- Stage 0 produces NaN or a traceback (`FileNotFoundError` naming the stats path means
  section 2 was skipped or the path is wrong);
- Stage 0 does not show `whiten_active=1.0`;
- Stage 0 does not show `recon_target_residual=0.0` (this run is the CLEAN arm — if this reads
  1.0 you launched the residual variant);
- Stage 0 does not show `recon_mean_norm=0.0` (mean tracker inactive without the residual
  target);
- Stage 0 does not show `present_recon_only=1.0` / `prediction_active=0.0`;
- Stage 0 does not show `L_flow=0.0` / `L_recon_pred=0.0`;
- `L_cov`, `L_var`, `L_sigreg` are not all finite and positive (they must be COMPUTED — the
  weights being nonzero is what adds them to the loss).

At step 0, `recon_scale=0.0` (2000-step recon warmup) and `sigreg_scale=0.0` (2000-step
sigreg warmup) — expected; both ramp to 1.0 by step 2000. `L_var`/`L_cov` enter the loss at
full weight from step 0 (no warmup on those two).

## 4. Launch On 1x A100

Use one tmux session:

```bash
tmux new -s inv015_whiten_abs_recon_geom
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv015_ae_latent_stack_whiten
export WANDB_NAME=ae_latent_stack_whiten_abs_recon_geom
mkdir -p logs /workspace/ckpt/inv015_whiten_abs_recon_geom

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data ssv2 \
  --steps 15000 \
  --seed 42 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.5 \
  --lambda-sigreg 5.0 \
  --sigreg-warmup-steps 2000 \
  --lambda-cov 0.01 \
  --lambda-slot 0.0 \
  --lambda-recon 0.05 \
  --lambda-recon-pred 0.0 \
  --recon-loss-mode cosine \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ssv2_train_seed42.pt \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv015_whiten_abs_recon_geom \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv015_whiten_abs_recon_geom.log 2>&1 &

echo "launched inv015 run 056 (geometry-regularized clean whitened AE) on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Do NOT pass `--resume` with any checkpoint from runs 052/053 (old bottleneck architecture,
rejected by design) or runs 054/055 (this run must train from scratch to be comparable — same
seed, same init path).

## 5. Early Tripwires

```bash
grep -m1 "step=0 " logs/inv015_whiten_abs_recon_geom.log
```

Expected at step 0: `whiten_active=1.0`, `recon_target_residual=0.0`, `recon_mean_norm=0.0`,
`present_recon_only=1.0`, `prediction_active=0.0`, `L_flow=0.0`, `L_recon_pred=0.0`, finite
positive `L_var`/`L_cov`/`L_sigreg`, `recon_scale=0.0`, `sigreg_scale=0.0`, `grad_skipped=0`.

Confirm the three weights landed in the W&B run config: `lambda_var=0.5`, `lambda_sigreg=5`,
`lambda_cov=0.01`. If any reads 0, the loss is the run-055 baseline, not this run.

Mechanical, do not read as progress: raw `c_slot_diversity_rank` near 32 and pooled
`c_effective_rank` near 31 at init (fixed slot identities — run 054's measurement caveat).
Judge slots on `c_slot_diversity_rank_centered` only.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv015_whiten_abs_recon_geom.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

## 6. Monitor

```bash
watch -n 5 nvidia-smi
tail -f logs/inv015_whiten_abs_recon_geom.log
```

In W&B group `inv015_ae_latent_stack_whiten`, overlay THIS run against run 055 (`nzz64pl6`,
same base, no geometry terms) and inv011 run 047 (`az60m6mx`, same geometry bundle, old
architecture / raw space). Decisive panels:

- Geometry (the headline): `c_effective_rank`, `c_slot_diversity_rank_centered`,
  `c_cross_video_cosine`, `c_std_mean`, `c_dead_dim_frac`. Watch for run 055's post-2.5k
  contraction being HELD or reversed.
- Honesty (must survive): `L_recon_shuffled_c`, `L_recon_video_gap`, `L_recon_present`,
  `L_recon`.
- Regularizer terms: `L_var`, `L_cov`, `L_sigreg` (should fall as the bundle bites),
  `sigreg_scale`, `recon_scale`.
- Attention: `c_attn_entropy`, `c_attn_entropy_min`.
- Wiring: `whiten_active` (=1), `recon_target_residual` (=0), `recon_mean_norm` (=0).
- Stability: `grad_norm`, `grad_skipped`, `grad_has_nan`, `instability_warn`, `agc_B_*`,
  `agc_D_*`.

Ignore `coarse_*` panels — prediction is intentionally inactive.

## 7. Verdict Rules

Judged jointly on GEOMETRY (does the anti-collapse bundle lift it) and HONESTY (does it
survive), per the priors in [`HYPOTHESIS.md`](HYPOTHESIS.md):

- **Bundle works, honesty survives (the target win):** `c_effective_rank` holds/climbs well
  above 40 (not contracting post-peak), `c_std_mean` → ~1.0, `c_cross_video_cosine` well
  below run 055's 0.82 toward the sweep's ~0.08, AND `L_recon_shuffled_c` stays high (≈ run
  055's 0.95+) with the video-conditioned share ≥ ~86%. First present-only run with both
  honest recon and healthy geometry — next step: transfer to full prediction.
- **Geometry lifts but honesty breaks:** rank/std/cosine improve but the video gap collapses
  (`L_recon_shuffled_c` falls toward `L_recon_present`). Signals a geometry term shredding
  content routing — diagnose which (drop cov vs sigreg) and reconsider the residual base.
- **Geometry does not lift:** the bundle fails to overcome the whitened-space equilibrium
  (unexpected given inv011). Check `L_cov`/`L_sigreg` actually fell; if they did and rank
  didn't move, the whitened base resists the raw-space lever — a real finding.
- **Validity gate:** gradients finite, no sustained skips, AGC quiet, wiring flags correct.

Interpretation guards:

- `L_recon_present` is comparable to run 055 only loosely (both clean whitened space); the
  geometry terms make the code carry more, which may raise `L_recon_present` somewhat — read
  the honesty SHARE and geometry curves, not the raw recon value. Never compare to raw-space
  runs 052/053, nor to inv011's raw-space `L_recon_present`.
- `c_*` metrics use the same formulas as every prior run and remain cross-run evidence.
