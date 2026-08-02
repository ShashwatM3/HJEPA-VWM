# GUIDE - run 055 `ae_latent_stack_whiten_abs_recon` (absolute-target ablation of run 054)

This launches investigation_015 run 055 on a 1x A100 pod:

```text
present-only reconstruction
cosine loss
ABSOLUTE whitened-feature target (residual target REMOVED - the only change vs run 054)
FIXED offline feature whitening (reuses run 054's stats file)
Perceiver latent-stack bottleneck (default architecture, 3 blocks)
no prediction, no SIGReg, no variance floor, no covariance, no slot penalty
```

Identical to [`../GUIDE.md`](../GUIDE.md) (run 054) except two flags are dropped:
`--recon-residual-target` and `--recon-mean-momentum 0.99`. How to read the result:
[`DESCRIPTION.md`](DESCRIPTION.md) priors in [`OBSERVATIONS.md`](OBSERVATIONS.md).

**Entry point:** [`AGENT_FILES/SETUPS/NEW_POD.md`](../../../../AGENT_FILES/SETUPS/NEW_POD.md)
sections 0-6 completed (SSH, apt, `HF_HOME`, repo pull, pip install, `wandb login`,
data verified). Do NOT run NEW_POD's section 7 Stage 0 — use section 2 below.

Repo path on RunPod:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

## 0. What This Run Is Testing

Keep every geometry/collapse regularizer off. The question is honesty, not geometry:
does whitening ALONE block the run-052 template shortcut, or is the residual target
still required?

```text
lambda_recon = 0.05           lambda_var = 0.0
lambda_recon_pred = 0.0       lambda_cov = 0.0
present_recon_only = true     lambda_sigreg = 0.0
recon_residual_target = FALSE lambda_slot = 0.0
whiten_features = true        bottleneck_latent_blocks = 3 (default)
```

Do not add geometry terms on this run; they would confound the honesty readout.

## 1. Verify Code Version

Same commit family as run 054 (latent stack + whitening must be present):

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git log --oneline -3
python3 -c "from models import BottleneckLatentBlock, FeatureWhitener; print('modules OK')"
pytest -q
python3 -c "from models import smoke_test_models; smoke_test_models()"
python3 -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

Stop on any failure.

## 2. Whitening Statistics (REUSE run 054's file)

The stats file is one-time per dataset and MUST be the same file run 054 used —
never recompute it mid-experiment:

```bash
ls -la /workspace/hierarchal-jepa-flow-world-model/logs/whiten/whiten_stats_ssv2_train_seed42.pt
```

If (and only if) the file is missing — e.g. a fresh pod volume — recompute it exactly
as run 054's guide section 2 did, in tmux:

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
  --lambda-var 0.0 \
  --lambda-sigreg 0.0 \
  --lambda-cov 0.0 \
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
- Stage 0 does not show `recon_target_residual=0.0` (THE run-defining flag — if this
  reads 1.0 you launched run 054 again);
- Stage 0 does not show `recon_mean_norm=0.0` (mean tracker inactive without the
  residual target);
- Stage 0 does not show `present_recon_only=1.0` / `prediction_active=0.0`;
- Stage 0 does not show `L_flow=0.0` / `L_recon_pred=0.0`.

At step 0, `recon_scale=0.0` (2000-step recon warmup) — expected.

## 4. Launch On 1x A100

Use one tmux session:

```bash
tmux new -s inv015_whiten_abs_recon
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv015_ae_latent_stack_whiten
export WANDB_NAME=ae_latent_stack_whiten_abs_recon
mkdir -p logs /workspace/ckpt/inv015_whiten_abs_recon

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data ssv2 \
  --steps 15000 \
  --seed 42 \
  --horizon-k 12 \
  --lr-bottleneck 1e-4 \
  --lr-coarse-flow 1e-4 \
  --lambda-var 0.0 \
  --lambda-sigreg 0.0 \
  --lambda-cov 0.0 \
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
  --checkpoint-dir /workspace/ckpt/inv015_whiten_abs_recon \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv015_whiten_abs_recon.log 2>&1 &

echo "launched inv015 run 055 (absolute whitened target) on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Do NOT pass `--resume` with any checkpoint from runs 052/053 (old bottleneck
architecture, rejected by design) or run 054 (this run must train from scratch to be
comparable — same seed, same init path).

## 5. Early Tripwires

```bash
grep -m1 "step=0 " logs/inv015_whiten_abs_recon.log
```

Expected at step 0: `whiten_active=1.0`, `recon_target_residual=0.0`,
`recon_mean_norm=0.0`, `present_recon_only=1.0`, `prediction_active=0.0`, `L_flow=0.0`,
`L_recon_pred=0.0`, finite `L_recon`, `recon_scale=0.0`, `grad_skipped=0`.

Mechanical, do not read as progress: raw `c_slot_diversity_rank` near 32 from the first
diagnostic, and pooled `c_effective_rank` near 31 at init (fixed slot identities —
run 054's measurement caveat). Judge slots on `c_slot_diversity_rank_centered` only.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv015_whiten_abs_recon.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

## 6. Monitor

```bash
watch -n 5 nvidia-smi
tail -f logs/inv015_whiten_abs_recon.log
```

In W&B group `inv015_ae_latent_stack_whiten`, the decisive panels for THIS run are the
honesty probes, overlaid against run 054 (`lx1b6gw2`) and run 052 (`662hfy3c`):

- `L_recon_shuffled_c` (run 054: pinned 0.969-0.975; run 052 signature: falls with
  `L_recon_present`)
- `L_recon_video_gap`, `L_recon_present`, `L_recon`
- `c_effective_rank`, `c_slot_diversity_rank_centered`, `c_cross_video_cosine`,
  `c_std_mean`, `c_dead_dim_frac`
- `c_attn_entropy`, `c_attn_entropy_min`
- `whiten_active`, `recon_target_residual` (must stay 0), `recon_mean_norm` (must stay 0)
- `grad_norm`, `grad_skipped`, `grad_has_nan`, `instability_warn`, `agc_B_*`, `agc_D_*`

Ignore `coarse_*` panels — prediction is intentionally inactive.

## 7. Verdict Rules

This run is judged primarily on HONESTY, with geometry as the secondary read
(pre-registered priors in [`OBSERVATIONS.md`](OBSERVATIONS.md)):

- Residual target redundant in whitened space: `L_recon_shuffled_c` pinned near ~1.0
  all run, video-conditioned share of decoder improvement ~90%+ (run 054 level).
- Residual target still required: `L_recon_shuffled_c` declines materially alongside
  `L_recon_present`; the video-conditioned share lands far below 92% (toward run 052's
  ~15% in the worst case).
- Intermediate shares are still decisive for the recipe: any material honesty loss vs
  run 054 keeps `--recon-residual-target` in the recipe (it costs nothing).
- Geometry: compare the equilibrium (rank/cosine/std at 14.5k) against run 054's
  21.9 / 0.820 / 0.397 — contraction itself is expected (H2 settled).
- Validity gate: gradients finite, no sustained skips, AGC quiet, wiring flags correct.

Interpretation guards:

- `L_recon_present` is easier under the absolute target than run 054's residual
  target — compare shapes and shares, not raw values; never compare values to raw-space
  runs 052/053.
- `c_*` metrics use the same formulas as every prior run and remain cross-run evidence.
