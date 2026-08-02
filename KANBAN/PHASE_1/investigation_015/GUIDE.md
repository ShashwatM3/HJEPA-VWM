# GUIDE - latent-stack bottleneck + whitened features, AE-only run (run 054)

This launches investigation_015 run 054 on a 1x A100 pod:

```text
present-only reconstruction
cosine loss
residual reconstruction target (e_w - mean_w)
FIXED offline feature whitening (NEW - has a one-time prerequisite, section 2)
Perceiver latent-stack bottleneck (new default architecture, 3 blocks)
no prediction, no SIGReg, no variance floor, no covariance, no slot penalty
```

This is the direct successor to inv013 run 053: the identical recipe, on the new
bottleneck, in whitened feature space. How to read the two changes independently:
[`README.md`](README.md).

**Entry point:** this guide assumes you have completed
[`AGENT_FILES/SETUPS/NEW_POD.md`](../../../AGENT_FILES/SETUPS/NEW_POD.md) sections 0-6
(SSH, apt packages, `HF_HOME`, repo pull, pip install, `wandb login`, data verified).
Do NOT run NEW_POD's section 7 Stage 0 — Stage 0 for this run needs the exact recipe
flags and the whitening stats file, both below.

Repo path on RunPod:

```bash
/workspace/hierarchal-jepa-flow-world-model
```

## 0. What This Run Is Testing

Keep every geometry/collapse regularizer off — the question is whether whitening (+ the
architecture fix) holds geometry WITHOUT them:

```text
lambda_recon = 0.05          lambda_var = 0.0
lambda_recon_pred = 0.0      lambda_cov = 0.0
present_recon_only = true    lambda_sigreg = 0.0
recon_residual_target = true lambda_slot = 0.0
whiten_features = true       bottleneck_latent_blocks = 3 (default)
```

Do not add covariance or SIGReg on this run; they would confound the whitening readout.

## 1. Verify Code Version

The latent-stack + whitening implementation must be on the pod. After NEW_POD section 3:

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git log --oneline -3
ls whiten_stats.py tests/test_whitening.py
python3 -c "from models import BottleneckLatentBlock, FeatureWhitener; print('new modules OK')"
```

Stop if `whiten_stats.py` is missing — the branch commit with the implementation has not
been pushed/pulled. Do not hand-copy files onto the pod; the run must be reproducible
from a commit.

Then the standard smoke gate:

```bash
pytest -q
python3 -c "from models import smoke_test_models; smoke_test_models()"
python3 -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

Stop on any failure.

## 2. PREREQUISITE - One-Time Whitening Statistics (NEW)

`train.py --whiten-features` refuses to start without the offline stats file. Compute it
once per dataset (this pass runs the frozen encoder over ~200 train batches; expect
roughly 30-60 min on an A100). Use tmux so an SSH drop cannot kill it:

```bash
tmux new -s inv015_whiten_stats
cd /workspace/hierarchal-jepa-flow-world-model
export HF_HOME=/workspace/hf_cache

python3 whiten_stats.py --data ssv2 --split train --max-batches 200 --seed 42
```

Expected output file (keep this exact path for every later step):

```text
logs/whiten/whiten_stats_ssv2_train_seed42.pt
```

Sanity on the printed summary:

- `rows` is in the millions (200 batches x 64 clips x 1024 tokens ~ 13.1M);
- top-5 eigenvalues are finite and the min eigenvalue is >= 0 (tiny is fine —
  `--whiten-eps` floors it at load time; default 1e-4).

Detach (`Ctrl-B`, `D`) or exit the session when it finishes. Notes:

- One-time per dataset: later runs on `ssv2` REUSE this file. Never recompute it
  mid-experiment (fixed statistics are the whole point — no per-batch whitening).
- The file is gitignored (`*.pt`); it lives on the pod volume like checkpoints do.
- Every checkpoint also embeds the whitener, so offline evaluation (`drift_probe.py`)
  never needs this file.

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
  --recon-residual-target \
  --recon-mean-momentum 0.99 \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ssv2_train_seed42.pt \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32
```

Stop if:

- Stage 0 produces NaN or a traceback (a `FileNotFoundError` naming `whiten_stats.py`
  means section 2 was skipped or the path is wrong);
- Stage 0 does not show `whiten_active=1.0`;
- Stage 0 does not show `recon_target_residual=1.0`;
- Stage 0 does not show `present_recon_only=1.0`;
- Stage 0 does not show `prediction_active=0.0`;
- Stage 0 does not show `L_flow=0.0`;
- Stage 0 does not show `L_recon_pred=0.0`.

At step 0, `recon_scale=0.0` (2000-step recon warmup) — expected. `recon_mean_norm`
will be much smaller than run 053's ~1624 because the mean is tracked in WHITENED
space — expected, and a useful confirmation whitening is live.

## 4. Launch On 1x A100

Use one tmux session:

```bash
tmux new -s inv015_latent_stack_whiten
cd /workspace/hierarchal-jepa-flow-world-model

export HF_HOME=/workspace/hf_cache
export WANDB_RUN_GROUP=inv015_ae_latent_stack_whiten
export WANDB_NAME=ae_latent_stack_whiten_recon_only
mkdir -p logs /workspace/ckpt/inv015_ae_latent_stack_whiten

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
  --recon-residual-target \
  --recon-mean-momentum 0.99 \
  --recon-warmup-steps 2000 \
  --present-recon-only \
  --whiten-features \
  --whiten-stats-path logs/whiten/whiten_stats_ssv2_train_seed42.pt \
  --decoder-dim 512 \
  --decoder-blocks 4 \
  --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv015_ae_latent_stack_whiten \
  --log-every 50 \
  --diag-every 500 \
  > logs/inv015_ae_latent_stack_whiten.log 2>&1 &

echo "launched inv015 latent-stack + whitening recon-only run on GPU 0 (pid $!)"
echo "detach with Ctrl-B then D, then verify with: tmux ls"
```

Do NOT pass `--resume` with any pre-existing checkpoint: the bottleneck architecture
changed, and old checkpoints fail to load by design (no migration layer).

## 5. Early Tripwires

Step 0 should appear within a few minutes:

```bash
grep -m1 "step=0 " logs/inv015_ae_latent_stack_whiten.log
```

Expected at step 0:

- `whiten_active=1.0`;
- `recon_target_residual=1.0`; finite, small `recon_mean_norm`;
- `present_recon_only=1.0`; `prediction_active=0.0`;
- `L_flow=0.0`; `L_recon_pred=0.0`;
- finite `L_recon` (around ~1.0 — cosine loss of an untrained decoder against
  near-isotropic whitened residuals);
- `recon_scale=0.0`; `sigreg_scale=0.0`; `grad_skipped=0`.

Also expected and mechanical, do not read as progress:

- `c_slot_diversity_rank` (raw) near 32 from the first diagnostic — fixed slot
  identities sit inside the output by construction. Judge slots ONLY on
  `c_slot_diversity_rank_centered`.

Step-500 check:

```bash
sleep 900
grep -q "step=500 " logs/inv015_ae_latent_stack_whiten.log \
  && echo "OK step 500 logged" \
  || echo "STALL before step 500 -- inspect log and RunPod events"
```

If it stalls:

```bash
tail -n 100 logs/inv015_ae_latent_stack_whiten.log
nvidia-smi
```

## 6. Monitor

Terminal:

```bash
watch -n 5 nvidia-smi
tail -f logs/inv015_ae_latent_stack_whiten.log
```

In W&B group `inv015_ae_latent_stack_whiten`, chart (attribution mapping per metric in
[`README.md`](README.md)):

- `L_recon_present`, `L_recon`, `L_recon_shuffled_c`, `L_recon_video_gap`
- `whiten_active`, `recon_target_residual`, `recon_mean_norm`
- `c_effective_rank`, `c_cross_video_cosine`, `c_std_mean`, `c_std_median`,
  `c_dead_dim_frac`
- `c_slot_diversity_rank`, `c_slot_diversity_rank_centered`
- `c_attn_entropy`, `c_attn_entropy_min`
- `grad_norm`, `grad_skipped`, `grad_has_nan`, `instability_warn`
- `agc_B_clipped`, `agc_B_max_ratio`, `agc_D_clipped`, `agc_D_max_ratio`

Ignore `coarse_*` panels — prediction is intentionally inactive.

The decisive window is steps ~4,000-12,000: that is where runs 052/053 entered their
geometry contraction phase while recon kept improving.

## 7. Verdict Rules

Success as a geometry-holding AE test requires ALL of:

- `L_recon_present` falls materially and monotonically after warmup;
- `L_recon_video_gap` clearly positive and growing (honesty carried over from 053);
- `c_effective_rank` does NOT contract through the 4k-12k window (053: 19.9 -> 10.5);
- `c_slot_diversity_rank_centered` does NOT decline monotonically (053: 12.5 -> 8.5);
- `c_cross_video_cosine` well below the 052/053 regime (~0.91-0.93);
- gradients finite, no sustained skips, AGC quiet.

Partial outcomes are still informative — attribute them with the 2x2 matrix in
[`README.md`](README.md) before deciding the follow-up (pre-registered options in
[`NEXT_STEPS.md`](NEXT_STEPS.md)).

Interpretation guards:

- `L_recon_*` absolute values are NOT comparable to runs 052/053 (whitened target
  space, different objective). Compare shapes, trends, and the honesty gap.
- `c_*` metrics use the same formulas on `c_t` as every prior run and remain the
  cross-run geometry evidence.
- A mid-run drift-probe checkpoint read is optional but supported: `drift_probe.py`
  rebuilds the whitener from the checkpoint automatically.
