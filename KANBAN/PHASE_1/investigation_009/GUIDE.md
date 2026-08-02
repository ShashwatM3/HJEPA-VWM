# GUIDE — running the investigation_009 wave on a 2× A100 pod (end to end)

Execution walkthrough for the residual-prediction wave. Design/why is in
[`DESCRIPTION.md`](DESCRIPTION.md); this file is *how*. Same approach as inv007/008:
**independent runs in parallel, one config per GPU — NOT distributed (DDP)**, each pinned with
`CUDA_VISIBLE_DEVICES`, its own checkpoint dir + log, grouped in W&B. Hardened against the inv007
Wave-2 synchronized step-200 pod death.

**This GUIDE picks up where [`AGENT_FILES/SETUPS/NEW_POD.md`](../../../AGENT_FILES/SETUPS/NEW_POD.md)
leaves off.** Do NEW_POD §0–§7 first (system packages, `HF_HOME`, clone/pull, `pip install -r
requirements.txt`, `wandb login`, data verify, `--stage0-only` preflight). Then come here for the
sync-to-inv009 + smoke + 2-wide launch.

**Prerequisite context:** 2× A100 pod (40/80 GB — VRAM is a non-issue, inv007 GUIDE §6), network
volume at `/workspace` with the repo at `/workspace/hierarchal-jepa-flow-world-model` and SSv2 under
`/workspace/data`, and a `WANDB_API_KEY` in the pod. No DDP, so SXM/NVLink is irrelevant; pinning is
`CUDA_VISIBLE_DEVICES`.

---

## 0. ⛔ Prerequisite — the inv009 code must be committed + pushed FIRST

This wave needs code that is **implemented but uncommitted** on `phase1-v0.2-frozen-encoder`:
`cfg.train.predict_residual`, the `--predict-residual` flag, `losses.residual_target`, the residual
branches in `train.py`, and the residual-aware `diagnostics.coarse_baselines`. **Commit + push it**
(per [`NEXT_STEPS.md`](NEXT_STEPS.md) Tier 0) before deploying the pod, or the `git pull` below won't
get it. The flag defaults to the current behavior, so the commit is safe for any other run.

## 1. Sync code + smoke (do once)

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519     # from the RunPod Connect tab
nvidia-smi                                            # confirm 2 GPUs, indices 0..1

cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git checkout phase1-v0.2-frozen-encoder && git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline                                 # must show the commit with --predict-residual

export HF_HOME=/workspace/hf_cache                   # if a fresh shell (NEW_POD §2)

# (a) Model build + ALL gradient contracts incl. the NEW residual path (ĉ=c_t+Δ̂ reaches
#     D, F_c, B, never the EMA bottleneck). ~seconds, no GPU/encoder download.
python -c "from models import smoke_test_models; smoke_test_models()"

# (b) Residual train-step on the REAL encoder + synthetic batch (load/forward/backward/EMA).
#     This is the end-to-end residual sanity — it exercises train_step with predict_residual=True.
python train.py --stage0-only --predict-residual

# (c) Full-latent stage0 still byte-identical (sanity that the flag is truly off by default).
python train.py --stage0-only
```

If (a) or (b) raises, the residual wiring is wrong — fix before burning GPU-hours. If `--predict-residual`
is an "unrecognized argument", the `git pull` didn't get the commit (§0).

## 2. Launch the wave (2 configs, 2-wide, in tmux)

The two runs differ in more than one knob (this is a 2-probe wave, not a clean OFAT sweep), so launch
them as two explicit commands. Each pins its GPU, writes its **own** checkpoint dir (non-negotiable —
a shared dir clobbers `phase1_step*.pt`), logs to its own file, and groups via `WANDB_RUN_GROUP`.

```bash
tmux new -s wave009
cd /workspace/hierarchal-jepa-flow-world-model
mkdir -p logs
export HF_HOME=/workspace/hf_cache             # fresh tmux shell loses your env — re-export
export WANDB_RUN_GROUP=inv009_residual         # W&B auto-groups both runs in this shell

# --- GPU 0 · Run 1 — SIGReg-only substrate (full latent, NO recon) ---
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-sigreg 6.0 --lambda-var 0.5 \
  --lambda-recon 0 --lambda-recon-pred 0 \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv009_run1_sigreg6_full \
  --log-every 50 --diag-every 500 \
  > logs/inv009_run1_sigreg6_full.log 2>&1 &
echo "launched GPU 0 -> Run 1 (pid $!)"

# --- GPU 1 · Run 2 — residual prediction + recon ---
CUDA_VISIBLE_DEVICES=1 python train.py \
  --data ssv2 --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
  --lambda-sigreg 5.0 --lambda-var 0.5 \
  --lambda-recon 0.05 --lambda-recon-pred 0.05 --recon-warmup-steps 2000 \
  --predict-residual \
  --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
  --checkpoint-dir /workspace/ckpt/inv009_run2_sigreg5_residual \
  --log-every 50 --diag-every 500 \
  > logs/inv009_run2_sigreg5_residual.log 2>&1 &
echo "launched GPU 1 -> Run 2 (pid $!)"

echo "both launched — DETACH with Ctrl-B then D before closing SSH"
```

(Control for both = inv008 full-latent history, already on W&B — not re-run here.)

## 3. Hardening (the whole reason inv007 Wave 2 was wasted — do all three)

1. **Detach properly.** `Ctrl-B` then `D`, then **verify**: `tmux ls` must still list `wave009`.
   *Only then* close SSH. (Leading suspect for the inv007 step-200 death was an SSH close without
   detach taking the process group down.)

2. **Step-0 component check (NEW — residual-aware, ~2 min in).** Confirm the residual run isn't
   pathological before committing 8h. Step 0 logs diagnostics (`diag_every=500` ⇒ step 0 logs):
   ```bash
   for f in logs/inv009_run1_sigreg6_full.log logs/inv009_run2_sigreg5_residual.log; do
     echo "== $f"; grep -m1 "step=0 " "$f"; done
   ```
   Sanity for **Run 2**: `predict_residual` is in the config dump; `L_flow` and `loss` are finite (no
   NaN); `coarse_copy_loss` is at the **natural Δ scale (~0.1–0.25, smaller than full-latent's ~0.6)**
   — that confirms the noise-scaling path is live. `coarse_vs_copy_ratio` is comparable to Run 1 (the
   σ-scaling cancels). If `L_flow` is NaN or `coarse_copy_loss` is ~1+ (unscaled noise leaked in),
   stop and recheck `residual_target` / the eps scaling.

3. **Step-600 tripwire (~15 min in).** Flags any run that never logged its first post-init diagnostic
   (the silent-early-death signature):
   ```bash
   sleep 900 && for f in logs/inv009_run1_sigreg6_full.log logs/inv009_run2_sigreg5_residual.log; do
     grep -q "step=500 " "$f" && echo "OK   $f" || echo "STALL $f  <-- investigate"
   done
   ```
   Any `STALL` → `tail -n 50` that log (traceback vs bare `Killed`/SIGTERM vs `CUDA out of memory`) and
   check RunPod console → Pod → events for a stop/reclaim.

VRAM/OOM is **not** expected (frozen ViT-L dominates; the residual path adds one cheap `B_EMA(e_t)`
forward + a scalar). If a run OOMs, lower `global_batch` / `num_workers` in `config.py` (no CLI flag)
and relaunch that run alone.

## 4. Monitor

```bash
watch -n 5 nvidia-smi                                          # each GPU ~100% util + a python proc
tail -f logs/inv009_run2_sigreg5_residual.log                 # follow the residual run
grep -H "step=15000" logs/*.log                               # completion check
```

In W&B both appear live under **`inv009_residual`** (project `hjepa-vwm`). Put
`coarse_vs_copy_ratio`, `c_effective_rank`, `L_recon_chat`, `L_recon_present`, `L_flow`, `L_sigreg`,
`c_std_mean`, and `c_cross_video_cosine` on a grouped chart to read both trajectories + the inv008
full-latent control at once.

## 5. Read the results (the actual point)

Per [`DESCRIPTION.md`](DESCRIPTION.md) / [`OBSERVATIONS.md`](OBSERVATIONS.md), in priority order:

1. **Run 2 — did `coarse_vs_copy_ratio` fall below the full-latent baseline (toward <1)?** This is the
   decision metric and it is **comparable across runs** (same `‖Δ‖²` copy baseline). Ratio down =
   residual prediction extracted motion → a real lever → k-sweep / anti-collapse-on-Δ follow-up. Ratio
   ≥ 1 tracking full-latent = residual alone insufficient → the horizon/rollout pivot.
   - Read the **ratio**, not the absolutes: Run 2's `L_flow` / `coarse_copy_loss` are at natural Δ
     scale (smaller than full-latent) — that scale difference is expected, not a regression.
2. **Run 2 — did `L_recon_chat` drop** under the residual decode `ĉ=c_t+Δ̂`? Likely still ~0.585
   (inv007 recon-blindness); a drop would resurrect the reconstruction-with-residuals thesis.
3. **Run 1 — is SIGReg-only a clean substrate?** `c_effective_rank` ~55 (watch if it clears 60),
   `c_std_mean` ~0.90 (floor active), `c_dead_dim_frac=0`, no cliff. Its `coarse_vs_copy_ratio` is
   expected **worse** than baseline (the inv008 law) — that's characterization, not failure.
   (Run 1's `L_recon_*` are meaningless — `λ_recon=0`, untrained decoder. Ignore them.)
4. **Health (both):** `L_flow` not blown up, `c_cross_video_cosine` < 0.5 (not climbing), and the
   gradient in band — read **`grad_norm`** (pre-clip, logged every step), **not**
   `grad_global_norm_postclip` (pinned near the 0.5 clip; WALK_FIXES F1).

Cut at the `c_effective_rank` / `coarse_vs_copy_ratio` plateau (~8–9k as in inv008; let it flatten
rather than fixing a hard step). Then write the synthesis into [`OBSERVATIONS.md`](OBSERVATIONS.md)
and record each run's W&B name+id in [`DESCRIPTION.md`](DESCRIPTION.md)'s Runs table.

## 6. End-to-end recap

NEW_POD §0–§7 → commit+push inv009 → SSH → `nvidia-smi` (2 GPUs) → `git pull` + smoke (incl.
`smoke_test_models` residual contract + `--stage0-only --predict-residual`) → `tmux` + export
`WANDB_RUN_GROUP=inv009_residual` → launch 2-wide (per-GPU pin, per-run ckpt dir) → detach (verify
`tmux ls`) → step-0 component check → step-600 tripwire → ~8h → read `coarse_vs_copy_ratio` (Run 2 vs
full-latent) per §5 → write up → Tier-2 follow-up.

---

### Sources
- RunPod — [LLM Training with Runpod GPU Pods](https://www.runpod.io/articles/guides/llm-training-with-pod-gpus), [Connect to a Pod](https://docs.runpod.io/pods/connect-to-a-pod)
- W&B — [Organize runs into groups](https://docs.wandb.ai/models/runs/grouping) (`WANDB_RUN_GROUP`)
- Design + residual rationale: [`DESCRIPTION.md`](DESCRIPTION.md); sibling GUIDEs
  [`investigation_008/GUIDE.md`](../investigation_008/GUIDE.md), [`investigation_007/GUIDE.md`](../investigation_007/GUIDE.md)
