# GUIDE — running the investigation_008 SIGReg sweep on 4× A100 (end to end)

Execution walkthrough for the `λ_sigreg` sweep. Design/why is in
[`SIGREG_DESIGN.md`](SIGREG_DESIGN.md); this file is *how*. Same approach as inv007:
**4 independent runs in parallel, one config per GPU — NOT distributed (DDP)**, each
pinned with `CUDA_VISIBLE_DEVICES`, each with its own checkpoint dir + log, grouped in
W&B. Hardened against the inv007 Wave-2 synchronized step-200 pod death.

**Prerequisite context:** 4× A100 pod (40/80 GB — VRAM is a non-issue here, inv007 GUIDE
§6), network volume at `/workspace` with the repo at
`/workspace/hierarchal-jepa-flow-world-model` and SSv2 under `/workspace/data`, and a
`WANDB_API_KEY` in the pod. GPU *type* changes nothing: no DDP, so SXM/NVLink is
irrelevant; pinning is still `CUDA_VISIBLE_DEVICES`; no precision/flag changes.

---

## 0. ⛔ Prerequisite — SIGReg must be implemented and committed FIRST

This sweep needs code that does not exist yet on `phase1-v0.2-frozen-encoder`:
`losses.sigreg_loss`, `cfg.train.lambda_sigreg`, the `--lambda-sigreg` flag, and the
`L_sigreg` metric. Implement + unit-test + commit per
[`NEXT_STEPS.md`](NEXT_STEPS.md) Tier 0 and [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md) §2
**before** deploying the pod. The pod `git pull` then picks it up. Do not launch against
a build where `--lambda-sigreg` is unknown — the run would error at arg-parse (good) or,
worse, silently ignore it.

## 1. Connect + sync + smoke (do once)

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519     # from the RunPod Connect tab
nvidia-smi                                            # confirm 4 GPUs, indices 0..3

cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git checkout phase1-v0.2-frozen-encoder && git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline                                 # confirm the commit with sigreg_loss + --lambda-sigreg

python -c "from models import smoke_test_models; smoke_test_models()"   # model build / grad contract
python -c "import torch, losses; \
  z=torch.randn(64,32,256); print('sigreg N(0,I) ~', float(losses.sigreg_loss(z))); \
  print('sigreg low-rank >', float(losses.sigreg_loss(z[:, :, :8].repeat(1,1,32))))"
```

The second line is a SIGReg sanity: on `N(0,I)` it should be ~0; on a degenerate low-rank
input it should be clearly larger. If the flag/loss is missing, `git pull` didn't get the
commit — fix before continuing.

## 2. Launch the wave (4 configs, 4-wide, in tmux)

One process per GPU, each with its **own** checkpoint dir (non-negotiable — a shared dir
clobbers `phase1_step*.pt` across runs), each logging to its own file, grouped via
`WANDB_RUN_GROUP`. Only `λ_sigreg` varies; everything else is the eager-plant-22
background.

```bash
tmux new -s sweep008
cd /workspace/hierarchal-jepa-flow-world-model
mkdir -p logs
export HF_HOME=/workspace/hf_cache             # fresh tmux shell loses your env — re-export
                                               # (without it the runs miss the warm V-JEPA cache)
export WANDB_RUN_GROUP=inv008_sigreg           # W&B auto-groups all runs in this shell

# only column that varies: lambda_sigreg. Background is eager-plant-22 (decoder 512x4).
sigregs=( 0.3 1.0 3.0 10.0 )
for i in "${!sigregs[@]}"; do
  L="${sigregs[$i]}"
  tag="sigreg${L}_D512x4_nc32"
  CUDA_VISIBLE_DEVICES=$i python train.py \
    --data ssv2 --steps 15000 --horizon-k 12 --lr-coarse-flow 1e-4 \
    --lambda-var 0.5 --lambda-sigreg $L \
    --lambda-recon 0.05 --lambda-recon-pred 0 --recon-warmup-steps 2000 \
    --decoder-dim 512 --decoder-blocks 4 --n-c 32 \
    --checkpoint-dir /workspace/ckpt/$tag \
    --log-every 50 --diag-every 500 \
    > logs/$tag.log 2>&1 &
  echo "launched GPU $i -> $tag (pid $!)"
done
echo "all 4 launched — DETACH with Ctrl-B then D before closing SSH"
```

(`λ_sigreg=0` control = eager-plant-22, `591mt31k`, already on W&B — not re-run here.)

## 3. Hardening (the whole reason inv007 Wave 2 was wasted — do all three)

1. **Detach properly.** `Ctrl-B` then `D`, then **verify**: `tmux ls` must still list
   `sweep008`. *Only then* close SSH. (Leading suspect for the inv007 step-200 death was
   an SSH close without detach taking the process group down.)

2. **Step-0 λ-calibration check (NEW, ~2 min in).** Confirm `λ_sigreg·L_sigreg` is a
   sane fraction of `L_flow` before committing 8h — if SIGReg is inert even at λ=10 or
   dominates at λ=0.3, re-center the ladder (SIGREG_DESIGN §3) and relaunch:
   ```bash
   for f in logs/sigreg*.log; do echo "== $f"; grep -m1 -E "L_sigreg|L_flow" "$f"; done
   ```

3. **Step-600 tripwire (~15 min in).** Flags any run that never logged its first
   diagnostic (step 500) — the silent-early-death signature:
   ```bash
   sleep 900 && for f in logs/sigreg0.3_D512x4_nc32.log logs/sigreg1.0_D512x4_nc32.log \
                        logs/sigreg3.0_D512x4_nc32.log logs/sigreg10.0_D512x4_nc32.log; do
     grep -q "step.*500" "$f" && echo "OK   $f" || echo "STALL $f  <-- investigate"
   done
   ```
   Any `STALL` → `tail -n 50` that log (traceback vs bare `Killed`/SIGTERM vs `CUDA out of
   memory`) and check RunPod console → Pod → events for a stop/reclaim.

VRAM/OOM is **not** expected (frozen ViT-L dominates; SIGReg's pairwise term is
subsampled to ~0.13 GB, freed each step). If a run *does* OOM, lower `sigreg_loss`'s
`max_rows`/`n_projections`, or `global_batch`/`num_workers` in `config.py` (no CLI flag),
and relaunch that run alone.

## 4. Monitor

```bash
watch -n 5 nvidia-smi                                  # each GPU ~100% util + a python proc
tail -f logs/sigreg1.0_D512x4_nc32.log                 # follow one run (loss, diag dumps)
grep -H "step=15000" logs/*.log                        # completion check
```

In W&B, all 4 appear live under **`inv008_sigreg`** (project `hjepa-vwm`). Put
`c_effective_rank`, `coarse_vs_copy_ratio`, `L_recon_present`, `c_cross_video_cosine`,
`L_flow`, and `L_sigreg` on a grouped chart to read all 4 trajectories + the
eager-plant-22 control at once.

## 5. Read the results (the actual point)

Per run, against the matrix in [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md) §6, in priority
order:
1. **Did `c_effective_rank` break ~13 and climb toward 60+?** Which λ moved it most? If
   none did even at λ=10 → the bottleneck `B` binds rank (architectural), not the
   regularizer.
2. **If rank rose, did `coarse_vs_copy_ratio` fall toward <1?** Rank up *and* copy down =
   utilization was a real prediction lever → build on it. Rank up but copy still >1 =
   disease is temporal → the prediction pivot is unimpeachable (→ investigation_009).
3. **Health:** `L_flow` not blown up, `c_cross_video_cosine` low (falling, not rising).

Cut at the `c_effective_rank` plateau (~8–9k as in Wave 1; let it flatten rather than
fixing a hard step). Then write the cross-run synthesis into
[`OBSERVATIONS.md`](OBSERVATIONS.md) and record each run's W&B name+id in
[`DESCRIPTION.md`](DESCRIPTION.md)'s Runs table.

## 6. End-to-end recap

implement+commit SIGReg → deploy 4× A100 + volume → SSH → `nvidia-smi` → `git pull` +
smoke (incl. SIGReg sanity) → `tmux` + export `WANDB_RUN_GROUP=inv008_sigreg` → launch
4-wide (per-GPU pin, per-run ckpt dir) → detach (verify `tmux ls`) → step-0 λ check →
step-600 tripwire → ~8h → read `c_effective_rank` ∧ `coarse_vs_copy_ratio` per the matrix
→ write up → Tier-2 follow-up.

---

### Sources
- RunPod — [LLM Training with Runpod GPU Pods](https://www.runpod.io/articles/guides/llm-training-with-pod-gpus), [Connect to a Pod](https://docs.runpod.io/pods/connect-to-a-pod)
- W&B — [Organize runs into groups](https://docs.wandb.ai/models/runs/grouping) (`WANDB_RUN_GROUP`)
- SIGReg — LeJEPA (Balestriero & LeCun, 2025); implementation spec in [`SIGREG_DESIGN.md`](SIGREG_DESIGN.md) §2
