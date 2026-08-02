# GUIDE — running the investigation_007 sweep on a 6–8 GPU pod (end to end)

Execution walkthrough for the capacity-floor sweep. Design/why is in
[`SWEEP_PLAN_decoder_capacity.md`](SWEEP_PLAN_decoder_capacity.md); this file is *how*. The
approach is **8 independent runs in parallel, one config per GPU — NOT distributed (DDP)
training**, because for a sweep that keeps each run's dynamics identical to prior runs and needs
zero distributed-training code.

Official docs this guide is grounded in (cited inline):
- RunPod multi-GPU pods & GPU exposure — RunPod docs/guides.
- W&B run grouping (`group=` / `WANDB_RUN_GROUP`) — W&B docs.

---

## 0. What you need before starting

- A RunPod **GPU Pod with 6–8 GPUs**. RunPod supports **up to 8 GPUs per single-node pod**
  (common 2×/4×/8× configs), and the GPUs are **indexed from 0** and addressed with the standard
  `CUDA_VISIBLE_DEVICES` env var — no RunPod-specific API.
  (RunPod: ["LLM Training with Runpod GPU Pods"](https://www.runpod.io/articles/guides/llm-training-with-pod-gpus),
  ["Manage Pods"](https://docs.runpod.io/pods/manage-pods).)
- Your **network volume** mounted (the RunPod convention mounts it at **`/workspace`**), with the
  repo already on it at `/workspace/hierarchal-jepa-flow-world-model` and the SSv2 data under
  `/workspace/data`.
- A `WANDB_API_KEY` available in the pod (env var or `wandb login`).

> Sanity-confirm the volume mount path in the RunPod console (Pod → Storage) — it's `/workspace`
> for network volumes by default, which is what the repo paths assume.

## 1. Connect to the pod

Get the exact SSH command from the pod's **Connect** tab in the RunPod console (it gives an
`ssh root@<ip> -p <port> -i <key>` line). See RunPod
["Connect to a Pod"](https://docs.runpod.io/pods/connect-to-a-pod). Then:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519     # from the Connect tab
nvidia-smi                                            # confirm all 6–8 GPUs are visible (indices 0..N-1)
```

`nvidia-smi` should list every GPU. If it shows fewer than expected, the pod wasn't deployed with
that GPU count — redeploy (you can't add GPUs to a running pod).

## 2. Sync code + sanity check (do this ONCE)

```bash
cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git checkout phase1-v0.2-frozen-encoder && git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline          # confirm the commit with the new sweep flags

# Gradient-contract smoke test (no GPU needed; ~seconds). Verifies option-1/option-3 routing
# and that the model builds — catches any flag wiring problem before you burn 8 GPUs for hours.
python -c "from models import smoke_test_models; smoke_test_models()"
```

If the smoke test passes you're safe to launch. (It builds `B`, `F_c`, `D`, runs the
forward/backward, and asserts the recon gradients reach the right modules.)

## 3. Launch the Stage-1 wave (8 configs, 8-wide, in tmux)

One process per GPU, pinned with `CUDA_VISIBLE_DEVICES`, each with its **own checkpoint dir**
(non-negotiable — a shared dir clobbers `phase1_step*.pt` across runs in real time), each logging
to its own file, all backgrounded, grouped in W&B via `WANDB_RUN_GROUP`.

```bash
tmux new -s sweep007
cd /workspace/hierarchal-jepa-flow-world-model
mkdir -p logs
export HF_HOME=/workspace/hf_cache             # fresh tmux shell loses your step-2 export;
                                               # without it the runs miss the warm V-JEPA cache
export WANDB_RUN_GROUP=inv007_capacity_floor   # W&B auto-groups all runs in this shell
                                               # (W&B: "Specify the experiment name to
                                               #  automatically group runs together.")

# columns: lambda_recon  decoder_dim  decoder_blocks  n_c   (lambda_recon_pred = 0 throughout)
configs=(
  "0.1  256 2 32"     # weight
  "0.2  256 2 32"     # weight
  "0.5  256 2 32"     # weight (aggressive)
  "0.05 512 2 32"     # decoder width
  "0.05 512 4 32"     # decoder depth
  "0.05 256 2 64"     # latent 2x
  "0.05 256 2 128"    # latent 4x
  "0.2  512 2 64"     # combined "all bigger"
)
for i in "${!configs[@]}"; do
  read L DDIM DBLK NC <<< "${configs[$i]}"
  tag="L${L}_D${DDIM}x${DBLK}_nc${NC}"
  CUDA_VISIBLE_DEVICES=$i python train.py \
    --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
    --lambda-recon $L --lambda-recon-pred 0 --recon-warmup-steps 2000 \
    --decoder-dim $DDIM --decoder-blocks $DBLK --n-c $NC \
    --checkpoint-dir /workspace/ckpt/$tag \
    --log-every 50 --diag-every 500 \
    > logs/$tag.log 2>&1 &
  echo "launched GPU $i -> $tag (pid $!)"
done
wait        # block until all 8 finish (~3–4h)
echo "ALL RUNS DONE"
```

- **Detach** (leave it running, safe to disconnect SSH): `Ctrl+B` then `D`.
- **Reattach:** `tmux attach -t sweep007`.
- **6-GPU pod:** use 6 of the 8 configs (drop the two least informative, e.g. one weight value
  and the combined run), or run the remaining 2 as a second wave after the first finishes.

Why `CUDA_VISIBLE_DEVICES=$i`: it makes process `i` see **only** physical GPU `i` (remapped to
its local `cuda:0`), so each run is isolated to one card. This is the standard CUDA mechanism
RunPod exposes; no framework changes needed.

### 3b. 5-GPU pod (e.g. A100 SXM) — two waves of 5 (10 configs)

With 5 GPUs we run **two sequential waves of 5** instead of one 8-wide pass. Since the second
wave would otherwise leave GPUs idle, we extend the design to **10 configs**, adding two
axis-saturation extremes (`lambda_recon=1.0`, `n_c=256`) that close the OFAT blind spot where a
"flat" axis is ambiguous with "not pushed hard enough" (rationale in
[`SWEEP_PLAN`](SWEEP_PLAN_decoder_capacity.md) §3 / §4b).

**GPU *type* changes nothing here:** VRAM is a non-issue on A100 (40/80 GB — same as on the RTX
PROs, §6), there is **no DDP** so SXM/NVLink interconnect is irrelevant, pinning is still
`CUDA_VISIBLE_DEVICES`, and no precision/flag changes are needed. Fewer concurrent runs also
means *less* network-volume contention, so each run may step slightly faster than 8-wide.

```bash
tmux new -s sweep007
cd /workspace/hierarchal-jepa-flow-world-model
mkdir -p logs
export HF_HOME=/workspace/hf_cache             # fresh tmux shell — re-export (see 3a note)
export WANDB_RUN_GROUP=inv007_capacity_floor

launch() {   # args: lambda decoder_dim decoder_blocks n_c gpu_index
  local L=$1 DDIM=$2 DBLK=$3 NC=$4 G=$5
  local tag="L${L}_D${DDIM}x${DBLK}_nc${NC}"
  CUDA_VISIBLE_DEVICES=$G python train.py \
    --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
    --lambda-recon $L --lambda-recon-pred 0 --recon-warmup-steps 2000 \
    --decoder-dim $DDIM --decoder-blocks $DBLK --n-c $NC \
    --checkpoint-dir /workspace/ckpt/$tag \
    --log-every 50 --diag-every 500 > logs/$tag.log 2>&1 &
  echo "launched GPU $G -> $tag (pid $!)"
}

# --- Wave 1: weight axis (3) + decoder axis (2) ---
wave1=( "0.1 256 2 32" "0.2 256 2 32" "0.5 256 2 32" "0.05 512 2 32" "0.05 512 4 32" )
for i in "${!wave1[@]}"; do read L D B N <<< "${wave1[$i]}"; launch $L $D $B $N $i; done
wait; echo "WAVE 1 DONE"

# --- Wave 2: latent axis (2) + combined + saturation extremes (NEW) ---
wave2=( "0.05 256 2 64" "0.05 256 2 128" "0.2 512 2 64" "1.0 256 2 32" "0.05 256 2 256" )
for i in "${!wave2[@]}"; do read L D B N <<< "${wave2[$i]}"; launch $L $D $B $N $i; done
wait; echo "WAVE 2 DONE — ALL 10 RUNS COMPLETE"
```

Total wall-clock ≈ 2× a single wave (~6–8h). **Watch the two new runs specifically:** `L1.0...`
for `L_flow` degradation (recon now equals the flow weight), and `...nc256` for slot-collapse
(`c_slot_diversity_rank`, `c_cross_video_cosine`) — both flagged in SWEEP_PLAN §4b as the only
elevated-risk points. All 10 still group live under `inv007_capacity_floor` in W&B.

### 3c. ⭐ LIVE PLAN — 4-GPU Wave-2 re-run (this is what to launch now)

**Context:** the original 5-wide Wave 2 (§3b) **died at step 200** in a synchronized whole-pod
death — no usable data ([`END_OF_WAVE_2.md`](END_OF_WAVE_2.md) §1). Those 5 failed runs are being
**deleted from W&B**. This is the hardened re-run on a **4-GPU pod**.

**Why these 4 (and why drop `λ=1.0`):** Wave 1 already killed the weight axis with three clean
points (λ 0.1→0.2→0.5 → `L_recon_present` 0.596→0.592→0.586, a flat −0.005/doubling line) and
confirmed live that `L_flow` *didn't even degrade* at λ=0.5 (≈0.42, same as λ=0.1). So the
`λ=1.0` saturation run (`helpful-snow-25`) is the single most predictable run in the wave — floor
~0.581, `L_flow` fine, `copy_ratio` flat — **zero expected information**, and it's dropped. The
4 we keep are: the **full latent ladder `n_c` 64/128/256** (the only axis Wave 2 exists to test —
read a *trend*, not just bookends) plus the **combined "all bigger"** run as the one
interaction-effect check. Strategy detail: [`END_OF_WAVE_2.md`](END_OF_WAVE_2.md) §1.4 /
[`wave_2/NEXT_STEPS.md`](wave_2/NEXT_STEPS.md).

| GPU | λ_recon · decoder · n_c | Role |
|---|---|---|
| 0 | 0.05 · 256×2 · **64**  | latent bookend — low ⭐ |
| 1 | 0.05 · 256×2 · **128** | latent interpolation |
| 2 | 0.05 · 256×2 · **256** | latent bookend — high ⭐ (highest value; watch OOM + slot collapse) |
| 3 | 0.2 · 512×2 · **64**   | combined "all bigger" — interaction insurance |
| ~~—~~ | ~~1.0 · 256×2 · 32~~ | **DROPPED** — predictable closure, no GPU for it |

```bash
tmux new -s sweep007
cd /workspace/hierarchal-jepa-flow-world-model
mkdir -p logs
export HF_HOME=/workspace/hf_cache             # fresh tmux shell — re-export (see 3a note)
export WANDB_RUN_GROUP=inv007_capacity_floor   # same group → sits next to Wave 1 in W&B

# columns: lambda_recon  decoder_dim  decoder_blocks  n_c   (lambda_recon_pred = 0 throughout)
configs=(
  "0.05 256 2 64"     # n_c=64  — latent bookend (low)
  "0.05 256 2 128"    # n_c=128 — latent interpolation
  "0.05 256 2 256"    # n_c=256 — latent bookend (high); most VRAM-hungry (OOM not expected — see §6)
  "0.2  512 2 64"     # combined "all bigger" — interaction check
)
for i in "${!configs[@]}"; do
  read L DDIM DBLK NC <<< "${configs[$i]}"
  tag="L${L}_D${DDIM}x${DBLK}_nc${NC}"
  CUDA_VISIBLE_DEVICES=$i python train.py \
    --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
    --lambda-recon $L --lambda-recon-pred 0 --recon-warmup-steps 2000 \
    --decoder-dim $DDIM --decoder-blocks $DBLK --n-c $NC \
    --checkpoint-dir /workspace/ckpt/$tag \
    --log-every 50 --diag-every 500 \
    > logs/$tag.log 2>&1 &
  echo "launched GPU $i -> $tag (pid $!)"
done
echo "all 4 launched — DETACH with Ctrl-B then D before closing SSH"
```

**Hardening (the whole reason the last wave was wasted) — do all three:**

1. **Detach properly.** `Ctrl-B` then `D`, then **verify** the session survived: `tmux ls` should
   still list `sweep007`. Only *then* close SSH. (The leading suspect for the step-200 death is an
   SSH close without detach taking the process group down.)
2. **Step-600 tripwire.** ~15 min after launch, run this once — it flags any run that hasn't yet
   logged its first diagnostic (step 500), which is exactly the silent-early-death signature:
   ```bash
   sleep 900 && for f in logs/L0.05_D256x2_nc64.log logs/L0.05_D256x2_nc128.log \
                        logs/L0.05_D256x2_nc256.log logs/L0.2_D512x2_nc64.log; do
     grep -q "step.*500" "$f" && echo "OK   $f" || echo "STALL $f  <-- investigate"
   done
   ```
   Any `STALL` → `tail -n 50` that log immediately (traceback vs bare `Killed`/SIGTERM vs `CUDA out
   of memory`) and check the RunPod console → Pod → events for a stop/reclaim.
3. **n_c=256 OOM watch.** It's the only run that grows VRAM, but OOM is **not expected** — the
   frozen ViT-L dominates memory and n_c only adds a small latent/KV tensor (§6: "VRAM is a
   non-issue" on A100/H100). There is **no CLI batch flag**; if `logs/...nc256.log` *does* show
   `CUDA out of memory`, the only knobs are `global_batch` (`cfg.train`, default 64) and
   `num_workers` (`cfg.data`, default 8) in `config.py` — lower one and relaunch that run alone.

Cut at the `L_recon_present` plateau (~8–9k as in Wave 1), but let the latent runs flatten rather
than fixing a hard step — more slots may reorganize slightly later. Read each run against the
triad in §5: **`L_recon_present` ∧ `c_effective_rank` ∧ `coarse_vs_copy_ratio`.**

## 4. Monitor

```bash
# In another shell on the pod (or split tmux pane):
watch -n 5 nvidia-smi                 # every GPU should show ~100% util + a python process
tail -f logs/L0.5_D256x2_nc32.log     # follow one run's stdout (loss, diag dumps)
grep -H "step=15000\|ALL RUNS DONE" logs/*.log   # completion check
```

In **W&B**, all 8 appear live under the **`inv007_capacity_floor`** group (project `hjepa-vwm`).
You can view/filter runs by group in the workspace (W&B:
["Organize runs into groups"](https://docs.wandb.ai/models/runs/grouping)). Add the key metrics
(`L_recon_present`, `coarse_vs_copy_ratio`, `c_effective_rank`, `c_slot_diversity_rank`) to a
grouped chart to read all 8 trajectories at once.

## 5. Read the results (the actual point)

Per run, against the interpretation matrix in SWEEP_PLAN §4:
1. **Did `L_recon_present` drop below ~0.55?** Identify *which axis* (weight / decoder / `n_c`)
   moved it. If none did → reconstruction is the wrong lever (pivot to horizon/task).
2. **If it dropped, did `coarse_vs_copy_ratio` fall toward <1?** Floor down *and* copy-ratio down
   = reconstruction was the right lever → Stage 2 (re-add option 3 on the unsaturated config).
   Floor down but copy-ratio still failed = prediction isn't reconstruction-bound.

Then write the cross-run synthesis into `investigation_007/OBSERVATIONS.md` and create per-run
folders (or one wave write-up).

## 6. Gotchas & tuning (read before launching)

- **Per-run `--checkpoint-dir` is mandatory.** Without the distinct `$tag` dir, 8 runs write
  `/workspace/ckpt/phase1_step2500.pt` simultaneously and corrupt each other.
- **Dataloader / network-IO contention is the real bottleneck**, not GPU. 8 runs × `num_workers`
  each = many workers all decoding video off the **network volume** → network IO may cap the
  speedup below a clean 8×. Mitigations, in order of payoff:
  1. **Cache the frozen-encoder features** (precompute `e_t`/`e_{t+k}` once — the encoder is
     frozen, so they're deterministic — and train on cached tensors). Makes runs GPU-bound and
     barely touch the dataset. Biggest win; turns ~6h runs much shorter. *(Not yet built — ask if
     you want it.)*
  2. Stage the dataset to **local NVMe** on the pod (if available) instead of the network volume.
  3. Lower per-run `num_workers` (e.g. 4) so 8 runs don't oversubscribe CPU.
- **VRAM is a non-issue** — each run is the frozen ViT-L + ~4M trainable at batch 64; RTX PRO
  cards have far more than enough. One run per GPU (compute-bound) is correct; don't pack two.
- **`--resume` is NOT valid across architecture changes.** `--decoder-dim/-blocks/--n-c` change
  tensor shapes, so a checkpoint from one config can't resume another. Fresh runs only.
- **`WANDB_PROJECT`** is already `hjepa-vwm` in code; `WANDB_RUN_GROUP` (set above) is the only
  env var you need for grouping (W&B reads it automatically).

## 7. End-to-end recap

deploy 8-GPU pod + network volume → SSH in → `nvidia-smi` (confirm GPUs) → `git pull` + smoke
test → `tmux` + export `WANDB_RUN_GROUP` → launch 8-wide loop (per-GPU pin, per-run checkpoint
dir) → detach → ~3–4h → read `L_recon_present` + `coarse_vs_copy_ratio` per the matrix → write up
→ Stage 2.

---

### Sources
- RunPod — [LLM Training with Runpod GPU Pods](https://www.runpod.io/articles/guides/llm-training-with-pod-gpus) (up to 8 GPUs/node, CUDA_VISIBLE_DEVICES), [Manage Pods](https://docs.runpod.io/pods/manage-pods), [Connect to a Pod](https://docs.runpod.io/pods/connect-to-a-pod)
- W&B — [Organize runs into groups](https://docs.wandb.ai/models/runs/grouping), [Environment variables](https://docs.wandb.ai/guides/track/environment-variables/) (`WANDB_RUN_GROUP`: "Specify the experiment name to automatically group runs together"; `WANDB_PROJECT`)
