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
