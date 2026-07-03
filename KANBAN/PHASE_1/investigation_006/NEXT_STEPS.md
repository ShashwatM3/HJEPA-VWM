# Next Steps - investigation_006

<!-- AUTO-GENERATED-WANDB-KANBAN -->

## Current Recommendation

The project moved into decoder capacity and latent-utilization sweeps to determine whether the reconstruction floor was architectural capacity or c-space utilization.

## Closure / Carry-Forward Status

- Status: **CLOSED**.
- Conclusion to carry forward: Reconstruction improved stability/readouts but did not break the rank ceiling or make F_c beat copy. Prediction-side reconstruction also showed that the decoder could be blind to whether c_hat was actually a good future latent.
- Runs covered: 018, 019.

## Follow-Up Chain

This investigation feeds into `investigation_007`: decoder capacity, reconstruction weight, n_c, and reconstruction-floor diagnosis. The reason is: Investigation 006 showed reconstruction was helpful but capped. This branch tested whether simply changing decoder/reconstruction capacity could lower the floor or reveal a richer c_t.

## Guardrails For Future Reuse

- Do not cite a present-only result as a prediction success.
- Do not cite a low `L_flow` as success without the copy and batch-mean gates.
- Do not compare residual-mode copy ratios against full-latent copy ratios without naming the mode difference.
- When reviving this branch, start from the exact run folder and W&B id, not a remembered nickname.

## Original Notes Preserved

# Next steps — investigation 006

## Experiment ladder

| # | Run | `lambda_recon` | Result |
|---|---|---|---|
| 1 | [`fanciful-lake-18`](run_018_fanciful-lake-18/OBSERVATIONS.md) | 0.05 (present anchor) | **DONE.** Cliff removed; rank ceiling NOT broken (~13.3); copy gate failed (2.59). |
| 2 | _(next — spawn folder)_ | 0.05 present + **0.05 predicted (option 3, through-`F_c`)** | Test whether the predicted-latent anchor fixes the copy gate. |

**Decision after run 1:** proceed to **option 3 as an added branch** (present + predicted,
simultaneously — the tech-lead's joint-objective design). The pre-registered gate's literal
reading said "don't" (rank flat, `chat−cplus` gap tiny), but run 1 **falsified the gate's
premise**: `F_c` loses to copy while `c_hat` reconstructs as well as `c_plus`, proving
present-anchored recon is blind to prediction error. Full reasoning in
[`fanciful-lake-18/NEXT_STEPS.md`](run_018_fanciful-lake-18/NEXT_STEPS.md) §"The option-3 decision".

No calibration step: `reconstruction_loss` is a **scale-free relative MSE** (~1.0
baseline), so `lambda_recon=0.05` is a fixed, meaningful weight. The linear ramp over
`recon_warmup_steps` (default 2000) protects the fragile early phase.

---

## How to run the new code on the pod (Path B — code already deployed)

Repo on pod: `/workspace/hierarchal-jepa-flow-world-model`. Full operator guide:
[`AGENT_FILES/SETUPS/SETUP.md`](../../../AGENT_FILES/SETUPS/SETUP.md) Path B.

### 1. Laptop — commit & push

```bash
cd /Users/gobus/Desktop/main/projects/NURON/HJEPA-VWM
git add -A
git commit -m "Add option-1 reconstruction anchor (decoder D, lambda_recon, split readouts)"
git push origin phase1-v0.2-frozen-encoder   # or merge to main first, then push main
```

### 2. Pod — SSH in & pull

```bash
ssh runpod-jepa                 # or the exact command from the RunPod Connect tab
cd /workspace/hierarchal-jepa-flow-world-model
git fetch origin
git checkout phase1-v0.2-frozen-encoder && git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline            # must match the laptop commit
# requirements.txt unchanged -> no pip install needed
```

### 3. Pod — sanity (the new gradient contract runs here)

```bash
# Option-1 routing contract: recon grad reaches D + B, never F_c / EMA (asserted in smoke test)
python -c "from models import smoke_test_models; smoke_test_models()"
# Encoder-loaded forward/backward/EMA sanity (~2 min)
python train.py --stage0-only
```

### 4. Pod — first active run (mirrors `royal-cherry-17` regime for a clean A/B)

```bash
tmux new -s recon_run
cd /workspace/hierarchal-jepa-flow-world-model
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 \
  --lr-coarse-flow 1e-4 --lambda-recon 0.05 --recon-warmup-steps 2000 \
  --log-every 50 --diag-every 500
# Detach: Ctrl+B then D.  Reattach: tmux attach -t recon_run
```

Checkpoints land at `/workspace/checkpoints/phase1_step*.pt`. Resume with
`--resume /workspace/checkpoints/phase1_stepXXXX.pt` (decoder is loaded if present,
else left at init — backward compatible with pre-recon checkpoints).

---

## After launch (KANBAN hygiene)

When the run gets a W&B name, create `investigation_006/<wandb-name>/` with the triad
and fill its `DESCRIPTION.md` (hypothesis, exact command, config delta) per
[`../PROTOCOL.md`](../../PROTOCOL.md).

## Watch / abort rules

**Success signals:** `c_effective_rank` climbs past ~13 (toward >60); `coarse_vs_copy_ratio`
stays ≤ 1 and does not rise; `L_flow` does not deteriorate vs `royal-cherry-17`; the
8000–9000 window does not cliff.

**Abort if** (reuse the [005](../investigation_005/run_017_royal-cherry-17/NEXT_STEPS.md) rules):
- `L_flow > 1.5` for 200 consecutive steps
- `c_effective_rank` drops > 3 points in 500 steps
- `agc_Fc_max_ratio` median > 200 over any 500-step window
- new: `agc_D_max_ratio` median > 200 (decoder destabilizing) or `L_recon` rising while `L_flow` rises

## Decision gate for option 3

Promote to the through-`F_c` future anchor only if run 1 (a) lifts rank without
`L_flow` harm AND (b) shows a large `L_recon_chat` − `L_recon_cplus` gap (F_c's guess
lands where the representation reconstructs poorly). Otherwise the indirect route
sufficed, or recon needs re-tuning first.
