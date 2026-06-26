# Next steps — investigation 007

## Status

Code ready (flags added + audited, committed). **Stage-1 wave not yet launched.** Awaiting the
6–8 GPU pod.

## Immediate next action — launch the Stage-1 wave

Follow [`GUIDE.md`](GUIDE.md) end-to-end. In brief:
1. Pod up, network volume mounted, repo at `/workspace/...`; `git pull` the branch.
2. `python -c "from models import smoke_test_models; smoke_test_models()"` (gradient contracts).
3. Launch the 8-config wave **8-wide in parallel** (one config per GPU, `CUDA_VISIBLE_DEVICES`,
   per-run `--checkpoint-dir`, in tmux). Exact script + the 8 configs: GUIDE §3 / SWEEP_PLAN §3.
4. ~3–4h. Then per-run, read **`L_recon_present`** (floor moved?) and **`coarse_vs_copy_ratio`**
   against the interpretation matrix (SWEEP_PLAN §4).

## After the wave (KANBAN hygiene)

- Create `investigation_007/<wandb-name>/` for each run (triad), or one combined results write-up
  if that's cleaner for a wave — fill `OBSERVATIONS.md` here with the cross-run synthesis:
  **which axis (if any) moved the floor**, and whether the copy gate followed.
- Update `DESCRIPTION.md` Runs table + the README run index.

## Stage 2 (conditional on Stage-1 result)

- **An axis moved the floor** → push that axis further AND re-introduce option 3
  (`--lambda-recon-pred 0.05`) on the now-unsaturated decoder — does the prediction branch
  finally bite?
- **Nothing moved the floor** → reconstruction is the wrong lever. Close the reconstruction line;
  open a new investigation on the **horizon/task** (the copy baseline is strong because `c`
  barely moves over horizon-12). Possibly also revisit whether the `>60` rank gate is right at
  128:1 compression.

## Watch / abort (per run; inv_005 rules + sweep-specific)

- `L_flow > 1.5` for 200 consecutive steps; `c_effective_rank` drops >3 in 500 steps;
  `agc_Fc_max_ratio` median >200 over any 500-step window.
- Sweep-specific: on `n_c=128` watch `c_slot_diversity_rank` / `c_cross_video_cosine` (slot
  collapse); on `lambda_recon=0.5` watch `L_flow` (recon over-competing with prediction).
