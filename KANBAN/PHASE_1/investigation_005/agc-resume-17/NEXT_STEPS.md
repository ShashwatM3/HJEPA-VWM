# Next steps — agc-resume-17

## Before launch

1. Pod: `git pull` branch `phase1-v0.2-frozen-encoder` (commit ≥ `1ae2e09`).
2. `python train.py --stage0-only`
3. Confirm `/workspace/checkpoints/phase1_step7500.pt` exists.
4. Launch in `tmux` (see command in `DESCRIPTION.md`).
5. Rename this folder to the W&B display name once assigned.

## On success

1. Record metrics vs [`AGENT_FILES/PHASES/PHASE_1.md`](../../../AGENT_FILES/PHASES/PHASE_1.md) §12.
2. **Close** [investigation_005](../DESCRIPTION.md).
3. Proceed toward Phase 2 per [`AGENT_FILES/PHASES/PHASE_2.md`](../../../AGENT_FILES/PHASES/PHASE_2.md).

## On failure (grad-skip recurs despite AGC)

1. Do **not** resume from post-spike checkpoints.
2. Retry from `phase1_step7500.pt` with `--lr-coarse-flow 1e-4` **and** AGC (belt +
   suspenders), or tighten `--agc-lambda-coarse-flow` (e.g. 0.05).
3. New run folder per [`KANBAN/PROTOCOL.md`](../../../PROTOCOL.md).
