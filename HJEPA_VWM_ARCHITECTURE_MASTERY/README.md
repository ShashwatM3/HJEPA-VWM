# HJEPA-VWM architecture mastery corpus

This folder is a code-grounded course for being able to explain the HJEPA-VWM system from disk bytes
to gradients, checkpoints, remote execution, and the planned coarse-to-fine hierarchy without
looking anything up.

**Delivery format:** all 45 artifacts in this folder are Markdown files. Diagrams are embedded as
Mermaid blocks; there are no separate HTML, CSS, JavaScript, image, or binary dependencies.

It was built from the repository working tree on **2026-07-25 15:53 IST**, branch
`phase1-v0.2-frozen-encoder`, base commit
`771cbba077d9f846bdf7a7dd48e12dbf29d54b49`. The worktree was dirty and already contained the
implemented pinned DINOv3 adapter plus current Investigation 016/017 records. This corpus describes
that working-tree snapshot, not merely the base commit. See [SOURCE_LEDGER.md](SOURCE_LEDGER.md) and
the independent [validation report](VALIDATION.md).

## The four truth labels

Every architectural claim belongs to one of four layers:

| Label | Meaning | Authority |
|---|---|---|
| **IMPLEMENTED** | Executes in the current working tree | code and tests |
| **SHIPPED DEFAULT** | Value produced by `Config()` before CLI overrides | `config.py` |
| **EXPERIMENT RECIPE** | A deliberate override used or planned in KANBAN | run guide / provenance |
| **PLANNED** | Design intent not present in the executable Phase 1 stack | current brief, then original brief |

Never collapse these labels. A common example is:

- shipped default: `M=256`, `N_c=32`, `D_c=256`, `k=4`, `lambda_var=0.1`;
- Investigation 017 recipe: `M=512`, `N_c x D_c` grid, `k` irrelevant because it is present-only,
  `lambda_var=0.5`, `lambda_cov=0.01`;
- planned future: `F_e`, a pixel/VAE-latent generator, inference integration, and horizon embeddings.

## Recommended reading order

### Pass 1 — tell the whole story

1. [00 — Truth map and notation](chapters/00_TRUTH_MAP_AND_NOTATION.md)
2. [01 — System overview](chapters/01_SYSTEM_OVERVIEW.md)
3. [20 — Five end-to-end traces](chapters/20_END_TO_END_TRACES.md)
4. Open [the visual architecture map](reference/architecture-map.md).

### Pass 2 — know every subsystem

5. [02 — Data and temporal windows](chapters/02_DATA_AND_TEMPORAL_WINDOWS.md)
6. [03 — Frozen encoder seam](chapters/03_FROZEN_ENCODER_SEAM.md)
7. [04 — Bottleneck internals](chapters/04_BOTTLENECK_INTERNALS.md)
8. [05 — EMA target and coarse flow](chapters/05_EMA_TARGET_AND_COARSE_FLOW.md)
9. [06 — Decoder, whitening, and reconstruction](chapters/06_DECODER_WHITENING_RECONSTRUCTION.md)
10. [07 — Losses and gradient routing](chapters/07_LOSSES_AND_GRADIENT_ROUTING.md)
11. [08 — Training step and schedules](chapters/08_TRAINING_STEP_AND_SCHEDULES.md)

### Pass 3 — operate and defend it

12. [09 — Diagnostics and how to read a run](chapters/09_DIAGNOSTICS_AND_RUN_READING.md)
13. [10 — Checkpoints, resume, RNG, and provenance](chapters/10_CHECKPOINT_RESUME_REPRODUCIBILITY.md)
14. [11 — Dataset construction](chapters/11_DATASET_CONSTRUCTION.md)
15. [12 — Offline probes](chapters/12_OFFLINE_PROBES.md)
16. [13 — Remote MLOps and W&B](chapters/13_REMOTE_MLOPS_AND_WANDB.md)
17. [14 — Test contracts](chapters/14_TEST_CONTRACTS.md)
18. [15 — Research evolution](chapters/15_RESEARCH_EVOLUTION.md)
19. [16 — Planned future hierarchy](chapters/16_PLANNED_FUTURE_HIERARCHY.md)

### Pass 4 — memorize the numbers

20. [17 — Exact-number atlas](chapters/17_EXACT_NUMBER_ATLAS.md)
21. [18 — Configuration and CLI](chapters/18_CONFIGURATION_AND_CLI.md)
22. [19 — File-by-file map](chapters/19_FILE_BY_FILE_MAP.md)
23. [21 — Failure modes and quiz traps](chapters/21_FAILURE_MODES_AND_QUIZ_TRAPS.md)
24. [Parameter formulas](appendices/PARAMETER_FORMULAS.md)
25. [Contradictions and stale prose](appendices/CONTRADICTIONS_AND_STALE_PROSE.md)

## Active recall

- Start the short interactive sequence from [the lesson index](lessons/README.md).
- The large oral-exam bank is [drills/QUIZ_BANK.md](drills/QUIZ_BANK.md).
- Answers are separated in [drills/ANSWER_KEY.md](drills/ANSWER_KEY.md).
- Fast repetition prompts are in [drills/FLASHCARDS.md](drills/FLASHCARDS.md).
- Whiteboard exercises are in [drills/WHITEBOARD_DRILLS.md](drills/WHITEBOARD_DRILLS.md).
- A suggested review cadence is in [drills/SPACED_REPETITION.md](drills/SPACED_REPETITION.md).

No `learning-records/` directory is included yet. The teaching workflow reserves learning records
for demonstrated learner evidence; generating fake evidence would defeat their purpose.

## Fastest “quiz tomorrow” route

If time is short:

1. memorize the [tensor atlas](reference/tensor-atlas.md);
2. reproduce the [gradient map](reference/gradient-map.md) from memory;
3. explain the [training schedule](reference/training-schedule.md);
4. answer Quiz Bank sections A–H aloud;
5. use the answer key only after committing to an answer.

## Refresh rule

Before trusting this corpus after code changes:

```bash
git diff -- config.py data.py encoders.py models.py losses.py diagnostics.py provenance.py train.py
shasum -a 256 config.py data.py encoders.py models.py losses.py diagnostics.py provenance.py train.py
```

Compare those hashes with [SOURCE_LEDGER.md](SOURCE_LEDGER.md). A changed hash means the relevant
chapter must be re-audited; it does not automatically mean the prose is wrong.
