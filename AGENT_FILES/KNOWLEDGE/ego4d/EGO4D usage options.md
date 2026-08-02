# EGO4D usage options — how to actually bring it into the training pipeline

> Purpose: given the decision to migrate toward EGO4D (recorded in
> [`EGO4D_DATASET_CHAT_CONTEXT.md`](EGO4D_DATASET_CHAT_CONTEXT.md) and grounded in the dataset
> facts in [`EGO4D_dataset_understanding.md`](EGO4D_dataset_understanding.md)), this document
> answers the *mechanics* question: how should EGO4D actually enter the training pipeline
> relative to SSv2 — replace it, sit alongside it, or sequence with it? Grounded in this
> repository's actual dataset abstraction (`config.py`, `data.py`, `train.py`) and in the full
> Phase 1 experiment history (`KANBAN/PHASE_1/`), not in generic dataset-migration habits.

## 1. Why this question is separate from "which dataset"

The dataset choice is already made and reasoned through in
[`EGO4D_DATASET_CHAT_CONTEXT.md`](EGO4D_DATASET_CHAT_CONTEXT.md): SSv2 is being supplemented
(not necessarily discarded) because its exocentric, static-camera, small-object-manipulation
footage makes consecutive frames nearly identical, which is the direct mechanical cause of the
`coarse_vs_copy_ratio` failure this project has chased since investigation_001. What has not
been decided is the *integration strategy* — and that choice has real consequences for what a
future run can and cannot prove, because of how this codebase's dataset abstraction and Phase 1
research method both work.

Two facts about the current codebase make this decision more tractable than it might seem:

1. **The dataset selector is already a clean, additive abstraction.** `config.py`'s
   `DataConfig.dataset_root()` is a simple string-keyed branch (`"ssv2"` → `full_root`,
   `"ssv2_tiny"` → `tiny_root`, else `ValueError`), and `train.py`'s `--data` CLI flag
   (`choices=["ssv2", "ssv2_tiny"]`) is the only place that string is set. Adding `"ego4d"` and
   `"ego4d_tiny"` as new branches and new CLI choices — exactly as
   `EGO4D_DATASET_CHAT_CONTEXT.md` already specifies — is a few-line, purely additive change.
   Nothing about SSv2's code path is touched. **This means "add a CLI flag to pick the dataset"
   is not really one option among several — it is the mechanism every option below is built on
   top of.** The real decision is what a *run* does with that flag, not whether the flag exists.
2. **Training is step-based and per-run stateless with respect to "which video was seen when."**
   `data.py`'s `build_dataloader` uses `DataLoader(..., shuffle=(split == "train"))`, so within
   a single run, which specific video lands on which specific step is randomized every epoch —
   there is no fixed, meaningful "step N is video N" correlation to preserve or break in the
   first place. This matters directly for one of the options raised below (§3, interleaving).

## 2. What the research history says the *next* experiment needs to prove

This is the part that should drive the decision more than any general data-engineering
principle, so it is worth stating plainly from the KANBAN record: **every full-prediction run
from investigation_001 through investigation_011 (runs 1–40) failed the same two gates**
(`coarse_vs_copy_ratio ≤ 0.70`, `coarse_vs_batch_mean_ratio ≤ 0.50`), and the most information-
dense negative result in the whole program, run 037 (`soft-universe-37`), proved that a
textbook-healthy representation (`c_effective_rank≈61/256`, `c_cross_video_cosine≈0.16`,
`c_std_mean≈1.0`) still does not make `F_c` beat copy — `coarse_vs_copy_ratio` landed at 1.06.
Fifteen investigations of regularizer, decoder, whitening, and architecture changes have not
moved that number below 1.0 on SSv2. The EGO4D migration is a bet that the reason is not the
model but the *data*: SSv2 makes "copy the present forward" a genuinely strong prediction
because present and future are almost the same frame, no matter how healthy `c_t` is.

That bet is only informative if the next experiment can cleanly attribute a change in
`coarse_vs_copy_ratio` (or its continued failure) to **the dataset alone**, holding everything
else fixed. Any integration strategy that mixes SSv2 and EGO4D content within the same training
run, or that carries forward SSv2-adapted weights into an EGO4D run, makes that attribution
ambiguous — an improvement (or lack of one) could always be explained away as "more total
training" or "a blend of easy and hard examples" rather than as evidence about EGO4D itself.
**This is the central criterion behind the ranking below, more than engineering convenience.**

## 3. The three options, ranked

### Option 1 (recommended): Full sibling dataset behind the existing `--data` flag, launched as a clean, single-variable A/B run — do not touch SSv2

Concretely: build `/workspace/data/ego4d/{train,validation}/` as a symlink directory with the
exact same internal shape SSv2 already uses (mirroring the `ssv2`/`ssv2_tiny` pattern precisely,
as `EGO4D_DATASET_CHAT_CONTEXT.md` already specifies), add `"ego4d"`/`"ego4d_tiny"` to
`dataset_root()` and the CLI `--data` choices, generalize the loader's glob from `*.webm` to
also match `*.mp4`, and leave every SSv2 path, file, and prior checkpoint completely untouched.
SSv2 stays fully queryable and comparable forever; it costs only disk space to keep (the raw
backing files already exist and are read-only). Then launch a **fresh** run (no `--resume`) on
`ego4d` with the *exact* config of the strongest recent comparable SSv2 run — the natural choice
is investigation_010's run 037 recipe (`predict_residual=true`, the cleaned optimizer/AGC
plumbing, `horizon_k` chosen per the fps-handling decision in
[`EGO4D_dataset_understanding.md`](EGO4D_dataset_understanding.md) §7) — changing nothing except
the dataset and whatever `horizon_k`/`frame_stride` adjustment the fps mismatch requires. Compare
`coarse_vs_copy_ratio`, `coarse_vs_batch_mean_ratio`, and `coarse_copy_loss`'s trend directly
against run 037's numbers.

**Why this ranks first.** It is the only option that gives an unambiguous answer to the question
the migration exists to answer: does removing SSv2's frame-similarity problem, and nothing else,
fix the copy-ratio gate? A single changed variable (the dataset) against a well-documented,
numerically pinned-down baseline (run 037) is exactly the kind of comparison this project's own
experiment-lifecycle discipline (`GUIDES/EXPERIMENT_LIFECYCLE.md`, `KANBAN/PROTOCOL.md`) is built
around, and it is what `EGO4D_DATASET_CHAT_CONTEXT.md` itself already recommends ("Commit to one
dataset, not two... Prove the hierarchy on one clean egocentric source first"). It also happens
to be the cheapest option to implement: additive-only code changes, no migration of existing
history, no new resume/scheduling logic, and no risk of quietly corrupting the SSv2 experiment
record this project has fifteen investigations invested in.

**Cost / risk.** A second copy of the data pipeline exists in parallel (more disk on the network
volume, one more code path to keep in sync going forward), and this option does not, by itself,
give a combined SSv2+EGO4D-scale model — it answers the causal question first and leaves scale
questions for later, deliberately.

### Option 2: Sequential — pretrain/warm-start on SSv2, then continue training on EGO4D via `--resume`

Concretely: take an existing SSv2 checkpoint (or run a fresh SSv2 warm-start to a chosen step
count) and resume training with `--data ego4d --resume <ckpt>`, continuing the same `B`, `B_EMA`,
`F_c`, and (if active) `D` weights and optimizer state under the EGO4D dataset from that point
forward. This is the literal implementation of "resume from a checkpoint and continue only with
EGO4D," and of "go through SSv2 first, then EGO4D" if that sequencing is meant to span a single
long training arc rather than two fully separate runs.

**Why this ranks second, not first.** It confounds two variables that Option 1 keeps separate:
by the time an EGO4D-only phase starts, `B` has already been shaped by ~15k steps of SSv2-style
optimization (whatever representation habits that induced — including, per the KANBAN record,
a persistent tendency toward the ~13–60/256 rank band and the specific regularizer equilibria
each investigation fought to reach), and any subsequent change in `coarse_vs_copy_ratio` is now
explainable by *either* "EGO4D fixed the copy-ratio problem" *or* "more total optimizer steps
helped" *or* "the SSv2-tuned bottleneck geometry transfers imperfectly and the metric shift is
transient adaptation noise," with no clean way to tell these apart from the run alone. There are
also mechanical snags: `B_EMA`'s momentum schedule (`ema_m_start=0.996` ramping to
`ema_m_end=0.9999` over `ema_schedule_steps=105000`) and the cosine LR schedule
(`stage1_steps`-denominated) are both functions of the *global* step count, so resuming into a
new dataset mid-schedule means the EGO4D phase inherits whatever point on those schedules SSv2
training left off at, rather than getting its own clean warmup/EMA ramp — a real, if fixable
(by resetting schedules and step-zeroing on resume), confound.

**When this becomes the right move.** After Option 1 has produced a clean answer, sequential
fine-tuning is a legitimate and cheaper way to ask a *second-order* question — "does an SSv2-
pretrained bottleneck adapt well to EGO4D, or does it need to start over" — which is a real
question for the eventual world-action-model roadmap (`EGO4D_DATASET_CHAT_CONTEXT.md`'s
long-term framing), just not the question this migration was undertaken to answer first.

### Option 3 (not recommended as a first move): Curated replacement or interleaving — swap specific SSv2 categories/videos for EGO4D content, or mix both sources within one training pool

Concretely: either delete a subset of "too simple" SSv2 action classes and splice in EGO4D
clips to backfill the directory, or build a single mixed `train/` directory drawing from both
sources every epoch. As covered in §1, the "steps are correlated with videos" concern that makes
this option feel risky is **not actually a property of this codebase** — `build_dataloader`
reshuffles every epoch (`shuffle=True`), so there is no fixed step-to-video mapping to disturb
either way, and a mixed-source directory would shuffle exactly as cleanly as a single-source one
mechanically speaking.

**Why it still ranks last.** The real problem is not scheduling disruption, it is diagnostic
confounding, and it is worse here than in Option 2. A single training run mixing two visually
and dynamically different distributions can never attribute a `coarse_vs_copy_ratio` change to
either source individually — a batch-level average metric computed across a blended distribution
answers a blended question. It also actively works against the reconstruction-honesty and
copy/batch-mean diagnostics this project's entire recent research arc (investigations 012–015)
was built around: those diagnostics assume a roughly homogeneous per-video difficulty
distribution when interpreting `L_recon_video_gap`, `c_cross_video_cosine`, and the copy
baseline's strength, and a deliberately curated mix (dropping "too simple" SSv2 classes) directly
introduces sampling bias into exactly the kind of before/after comparison Option 1 is designed to
give a clean version of. Preserve SSv2 in full and do not curate it — its "too simple" clips are
not garbage data, they are the known, well-characterized baseline against which fifty-plus prior
runs are already indexed in `KANBAN/PHASE_1/README.md`; deleting or replacing parts of it would
also silently invalidate the ability to re-run or sanity-check any of that prior history.

**When this becomes reasonable.** Once EGO4D-only training (Option 1) is validated as sound and
scaled, a genuinely diverse, multi-domain final training mix (SSv2 plus EGO4D plus whatever else)
is a legitimate later objective for maximizing generalization — but as a deliberate, later design
choice made from a position of already knowing each source's individual effect, not as the
mechanism for finding that effect out.

## 4. Direct answers to the specific questions raised

- **"Are we completely replacing SSv2 with EGO4D — same folder structure, different videos, on
  the network volume?"** Structurally yes (mirror the `ssv2`/`ssv2_tiny` symlink pattern
  exactly, per `EGO4D_dataset_understanding.md` §3 and §7), but as a **sibling**, not a
  replacement — `data/ego4d/` next to `data/ssv2/`, both permanently present on the volume.
  "Replacing" only describes which dataset future runs default to choosing, never a deletion of
  SSv2's data or its comparability.
- **"Will we need a CLI flag to indicate which dataset to use?"** Yes, and this is not an
  incremental cost — it is the same mechanism `--data ssv2 | ssv2_tiny` already is, extended with
  two more choices. This is true under every option in this document, including the ones ranked
  lower.
- **"Do we want to decidedly remove certain SSv2 videos/categories and replace them with EGO4D
  videos?"** No — recommended against (§3, Option 3) — both for the diagnostic-confounding reason
  above and because it would silently break comparability with the 57 runs already indexed in
  `KANBAN/PHASE_1/README.md`.
- **"Will that disrupt the step-to-video correlation?"** No such correlation exists to disrupt
  in this codebase — the dataloader reshuffles every epoch regardless of dataset composition
  (§1). This is not the reason to avoid mixing; diagnostic confounding is.
- **"Do we want to keep it sequential — SSv2 first, then EGO4D?"** Covered as Option 2; ranked
  second, appropriate as a follow-up question about representation transfer, not as the first
  experiment.
- **"Do we resume from a checkpoint and continue only with EGO4D?"** This is the concrete
  mechanism of Option 2 (`--resume <ssv2_ckpt> --data ego4d`); same ranking and caveats apply.
- **"Do we really want to preserve SSv2?"** Yes, unconditionally — it costs only disk space,
  remains the only apples-to-apples baseline the copy-ratio hypothesis can be tested against, and
  is the anchor for every prior KANBAN investigation's numbers.

## 5. Summary recommendation

Implement the dataset abstraction as fully additive (new `ego4d`/`ego4d_tiny` config branches and
CLI choices, generalized glob, new UID-selection and chunker scripts — the code deltas already
scoped in `EGO4D_DATASET_CHAT_CONTEXT.md`), leave every SSv2 path and prior run untouched, and
launch the first EGO4D run as a **clean, fresh, single-variable-changed A/B** against the
strongest directly comparable SSv2 run (run 037, `soft-universe-37`) already on record. Only
after that comparison produces a real answer about whether EGO4D fixes the copy-ratio problem
does it become worth spending compute on sequential fine-tuning (Option 2) or a deliberately
diverse final training mix (Option 3's later-stage version).
