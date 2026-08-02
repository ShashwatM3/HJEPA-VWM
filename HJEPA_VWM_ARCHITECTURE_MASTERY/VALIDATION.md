# Validation report

This report records the checks used to keep the corpus tied to executable repository truth. It is
not a claim that a fresh paid GPU run was launched; no SSH or remote command was used.

## Snapshot boundary

| Field | Audited value |
|---|---|
| Date | 2026-07-25 |
| Branch | `phase1-v0.2-frozen-encoder` |
| Base commit | `771cbba077d9f846bdf7a7dd48e12dbf29d54b49` |
| Source state | dirty working tree, intentionally documented as such |
| Python | `/usr/local/bin/python` 3.12.4 |
| Torch / NumPy | 2.10.0 / 1.26.4 |
| required / locally installed Transformers | 4.57.6 / 5.5.3 |

Every core hash was recomputed after writing the corpus and still matches
[SOURCE_LEDGER.md](SOURCE_LEDGER.md). This includes config, data, encoder adapters, modules, losses,
diagnostics, provenance, trainer, all three offline probes, requirements, agent/remote guides,
Phase 1 KANBAN index, and the original brief PDF.

## Coverage inventory

| Layer | Delivered material |
|---|---|
| executable architecture | 22 ordered chapters plus four Markdown diagram/reference sheets |
| formulas and exact numbers | parameter/memory appendix and exact-number atlas |
| training and objectives | schedule, step state machine, all loss/gradient routes, diagnostics |
| reproducibility | checkpoint, sampler, RNG, provenance, resume, environment pins |
| MLOps | dataset factories, probes, W&B, RunPod/tmux protocol, volume/recovery boundaries |
| research context | experiment chronology, contradictions, current-versus-planned graph |
| retrieval practice | five Markdown retrieval lessons, 120-question oral bank and 120-answer key |
| embodied recall | flashcards, 15 whiteboard drills, spaced-repetition plan |

The corpus deliberately omits `learning-records/`: those files require demonstrated learner
evidence and must not be fabricated.

## Live-module parameter verification

The audit instantiated current `Bottleneck`, `CoarseFlow`, and `Decoder` classes against both
implemented encoder geometries. Counts include trainable parameters only; the final column also
includes the frozen-gradient EMA copy of `B`.

| Geometry | `M` | `B` | `F_c` | `D` | trainable total | bundle + `B_EMA` |
|---|---:|---:|---:|---:|---:|---:|
| V-JEPA | 256 | 4,837,123 | 7,643,904 | 2,172,160 | 14,653,187 | 19,490,310 |
| V-JEPA | 512 | 18,324,739 | 7,643,904 | 2,172,160 | 28,140,803 | 46,465,542 |
| V-JEPA | 1,024 | 70,727,427 | 7,643,904 | 2,172,160 | 80,543,491 | 151,270,918 |
| frame encoder | 256 | 5,033,731 | 7,643,904 | 2,106,368 | 14,784,003 | 19,817,734 |
| frame encoder | 512 | 18,717,955 | 7,643,904 | 2,106,368 | 28,468,227 | 47,186,182 |
| frame encoder | 1,024 | 71,513,859 | 7,643,904 | 2,106,368 | 81,264,131 | 152,777,990 |

These values match the parameter appendix, tensor atlas, chapters, answer key, and flashcards.

## Live-schedule verification

The exact current functions `lr_scale`, `linear_ramp_scale`, and `ema_cosine` were evaluated:

| step | LR scale | 2k loss ramp | EMA momentum |
|---:|---:|---:|---:|
| 0 | 0.000666666667 | 0 | 0.996000000000 |
| 1,000 | 0.667333333333 | 0.5 | 0.996000872757 |
| 1,499 | 1 | 0.7495 | 0.996001960904 |
| 1,500 | 1 | 0.75 | 0.996001963520 |
| 2,000 | 0.996619178871 | 1 | 0.996003490247 |
| 7,500 | 0.586824088833 | 1 | 0.996048890571 |
| 14,999 | 0.000000013539 | 1 | 0.996193085394 |
| 15,000 | 0 | 1 | 0.996193110708 |
| 105,000 | 0 | 1 | 0.999900000000 |

This explicitly verifies that LR, loss ramps, and EMA use different step-zero behavior and
denominators.

## Code validation

- `python -m py_compile` passed for the training, data, encoder, model, provenance, probe, and
  dataset-builder scripts.
- `models.smoke_test_models()` passed.
- `python -m pytest -q` produced **181 passed, 1 failed**.
- The single failure is the repository's intentional environment-pin assertion: the repository
  requires Transformers 4.57.6, while the local interpreter currently has 5.5.3. The pin was not
  weakened to make the test green.
- Invoking the standalone `pytest` executable selected a different Python 3.11 installation without
  NumPy and failed during collection. The module invocation above is the valid test result for the
  audited interpreter.

## Markdown-only validation

- Every delivered file has the `.md` suffix; no HTML, CSS, JavaScript, image, or binary artifact is
  required by the course.
- Every local Markdown target resolves; external links were classified separately.
- All fenced code/Mermaid blocks are balanced.
- The four reference sheets and five lessons use portable Markdown headings, tables, blockquotes,
  code blocks, and Mermaid diagrams.
- Each lesson keeps retrieval answers after an explicit divider, so active recall remains possible
  without JavaScript disclosure controls.
- The quiz bank contains 120 unique IDs; the answer key contains the same 120 IDs in the same order.

## Interpretation boundary

The corpus verifies code structure, calculations, source identities, tests, and Markdown integrity.
It does not assert:

- that remote credentials or a particular RunPod pod are currently available;
- that datasets/checkpoints named in operator guides still exist on a live volume;
- that a new training run reproduces historical W&B metrics without executing that run;
- that planned `F_e`, pixel generation, horizon embeddings, samplers, or rollout integration are
  implemented.

Those boundaries are why every chapter separates **IMPLEMENTED**, **SHIPPED DEFAULT**,
**EXPERIMENT RECIPE**, and **PLANNED** claims.
