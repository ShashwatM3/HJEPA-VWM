# Spaced-repetition plan

The goal is retrieval, not rereading.

## Day 0 — build the map

- Read Chapters 00,01,20.
- Draw Whiteboard Drills 1 and 11.
- Answer Quiz A and B.
- Review missed answers immediately, then answer them again from a blank page.

## Day 1 — tensor machinery

- Read Chapters 03–06.
- Draw Drills 2,4,5.
- Answer Quiz C–F.
- Run all shape/parameter flashcards twice.

## Day 3 — gradients and time

- Read Chapters 07–09.
- Draw Drills 6,7,8.
- Answer Quiz G–I.
- Re-answer every Day-1 miss before opening notes.

## Day 7 — reproducibility and operations

- Read Chapters 10–14.
- Draw Drills 9,10,14.
- Answer Quiz J–K.
- Explain exact resume to a partner in under three minutes.

## Day 14 — research judgment

- Read Chapters 15,16,21 and contradictions appendix.
- Draw Drill 12.
- Answer Quiz L.
- Give verdicts for three historical run patterns without using metric names alone.

## Day 30 — full technical-lead simulation

1. Draw current graph in eight minutes.
2. Draw planned graph in four minutes.
3. Have a partner choose 30 random Quiz Bank prompts.
4. Derive one module count and one schedule value live.
5. Diagnose one fictional W&B run.
6. Explain remote launch/resume/completion.

Pass criterion:

- 90% factual accuracy;
- no truth-layer conflations;
- no missed stop-gradient/EMA boundary;
- both prediction gates stated correctly;
- every numerical answer includes its axis/unit.

## Ongoing maintenance

Once per month or after core code changes:

```bash
git diff -- config.py data.py encoders.py models.py losses.py diagnostics.py provenance.py train.py
```

Compare hashes with `SOURCE_LEDGER.md`, update affected cards, and retire cards whose behavior no
longer executes. Never memorize stale values more efficiently.

## Miss log

Keep a private table:

| Date | Prompt | Wrong answer | Correct distinction | Next review |
|---|---|---|---|---|

Do not create a formal course learning record until there is actual demonstrated evidence. A miss
log is personal study state; the repository corpus intentionally contains no fabricated learner
record.
