# Lesson 4 — Read the run in order

**Time:** 20 minutes

**Goal:** form one causal verdict instead of cherry-picking a smooth curve.

[Course home](../README.md) · [Previous](03_follow_the_gradient.md) ·
[Deep chapter](../chapters/09_DIAGNOSTICS_AND_RUN_READING.md) ·
[Next](05_resume_without_drift.md)

## The eight-question cycle

| Question | Test | Core evidence |
|---:|---|---|
| Q1 | stable? | finite losses/norms; no NaN or skip spiral |
| Q2 | alive? | standard deviation 0.8–1.2; cosine below 0.5; dead dimensions near zero |
| Q3 | rich? | effective rank above 60; EMA follows online state |
| Q4 | dynamic? | copy-loss trend, rank, and model/copy ratio together |
| Q5 | forecasts? | model/copy ratio at most 0.70 |
| Q6 | conditions? | model/batch-mean ratio at most 0.50 |
| Q7 | reconstruction honest? | wrong/shuffled `c_hat` must separate from correct target |
| Q8 | verdict? | one named success/failure class explaining the joint evidence |

Run 037 is the canonical trap:

```text
effective rank ≈ 61
standard deviation ≈ 1
cross-video cosine ≈ .16
model/copy ratio ≈ 1.06
```

The correct verdict is **healthy representation, no predictor**. Representation geometry and future
forecasting are separate claims.

## Challenge cases

Answer before reading the key.

1. Effective rank rises, copy loss falls, but model/copy ratio stays around 1. What happened?
2. Model/batch-mean ratio is 0.4 while model/copy ratio is 0.95. Does the run pass?
3. Present reconstruction is 0.300 and shuffled reconstruction is 0.301. What does the low raw
   reconstruction loss establish?
4. Can prediction copy gates be used for present-only runs?
5. Why can the EGO diagnostic called “cross-video cosine” be mislabeled?

---

## Answer key

1. The static-`c` trap: extra dimensions likely encode static appearance, making present and future
   more alike without improving prediction.
2. No. The model uses some per-video signal but fails the main forecast-versus-copy gate.
3. Almost nothing about video content. The approximately 0.001 correct-versus-shuffled gap exposes a
   template shortcut.
4. No. `F_c` is absent. Judge stability, geometry, correct-versus-shuffled reconstruction, and the
   controlled present-representation hypothesis.
5. The fixed first 16 EGO chunks can share a single source UID, turning it into within-source
   cross-chunk alignment rather than genuinely cross-video alignment.

## Exit ticket

Say all eight questions in order and give the two numeric prediction gates.

Continue to [Lesson 5](05_resume_without_drift.md).
