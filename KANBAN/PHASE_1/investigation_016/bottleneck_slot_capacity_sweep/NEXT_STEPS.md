# Next steps — bottleneck slot-capacity sweep

1. ~~Launch and finish the clean-commit EGO4D `N_c=32/64/128` core sweep.~~ Complete as W&B
   `x03xlpyl`, `evyokqrm`, and `7pmvxrxi` with final checkpoints and zero instability.
2. ~~Apply Reading Cycle B separately and compare the preregistered late windows.~~ Complete in
   [`ANALYSIS.md`](ANALYSIS.md). The curve does not support a healthy slot-capacity win.
3. Do **not** launch the conditional 256-slot arm. The modest loss reduction fails the
   representation-health guard, so another slot-count point would not resolve the leading issue.
4. Run the orthogonal no-whitening EGO4D present-reconstruction arm at the 32-slot control shape.
   Keep encoder, decoder, schedule, absolute target, and geometry weights fixed.
5. Separately test the early 1,024-to-256 channel squeeze / `D_c` capacity axis. Choose values only
   after confirming how `D_c` and bottleneck mixer width are coupled in the current implementation;
   do not describe another `N_c` ladder as a channel-width test.
6. Keep alternate-encoder work separate so whitening, capacity, and encoder substrate remain
   attributable.

No result from this present-only sweep licenses a prediction claim. Because the fixed EGO4D
diagnostic batch is one source UID, retain the source-diverse measurement repair in the parent
investigation plan.
