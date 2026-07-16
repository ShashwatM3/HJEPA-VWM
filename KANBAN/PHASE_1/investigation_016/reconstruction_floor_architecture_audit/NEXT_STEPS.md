# Next Steps — reconstruction-floor architecture audit

Do these in order. Do not spend another full run on a broad hyperparameter sweep before
the first two stages make the floor identifiable.

## 1. Repair the measurement contract first

1. Build the fixed EGO4D validation batch with one seeded chunk per source UID; assert
   all source IDs are unique for cross-source metrics.
2. Replace unconditional `torch.roll` with a verified different-source derangement.
3. Rename the existing metric to `cross_sample_cosine` unless the batch contract proves
   cross-source identity.
4. Log both cross-source and same-source controls: cosine, shuffled reconstruction, and
   reconstruction gap.
5. Add `zero-c`, mean-`c`, `B(zeros)`, and per-position target-mean reconstruction
   baselines. These directly measure the template channel.

This is a correctness fix, not a model ablation. Re-read the run-058 checkpoint with the
new diagnostics if a compatible checkpoint is available; otherwise apply them to the
live run and all future runs.

## 2. Run cached-feature capacity probes

Use frozen, source-diverse train/validation feature caches so these probes take minutes,
not a paid 15,000-step run.

1. Compute the exact best constant predictor for cosine loss at each detailed position.
2. Measure rank-`r` linear/PCA oracles for `r = 64, 128, 256, 512, 1024` in both raw and
   whitened spaces.
3. Overfit one fixed batch, in this order:
   - optimize a free per-sample `32 x 256` latent plus D;
   - train B+D with `lambda_var=lambda_cov=0`, no crop/jitter variation, and flat LR;
   - add covariance/variance;
   - restore augmentation and the production schedule.
4. Log trainable-output norms, target norms, per-module update/weight ratios, per-loss B
   gradient norms, and gradient cosines `recon-vs-var` and `recon-vs-cov`.

Decision rules:

- Free latent+D cannot overfit: decoder/latent bandwidth is the first failure.
- Free latent+D succeeds but B+D fails: the bottleneck/content path is the first failure.
- Fixed-batch B+D succeeds but corpus training stalls: information budget,
  augmentation/generalization, regularizer conflict, or schedule is the first failure.
- PCA-256 itself sits near the observed band: full whitening plus the 1,024-to-256
  channel squeeze is confirmed as the dominant floor.

## 3. Only then run a small capacity factorial

Prefer orthogonal arms over another scalar-weight ladder:

1. control: `n_c=32`, `d_c=256`, mixer 256;
2. channel-preserving input: mixer 1,024 with the same slot code;
3. slot-wide: `n_c=128`, `d_c=256`;
4. wider code: `n_c=32`, `d_c=512`;
5. no/full/partial whitening on the same source-diverse substrate.

Read raw reconstruction together with cross-source honesty and geometry. A lower loss
that comes only from a constant template is not a capacity win.

## 4. Interpret the live weight-1 arm narrowly

Allow run 060 (`2423b84g`) to finish if operationally desired, but do not call it a
strict `lambda_recon` causal replay of run 058: module initialization, data order and
augmentation RNG, encoder pinning, whitening artifact binding, and provenance code
changed between the commits. Also, multiplying reconstruction by 20 is largely canceled
inside Adam for D, which receives only reconstruction gradients; it mainly changes B's
gradient mixture and early clipping dynamics.

## 5. Full-run acceptance gate

Launch the next paid full run only after the chosen arm has:

- a source-diverse fixed validation batch;
- a clearly positive cross-source shuffled-code gap;
- a fixed-batch capacity result explaining where its floor should lie;
- stable per-loss gradient and update ratios;
- a schedule with non-negligible LR remaining during the intended convergence window.
