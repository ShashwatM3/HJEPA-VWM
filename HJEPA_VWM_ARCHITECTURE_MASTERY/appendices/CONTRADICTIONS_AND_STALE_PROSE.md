# Appendix — Contradictions, stale prose, and authority decisions

This appendix records places where a reader can reach a wrong answer by trusting an older narrative
or comment over executable code.

## Authority order used by this corpus

For what executes:

```text
tests + current code
→ current config
→ current dated KANBAN analysis/observations
→ current guides
→ latest narrative brief
→ original brief
→ stale/generated/personal files
```

For why a decision was made, dated KANBAN analyses and guides can outrank terse code comments. For
current tensor behavior, code/tests win.

## Original brief versus Phase 1 implementation

| Original design | Current implementation |
|---|---|
| four 128×128 frames | eight 256×256 frames |
| trainable compact encoder | frozen pinned pretrained encoder |
| EMA target encoder | no encoder copy; EMA bottleneck only |
| ~256 detailed tokens at width 384 | 1024×1024 or 2048×768 |
| coarse + fine flows | coarse flow only |
| frame/VAE-latent generator | absent |
| future inference hierarchy | only diagnostic one-step endpoint |

The brief is architecture ancestry and future intent, not an executable specification.

## “One epoch”

Narrative prose has described the 15k schedule as one epoch. Actual loader arithmetic:

```text
SSv2 tiny: 62 batches/epoch → 241 epochs + 58 batches
SSv2 full: 2639 batches/epoch → ~5.684 epochs
```

There is no supported dataset for which 15,000 default steps at batch 64 equals one epoch.

## “Target encoder”

Some historical language says target encoder/EMA encoder. Current code:

```text
same frozen E on both clips
B online for context
B_EMA for target
```

Checkpoint absence of encoder weights and tests for frozen identity confirm this.

## “Decoder/frame generator”

Older phase documents can call reconstruction/generation loosely. Current `Decoder` outputs
`(B,N_e,D_e)` encoder features. The Phase 3 frame generator is planned and absent.

## `f_c_dim`

`ModelConfig.f_c_dim=256` survives as a legacy-looking field. Current `CoarseFlow` builds all
projections at `cfg.d_c`. Changing only `f_c_dim` would not change the executable flow architecture.

## `total_latent_steps` comment

The config comment describes `total_latent_steps=105000` as a cosine LR/EMA denominator. Current
code uses:

```text
LR denominator  = stage1_steps = 15000
EMA denominator = ema_schedule_steps = 105000
```

`total_latent_steps` is not consulted by `lr_scale`.

## Reconstruction comment saying future anchor is unwired

An older `TrainConfig.lambda_recon` comment says the through-`F_c` future anchor is intentionally not
wired. Current code has a separate `lambda_recon_pred` branch that is wired through `c_hat`, `F_c`,
and `B`. Treat the comment as historical context for when the option was proposed.

## `train_step` docstring tuple/optimizer wording

The executable function accepts five modules:

```text
(encoder,bottleneck,target_bottleneck,coarse_flow,decoder)
```

Some docstring lines still describe four modules or optimizer over `B`/`F_c`. Actual optimizer also
contains `D`. Tests and construction code are authoritative.

## “Variance floor replaces SIGReg”

The top of `losses.py` retains evolutionary wording about the variance floor replacing SIGReg, while
the same file implements active optional SIGReg. Current shipped behavior is:

- variance floor weight 0.1;
- SIGReg computed every step but weight zero;
- SIGReg can be enabled/ramped as an experiment.

## Full EGO counts

Volume documentation uses estimates around 170k train/18–19k validation. Exact counts are generated
by selected source duration, redactions, failed/missing downloads, and chunking. Only the built
manifest/filesystem fingerprint is exact.

## Checkpoint directory

Config default is `/workspace/checkpoints`, but volume/MLOps guides classify it as legacy. Real runs
must use a unique `/workspace/ckpt/<run_tag>` override. Both statements are true at different layers:
one is shipped default; one is operator policy.

## W&B run table snapshots

The Phase 1 index contains auto-generated and dated snapshots with historical “running/planned”
states. Later dated reconciliation sections supersede earlier state counts. Any question about
current live W&B must query W&B; this corpus only reports the dated repository snapshot.

## Investigation 017 status

The audited local record dated 2026-07-21 says 27 latent-shape runs were planned and unlaunched. This
corpus does not promote that into a live claim as of 2026-07-25 because no live query was part of the
task.

## “Cross-video” on historical EGO fixed batch

Metric implementation compares batch examples, but all first 16 EGO chunks in the recorded batch can
share a source UID. The numerical metric is valid for those examples; the global cross-video label is
not. Interpret as within-source adjacent-chunk until sampling is repaired.

## Rank dimensionality

Some prose informally refers to rank out of full flattened `N_c*D_c=8192`. Current
`effective_rank` pools slots and forms covariance over `D_c`, so its maximum is 256. The variance
floor, by contrast, does flatten to 8,192 coordinates across examples.

## Standard-deviation correction

Different functions intentionally use different estimators:

- variance floor/diagnostic std: `unbiased=False` population std;
- residual noise scale: PyTorch `std()` default sample correction;
- covariance loss/effective rank: divide by `N-1`;
- whitening offline covariance: biased maximum-likelihood covariance.

Saying “the code uses standard deviation/covariance” without the axis and denominator is incomplete.

## Transformer version

The environment used for documentation had Transformers 5.5.3, while repository requirement/test
pins 4.57.6. 181 test cases passed and the pin test failed exactly for this reason. The correct
action for reproduction is installing the required version, not updating documentation to 5.5.3.

## Dirty working tree

The corpus describes the audited working tree, including an implemented uncommitted DINOv3 adapter,
not only base commit `771cbba`. A fresh checkout of that commit alone is not guaranteed to match the
documented snapshot. File hashes in `SOURCE_LEDGER.md` are the stronger local join.

## Stale personal/scratch documents

`YOUR_FILES/` and `tmp/` contain useful history but are not current authority. The corpus did not use
their claims unless independently confirmed by current code, tests, or authoritative repository
guides.

## Safe answer pattern when sources disagree

Use:

> The current code does X with shape/value Y. The shipped default is Z. An older brief proposed A,
> and Investigation NN used override B. The older statement is historical/planned, not what executes
> in this snapshot.

That answer preserves every truth layer without pretending the disagreement does not exist.
