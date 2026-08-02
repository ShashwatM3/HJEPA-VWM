# Plan — residual full-prediction arm

## Shared implementation contract

The launch implementation provides a true warm start in `train.py`/`provenance.py`:

1. add operator flag `--warm-start-from PATH`, mutually exclusive with `--resume`;
2. accept only current `hjepa-phase1-checkpoint-v2` sources;
3. validate source SHA, exact encoder fingerprint, dataset fingerprint, feature-space mode,
   bottleneck/decoder keys and shapes, selected `N_c,D_c,M`, and latent/decoder architecture before
   mutating a module;
4. load source online `bottleneck` and `decoder`;
5. copy the loaded online bottleneck into a fresh `target_bottleneck`;
6. leave `coarse_flow`, optimizer, RNG, sampler, W&B ID, and step fresh;
7. compute the trainable initialization hash after warm start;
8. record a `warm_start` provenance block with source path, SHA-256, W&B ID, feature/dataset
   fingerprints, transferred components, fresh components, and target-copy policy;
9. make warm start available to preflight, resource preflight, and training;
10. reject any attempt to use warm start and resume together.

Transfer the decoder because it is the matched readout that certified the bottleneck's
reconstruction ability. It remains trainable. Do not load the source `F_c`, even if present-only
training left it unchanged.

## Parallel target override

The scientific override is:

```text
--temporal-target {residual,full_latent}
```

It must set only `cfg.train.predict_residual`. This explicit override is required because the
repository has one YAML and both modes must run concurrently from one clean worktree.

Update:

- `train.build_arg_parser` and `train.main`;
- config/CLI documentation in `AGENT_FILES/AGENTS.md`, `config.py`, and `configs/train.yaml`;
- `GUIDES/CODEBASE_STRUCTURE.md` only if a file is added;
- `GUIDES/READING_EXPERIMENTS.md` only if acceptance behavior changes.

## Provenance parity

The narrow comparator permits exactly one scientific difference:

```text
train.predict_residual: false <-> true
```

Tracking names, W&B IDs, output paths, resource measurements, and the target flag may differ.
Encoder, dataset, shape, source checkpoint SHA, initialization hash, seed streams, optimizer,
losses, schedules, and all other fields must match.

## Tests

Add targeted tests proving:

- warm start and resume are mutually exclusive;
- validation occurs before the first live parameter mutation;
- encoder/dataset/shape/whitening mismatches fail;
- `B` and `D` load exactly;
- `B_EMA` equals the loaded online `B`, not the source EMA;
- `F_c` remains byte-identical to fresh initialization;
- optimizer has no restored moments;
- step, sampler, RNG, and W&B identity start fresh;
- provenance contains the source SHA and transfer policy;
- same source/seed yields the same step-0 trainable hash in both modes;
- `--temporal-target residual` changes only `predict_residual`;
- residual gradient routing remains `L_flow -> B,F_c`, not encoder or `B_EMA`.

Run:

```bash
pytest tests/test_optimizer_and_flow.py tests/test_encoder_run_contract.py tests/test_present_recon_only.py -q
pytest -q
python -m py_compile config.py provenance.py train.py
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

## Locked arm config

Use the parent recipe plus:

```text
temporal_target = residual
predict_residual = true
```

No arm-specific code or coefficient change is permitted after the preflight.

## Analysis

Use Reading Cycle A. The primary failure discriminator is:

```text
rising coarse_copy_loss + ratio near 1
    => dynamic latent, zero-residual/no-predictor failure
```

Compare ratios and verdicts to the paired full-latent arm; do not compare absolute flow losses.
