# 14 — Test contracts

The test suite is executable architecture documentation. The audited environment collected 181
passing cases and one deliberate environment-pin failure: installed Transformers 5.5.3 versus
required 4.57.6.

## Encoder contracts

`tests/test_encoders.py` covers:

- small public adapter surface;
- V-JEPA tubelet and frame-encoder time-major layouts;
- exactly-once FP32 input normalization;
- dense-output shape/order validation;
- FP32 disabling caller autocast;
- BF16 rejection off CUDA;
- frame microbatch equivalence;
- fingerprint stability and sensitivity to every identity field;
- sticky freeze/eval behavior;
- pinned DINO revision;
- Transformers dependency pin;
- DINO special-token stripping and malformed-output rejection;
- SigLIP vision-tower-only loading and patch order;
- unknown aliases/mutable revisions failing before load;
- pinned/cached V-JEPA load;
- input/output error paths;
- invalid spec/layout rejection.

These tests make the seam semantic, not merely shape-compatible.

## Bottleneck contracts

`tests/test_bottleneck_attention.py` proves:

- `M` spans the entire internal stream;
- an input-dependent path opens after an update;
- initialization is normalized slot identity, independent of input;
- cross-attention, competition/self-attention, and refinement/MLP bridges all train open;
- sharpened-attention temperature gets gradient once the zero bridge opens;
- latent-block count must be positive;
- raw and centered slot-rank diagnostics both exist.

## Flow and optimizer contracts

`tests/test_optimizer_and_flow.py` and `tests/test_agc.py` cover:

- geometry/zero-init parameters excluded from weight decay;
- slot/type embeddings are learned and affect flow output;
- loss-ramp edge cases;
- EMA-transition comparison tolerates legitimate FP32 rounding;
- a missing representable EMA update fails;
- incompatible optimizer load requires explicit reset;
- old learned-query decoder state is rejected;
- AGC scales oversized but not small gradients;
- disabling AGC is a no-op;
- bias/norm/zero-init exclusions;
- decay/no-decay forms an exact partition.

## Phase 1 shape/config/data contracts

`tests/test_phase1_contract.py` covers:

- CLI width/latent-shape overrides are applied before Stage 0;
- invalid `M,N_c,D_c` fail;
- the full external latent grid constructs successfully;
- locked shipped constants remain pinned;
- all four dataset roots resolve;
- `.webm`/`.mp4` indexing is deterministic;
- dataset emits context-only or a shared transformed pair as configured;
- stats loader can retain a final partial batch;
- SSv2 tiny subset is stratified, symlinked, and manifested;
- required scaffolding files exist.

## Decoder and reconstruction contracts

`tests/test_decoder.py` proves:

- decoder position is a fixed buffer, not learned output queries;
- position alone cannot create output content;
- reconstruction gradients reach input latent and trainable decoder weights;
- bottleneck/decoder derive detailed geometry from each encoder spec.

`tests/test_reconstruction_loss.py` covers tokenwise cosine semantics, legacy relative MSE,
magnitude invariance, detached target/active prediction gradient, and invalid mode rejection.

## Present-only and residual-feature contracts

`tests/test_present_recon_only.py` proves the mode skips prediction and leaves `F_c` unchanged, and
requires a present-reconstruction weight.

`tests/test_residual_recon_target.py` covers:

- first mean update copies, later update EMA-lerps;
- tracker has buffers only and preserves input dtype;
- momentum/shape validation;
- training actually changes the target;
- active mode requires tracker;
- predicted anchor reaches `F_c`;
- config pairing rules;
- checkpoint tracker round-trip and old-checkpoint fallback;
- diagnostics use residual targets and report video gap;
- default config leaves baseline behavior untouched.

## Whitening contracts

`tests/test_whitening.py` proves:

- no trainable parameters and configure-before-use;
- round-trip and unit covariance on controlled data;
- dtype preservation and stats validation;
- config validation;
- artifact loading/shape checks;
- resume prefers and verifies embedded checkpoint whitener;
- stats use context-only path and exact clip/token-row counts;
- both train branches whiten consistently and checkpoint state persists.

## Provenance and resume contracts

`tests/test_provenance.py` covers encoder-spec round-trip/fingerprint, dataset inventory plus frame
counts, atomic whitening envelopes, semantic rejection despite matching shape, feature-cache
identity/dtype/offset binding, tensor validation, paired encoder common identity, and narrow
dataset-transfer rules.

`tests/test_encoder_run_contract.py` adds:

- equal-geometry encoder names do not perturb trainable initialization;
- encoder mismatch fails before state mutation;
- checkpoint next-step resume;
- exact dataloader position restoration;
- resource-throughput arithmetic;
- auxiliary and optimizer tensor validation before mutation;
- complete CLI operator surface;
- batch one rejection for shuffled-video diagnostics;
- transfer keeps non-dataset guards;
- diagnostics cannot perturb future training draws;
- interrupted resume matches uninterrupted next update.

The last test is the strongest end-to-end exact-resume contract.

## Offline-probe contracts

`tests/test_rank_probe.py` covers spectrum/effective-rank equivalence, energy ranks, JSON report
views/ceilings, pre-concatenation frame statistics, generic tubelet behavior, shape/empty rejection,
and rank/drift manifest/cache compatibility.

`tests/test_drift_probe.py` covers offset parsing, non-overlap boundary, exact temporal indices,
deterministic probe selection, shared cache path, known cosine-distance recovery, normalization,
aggregation axes, Spearman ties/endpoints, checkpoint config filtering, real bottleneck drift, and
whitener identity validation.

## Dataset-builder contracts

`tests/test_chunk_ego4d.py` proves:

- ffmpeg absence fails before work;
- worker count is capped;
- every ffmpeg encode gets one thread;
- any failed encode fails the batch.

`tests/test_select_ego4d_uids.py` proves grouped-video filtering, authoritative tier filtering, and
source-file hashes in the selection manifest.

## SIGReg contracts

`tests/test_sigreg.py` checks:

- near-zero discrepancy for isotropic Gaussian samples;
- stronger penalty for anisotropic low rank;
- finite gradient flow;
- degenerate batch behavior;
- reproducibility with an explicit generator.

## Validation commands

Meaningful local validation used:

```bash
python -m pytest -q
python -m py_compile config.py data.py encoders.py models.py losses.py \
  diagnostics.py provenance.py train.py whiten_stats.py rank_probe.py drift_probe.py
python -c 'from models import smoke_test_models; smoke_test_models()'
```

The standalone `pytest` executable resolved to a separate Python 3.11 environment without NumPy and
failed at collection. `python -m pytest` used the intended Python 3.12 environment.

## What tests do not prove

- Tests do not prove a 15k GPU run is stable.
- Synthetic shape tests do not prove Hugging Face weights are available in cache.
- Unit rank tests do not prove learned rank exceeds 60.
- Exact local resume does not guarantee cross-GPU bitwise determinism.
- Builder tests do not prove a remote corpus is complete; provenance scan does.
- Passing code tests cannot satisfy copy/batch-mean scientific gates.

The hierarchy is:

```text
unit/integration contracts
→ Stage 0 real-device sanity
→ recipe preflight
→ monitored training
→ late-window scientific acceptance
```
