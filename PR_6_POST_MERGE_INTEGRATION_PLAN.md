# PR #6 Post-Merge Integration Decision Map

Objective: make `dinov3_vitb16` a first-class third encoder choice beside
`vjepa2_vitl16` and `siglip2_vitb16`, while proving that every implemented Phase-1
pipeline still works and that incompatible artifacts cannot be mixed.

## Execution record — 2026-07-26

### Completed

- PR #6 was merged into `phase1-v0.2-frozen-encoder` as
  `36e7b1fa35535b6dba569e92c31a5ea3884d2498`.
- The merged tree was certified in an isolated worktree; the user's dirty research
  worktree was not reset, cleaned, stashed, or overwritten.
- The exact post-merge baseline passed 168 tests.
- Follow-up PR #7 closed the remaining code/MLOps gaps and was merged as
  `9a61583e12099f0e7018924e7e52d2965508fe09`.
- PR #7 added regression evidence that:
  - `--encoder dinov3_vitb16` uses pinned revision
    `5931719e67bbdb9737e363e781fb0c67687896bc` with no revision flag;
  - repeated DINO frames produce repeated patch blocks;
  - frame permutation only permutes complete time blocks;
  - frame microbatches 1, 3, and 8 agree;
  - special tokens stay out of dense output;
  - fp32 output is finite and DINO remains frozen/eval;
  - W&B exposes authoritative `resolved_encoder_spec`,
    `resolved_feature_fingerprint`, and `resolved_dataset_fingerprint` fields while
    retaining historical compatibility fields.
- The final source tree passed the full suite twice: 171 passed with only the three
  expected legacy-checkpoint warnings.
- The model smoke, diagnostic smoke, focused whitening/provenance/drift/checkpoint
  matrix (35 tests), changed-file Ruff, changed-file Black, and `git diff --check`
  all passed.
- The tested commit `9171388942d55015e077d1bd29bf520799bf7221` and merged commit
  `9a61583e12099f0e7018924e7e52d2965508fe09` have the same Git tree
  (`af2dcefa075cae692c75a52c2e75b29e29555e29`).

### External gate not executable on 2026-07-26

The configured SSH endpoint `hjepa-runpod` refused the TCP connection. A read-only
`runpodctl get pod` query returned zero pods for the configured RunPod account. Therefore
there is currently no target GPU runtime on which to run the authenticated real-adapter,
real-data, W&B, whitening/rank/drift, resource, or short checkpoint/resume gates.

This is not a code failure and it is not valid evidence that those gates passed. Repository
policy forbids an unattended agent from creating/redeploying a pod or guessing its volume,
GPU, template, and secret attachment. The remaining external gate requires a deployed
A100-class pod attached to the existing network volume and the approved secret mappings:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_dinov3_read }}
WANDB_API_KEY={{ RUNPOD_SECRET_wandb_hjepa }}
```

Once that pod exists and `hjepa-runpod` points to it, restart at `real-adapter-smokes`
using exact source commit `9a61583e12099f0e7018924e7e52d2965508fe09`. No 15,000-step DINO
prediction run is part of this remaining engineering certification.

Target selector:

```bash
python train.py --encoder vjepa2_vitl16 ...
python train.py --encoder siglip2_vitb16 ...
python train.py --encoder dinov3_vitb16 ...
```

This plan begins after PR #6 is merged. Perform the work in a fresh worktree or clean
clone so the current uncommitted research record is not overwritten.

## Credential state and exact requirement

Local audit on 2026-07-26:

- the repository `.env` has no `HF_TOKEN` or equivalent Hugging Face credential;
- the current process has no Hugging Face token environment variable;
- `hf auth whoami` reports `Not logged in`;
- the local Hugging Face cache does not contain the pinned DINOv3-B repository;
- this codebase does not call `load_dotenv`, so a value written only to `.env` would not be
  loaded automatically by `train.py` or `encoders.py`.

Therefore the code/PR does not need a secret committed or added to its architecture, but a
fresh machine cannot download the gated DINO weights in the current local credential state.

Exact operational contract for future RunPod use:

```text
RunPod secret name: hf_dinov3_read
Pod environment mapping:
HF_TOKEN={{ RUNPOD_SECRET_hf_dinov3_read }}
```

The secret value must be a fine-grained, read-only Hugging Face token whose authenticating
account already has access to:

```text
facebook/dinov3-vitb16-pretrain-lvd1689m
```

If the friend is supplying the authorized project runtime, ask her to create/install that
dedicated secret and confirm the secret name and mapping. Do not ask her to send the token
value for this repository's `.env`, chat, a shell command, or a KANBAN file. Her prior W&B
runs prove that she had working access when those runs executed; they do not prove that the
credential is currently attached to every future pod template.

The alternative is for the user to obtain gated access on their own Hugging Face account and
create the same fine-grained read token under that identity.

Credential readiness is proven on the target pod, without printing the token, by:

```bash
test -n "$HF_TOKEN"
hf auth whoami
python encoders.py --smoke --encoder dinov3_vitb16 --batch-size 1
```

The smoke must resolve revision
`5931719e67bbdb9737e363e781fb0c67687896bc`. An existing warm cache may allow the same old
pod to run without downloading again, but cache presence is not portable authorization for a
fresh pod.

## Success definition

Integration is complete only when:

1. All three aliases build through the same public `FrozenEncoder` factory.
2. V-JEPA remains the default and its existing behavior is unchanged.
3. DINO resolves its pinned SHA without requiring `--encoder-revision`.
4. Present-only and full-prediction short runs work for all three encoders.
5. DINO full-prediction resource usage is measured before a paid run.
6. Same-encoder checkpoints, whitening files, caches, rank probes, and drift probes work.
7. Cross-encoder checkpoint/stat/cache reuse fails before state mutation.
8. W&B records the authoritative resolved `EncoderSpec`, dataset identity, and artifact
   fingerprints.
9. Exact checkpoint resume remains deterministic.
10. The full local suite and changed-file formatting checks pass.
11. A fresh target pod receives `HF_TOKEN` through the approved secret mapping and passes the
    authenticated DINO smoke without exposing the token.

## Locked invariants

These are already settled by the codebase and are not open design questions:

- `FrozenEncoder` remains the only public detailed-encoder seam.
- DINO-specific model classes and token rules stay private to `encoders.py`.
- V-JEPA remains the default alias.
- The DINO alias stays pinned to
  `5931719e67bbdb9737e363e781fb0c67687896bc`; never fall back to Hub `main`.
- DINO retains all eight frames and returns `(B,2048,768)`.
- DINO strips exactly one CLS plus four register tokens per frame.
- The external abstract interface remains `(B,32,256)`.
- No loss, schedule, gradient-routing, optimizer, EMA, or diagnostic-gate change is
  part of encoder integration.
- Cross-encoder resume and cross-encoder whitening/cache reuse remain forbidden.
- Frozen encoder weights remain outside Phase-1 checkpoints.

## selector-contract: Keep One Three-Way Selector

Blocked by: none

Status: resolved

Type: Research

### Question

Should DINO use a new training path or the existing encoder selector?

### Answer

Use the existing selector. PR #6 already implements the intended public behavior:

```text
--encoder vjepa2_vitl16
--encoder siglip2_vitb16
--encoder dinov3_vitb16
```

Do not introduce DINO-specific branches in `data.py`, `models.py`, `losses.py`,
`diagnostics.py`, `train.py`, whitening tools, or probes.

## dino-contract: Keep The Adapter Strict And Private

Blocked by: none

Status: resolved

Type: Research

### Question

Should the DINO adapter be generalized to arbitrary DINO checkpoints?

### Answer

No. Keep the stable alias tied to the validated ViT-B/16 contract. Failing loudly on a
different register count, sequence length, or width is safer than silently corrupting the
spatial lattice. A future DINO size should receive a separate registration and contract.

## artifact-policy: Isolate Every Encoder’s Artifacts

Blocked by: none

Status: resolved

Type: Research

### Question

Can equal-shaped DINO and SigLIP features share whitening files or caches?

### Answer

No. Artifact compatibility is semantic, not shape-only. Preserve strict binding to encoder
family, repository, revision, preprocessing, layout, precision, frame microbatch, attention
implementation, dataset fingerprint, and transform seed.

## validation-policy: Use Offline CI Plus Gated Real Smokes

Blocked by: none

Status: resolved

Type: Research

### Question

Must public CI download the gated DINO checkpoint?

### Answer

No. Keep deterministic fake-backend contract tests in the normal suite. Require a
credentialed CUDA smoke whenever the DINO revision, Transformers pin, adapter contract, or
shared encoder seam changes. The credential may belong to the authorized project operator
rather than every contributor, but the execution environment must receive it through an
approved secret mechanism. Record the real smoke as provenance, not as a public-secret CI
dependency.

## merge-snapshot: Establish The Exact Post-Merge Baseline

Blocked by: none

Status: open

Type: Research

### Question

What exact source tree and dependency environment will every later gate certify?

### Answer

To be filled during execution.

Acceptance evidence:

- record the merge commit SHA and confirm a clean worktree;
- record Python, PyTorch, Transformers, CUDA, cuDNN, and GPU model;
- confirm `transformers==4.57.6`;
- save `git diff --check`, full pytest, and changed-Python Black results;
- confirm no local KANBAN/research changes were overwritten;
- use this same commit for all three real-adapter comparisons unless a follow-up fix is
  required, in which case restart the certification from the new fix commit.

## default-alias-proof: Test DINO Without An Explicit Revision

Blocked by: merge-snapshot

Status: open

Type: Prototype

### Question

Does the public factory actually make DINO as easy to select as V-JEPA and SigLIP?

### Answer

To be filled during execution.

Required regression:

1. Monkeypatch the DINO loader.
2. Construct `EncoderConfig(alias="dinov3_vitb16", revision=None)`.
3. Build through `build_frozen_encoder`.
4. Assert requested and resolved revisions both equal the pinned SHA.
5. Assert the factory passes that exact SHA to `from_pretrained`.
6. Keep explicit-revision override coverage.
7. Assert mutable/non-40-character revisions remain rejected.

Acceptance: the DINO selector works with only `--encoder dinov3_vitb16`.

## temporal-contract-tests: Close The Remaining Frame Semantics

Blocked by: merge-snapshot

Status: open

Type: Prototype

### Question

Does the frame-native path preserve content and time ordering exactly?

### Answer

To be filled during execution.

Add or confirm tests for:

- repeated input frames produce repeated per-frame patch features;
- permuting frames only permutes the eight feature-time blocks;
- special tokens never appear in patch output;
- frame microbatch `1`, an intermediate value, and unmicrobatched execution agree;
- parent `.train()` cannot unfreeze DINO or leave it in training mode;
- fp32 and bf16 paths return the declared dtype/finite output contract;
- token identity is checked, not only shape.

Acceptance: DINO behaves as eight ordered independent images before downstream temporal
positioning.

## three-encoder-offline-matrix: Prove Shared-Code Non-Regression

Blocked by: default-alias-proof, temporal-contract-tests

Status: open

Type: Prototype

### Question

Did enabling DINO accidentally change the two existing encoder paths or downstream module
construction?

### Answer

To be filled during execution.

Run the full synthetic/offline matrix:

| Gate | V-JEPA2 | SigLIP 2 | DINOv3 |
|---|---:|---:|---:|
| Factory/spec/fingerprint | required | required | required |
| Raw `[0,1]` validation | required | required | required |
| Normalization exactly once | required | required | required |
| Frozen/eval/zero trainables | required | required | required |
| Frame/tubelet ordering | required | required | required |
| `B`, `B_EMA`, `D` construction | required | required | required |
| Present-only forward/backward | required | required | required |
| Full-prediction forward/backward | required | required | required |
| Diagnostic RNG isolation | required | required | required |

Expected resolved shapes:

```text
V-JEPA2: layout 4x16x16, features (B,1024,1024)
SigLIP2: layout 8x16x16, features (B,2048,768)
DINOv3:  layout 8x16x16, features (B,2048,768)
all B outputs: (B,32,256)
```

Acceptance: the full repository suite passes and V-JEPA/SigLIP expected specs and
fingerprints are unchanged.

## authoritative-run-config: Remove The W&B Geometry Ambiguity

Blocked by: merge-snapshot

Status: open

Type: Prototype

### Question

How do operators and automation avoid reading legacy `model.d_e=1024` as DINO’s real width?

### Answer

To be filled during execution.

Conservative implementation:

- do not delete compatibility fields required by historical checkpoints;
- continue constructing production modules only from the resolved `EncoderSpec`;
- make `resolved_provenance.encoder_spec` the documented authoritative W&B config block;
- add a regression proving DINO W&B/provenance records `feature_dim=768`,
  `layout.n_tokens=2048`, and the DINO feature fingerprint;
- if a flat summary field is useful, log an explicitly named resolved value such as
  `resolved_feature_dim`, not a second ambiguous `d_e`;
- update comparison scripts to read resolved provenance rather than legacy `model.d_e`.

Acceptance: no dashboard or automation maintained by the project needs to infer DINO
geometry from compatibility fields.

## artifact-isolation-matrix: Prove Fail-Closed Compatibility

Blocked by: three-encoder-offline-matrix

Status: open

Type: Prototype

### Question

Can any checkpoint, whitening file, or feature cache be silently consumed by the wrong
encoder?

### Answer

To be filled during execution.

Required matrix:

| Producer | Consumer | Expected |
|---|---|---|
| V-JEPA checkpoint | V-JEPA | load |
| SigLIP checkpoint | SigLIP | load |
| DINO checkpoint | DINO | load |
| V-JEPA checkpoint | SigLIP/DINO | reject before mutation |
| SigLIP checkpoint | DINO | reject before mutation |
| DINO checkpoint | V-JEPA/SigLIP | reject before mutation |
| each whitening artifact | matching encoder/dataset identity | load |
| each whitening artifact | either wrong equal/different-shape encoder | reject |
| each rank/drift cache | matching feature fingerprint | load |
| each rank/drift cache | wrong encoder/revision/preprocessing | reject |

Also prove an explicit same-encoder resume does not require the external whitening file when
the checkpoint’s embedded whitener state is authoritative.

Acceptance: every incompatible case fails before loading trainable state or beginning a
paid run.

## deterministic-resume-matrix: Protect Training Continuity

Blocked by: artifact-isolation-matrix

Status: open

Type: Prototype

### Question

Does DINO preserve the repository’s exact resume contract?

### Answer

To be filled during execution.

For each encoder, compare an uninterrupted short run with a save/restart continuation:

- same next sample IDs and sampler position;
- same training RNG draws;
- same optimizer and EMA update count;
- same trainable state at the comparison step;
- same W&B run identity on resume;
- no repeated completed update;
- checkpoint inspection/compatibility occurs before state mutation.

Acceptance: DINO passes the same deterministic-resume tolerance as V-JEPA and SigLIP.

## real-adapter-smokes: Validate All Three Real Backbones On One Runtime

Blocked by: three-encoder-offline-matrix

Status: open

Type: Prototype

### Question

Do the real checkpoint implementations match the tested private contracts on the target
GPU?

### Answer

To be filled during execution.

On the same clean commit and A100-class runtime:

1. Authenticate with an account/project credential that is authorized for the gated DINO
   repository, without printing or logging the token.
2. Pre-download encoders sequentially to avoid cache races.
3. Run `encoders.py --smoke` for V-JEPA, SigLIP, and DINO.
4. For each, record requested/resolved revision, shape, dtype, finite output, parameter
   count, zero trainables, peak encoder memory, and throughput.
5. Run DINO once without `--revision` and verify it resolves the pinned SHA.
6. Run one repeated-frame and one permuted-frame real probe for DINO.

Acceptance: all three real backbones pass on the same dependency/runtime envelope.

## real-data-mode-matrix: Exercise Every Implemented Branch

Blocked by: real-adapter-smokes, deterministic-resume-matrix

Status: open

Type: Prototype

### Question

Does encoder switching work beyond isolated encoder calls?

### Answer

To be filled during execution.

Run short, from-scratch jobs for every encoder:

| Dataset | Mode | Required purpose |
|---|---|---|
| `ssv2_tiny` | present-only | context-only decode, `B+D`, diagnostics |
| `ssv2_tiny` | full prediction | context/future encode, `B_EMA+F_c`, copy/batch baselines |
| `ego4d_tiny` | present-only | MP4/data identity path |
| `ego4d_tiny` | full prediction | complete EGO4D temporal path |

For each run verify:

- correct `prediction_active`/`present_recon_only` flags;
- expected encoder shape and fingerprint;
- finite forward/backward;
- nonzero gradients only on intended modules;
- successful optimizer/EMA transition;
- checkpoint save/resume;
- W&B strict initialization and resolved config;
- no skipped/nonfinite update caused by integration wiring.

These are wiring smokes, not scientific verdicts. Do not compare raw reconstruction losses
across encoders.

Acceptance: all twelve encoder/dataset/mode combinations complete their registered short
smoke or have a diagnosed, explicitly scoped resource failure.

## dino-whitening-and-probes: Certify DINO’s Offline Toolchain

Blocked by: real-adapter-smokes, artifact-isolation-matrix

Status: open

Type: Prototype

### Question

Can DINO use the same strict whitening/rank/drift toolchain without a DINO-specific bypass?

### Answer

To be filled during execution.

Required evidence:

1. Fit a tiny deterministic DINO whitening artifact.
2. Inspect schema, encoder/dataset fingerprints, clip/token counts, finite tensors, and
   payload hash.
3. Load it into a present-only and full-prediction smoke.
4. Run `rank_probe.py` through the generic factory.
5. Run `drift_probe.py` with a DINO checkpoint and embedded whitener identity.
6. Prove SigLIP’s equal-width whitening file is rejected.
7. Prove V-JEPA’s different-width whitening file is rejected.

Acceptance: no offline tool imports or branches on DINO outside the encoder factory/spec.

## dino-full-resource-preflight: Find A Safe Full-Prediction Envelope

Blocked by: real-data-mode-matrix, dino-whitening-and-probes

Status: open

Type: Prototype

### Question

What batch/frame-microbatch combination safely supports DINO full prediction?

### Answer

To be filled during execution.

Run the exact full-prediction recipe with diagnostics measured separately. Start at the
project’s desired physical batch and descend only the documented frame-microbatch ladder:

```text
32 -> 16 -> 8 -> 4 -> 2 -> 1
```

Do not silently change logical batch or introduce naive gradient accumulation because
variance/covariance losses depend on the logical batch.

Record:

- encoder-only and total peak allocated/reserved memory;
- training-step time separately from diagnostic time;
- examples, input frames, and detailed tokens per second;
- headroom before launch;
- context-plus-future encoder cost;
- whether whitening changes peak usage;
- final physical batch and frame microbatch.

Acceptance: one exact configuration passes Stage 0, a training step, a diagnostic step, and
checkpoint save with documented headroom.

## post-merge-release-gate: Certify The Three-Way Switch

Blocked by: authoritative-run-config, dino-full-resource-preflight

Status: open

Type: Research

### Question

Is the implementation ready to be treated as a supported three-encoder system?

### Answer

To be filled during execution.

Release checklist:

- exact clean commit recorded;
- full local suite passes;
- default-alias and temporal-contract tests pass;
- all real adapters pass on the pinned runtime;
- all short real-data modes pass;
- same-encoder resume passes;
- cross-encoder artifact misuse fails closed;
- DINO whitening/rank/drift paths pass;
- W&B resolved configuration is unambiguous;
- DINO full-prediction resource envelope is recorded;
- no DINO conditional escaped `encoders.py`/the private registry;
- V-JEPA remains the default and its fingerprints/regressions are unchanged.

Acceptance: a new user can change only `--encoder` plus select the matching artifacts and
obtain the intended backend without code edits.

## paid-dino-prediction: Run Scientific Training Only After Certification

Blocked by: post-merge-release-gate

Status: deferred

Type: Research

### Question

Should the project spend GPU budget on a full 15,000-step DINO prediction run?

### Answer

Deferred by the user. It is outside the technical release gate and requires human
compute-budget approval only if the project later chooses to run it.

This is not necessary to prove the switch works. It is necessary to learn whether
frame-independent DINO features are a good temporal-prediction substrate. Register the
experiment separately and retain the existing copy and batch-mean acceptance gates.

## research-ledger-reconciliation: Preserve One Canonical Experiment History

Blocked by: none

Status: open

Type: Research

### Question

Which run numbering should become canonical after the PR history and newer local/W&B
history are reconciled?

### Answer

The inconsistencies and evidence gaps are reported, without a remediation design, in
`KANBAN_INCONSISTENCIES_AND_GAPS.md`. Canonical renumbering is intentionally left unresolved
and does not block code certification.

## Human-only authority and future choices

No unresolved model/code architecture decision blocks the post-merge work.

1. **Friend-operated/project credential:** ask the friend to install a dedicated
   fine-grained read token as RunPod secret `hf_dinov3_read` and map it to `HF_TOKEN`. What
   the user needs from her is confirmation that this secret is attached to the pod/template,
   not the raw token value. The present local `.env` is neither populated for Hugging Face
   nor automatically loaded by the training code.
2. **Independent operation:** if the friend will not own that project credential, the user
   needs their own approved Hugging Face identity and fine-grained read token. Gated access
   remains attached to the authenticating identity, not to the Git branch, repository, cached
   commit, or PR.
3. **Future paid science:** no 15,000-step DINO prediction run is authorized or required now.
   A later decision to run one is a compute-budget choice, not part of making the code
   experiment-ready.

KANBAN numbering and narrative inconsistencies are recorded separately and do not require a
design decision before technical certification.

The following do **not** require a new human design decision:

- whether DINO gets a separate pipeline: no;
- whether DINO becomes the default: no, V-JEPA remains default;
- whether to pool DINO frames: no, keep all eight;
- whether artifacts can cross encoders: no;
- whether to loosen immutable revision checks: no;
- whether to change losses/schedules for integration: no;
- whether to add a general arbitrary-DINO abstraction: no;
- whether a full prediction smoke is required: yes;
- whether a full paid prediction run is required merely to merge the switch: no.

## Rollback policy

The three-way switch is low-risk to existing users because V-JEPA remains the default.

If DINO-only validation fails:

- stop selecting `dinov3_vitb16`;
- preserve failed artifacts/logs for diagnosis;
- do not weaken fingerprint or shape checks to make them load;
- keep V-JEPA/SigLIP available;
- fix the private DINO adapter or runtime and restart certification from
  `merge-snapshot`.

If a shared-seam regression affects V-JEPA or SigLIP:

- treat it as a release blocker;
- revert or fix the shared change before any paid run;
- rerun the complete three-encoder matrix from a new clean commit.

Do not delete checkpoints, whitening artifacts, provenance, or W&B evidence as a rollback
mechanism.
