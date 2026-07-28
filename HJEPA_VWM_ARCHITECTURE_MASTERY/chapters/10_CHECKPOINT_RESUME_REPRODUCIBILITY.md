# 10 — Checkpoints, resume, RNG, and provenance

## Checkpoint schema

Current checkpoints declare:

```text
hjepa-phase1-checkpoint-v2
```

The payload includes:

| Category | Keys/content |
|---|---|
| progress | `next_step`, `completed_updates`, legacy `global_step` |
| learned state | online `bottleneck`, `target_bottleneck`, `coarse_flow`, `decoder` |
| optimization | full AdamW state and group metadata |
| configuration | nested serialized config |
| encoder identity | serialized `EncoderSpec`, feature fingerprint |
| data identity | exact dataset envelope/fingerprint |
| initialization | trainable initialization hash |
| tracking | W&B run ID, no credential |
| provenance | full run-provenance envelope |
| sampling | epoch and batch offset |
| randomness | Python, NumPy, Torch CPU, all CUDA RNG states |
| optional feature state | mean tracker and identity |
| optional whitening | whitener buffers and identity |

The frozen encoder weights are excluded. They are reloaded from the immutable repository/revision
bound by `EncoderSpec`.

## Atomic persistence

Checkpoint and artifact writes use a sibling temporary file, flush and `fsync`, then filesystem
replacement. An interrupted write should leave either the old complete destination or the new
complete destination, rather than a partially serialized final path.

Atomic replacement is not remote backup. If the volume disappears, locally atomic files disappear
with it.

## Step semantics

All three progress keys store the next update:

```text
checkpoint step 2500
→ updates 0..2499 complete
→ update 2500 is next
```

`completed_updates` equals `next_step`. The legacy `global_step` remains only for narrow historical
readers.

## Sampler-state derivation

For training count `N`, physical batch `B`, and drop-last:

```text
batches_per_epoch = floor(N/B)
epoch = next_step // batches_per_epoch
batch_offset = (next_step % batches_per_epoch)*B
```

The checkpoint stores `epoch` and `batch_offset`. Load recomputes the expected pair from saved step,
saved dataset count, and saved batch size; any mismatch fails.

On the first resumed loader, validated stored position is authoritative. Later epochs derive normally.
Dataset transfer intentionally discards sampler continuity because the old offset has no meaning in
the new inventory.

## Strict validation before mutation

Run preparation resolves live encoder, dataset, whitening, initialization, and run provenance before
checkpoint weights are allowed to modify the live stack. Load validates:

1. supported schema or explicit legacy permission;
2. exact encoder fingerprint;
3. exact dataset fingerprint unless explicit transfer;
4. allowed run-provenance equivalence;
5. sampler state consistency;
6. shape/key compatibility of each module state;
7. decoder-era compatibility;
8. optimizer group count, parameter mapping, and state tensor shapes;
9. feature-mean and whitening state hashes.

Only after validation are module/optimizer/buffer states loaded. This prevents a failed resume check
from partially mutating a fresh model.

## Exact random state

The checkpoint serializes:

- `random.getstate()`;
- NumPy RNG state;
- Torch CPU RNG;
- every CUDA generator state.

Training also reseeds per step, making its main random draws a pure function of seed and step.
Dataset transformations derive per-item seeds from sample/epoch. Diagnostics fork and restore RNG.
These layers jointly protect continuation.

Exact resume still depends on compatible software/hardware kernels. Provenance records runtime
identity so “same checkpoint” is not confused with “bit-identical execution everywhere.”

## Trainable initialization hash

Fresh `B`, `F_c`, and `D` state tensors are hashed by name, dtype, shape, and bytes. This makes
controlled encoder comparisons verify that their trainable stacks began identically. Since
encoder-specific input geometry changes bottleneck/decoder shapes, the project separates common
comparison identity from encoder identity deliberately.

## Dataset identity

For every split, the identity stores:

- sorted relative paths;
- resolved byte sizes;
- authoritative Decord frame count per clip folded into a fingerprint;
- clip count;
- total, minimum, and maximum frames;
- retained manifest paths, sizes, and SHA-256;
- preprocessing version;
- input geometry.

EGO completeness additionally checks:

- source UIDs exactly match the selection manifest;
- no train/validation source intersection;
- chunk-manifest counts match filesystem inventory;
- source-UID counts match.

For `ego4d_tiny`, parent split inventories are also bound.

A root path alone is therefore insufficient to impersonate a dataset.

## Run provenance

Schema:

```text
hjepa-run-provenance-v1
```

It contains:

- a common encoder-independent identity hash;
- resolved config;
- exact dataset identity;
- serialized encoder spec/fingerprint;
- optional whitening payload fingerprint;
- trainable initialization hash;
- seed-stream formulas;
- initial data-order identity;
- encoder runtime contract;
- runtime identity, including Git state;
- whitelisted W&B identity fields.

The runtime Git identity records dirty status but not the entire diff. This is why the mastery corpus
also records file hashes.

Tracking fields are restricted to `entity`, `project`, `group`, `name`, and `id`. Unknown fields are
refused to prevent credentials from entering provenance.

## Controlled encoder-arm comparison

The common comparison hash omits:

- encoder config;
- HF cache directory;
- checkpoint directory;
- whitening stats path.

It retains dataset, seed streams, trainable initialization, encoder execution contract, runtime, and
all common training controls. Two arms can be declared controlled only when their common identities
match.

This separation allows V-JEPA/SigLIP/DINO to differ in frozen representation while guarding every
non-encoder experimental axis.

## Dataset transfer policy

Normal resume requires exact common identity. Explicit dataset transfer removes only:

- dataset fingerprint;
- initial data order;
- data configuration.

Seeds, runtime, encoder execution, initialization, losses, schedules, and all non-data config remain
guarded. This is a narrow permission, not a general “ignore mismatches” switch.

An optimizer reset is similarly explicit. It changes the scientific continuation and must be
recorded.

## Whitening artifact

The versioned whitening envelope binds:

- mean, eigenvalues, eigenvectors;
- exact encoder spec/fingerprint;
- exact dataset identity;
- split;
- seed and exact clip budget;
- eigensolver identity;
- tensor payload fingerprint.

A fresh required-W&B whitening run uploads the stats artifact. A resume uses checkpoint-embedded
whitener buffers as the exact execution state and still carries the payload identity in provenance.

## Feature caches

Feature-cache readers validate:

- cache schema and producer identity;
- encoder fingerprint;
- dataset fingerprint;
- expected token shape;
- dtype;
- finite values.

Shape compatibility alone is not semantic compatibility. Two `(2048,768)` tensors from SigLIP2 and
DINOv3 must never be interchanged.

## Legacy checkpoints

Legacy load requires explicit permission in strict workflows and warns that encoder/dataset identity
is weaker. Two important compatibility cases:

- old checkpoints without decoder state may leave a fresh decoder, meaningful only if later trained;
- learned-output-query decoder checkpoints are incompatible with the fixed-position decoder because
  their parameter semantics differ even if some dimensions match.

Old three-group optimizer state is incompatible with current up-to-six-group decay partitioning.

## W&B continuation identity

The checkpoint stores the run ID. Resume uses it when no explicit ID is supplied and sets W&B
`resume="must"`. That prevents accidentally creating a new run that visually looks like a
continuation.

W&B is not checkpoint storage here. The run summary records final local path and SHA-256, while
checkpoints remain on the configured volume.

## Exact-resume checklist

Before calling a continuation exact, verify:

- current schema;
- encoder fingerprint match;
- dataset identity match;
- run-provenance match;
- module and optimizer compatibility;
- optional mean/whitener restored;
- sampler position validated;
- all RNG states restored;
- W&B continues the same run ID;
- next-step semantics are honored;
- final destination is durable.

Loading only `B.state_dict()` is transfer learning, not resume.
