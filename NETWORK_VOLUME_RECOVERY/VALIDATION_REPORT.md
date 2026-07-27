# Validation report and limits

Validation date: **2026-07-23 (Asia/Kolkata)**.

This report separates what was mechanically verified on this Mac from what requires the user's
private provider accounts, live volume, licensed datasets, or GPU. It is deliberately not a claim
that inaccessible bucket `4hzrwzk8ja` was inventoried.

## Outcome

The package is internally consistent and executable as a runbook:

- both before-deletion destinations—private AWS S3 and a Mac/external disk—have complete rescue,
  inventory, verification, credential-cleanup, and later restore paths;
- after-deletion handling separates complete AWS restore, complete local-disk restore, partial
  recovery, and honest ground-up derivation;
- current code's full `/workspace` contract is mapped, including non-dataset scientific state;
- helper scripts are fail-safe against parent traversal, symlink escape, incorrect links, missing
  targets, size mismatches, duplicate inventory paths, and regular-file collisions;
- exact restoration, functionally equivalent derivation, fresh start, and irrecoverable state are
  never treated as synonyms.

## Repository audit performed

The mandatory repository entry point and its required architecture/setup sources were read before
writing the recovery package, including:

- `AGENT_FILES/AGENTS.md`;
- `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`;
- `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md`;
- `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`, `SETUP.md`, and `NEW_POD.md`;
- `AGENT_FILES/KNOWLEDGE/ego4d/GUIDE.md` in full;
- remote-operation guides required by the repository instructions;
- current `config.py`, `data.py`, `provenance.py`, `train.py`, `encoders.py`, dataset builders,
  checkpoint/statistics/probe code, tests, and KANBAN launch/run records.

The reproducible literal search, excluding this generated recovery package and `.git`, found
`/workspace/` references in **125 files and 836 matching lines**. Dynamic paths constructed from
configuration and run tags were then resolved into the canonical families in
[`SHARED_REFERENCE/02_CODE_REFERENCED_PATH_CATALOG.md`](SHARED_REFERENCE/02_CODE_REFERENCED_PATH_CATALOG.md).

The audit established these non-obvious facts from current code:

- the volume is not “just datasets”; it can contain irreplaceable checkpoint/optimizer/RNG state,
  whitening envelopes, preflight evidence, logs, W&B offline queues, Git objects/dirty Pod state,
  generated EGO4D chunks, selection manifests, archives, and pinned model cache blobs;
- current `train.py` records checkpoint path/SHA metadata but does not upload checkpoint bytes as a
  W&B artifact;
- frozen encoder bytes are not duplicated into every checkpoint, so the pinned model snapshot and
  access remain part of resume readiness;
- the six large dataset views are intended symlink trees, so an S3 object copy alone is not proof of
  identical POSIX topology.

## Mechanical validation results

| Validation | Result |
|---|---|
| Recovery helper unit tests | **PASS: 14/14** |
| Python byte-compilation of all package scripts/tests | **PASS** |
| Ruff on all package scripts/tests | **PASS** |
| Recovery-helper `--help` entry points | **PASS: all 7 helpers** |
| Current project CLIs used by the runbooks | **PASS:** documented flags parse for encoder, SSv2/EGO4D selection/chunk/subset, and whitening tools |
| Fenced shell syntax | **PASS: 470/470 `bash` blocks parse with `bash -n`** |
| Fenced JSON syntax | **PASS: 1/1** |
| Markdown fence balance | **PASS** |
| Relative Markdown targets and local anchors | **PASS: 142/142 links resolve** |
| Official/external URL reachability | **PASS with expected authentication behavior:** 69 URLs exposed HTTP 200; unauthenticated RunPod S3 root returned expected HTTP 401 |
| Live AWS S3 `us-east-2` bulk price file | **PASS:** Standard $0.023/GB-month, PUT/LIST $0.005/1,000, GET/other $0.004/10,000, Standard-IA $0.0125/GB-month |
| Live AWS Data Transfer `us-east-2` bulk price file | **PASS:** $0.09/GB for the first 10 TB/month after the global free tier |
| Actual dirty-worktree reconstruction | **PASS:** all-ref bundle + separate index/worktree binary patches + untracked tar reproduced identical `git status --porcelain=v2 --untracked-files=all` at audited HEAD |
| macOS command assumptions | **PASS:** `diskutil list external physical`, `diskutil eject` syntax, `caffeinate -dimsu`, `shasum`, and `mktemp` behavior checked locally |

The actual Git round-trip was performed in an isolated temporary directory and removed afterward. It
used audited HEAD `771cbba077d9f846bdf7a7dd48e12dbf29d54b49`; the empty staged patch also verified why
`git apply --allow-empty` is required. No commit, reset, checkout of the working repository, push,
or remote mutation occurred.

The two large AWS CLI installer URLs returned HTTP 200 and began transferring successfully; the URL
validator intentionally stopped those binary downloads at its 20-second cap rather than downloading
about 59 MB and 73 MB merely to test reachability. All documentation/model/dataset/provider pages
completed normally. The private RunPod endpoint's unauthenticated 401 is not a dead link.

## Full repository test-suite result

The package's own tests pass. The repository-wide `pytest -q` was also attempted and **did not reach
project test execution** in the current Mac shell:

- 14 tests were skipped during collection;
- `tests/test_phase1_contract.py` failed to import because the `pytest` interpreter lacks `numpy`;
- result: one collection error, exit status 2.

The existing `.venv` contains NumPy/Torch but does not contain pytest, Transformers, or Decord. This
task did not mutate the user's Python environments or download large ML dependencies merely to mask
that pre-existing local setup mismatch. The fresh-Pod validation guide therefore makes a fully
installed `requirements.txt`, CUDA, `pytest -q`, model/data smokes, three real encoder smokes,
resource preflight, and disposable training run mandatory before stage 100%.

## Official-source validation

The evidence index is
[`SHARED_REFERENCE/03_OFFICIAL_SOURCES.md`](SHARED_REFERENCE/03_OFFICIAL_SOURCES.md). Its 70 unique
external URLs were checked on the audit date. Research used provider/publisher primary sources for:

- RunPod deletion/recovery policy, direct S3 access, endpoint mapping, limits, pricing, and volume
  lifecycle;
- AWS signup/Free-versus-Paid plan, India payment methods, account closure/retention, IAM, S3
  security, storage/request/egress prices, integrity tools, storage classes, and migration-off credit;
- current Qualcomm Something-Something V2 counts, format, downloads, extraction, and license route;
- EGO4D licensing/credential lifetime, CLI/tier/UID behavior, metadata, and video conventions;
- Hugging Face immutable revisions/authentication/gating;
- W&B file/artifact/offline-sync/restore semantics;
- rclone's S3, copy, check, inventory, and environment configuration semantics.

Prices are timestamped inputs, not timeless promises. The AWS bulk offer files and RunPod console
must be rechecked when a later restore is actually funded.

## Intentionally not executed

The following would require private credentials, a live provider resource, license acceptance,
money, or an external state change. They were not fabricated or claimed as tests:

- authenticated list/head/copy/check against old RunPod volume `4hzrwzk8ja`;
- creation of a RunPod S3 key, AWS account, IAM role, S3 bucket, EC2 instance, budget, or support case;
- any AWS/RunPod upload, deletion, key revocation, billing action, or volume creation;
- inspection of the user's W&B, Hugging Face, Qualcomm, EGO4D, GitHub-account-only, or external-disk
  private data;
- SSv2/EGO4D licensed downloads and hundreds-of-gigabytes preprocessing;
- CUDA decoder/encoder/training/checkpoint-resume validation on a RunPod GPU;
- a real source/destination inventory or proof of the old volume's exact contents/deletion status.

Those are execution-time gates in the runbooks. A bucket name, endpoint, path catalog, manifest SHA,
or W&B summary cannot substitute for authenticated object bytes.

## Final validation boundary

The documentation can take a user from either surviving old bytes or no old bytes to the strongest
honest end state. It cannot make an information-theoretically impossible promise:

- if a verified byte backup survives, exact object restoration and known topology reconstruction
  are testable;
- if no backup survives, current experiments can be rebuilt as a new derivation only while dataset
  and model entitlements still work;
- deleted checkpoints, optimizer/RNG state, Pod-only dirty code, unsynced W&B queues, logs, and exact
  generated EGO4D bytes cannot be reconstructed from metadata alone.

Final operational completion therefore occurs only when the applicable live-copy/rebuild steps and
all gates in
[`PERSPECTIVE_2_AFTER_DELETION/07_VALIDATE_EXPERIMENT_READINESS.md`](PERSPECTIVE_2_AFTER_DELETION/07_VALIDATE_EXPERIMENT_READINESS.md)
pass on the replacement volume.
