# 19 — File-by-file map

## Executable core

| File | Responsibility | Key seam |
|---|---|---|
| [`config.py`](../../config.py) | all shipped defaults and path contract | dataclasses before CLI overrides |
| [`data.py`](../../data.py) | deterministic window indices, decode, paired transforms, loaders | raw `(B,8,3,256,256)` |
| [`encoders.py`](../../encoders.py) | pinned adapters, `EncoderSpec`, normalization, freeze/fingerprint | `(B,N_e,D_e)` |
| [`models.py`](../../models.py) | `B`, `B_EMA`, `F_c`, `D`, mean tracker, whitener | learned tensor graph |
| [`losses.py`](../../losses.py) | flow, recon, variance, covariance, slot, SIGReg | pure tensor objectives |
| [`diagnostics.py`](../../diagnostics.py) | collapse/rank/attention/baseline/AGC helpers | metric semantics |
| [`provenance.py`](../../provenance.py) | dataset/artifact/run identities, atomic envelopes | strict semantic joins |
| [`train.py`](../../train.py) | CLI, construction, optimizer, step, diagnostics, save/resume, W&B | state machine |

If asked “where does behavior live?” prefer this ownership map over searching by generic ML concept.

## Dataset builders

| File | Responsibility |
|---|---|
| [`make_subset.py`](../../make_subset.py) | stratified SSv2 tiny symlink view + manifest |
| [`select_ego4d_uids.py`](../../select_ego4d_uids.py) | EGO filter/diversity/source split/download batches |
| [`chunk_ego4d.py`](../../chunk_ego4d.py) | privacy-aware four-second ffmpeg chunk creation |
| [`make_ego4d_subset.py`](../../make_ego4d_subset.py) | source-capped EGO tiny symlink view |

## Offline/statistics tools

| File | Responsibility |
|---|---|
| [`whiten_stats.py`](../../whiten_stats.py) | exact offline FP64 covariance eigen-statistics |
| [`rank_probe.py`](../../rank_probe.py) | detailed-feature covariance spectrum/rank |
| [`drift_probe.py`](../../drift_probe.py) | temporal offset drift for `e`, online `B`, or EMA `B` |
| [`run_history.py`](../../run_history.py) | unsampled W&B public-API history and acceptance snapshot |
| [`parse_logs.py`](../../parse_logs.py) | console metric log → structured JSON |
| [`fetch_drift_outputs.sh`](../../fetch_drift_outputs.sh) | proxy-SSH marker/base64 result transfer |

## Tests

| File | Primary contract |
|---|---|
| `test_encoders.py` | adapter identity/layout/normalization/freeze/errors |
| `test_bottleneck_attention.py` | internal width, init, bridge opening, temperature |
| `test_optimizer_and_flow.py` | optimizer exclusions, flow identity codes, EMA, compatibility |
| `test_agc.py` | clipping math and decay partition |
| `test_phase1_contract.py` | config, CLI, shapes, dataset/runtime scaffold |
| `test_decoder.py` | fixed position cannot become content |
| `test_reconstruction_loss.py` | cosine/legacy objective and detach |
| `test_present_recon_only.py` | prediction path truly absent |
| `test_residual_recon_target.py` | mean tracker, routing, diagnostics, checkpoints |
| `test_whitening.py` | ZCA math, fit/use/resume consistency |
| `test_provenance.py` | semantic artifact identity and atomicity |
| `test_encoder_run_contract.py` | pre-mutation validation and exact interrupted resume |
| `test_rank_probe.py` | spectrum/report/cache contract |
| `test_drift_probe.py` | temporal offsets/cache/drift math |
| `test_chunk_ego4d.py` | ffmpeg resource/failure rules |
| `test_select_ego4d_uids.py` | authoritative EGO selection identity |
| `test_sigreg.py` | distributional penalty behavior |

See [14](14_TEST_CONTRACTS.md) for the test-by-test teaching view.

## Top-level technical docs

| Path | Role |
|---|---|
| [`README.md`](../../README.md) | project entry and current high-level commands |
| [`AGENTS.md`](../../AGENTS.md) | redirects agents to authoritative instructions |
| [`AGENT_FILES/AGENTS.md`](../../AGENT_FILES/AGENTS.md) | repository truth entry, mandatory read order, full map |
| `AGENT_FILES/AGENT-BEHAVIOUR/` | protocol, workflow, naming/code-design rules |
| `AGENT_FILES/SETUPS/` | persistent volume and pod setup |
| `AGENT_FILES/KNOWLEDGE/` | current briefs, encoder studies, EGO research |
| [`GUIDES/latest_brief.md`](../../GUIDES/latest_brief.md) | current narrative intent; code wins on executable facts |
| [`GUIDES/original_brief.pdf`](../../GUIDES/original_brief.pdf) | founding architecture proposal |
| [`GUIDES/CODEBASE_STRUCTURE.md`](../../GUIDES/CODEBASE_STRUCTURE.md) | maintained codebase orientation |
| [`GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`](../../GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md) | metric/failure history |
| [`GUIDES/READING_EXPERIMENTS.md`](../../GUIDES/READING_EXPERIMENTS.md) | fixed run-reading cycles |
| [`GUIDES/EXPERIMENT_LIFECYCLE.md`](../../GUIDES/EXPERIMENT_LIFECYCLE.md) | hypothesis→run→analysis lifecycle |
| [`GUIDES/MLOPS.md`](../../GUIDES/MLOPS.md) | operator conventions |

## KANBAN

`KANBAN/PHASE_1/` is the experiment journal. Each investigation/run can own:

```text
DESCRIPTION.md  — question and controlled diff
PLAN.md         — preregistered method/gates
GUIDE.md        — executable remote recipe
OBSERVATIONS.md — raw outcome record
ANALYSIS.md     — interpretation
NEXT_STEPS.md   — ordered decision
```

`KANBAN/PROTOCOL.md` defines how records transition. `README_for_reading_experiments.md` explains the
human reading route. Investigation directories are historical evidence: later dated reconciliation
can supersede their original “planned/running” text.

## Remote-operation docs

| File | Role |
|---|---|
| [`AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md`](../../AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md) | connection and remote safety |
| [`AGENT_FILES/GUIDE_AUTONOMOUS_REMOTE_RUN.md`](../../AGENT_FILES/GUIDE_AUTONOMOUS_REMOTE_RUN.md) | unattended KANBAN execution |
| [`AGENT_FILES/SETUPS/NEW_POD.md`](../../AGENT_FILES/SETUPS/NEW_POD.md) | canonical fresh-pod bootstrap |
| [`AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`](../../AGENT_FILES/SETUPS/VOLUME_LAYOUT.md) | persistent storage tree |
| [`.agents/skills/run-remote-experiment/SKILL.md`](../../.agents/skills/run-remote-experiment/SKILL.md) | mandatory remote workflow adapter |

## Recovery docs

`NETWORK_VOLUME_RECOVERY/` is a forensic/recovery corpus:

- `README.md`: recovery entry and safe handling;
- `VALIDATION_REPORT.md`: recovered-content validation;
- supporting maps/catalogs: historical path/artifact relationships.

It is not part of the training import graph, but it is part of the MLOps knowledge needed when the
durable-volume assumption fails.

## Dependency/tooling files

| File | Role |
|---|---|
| [`requirements.txt`](../../requirements.txt) | pinned runtime packages, notably Transformers 4.57.6 |
| [`pyproject.toml`](../../pyproject.toml) | Ruff/project tooling configuration |
| [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) | local lint/format guard |
| [`.gitignore`](../../.gitignore) | keeps caches/secrets/run artifacts out of Git |

## Non-authoritative or local artifacts

- `YOUR_FILES/` contains stale/ignored personal guides; do not use it as architectural authority.
- `tmp/` contains scratch analysis, generated exports, and investigation helpers; validate before
  reuse.
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.venv/` are generated/local state.
- `.env`, SSH/key material, and local settings are never documentation sources and must not be
  copied into reports.
- `error.log`, `ssh.txt`, dated task notes, and unusual local files are workspace residue unless a
  current guide explicitly binds them.

## Import/dataflow map

```text
config ───────────────┐
                     ├→ encoders ───────┐
data ─────────────────┤                 │
losses ───────────────┼→ models ────────┼→ train
diagnostics ──────────┤                 │
provenance ───────────┘                 │
                                       ├→ checkpoints/W&B
builders → dataset roots → data ────────┘
probes → encoders/models/provenance → JSON/PNG/artifacts
```

Actual imports are more nuanced, but this shows ownership and runtime direction.

## Where to edit for a change

| Desired change | Primary file | Required companion |
|---|---|---|
| temporal sampling/augmentation | `data.py` | transform version, tests, provenance expectations |
| add encoder | `encoders.py` | spec/fingerprint tests, model geometry tests, requirements |
| bottleneck/flow/decoder architecture | `models.py` | parameter/count docs, tests, checkpoint compatibility |
| objective | `losses.py` + `train.py` | diagnostics and gradient-routing tests |
| metric | `diagnostics.py` + `train.py` | reading guide and W&B interpretation |
| resume/artifact identity | `provenance.py` + `train.py` | pre-mutation and round-trip tests |
| shipped default | `config.py` | CLI/help/docs/KANBAN recipe audit |
| dataset manufacturing | builder script | manifest schema, provenance, tests, volume guide |
