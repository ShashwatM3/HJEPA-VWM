# Source ledger

## Snapshot identity

| Field | Value |
|---|---|
| Captured | 2026-07-25 15:53:52 IST |
| Branch | `phase1-v0.2-frozen-encoder` |
| Base commit | `771cbba077d9f846bdf7a7dd48e12dbf29d54b49` |
| Working tree | Dirty; current working-tree files were used |
| Python used for derivations | `/usr/local/bin/python`, 3.12.4 |
| Local Torch / NumPy | 2.10.0 / 1.26.4 |
| Repository-required Transformers | 4.57.6 |
| Local Transformers during validation | 5.5.3 |

## Core file hashes

| File | SHA-256 |
|---|---|
| `config.py` | `e198405befe0af3a62cdc53c7f9cd65f1e7cdb196e780db6680fe1e9cb6874bb` |
| `data.py` | `81a2995b3497a0f29a8eb883af7582ea11b88ed1c6d49e6dd9cc3ba21412efbf` |
| `encoders.py` | `c76b1dadc664de6e6243ff6bc103d21914547fdce106376e704f70a08ef6a432` |
| `models.py` | `7ab0c443a3b2b43429d301046d3e5640730dd04458446514d08087a0d30f67e5` |
| `losses.py` | `a1edad6ea1f18cca9426eb63a442965fa7598c74d066f59ecd5236ea752ef3c9` |
| `diagnostics.py` | `80896b029b37b1f768bfa32106689f36aeb786661354189be0927a527d48f733` |
| `provenance.py` | `655644ddbf5b06928fcb79938ed1b603f62906db87def12c84f5a0974b783dc1` |
| `train.py` | `a7e0067ac89f2b6ff1c746ad4735bb5a22abb174e6636b30d8c4910a8596a6f0` |
| `whiten_stats.py` | `30bc3f718c68a5471176530c415aa7e2e1f4597a44141eeeccde18eb64ba5552` |
| `rank_probe.py` | `c6c0f06cc8ce92cc5cb6c55bc26a4f257b89d106dc838a3eaa7054eea00bffeb` |
| `drift_probe.py` | `6d7ff129368374fe1deae85419652115324b99da8b3b8e6151f605a1bbda325b` |
| `requirements.txt` | `ef711817d7627ca2a694dea412fbddb4a31afcdd24f1a2a305f7bb605b64e2f6` |
| `AGENT_FILES/AGENTS.md` | `08e392e08495f6c7c129ddc9528af67cb07e39c1e42aa268072ed6781b12ea2d` |
| `NEW_POD.md` | `e05789601d330045070407fb2b442490ecaf52d1852ddc5ca5c3bf143f7b3193` |
| SSH guide | `a40a78989bdd80199802bd176627dba3eca4ed2c7960038edd432c0d0a931297` |
| Autonomous run guide | `c118513165fd35a3522a624c5f7aa4a6e80305c9d33ccaf11d26aa998d9d5372` |
| Phase 1 KANBAN index | `2de1bf9541f171362f00986f431270a65827864f561d5255d2fea467bc4a1bf4` |
| Original brief PDF | `3d92bdb7922b524def053661831bf6db64e7a330039239d50e3a847622f2b099` |

## Validation evidence

- `python -m py_compile` passed for the core training, data, encoder, provenance, probe, and dataset
  scripts.
- `models.smoke_test_models()` passed.
- `python -m pytest -q` produced **181 passed, 1 failed**. The only failure was the intentional
  environment pin assertion: local Transformers `5.5.3` versus required `4.57.6`.
- Invoking the standalone `pytest` executable used a different Python 3.11 environment without
  NumPy and failed during collection. The module invocation is the meaningful result.

## High-value code anchors

| Concern | Anchor |
|---|---|
| Config dataclasses | [`config.py`](../config.py) |
| Deterministic dataset window | [`data.py`](../data.py) |
| `EncoderSpec` and fingerprint | [`encoders.py`](../encoders.py) |
| `Bottleneck` | [`models.py`](../models.py) |
| `TargetBottleneck` | [`models.py`](../models.py) |
| `CoarseFlow` | [`models.py`](../models.py) |
| `Decoder` | [`models.py`](../models.py) |
| `FeatureMeanTracker` / `FeatureWhitener` | [`models.py`](../models.py) |
| `train_step` | [`train.py`](../train.py) |
| checkpoint save/load | [`train.py`](../train.py) |
| diagnostics | [`train.py`](../train.py), [`diagnostics.py`](../diagnostics.py) |
| provenance envelopes | [`provenance.py`](../provenance.py) |

## Known dirty-snapshot implications

The DINOv3 adapter and its tests were uncommitted relative to the base commit, but implemented in the
working tree. Therefore:

- this corpus correctly lists DINOv3 as **IMPLEMENTED**;
- cloning only the base commit may expose DINOv3 differently;
- the exact working-tree state must be preserved before reproducing this snapshot elsewhere.
