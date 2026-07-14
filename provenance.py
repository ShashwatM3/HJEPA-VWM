"""Deterministic identities and atomic artifact envelopes for encoder experiments.

This flat deep module centralizes the facts shared by training, whitening, rank,
and drift. Callers provide resolved objects and receive validated dictionaries;
artifact schema, hashing, and atomicity stay local to this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from config import Config
from data import TRANSFORM_VERSION
from encoders import EncoderSpec, FeatureLayout

WHITENING_SCHEMA = "hjepa-whitening-v2"
DATASET_SCHEMA = "hjepa-dataset-v2"
FEATURE_CACHE_SCHEMA = "hjepa-feature-cache-v2"
WHITENING_EIGENSOLVER = {
    "name": "torch.linalg.eigh",
    "covariance": "biased-mle",
    "accumulation_dtype": "float64",
}


def canonical_json(value: Any) -> bytes:
    """Serialize JSON-compatible metadata into stable UTF-8 bytes.

    Canonical ordering makes logically identical metadata hash to the same identity
    regardless of dictionary insertion order.

    Args:
        value: JSON-compatible value to serialize.
    Returns:
        Deterministically encoded UTF-8 JSON bytes.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading it fully into memory.

    Streaming keeps checkpoint and artifact verification bounded in memory.

    Args:
        path: Artifact path whose exact bytes define the checksum.
    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_fingerprint(value: Any) -> str:
    """Return the canonical SHA-256 identity of JSON metadata.

    Args:
        value: JSON-compatible metadata tree.
    Returns:
        Lowercase hexadecimal metadata fingerprint.
    """
    return hashlib.sha256(canonical_json(value)).hexdigest()


def encoder_spec_to_dict(spec: EncoderSpec) -> dict[str, Any]:
    """Serialize a resolved encoder spec with its independently checked fingerprint.

    The explicit fingerprint travels with artifacts so shape-compatible but semantically
    different encoders cannot be confused.

    Args:
        spec: Fully resolved immutable encoder contract.
    Returns:
        JSON-compatible encoder metadata including the feature fingerprint.
    """
    payload = asdict(spec)
    payload["feature_fingerprint"] = payload.pop("fingerprint")
    return payload


def encoder_spec_from_dict(payload: dict[str, Any]) -> EncoderSpec:
    """Rebuild an immutable spec and reject altered serialized identity.

    Recomputing the dataclass fingerprint prevents callers from trusting a stale or
    hand-edited identity field.

    Args:
        payload: Serialized fields produced by :func:`encoder_spec_to_dict`.
    Returns:
        Validated immutable encoder specification.
    """
    values = dict(payload)
    expected = values.pop("feature_fingerprint", values.pop("fingerprint", None))
    layout = values.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("Serialized EncoderSpec is missing its layout dictionary.")
    values["layout"] = FeatureLayout(**layout)
    for field_name in ("normalization_mean", "normalization_std"):
        if field_name in values:
            values[field_name] = tuple(values[field_name])
    spec = EncoderSpec(**values)
    if expected is not None and expected != spec.fingerprint:
        raise ValueError(
            "Serialized EncoderSpec feature fingerprint does not match its resolved fields."
        )
    return spec


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Write a Torch artifact by fsyncing a sibling temporary file then replacing.

    A failed writer leaves the prior destination intact and cleans its temporary file.

    Args:
        payload: Python/Tensor payload accepted by ``torch.save``.
        path: Final artifact destination.
    Returns:
        None.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as reader:
            os.fsync(reader.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(payload: Any, path: str | Path) -> None:
    """Write JSON atomically with stable ordering.

    Reports and provenance sidecars therefore cannot be mistaken for complete files
    after an interrupted write.

    Args:
        payload: JSON-compatible report or provenance value.
        path: Final JSON destination.
    Returns:
        None.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        temporary.replace(destination)
    finally:
        handle.close()
        temporary.unlink(missing_ok=True)


def _manifest_identity(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    """Return path/size/hash metadata for a retained dataset manifest.

    Args:
        path: Manifest file to bind into dataset provenance.
        required: Whether a missing file is an error.
    Returns:
        Manifest identity, or ``None`` when an optional file is absent.
    """
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required dataset provenance file is missing: {path}")
        return None
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def _video_frame_count(path: Path) -> int:
    """Read the authoritative decoded-frame count from one video container.

    Dataset identity must distinguish containers that retain the same path and byte
    size but expose a different frame timeline. Opening only the container header is
    slower than stat-only inventory, but it happens before paid work and makes every
    downstream stats/cache/checkpoint join scientifically strict.

    Args:
        path: Resolved video path.
    Returns:
        Positive decoded-frame count reported by Decord.
    """
    try:
        from decord import VideoReader, cpu
    except ModuleNotFoundError as exc:  # pragma: no cover - requirements install error.
        raise RuntimeError(
            "Decord is required to fingerprint dataset frame counts. "
            "Install requirements.txt before materializing run provenance."
        ) from exc
    try:
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        frames = len(reader)
    except Exception as exc:
        raise RuntimeError(f"Could not read frame metadata for dataset clip: {path}") from exc
    if frames <= 0:
        raise RuntimeError(f"Dataset clip contains no decodable frames: {path}")
    return int(frames)


def _split_inventory(root: Path, split: str) -> dict[str, Any]:
    """Hash stable path, resolved byte size, and frame count for every split clip.

    Args:
        root: Dataset root containing split directories.
        split: Split directory name to inventory.
    Returns:
        Clip/frame totals, frame-count range, and deterministic inventory fingerprint.
    """
    split_dir = root / split
    files = sorted([*split_dir.glob("*.webm"), *split_dir.glob("*.mp4")])
    if not files:
        raise FileNotFoundError(f"No .webm or .mp4 clips found in {split_dir}")
    entries = []
    for path in files:
        try:
            resolved = path.resolve(strict=True)
            size = resolved.stat().st_size
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Broken dataset clip/symlink: {path}") from exc
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "size": size,
                "frames": _video_frame_count(resolved),
            }
        )
    frame_counts = [entry["frames"] for entry in entries]
    return {
        "count": len(entries),
        "total_frames": sum(frame_counts),
        "min_frames": min(frame_counts),
        "max_frames": max(frame_counts),
        "fingerprint": _metadata_fingerprint(entries),
    }


def _ego_source_uids(root: Path, split: str) -> set[str]:
    """Extract source UIDs from EGO4D ``<uid>_<chunk>.mp4`` filenames.

    Args:
        root: Chunked EGO4D dataset root.
        split: Split whose source coverage is being checked.
    Returns:
        Unique source-video identifiers present in the split.
    """
    return {
        path.name[: path.name.rfind("_")]
        for path in (root / split).glob("*.mp4")
        if "_" in path.name
    }


def _validate_ego4d_complete(cfg: Config, root: Path, manifests: dict[str, Any]) -> None:
    """Enforce the Stage-4D source coverage/count/leakage contract for full EGO4D.

    Args:
        cfg: Runtime configuration containing retained EGO4D manifest paths.
        root: Full chunked EGO4D dataset root.
        manifests: Partially built dataset identity with split inventories.
    Returns:
        None. Any incomplete or leaking dataset raises before stats or training.
    """
    selection_path = Path(cfg.data.ego4d_selection_manifest)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected = {"train": set(), "validation": set()}
    for record in selection.get("videos", []):
        expected[record["split"]].add(record["video_uid"])
    actual = {split: _ego_source_uids(root, split) for split in expected}
    for split in expected:
        if actual[split] != expected[split]:
            missing = sorted(expected[split] - actual[split])[:10]
            extra = sorted(actual[split] - expected[split])[:10]
            raise RuntimeError(
                f"EGO4D completeness failed for {split}: missing={missing}, extra={extra}."
            )
    if actual["train"] & actual["validation"]:
        raise RuntimeError("EGO4D source UID leakage exists across train/validation.")
    chunk_manifest = json.loads((root / "chunk_manifest.json").read_text(encoding="utf-8"))
    for split in expected:
        if chunk_manifest["splits"][split]["chunks"] != manifests["splits"][split]["count"]:
            raise RuntimeError(f"EGO4D chunk manifest count mismatch for {split}.")
        if chunk_manifest["splits"][split]["source_uids"] != len(actual[split]):
            raise RuntimeError(f"EGO4D source UID count mismatch for {split}.")


def build_dataset_identity(cfg: Config, *, require_complete: bool = False) -> dict[str, Any]:
    """Resolve one deterministic dataset/preprocessing identity before model state exists.

    The identity binds clip inventories, retained manifests, raw geometry, and the shared
    transform version so artifacts cannot cross dataset states silently.

    Args:
        cfg: Runtime configuration selecting the dataset and input geometry.
        require_complete: Enforce the final full-EGO4D coverage contract when applicable.
    Returns:
        JSON-compatible dataset identity with its canonical fingerprint.
    """
    root = Path(cfg.data.dataset_root())
    splits = {split: _split_inventory(root, split) for split in ("train", "validation")}
    manifest_paths: list[tuple[str, Path]] = []
    if cfg.data.dataset == "ssv2":
        manifest_paths.append(("labels", root / "labels.json"))
    elif cfg.data.dataset == "ssv2_tiny":
        manifest_paths.append(("tiny", root / "manifest.json"))
    elif cfg.data.dataset in {"ego4d", "ego4d_tiny"}:
        parent = Path(cfg.data.ego4d_root)
        manifest_paths.extend(
            [
                ("selection", Path(cfg.data.ego4d_selection_manifest)),
                ("download_tier", Path(cfg.data.ego4d_download_manifest)),
                ("chunks", parent / "chunk_manifest.json"),
            ]
        )
        if cfg.data.dataset == "ego4d_tiny":
            manifest_paths.append(("tiny", root / "manifest.json"))
    manifests = {name: _manifest_identity(path) for name, path in manifest_paths}
    body = {
        "schema": DATASET_SCHEMA,
        "dataset": cfg.data.dataset,
        "root": str(root),
        "preprocessing_version": TRANSFORM_VERSION,
        "input_geometry": [
            cfg.encoder.input_frames,
            cfg.encoder.input_height,
            cfg.encoder.input_width,
        ],
        "splits": splits,
        "manifests": manifests,
    }
    if cfg.data.dataset == "ego4d_tiny":
        parent = Path(cfg.data.ego4d_root)
        body["parent_splits"] = {
            split: _split_inventory(parent, split) for split in ("train", "validation")
        }
    if require_complete and cfg.data.dataset == "ego4d":
        _validate_ego4d_complete(cfg, root, body)
        body["completeness"] = "stage-4d-verified"
    body["fingerprint"] = _metadata_fingerprint(body)
    return body


def _update_tensor_hash(digest: Any, name: str, tensor: Tensor) -> None:
    """Fold one tensor's name, dtype, shape, and raw bytes into an artifact hash.

    Args:
        digest: Mutable hashlib-compatible digest.
        name: Stable tensor identity within the containing payload.
        tensor: Tensor whose exact CPU bytes are identity-bearing.
    Returns:
        None.
    """
    value = tensor.detach().cpu().contiguous()
    digest.update(
        canonical_json({"name": name, "dtype": str(value.dtype), "shape": list(value.shape)})
    )
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _artifact_fingerprint(metadata: dict[str, Any], tensors: dict[str, Tensor]) -> str:
    """Hash metadata and tensor bytes independently of Torch serialization details.

    Args:
        metadata: Canonical JSON-compatible artifact metadata.
        tensors: Named tensor payload.
    Returns:
        Lowercase hexadecimal payload fingerprint.
    """
    digest = hashlib.sha256(canonical_json(metadata))
    for name in sorted(tensors):
        _update_tensor_hash(digest, name, tensors[name])
    return digest.hexdigest()


def trainable_state_hash(modules: Any) -> str:
    """Hash ordered trainable tensors without depending on encoder identity.

    Module position and state-dict key are part of the identity. Frozen buffers
    are deliberately excluded; this proves that paired encoder arms start the
    trainable B/F/D stack from the same bytes.

    Args:
        modules: Ordered iterable of trainable Phase-1 modules.
    Returns:
        Lowercase hexadecimal initialization fingerprint.
    """
    digest = hashlib.sha256(b"hjepa-trainable-state-v1")
    for module_index, module in enumerate(modules):
        trainable_names = {
            name for name, parameter in module.named_parameters() if parameter.requires_grad
        }
        state = module.state_dict()
        for name in sorted(trainable_names):
            if name not in state:
                raise RuntimeError(f"Trainable parameter {name!r} is absent from state_dict().")
            _update_tensor_hash(digest, f"{module_index}:{name}", state[name])
    return digest.hexdigest()


def state_dict_hash(state: dict[str, Tensor]) -> str:
    """Hash every tensor in one module state dict for checkpoint provenance.

    Args:
        state: Named tensor state dictionary.
    Returns:
        Lowercase hexadecimal state fingerprint.
    """
    digest = hashlib.sha256(b"hjepa-module-state-v1")
    for name in sorted(state):
        _update_tensor_hash(digest, name, state[name])
    return digest.hexdigest()


def initial_data_order_identity(cfg: Config) -> dict[str, Any]:
    """Hash the exact epoch-0 train order and fixed validation batch identities.

    This makes paired-arm provenance sensitive to sampling order without embedding the
    full training permutation in every display surface.

    Args:
        cfg: Runtime configuration supplying dataset root, seed, and batch size.
    Returns:
        Train-order and validation-batch identities plus small audit prefixes.
    """
    root = Path(cfg.data.dataset_root())

    def sample_ids(split: str) -> list[str]:
        """Return deterministically sorted relative clip identities for one split.

        Args:
            split: Dataset split directory name.
        Returns:
            Stable relative clip paths in lexical order.
        """
        paths = sorted([*(root / split).glob("*.webm"), *(root / split).glob("*.mp4")])
        if not paths:
            raise FileNotFoundError(f"No clips found for order identity in {root / split}.")
        return [str(path.relative_to(root)) for path in paths]

    train_ids = sample_ids("train")
    generator = torch.Generator().manual_seed(cfg.seed)
    permutation = torch.randperm(len(train_ids), generator=generator).tolist()
    ordered = [train_ids[index] for index in permutation]
    validation = sample_ids("validation")[: min(16, cfg.train.global_batch)]
    return {
        "train_order_fingerprint": _metadata_fingerprint(ordered),
        "train_prefix": ordered[: min(32, len(ordered))],
        "validation_batch_fingerprint": _metadata_fingerprint(validation),
        "validation_batch": validation,
    }


def runtime_identity() -> dict[str, Any]:
    """Capture dependency/code identity without reading credentials or environment secrets.

    Runtime versions and the Git state make provenance auditable while deliberately
    excluding environment variables and authentication material.

    Returns:
        JSON-compatible dependency, platform, and Git identity.
    """
    try:
        transformers_version = __import__("transformers").__version__
    except ModuleNotFoundError:
        transformers_version = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    cuda_version = getattr(torch.version, "cuda", None)
    cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
    gpu_models: list[str] = []
    if torch.cuda.is_available():
        try:
            gpu_models = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
        except (AssertionError, RuntimeError):
            gpu_models = ["unavailable"]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda": cuda_version,
        "cudnn": cudnn_version,
        "gpu_models": gpu_models,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _seed_stream_identity(seed: int) -> dict[str, Any]:
    """Describe every deterministic seed stream used by the run.

    Args:
        seed: User-selected base seed.
    Returns:
        Explicit data, initialization, training-step, and diagnostic seed contract.
    """
    return {
        "base": seed,
        "data_transform": seed,
        "train_order_epoch0": seed,
        "validation_order_epoch0": seed + 10_000,
        "model_init": seed + 1_000,
        "training_step_formula": "base*1000003+step",
        "diagnostic": seed * 1_000_003 + 900_001,
    }


def build_run_provenance(
    cfg: Config,
    *,
    encoder_spec: EncoderSpec,
    dataset_identity: dict[str, Any],
    trainable_init: str,
    whitening_payload_fingerprint: str | None,
    tracking_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize paired-arm identity with one explicit common comparison hash.

    Encoder-specific facts stay outside the common hash, while all comparison controls
    remain identical across arms. Tracking metadata is whitelisted to exclude secrets.

    Args:
        cfg: Fully finalized runtime configuration.
        encoder_spec: Resolved immutable encoder identity.
        dataset_identity: Validated dataset and preprocessing identity.
        trainable_init: Fingerprint of the freshly initialized B/F/D stack.
        whitening_payload_fingerprint: Bound whitening artifact identity, if enabled.
        tracking_identity: Optional credential-free W&B entity/project/group/name/id fields.
    Returns:
        Versioned run provenance with common, encoder, data, and tracking identities.
    """
    order = initial_data_order_identity(cfg)
    config = json.loads(json.dumps(cfg, default=lambda value: value.__dict__))
    common_config = json.loads(json.dumps(config))
    common_config.pop("encoder", None)
    common_config.pop("hf_cache_dir", None)
    common_config.pop("checkpoint_dir", None)
    common_config.get("train", {}).pop("whiten_stats_path", None)
    encoder_runtime_contract = {
        "input_frames": encoder_spec.input_frames,
        "input_height": encoder_spec.input_height,
        "input_width": encoder_spec.input_width,
        "inference_precision": encoder_spec.inference_precision,
        "frame_microbatch": encoder_spec.frame_microbatch,
        "attention_implementation": encoder_spec.attention_implementation,
    }
    configured_runtime_contract = {
        "input_frames": cfg.encoder.input_frames,
        "input_height": cfg.encoder.input_height,
        "input_width": cfg.encoder.input_width,
        "inference_precision": cfg.encoder.precision,
        "frame_microbatch": cfg.encoder.frame_microbatch,
        "attention_implementation": cfg.encoder.attention_implementation,
    }
    if configured_runtime_contract != encoder_runtime_contract:
        raise ValueError(
            "Resolved EncoderSpec runtime fields disagree with EncoderConfig: "
            f"{encoder_runtime_contract} != {configured_runtime_contract}."
        )
    allowed_tracking_fields = {"entity", "project", "group", "name", "id"}
    provided_tracking = dict(tracking_identity or {})
    unknown_tracking_fields = set(provided_tracking) - allowed_tracking_fields
    if unknown_tracking_fields:
        raise ValueError(
            "Run provenance refuses unknown tracking fields (credentials must never be "
            f"serialized): {sorted(unknown_tracking_fields)}."
        )
    resolved_tracking = {
        field: provided_tracking.get(field) for field in sorted(allowed_tracking_fields)
    }
    common = {
        "dataset_fingerprint": dataset_identity["fingerprint"],
        "trainable_init_hash": trainable_init,
        "data_order": order,
        "config": common_config,
        "encoder_runtime_contract": encoder_runtime_contract,
        "runtime": runtime_identity(),
        "seed_streams": _seed_stream_identity(cfg.seed),
    }
    return {
        "schema": "hjepa-run-provenance-v1",
        "common_identity": _metadata_fingerprint(common),
        "common": common,
        "encoder_spec": encoder_spec_to_dict(encoder_spec),
        "feature_fingerprint": encoder_spec.fingerprint,
        "whitening_payload_fingerprint": whitening_payload_fingerprint,
        "dataset_identity": dataset_identity,
        "resolved_config": config,
        "tracking_identity": resolved_tracking,
    }


def compare_run_provenance(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Reject paired configurations unless every encoder-independent field matches.

    This is the final join guard before treating two encoder arms as a controlled pair.

    Args:
        left: First run-provenance envelope.
        right: Second run-provenance envelope.
    Returns:
        None. A mismatch raises instead of producing a misleading comparison.
    """
    for label, payload in (("left", left), ("right", right)):
        if payload.get("schema") != "hjepa-run-provenance-v1":
            raise ValueError(f"{label} provenance has an unsupported schema.")
    if left.get("common_identity") != right.get("common_identity"):
        raise ValueError("Encoder-arm provenance differs in encoder-independent fields.")


def compare_checkpoint_provenance(
    saved: dict[str, Any],
    expected: dict[str, Any],
    *,
    allow_dataset_transfer: bool = False,
) -> None:
    """Validate resume provenance with an explicit, narrow dataset-transfer policy.

    Normal resumes require the exact common identity. Dataset transfer removes only the
    dataset fingerprint, initial data order, and dataset configuration before comparing;
    seeds, runtime, encoder execution, initialization, losses, and schedules stay guarded.

    Args:
        saved: Run provenance embedded in the checkpoint.
        expected: Provenance resolved for the requested continuation.
        allow_dataset_transfer: Permit only dataset-specific common fields to differ.
    Returns:
        None. Any unapproved resume difference raises before state mutation.
    """
    for label, payload in (("saved", saved), ("expected", expected)):
        if not isinstance(payload, dict) or payload.get("schema") != "hjepa-run-provenance-v1":
            raise ValueError(f"Checkpoint {label} run provenance has an unsupported schema.")
    if not allow_dataset_transfer:
        if saved.get("common_identity") != expected.get("common_identity"):
            raise ValueError("Checkpoint run provenance differs in encoder-independent fields.")
        return

    def transfer_invariant(payload: dict[str, Any]) -> dict[str, Any]:
        """Remove exactly the dataset-owned fields from a common identity.

        Args:
            payload: Versioned run provenance.
        Returns:
            Deep-copied common fields that must remain equal during transfer.
        """
        common = json.loads(json.dumps(payload.get("common", {})))
        common.pop("dataset_fingerprint", None)
        common.pop("data_order", None)
        config = common.get("config")
        if isinstance(config, dict):
            config.pop("data", None)
        return common

    if transfer_invariant(saved) != transfer_invariant(expected):
        raise ValueError("Checkpoint non-dataset provenance differs during dataset transfer.")


def build_whitening_envelope(
    *,
    mean: Tensor,
    eigenvalues: Tensor,
    eigenvectors: Tensor,
    encoder_spec: EncoderSpec,
    dataset_identity: dict[str, Any],
    split: str,
    transform_seed: int,
    clip_count: int,
    token_row_count: int,
    eigensolver: dict[str, Any],
) -> dict[str, Any]:
    """Create a versioned whitening artifact bound to its exact feature space.

    Tensor bytes and all sampling/eigensolver identities are fingerprinted together so
    same-shaped features from another encoder or dataset cannot be reused.

    Args:
        mean: (D_e,) empirical feature mean.
        eigenvalues: (D_e,) covariance eigenvalues in ascending order.
        eigenvectors: (D_e,D_e) covariance eigenvectors by column.
        encoder_spec: Resolved encoder that produced the feature rows.
        dataset_identity: Dataset/split inventory identity used for sampling.
        split: Dataset split; whitening is restricted to ``train``.
        transform_seed: Deterministic data-transform seed.
        clip_count: Number of encoded clips.
        token_row_count: Number of flattened token rows accumulated.
        eigensolver: Exact covariance and eigensolver settings.
    Returns:
        Versioned whitening envelope with fp32 tensors and payload fingerprint.
    """
    if split != "train":
        raise ValueError("Whitening statistics must be fitted on the training split.")
    if clip_count <= 0 or token_row_count <= 0:
        raise ValueError("Whitening clip and token-row counts must be positive.")
    tensors = {
        "mean": mean.detach().cpu().float(),
        "eigenvalues": eigenvalues.detach().cpu().float(),
        "eigenvectors": eigenvectors.detach().cpu().float(),
    }
    d = encoder_spec.feature_dim
    if tensors["mean"].shape != (d,) or tensors["eigenvalues"].shape != (d,):
        raise ValueError("Whitening vectors do not match the encoder feature dimension.")
    if tensors["eigenvectors"].shape != (d, d):
        raise ValueError("Whitening eigenvectors do not match the encoder feature dimension.")
    metadata = {
        "encoder_spec": encoder_spec_to_dict(encoder_spec),
        "feature_fingerprint": encoder_spec.fingerprint,
        "dataset_identity": dataset_identity,
        "dataset_fingerprint": dataset_identity.get("fingerprint"),
        "split": split,
        "preprocessing_version": TRANSFORM_VERSION,
        "input_geometry": [
            encoder_spec.input_frames,
            encoder_spec.input_height,
            encoder_spec.input_width,
        ],
        "precision": encoder_spec.inference_precision,
        "frame_microbatch": encoder_spec.frame_microbatch,
        "transform_seed": transform_seed,
        "clip_count": clip_count,
        "token_row_count": token_row_count,
        "eigensolver": eigensolver,
        "creation_version": 2,
    }
    envelope = {"schema": WHITENING_SCHEMA, "metadata": metadata, "tensors": tensors}
    envelope["payload_fingerprint"] = _artifact_fingerprint(metadata, tensors)
    return envelope


def load_whitening_envelope(
    path: str | Path,
    *,
    expected_encoder_spec: EncoderSpec | None = None,
    expected_dataset_identity: dict[str, Any] | None = None,
    expected_transform_seed: int | None = None,
    expected_clip_count: int | None = None,
    expected_eigensolver: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and fully validate whitening metadata before callers construct state.

    Validation happens before model mutation so an incompatible or corrupted artifact
    cannot partially configure a training run.

    Args:
        path: Whitening artifact path.
        expected_encoder_spec: Optional exact encoder identity required by the caller.
        expected_dataset_identity: Optional exact dataset identity required by the caller.
        expected_transform_seed: Optional exact sampling/transform seed.
        expected_clip_count: Optional exact clip budget required by the recipe.
        expected_eigensolver: Optional exact covariance/eigensolver settings.
    Returns:
        Validated whitening envelope containing metadata and fp32 tensors.
    """
    envelope = torch.load(path, map_location="cpu")
    if not isinstance(envelope, dict) or envelope.get("schema") != WHITENING_SCHEMA:
        raise ValueError(f"Whitening artifact {path} is not schema {WHITENING_SCHEMA}.")
    metadata = envelope.get("metadata")
    tensors = envelope.get("tensors")
    if not isinstance(metadata, dict) or not isinstance(tensors, dict):
        raise ValueError(f"Whitening artifact {path} has a malformed envelope.")
    actual_payload = _artifact_fingerprint(metadata, tensors)
    if actual_payload != envelope.get("payload_fingerprint"):
        raise ValueError(f"Whitening artifact {path} payload fingerprint is invalid.")
    serialized_spec = encoder_spec_from_dict(metadata.get("encoder_spec", {}))
    if metadata.get("feature_fingerprint") != serialized_spec.fingerprint:
        raise ValueError("Whitening feature fingerprint disagrees with its EncoderSpec.")
    required_tensor_names = {"mean", "eigenvalues", "eigenvectors"}
    if set(tensors) != required_tensor_names or not all(
        isinstance(tensors[name], Tensor) for name in required_tensor_names
    ):
        raise ValueError("Whitening artifact must contain exactly three tensor payloads.")
    d = serialized_spec.feature_dim
    if (
        tensors["mean"].shape != (d,)
        or tensors["eigenvalues"].shape != (d,)
        or tensors["eigenvectors"].shape != (d, d)
    ):
        raise ValueError("Whitening tensor shapes do not match the resolved EncoderSpec.")
    for name in required_tensor_names:
        value = tensors[name]
        if not value.is_floating_point() or value.dtype != torch.float32:
            raise ValueError(f"Whitening tensor {name!r} must be fp32.")
        if not torch.isfinite(value).all():
            raise ValueError(f"Whitening tensor {name!r} contains non-finite values.")
    dataset_identity = metadata.get("dataset_identity")
    if not isinstance(dataset_identity, dict) or metadata.get(
        "dataset_fingerprint"
    ) != dataset_identity.get("fingerprint"):
        raise ValueError("Whitening dataset identity and fingerprint disagree.")
    if metadata.get("split") != "train":
        raise ValueError("Whitening artifact must be fitted on the training split.")
    if metadata.get("preprocessing_version") != TRANSFORM_VERSION:
        raise ValueError("Whitening preprocessing version is not supported by this code.")
    expected_geometry = [
        serialized_spec.input_frames,
        serialized_spec.input_height,
        serialized_spec.input_width,
    ]
    if metadata.get("input_geometry") != expected_geometry:
        raise ValueError("Whitening input geometry disagrees with its EncoderSpec.")
    if metadata.get("precision") != serialized_spec.inference_precision:
        raise ValueError("Whitening precision disagrees with its EncoderSpec.")
    if metadata.get("frame_microbatch") != serialized_spec.frame_microbatch:
        raise ValueError("Whitening frame microbatch disagrees with its EncoderSpec.")
    clip_count = metadata.get("clip_count")
    row_count = metadata.get("token_row_count")
    if (
        not isinstance(clip_count, int)
        or clip_count <= 0
        or not isinstance(row_count, int)
        or row_count != clip_count * serialized_spec.layout.n_tokens
    ):
        raise ValueError("Whitening clip/token-row counts are inconsistent.")
    if not isinstance(metadata.get("transform_seed"), int):
        raise ValueError("Whitening transform seed is missing or invalid.")
    eigensolver = metadata.get("eigensolver")
    if not isinstance(eigensolver, dict) or not eigensolver.get("name"):
        raise ValueError("Whitening eigensolver settings are missing or invalid.")
    if metadata.get("creation_version") != 2:
        raise ValueError("Whitening creation version is unsupported.")
    if (
        expected_encoder_spec is not None
        and serialized_spec.fingerprint != expected_encoder_spec.fingerprint
    ):
        raise ValueError(
            "Whitening feature fingerprint does not match the selected encoder: "
            f"{serialized_spec.fingerprint} != {expected_encoder_spec.fingerprint}."
        )
    if expected_dataset_identity is not None:
        expected = expected_dataset_identity.get("fingerprint")
        if metadata.get("dataset_fingerprint") != expected:
            raise ValueError("Whitening dataset fingerprint does not match the selected dataset.")
    if (
        expected_transform_seed is not None
        and metadata.get("transform_seed") != expected_transform_seed
    ):
        raise ValueError("Whitening transform seed does not match the selected run seed.")
    if expected_clip_count is not None and metadata.get("clip_count") != expected_clip_count:
        raise ValueError("Whitening clip count does not match the selected sampling budget.")
    if expected_eigensolver is not None and metadata.get("eigensolver") != expected_eigensolver:
        raise ValueError("Whitening eigensolver settings do not match the selected recipe.")
    return envelope


def build_feature_cache_envelope(
    *,
    features: dict[str, Tensor],
    encoder_spec: EncoderSpec,
    dataset_identity: dict[str, Any],
    probe_manifest: dict[str, Any],
    offsets: list[int],
    storage_dtype: str,
) -> dict[str, Any]:
    """Bind cached detailed features to every fact that can change their meaning.

    Probe caches are identity-bearing artifacts, not shape-only acceleration files.

    Args:
        features: Mapping from probe-window identity to (N_e,D_e) detailed features.
        encoder_spec: Resolved encoder that produced the features.
        dataset_identity: Dataset inventory and manifest identity.
        probe_manifest: Exact videos and window endpoints represented by the cache.
        offsets: Temporal offsets evaluated by the probe.
        storage_dtype: Explicit ``fp16`` or ``fp32`` cache storage policy.
    Returns:
        Versioned feature-cache envelope with a payload fingerprint.
    """
    if storage_dtype not in {"fp16", "fp32"}:
        raise ValueError("Feature-cache storage_dtype must be fp16 or fp32.")
    expected_dtype = torch.float16 if storage_dtype == "fp16" else torch.float32
    tensors: dict[str, Tensor] = {}
    for key, value in features.items():
        if value.ndim != 2 or value.shape != (
            encoder_spec.layout.n_tokens,
            encoder_spec.feature_dim,
        ):
            raise ValueError(f"Feature cache entry {key!r} has an incompatible shape.")
        if not torch.isfinite(value).all():
            raise ValueError(f"Feature cache entry {key!r} contains non-finite values.")
        tensors[key] = value.detach().cpu().to(expected_dtype).contiguous()
    metadata = {
        "encoder_spec": encoder_spec_to_dict(encoder_spec),
        "feature_fingerprint": encoder_spec.fingerprint,
        "dataset_fingerprint": dataset_identity.get("fingerprint"),
        "dataset_identity": dataset_identity,
        "probe_manifest_fingerprint": _metadata_fingerprint(probe_manifest),
        "offsets": sorted(set(offsets)),
        "storage_dtype": storage_dtype,
        "preprocessing_version": encoder_spec.preprocess_version,
        "precision": encoder_spec.inference_precision,
        "token_selection": encoder_spec.preprocess_version,
    }
    envelope = {"schema": FEATURE_CACHE_SCHEMA, "metadata": metadata, "features": tensors}
    envelope["payload_fingerprint"] = _artifact_fingerprint(metadata, tensors)
    return envelope


def load_feature_cache_envelope(
    path: str | Path,
    *,
    expected_encoder_spec: EncoderSpec,
    expected_dataset_identity: dict[str, Any],
    expected_probe_manifest: dict[str, Any],
    expected_offsets: list[int],
    expected_storage_dtype: str,
) -> dict[str, Any]:
    """Validate a feature cache fully, including explicit paths, before reuse.

    Every semantic input and every tensor is checked even when the user supplies the
    cache path explicitly.

    Args:
        path: Feature-cache artifact path.
        expected_encoder_spec: Exact encoder identity required by this probe.
        expected_dataset_identity: Exact dataset identity required by this probe.
        expected_probe_manifest: Exact selected videos and endpoints.
        expected_offsets: Temporal offsets requested by this probe.
        expected_storage_dtype: Explicit ``fp16`` or ``fp32`` storage policy.
    Returns:
        Validated feature-cache envelope.
    """
    envelope = torch.load(path, map_location="cpu")
    if not isinstance(envelope, dict) or envelope.get("schema") != FEATURE_CACHE_SCHEMA:
        raise ValueError(f"Feature cache {path} is not schema {FEATURE_CACHE_SCHEMA}.")
    metadata, features = envelope.get("metadata"), envelope.get("features")
    if not isinstance(metadata, dict) or not isinstance(features, dict):
        raise ValueError(f"Feature cache {path} has a malformed envelope.")
    if _artifact_fingerprint(metadata, features) != envelope.get("payload_fingerprint"):
        raise ValueError(f"Feature cache {path} payload fingerprint is invalid.")
    saved_spec = encoder_spec_from_dict(metadata.get("encoder_spec", {}))
    if saved_spec.fingerprint != expected_encoder_spec.fingerprint:
        raise ValueError("Feature cache feature fingerprint does not match the selected encoder.")
    if metadata.get("dataset_fingerprint") != expected_dataset_identity.get("fingerprint"):
        raise ValueError("Feature cache dataset fingerprint does not match this dataset.")
    if metadata.get("probe_manifest_fingerprint") != _metadata_fingerprint(expected_probe_manifest):
        raise ValueError("Feature cache probe manifest fingerprint does not match.")
    if metadata.get("offsets") != sorted(set(expected_offsets)):
        raise ValueError("Feature cache offset identity does not match the requested offsets.")
    if metadata.get("storage_dtype") != expected_storage_dtype:
        raise ValueError("Feature cache storage dtype does not match the requested dtype.")
    expected_dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(expected_storage_dtype)
    if expected_dtype is None:
        raise ValueError("Feature-cache expected_storage_dtype must be fp16 or fp32.")
    expected_shape = (
        expected_encoder_spec.layout.n_tokens,
        expected_encoder_spec.feature_dim,
    )
    for key, value in features.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError("Feature cache entries must map string identities to tensors.")
        if value.ndim != 2 or tuple(value.shape) != expected_shape:
            raise ValueError(f"Feature cache entry {key!r} has an incompatible shape.")
        if value.dtype != expected_dtype:
            raise ValueError(
                f"Feature cache entry {key!r} tensor dtype does not match "
                f"{expected_storage_dtype}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Feature cache entry {key!r} contains non-finite values.")
    return envelope
