"""Artifact and dataset identity contracts shared by training and offline tools."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")


def _spec(repo: str = "offline/frame-a"):
    from encoders import EncoderSpec, FeatureLayout

    return EncoderSpec(
        family="fake-frame",
        repo_id=repo,
        requested_revision="a" * 40,
        resolved_revision="a" * 40,
        input_frames=8,
        input_height=256,
        input_width=256,
        feature_dim=3,
        layout=FeatureLayout(8, 16, 16, "time_y_x", "frame", 1, 1),
        normalization_id="test",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        preprocess_version="test-v1",
        inference_precision="fp32",
        frame_microbatch=4,
        attention_implementation="sdpa",
        cache_dir="/tmp/cache",
        parameter_count=0,
    )


def test_encoder_spec_round_trip_recomputes_and_validates_fingerprint():
    import provenance

    spec = _spec()
    encoded = provenance.encoder_spec_to_dict(spec)
    assert provenance.encoder_spec_from_dict(encoded) == spec

    encoded["feature_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="feature fingerprint"):
        provenance.encoder_spec_from_dict(encoded)


def test_dataset_identity_binds_manifest_sorted_inventory_and_frame_counts(tmp_path, monkeypatch):
    import provenance
    from config import Config

    root = tmp_path / "data" / "ssv2_tiny"
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True)
        (root / split / f"{split}.webm").write_bytes(split.encode())
    (root / "manifest.json").write_text(json.dumps({"seed": 42, "clips": ["a"]}))
    cfg = Config()
    cfg.data.data_root = str(tmp_path / "data")
    cfg.data.dataset = "ssv2_tiny"
    frame_counts = {"train.webm": 17, "validation.webm": 19}
    monkeypatch.setattr(
        provenance,
        "_video_frame_count",
        lambda path: frame_counts[path.name],
    )

    first = provenance.build_dataset_identity(cfg)
    second = provenance.build_dataset_identity(cfg)
    assert first == second
    assert first["schema"] == "hjepa-dataset-v2"
    assert first["dataset"] == "ssv2_tiny"
    assert first["splits"]["train"]["count"] == 1
    assert first["splits"]["train"]["total_frames"] == 17

    # Container metadata is identity-bearing even if path and byte size do not change.
    frame_counts["train.webm"] = 18
    frame_changed = provenance.build_dataset_identity(cfg)
    assert frame_changed["fingerprint"] != first["fingerprint"]

    (root / "train" / "train.webm").write_bytes(b"changed-size")
    assert provenance.build_dataset_identity(cfg)["fingerprint"] != frame_changed["fingerprint"]


def test_parallel_split_inventory_preserves_canonical_sorted_fingerprint(tmp_path, monkeypatch):
    import provenance

    root = tmp_path / "dataset"
    split = root / "train"
    split.mkdir(parents=True)
    for name, payload in (
        ("clip-c.mp4", b"ccc"),
        ("clip-a.mp4", b"a"),
        ("clip-b.mp4", b"bb"),
    ):
        (split / name).write_bytes(payload)
    frame_counts = {"clip-a.mp4": 11, "clip-b.mp4": 12, "clip-c.mp4": 13}
    monkeypatch.setattr(
        provenance,
        "_video_frame_count",
        lambda path: frame_counts[path.name],
    )

    inventory = provenance._split_inventory(root, "train")
    canonical_entries = [
        {"path": "train/clip-a.mp4", "size": 1, "frames": 11},
        {"path": "train/clip-b.mp4", "size": 2, "frames": 12},
        {"path": "train/clip-c.mp4", "size": 3, "frames": 13},
    ]
    assert inventory["count"] == 3
    assert inventory["total_frames"] == 36
    assert inventory["fingerprint"] == provenance._metadata_fingerprint(canonical_entries)


def test_whitening_envelope_is_atomic_and_rejects_shape_matched_wrong_encoder(tmp_path):
    import provenance

    dataset = {"dataset": "fake", "split": "train", "fingerprint": "d" * 64}
    spec_a = _spec("offline/frame-a")
    spec_b = _spec("offline/frame-b")
    path = tmp_path / "stats.pt"
    envelope = provenance.build_whitening_envelope(
        mean=torch.zeros(3),
        eigenvalues=torch.ones(3),
        eigenvectors=torch.eye(3),
        encoder_spec=spec_a,
        dataset_identity=dataset,
        split="train",
        transform_seed=42,
        clip_count=2,
        token_row_count=4096,
        eigensolver={"name": "torch.linalg.eigh", "accumulation_dtype": "float64"},
    )
    provenance.atomic_torch_save(envelope, path)

    loaded = provenance.load_whitening_envelope(
        path,
        expected_encoder_spec=spec_a,
        expected_dataset_identity=dataset,
        expected_transform_seed=42,
        expected_clip_count=2,
        expected_eigensolver={
            "name": "torch.linalg.eigh",
            "accumulation_dtype": "float64",
        },
    )
    assert loaded["metadata"]["clip_count"] == 2
    assert loaded["metadata"]["token_row_count"] == 4096
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(ValueError, match="feature fingerprint"):
        provenance.load_whitening_envelope(
            path, expected_encoder_spec=spec_b, expected_dataset_identity=dataset
        )
    with pytest.raises(ValueError, match="transform seed"):
        provenance.load_whitening_envelope(path, expected_transform_seed=43)
    with pytest.raises(ValueError, match="clip count"):
        provenance.load_whitening_envelope(path, expected_clip_count=12_800)
    with pytest.raises(ValueError, match="eigensolver settings"):
        provenance.load_whitening_envelope(
            path,
            expected_eigensolver={"name": "different"},
        )


def test_whitening_loader_rejects_self_consistent_invalid_metadata_and_tensors(tmp_path):
    import provenance

    dataset = {"dataset": "fake", "split": "train", "fingerprint": "d" * 64}
    spec = _spec()
    envelope = provenance.build_whitening_envelope(
        mean=torch.zeros(3),
        eigenvalues=torch.ones(3),
        eigenvectors=torch.eye(3),
        encoder_spec=spec,
        dataset_identity=dataset,
        split="train",
        transform_seed=42,
        clip_count=2,
        token_row_count=2 * spec.layout.n_tokens,
        eigensolver={"name": "torch.linalg.eigh", "accumulation_dtype": "float64"},
    )

    bad_metadata = dict(envelope["metadata"])
    bad_metadata["preprocessing_version"] = "wrong-transform"
    envelope["metadata"] = bad_metadata
    envelope["payload_fingerprint"] = provenance._artifact_fingerprint(
        bad_metadata, envelope["tensors"]
    )
    metadata_path = tmp_path / "bad-metadata.pt"
    provenance.atomic_torch_save(envelope, metadata_path)
    with pytest.raises(ValueError, match="preprocessing version"):
        provenance.load_whitening_envelope(metadata_path)

    valid = provenance.build_whitening_envelope(
        mean=torch.zeros(3),
        eigenvalues=torch.ones(3),
        eigenvectors=torch.eye(3),
        encoder_spec=spec,
        dataset_identity=dataset,
        split="train",
        transform_seed=42,
        clip_count=2,
        token_row_count=2 * spec.layout.n_tokens,
        eigensolver={"name": "torch.linalg.eigh", "accumulation_dtype": "float64"},
    )
    valid["tensors"]["mean"] = torch.zeros(4)
    valid["payload_fingerprint"] = provenance._artifact_fingerprint(
        valid["metadata"], valid["tensors"]
    )
    tensor_path = tmp_path / "bad-tensor.pt"
    provenance.atomic_torch_save(valid, tensor_path)
    with pytest.raises(ValueError, match="tensor shapes"):
        provenance.load_whitening_envelope(tensor_path)


def test_feature_cache_envelope_binds_encoder_dataset_manifest_offsets_and_dtype(tmp_path):
    import provenance

    spec = _spec()
    dataset = {"dataset": "fake", "fingerprint": "d" * 64}
    manifest = {"dataset": "fake", "videos": [{"path": "a", "anchor_end": 7}]}
    features = {"a::end7": torch.ones(spec.layout.n_tokens, spec.feature_dim)}
    envelope = provenance.build_feature_cache_envelope(
        features=features,
        encoder_spec=spec,
        dataset_identity=dataset,
        probe_manifest=manifest,
        offsets=[2, 4],
        storage_dtype="fp32",
    )
    path = tmp_path / "features.pt"
    provenance.atomic_torch_save(envelope, path)
    loaded = provenance.load_feature_cache_envelope(
        path,
        expected_encoder_spec=spec,
        expected_dataset_identity=dataset,
        expected_probe_manifest=manifest,
        expected_offsets=[2, 4],
        expected_storage_dtype="fp32",
    )
    assert torch.equal(loaded["features"]["a::end7"], features["a::end7"])
    with pytest.raises(ValueError, match="offset"):
        provenance.load_feature_cache_envelope(
            path,
            expected_encoder_spec=spec,
            expected_dataset_identity=dataset,
            expected_probe_manifest=manifest,
            expected_offsets=[8],
            expected_storage_dtype="fp32",
        )


@pytest.mark.parametrize(
    ("bad_features", "message"),
    [
        ({"a::end7": torch.ones(2047, 3)}, "incompatible shape"),
        ({"a::end7": torch.ones(2048, 3, dtype=torch.float16)}, "tensor dtype"),
        (
            {"a::end7": torch.full((2048, 3), float("nan"))},
            "non-finite values",
        ),
    ],
)
def test_feature_cache_loader_rejects_self_consistent_invalid_tensors(
    tmp_path, bad_features, message
):
    import provenance

    spec = _spec()
    dataset = {"dataset": "fake", "fingerprint": "d" * 64}
    manifest = {"dataset": "fake", "videos": [{"path": "a", "anchor_end": 7}]}
    envelope = provenance.build_feature_cache_envelope(
        features={"a::end7": torch.ones(spec.layout.n_tokens, spec.feature_dim)},
        encoder_spec=spec,
        dataset_identity=dataset,
        probe_manifest=manifest,
        offsets=[2, 4],
        storage_dtype="fp32",
    )
    envelope["features"] = bad_features
    envelope["payload_fingerprint"] = provenance._artifact_fingerprint(
        envelope["metadata"], bad_features
    )
    path = tmp_path / "malformed.pt"
    provenance.atomic_torch_save(envelope, path)

    with pytest.raises(ValueError, match=message):
        provenance.load_feature_cache_envelope(
            path,
            expected_encoder_spec=spec,
            expected_dataset_identity=dataset,
            expected_probe_manifest=manifest,
            expected_offsets=[2, 4],
            expected_storage_dtype="fp32",
        )


def test_paired_provenance_allows_backend_identity_but_binds_runtime_feature_contract(
    tmp_path,
):
    import provenance
    from config import Config

    root = tmp_path / "data" / "ssv2_tiny"
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True)
        (root / split / f"{split}.webm").write_bytes(split.encode())

    left_cfg = Config()
    left_cfg.data.data_root = str(tmp_path / "data")
    left_cfg.encoder.alias = "siglip2_vitb16"
    left_cfg.encoder.precision = "fp32"
    left_cfg.encoder.frame_microbatch = 4
    left_cfg.experiment_config_path = "/workspace/configs/siglip.yaml"
    left_cfg.experiment_config_sha256 = "a" * 64
    right_cfg = Config()
    right_cfg.data.data_root = str(tmp_path / "data")
    right_cfg.encoder.alias = "dinov3_vitb16"
    right_cfg.encoder.precision = "fp32"
    right_cfg.encoder.frame_microbatch = 4
    right_cfg.experiment_config_path = "/workspace/configs/dino.yaml"
    right_cfg.experiment_config_sha256 = "b" * 64
    dataset = {"dataset": "fake", "fingerprint": "d" * 64}
    left_spec = _spec("offline/siglip")
    right_spec = replace(left_spec, repo_id="offline/dino")

    left = provenance.build_run_provenance(
        left_cfg,
        encoder_spec=left_spec,
        dataset_identity=dataset,
        trainable_init="i" * 64,
        whitening_payload_fingerprint="s" * 64,
        tracking_identity={
            "entity": "team",
            "project": "hjepa-vwm",
            "group": "encoder-pair",
            "name": "Investigation 17 · Encoder substrate · SigLIP 2",
            "id": None,
        },
    )
    right = provenance.build_run_provenance(
        right_cfg,
        encoder_spec=right_spec,
        dataset_identity=dataset,
        trainable_init="i" * 64,
        whitening_payload_fingerprint="t" * 64,
        tracking_identity={
            "entity": "team",
            "project": "hjepa-vwm",
            "group": "encoder-pair",
            "name": "Investigation 17 · Encoder substrate · DINOv3",
            "id": None,
        },
    )
    assert left["tracking_identity"]["name"].endswith("SigLIP 2")
    assert right["tracking_identity"]["name"].endswith("DINOv3")
    assert left["resolved_config"]["experiment_config_sha256"] == "a" * 64
    assert right["resolved_config"]["experiment_config_sha256"] == "b" * 64
    assert "experiment_config_path" not in left["common"]["config"]
    assert "experiment_config_sha256" not in left["common"]["config"]
    assert left["common"]["seed_streams"] == {
        "base": 42,
        "data_transform": 42,
        "train_order_epoch0": 42,
        "validation_order_epoch0": 10_042,
        "model_init": 1_042,
        "training_step_formula": "base*1000003+step",
        "diagnostic": 42 * 1_000_003 + 900_001,
    }
    runtime = left["common"]["runtime"]
    assert {"cuda", "cudnn", "gpu_models"} <= runtime.keys()
    provenance.compare_run_provenance(left, right)

    right_cfg.encoder.frame_microbatch = 8
    wrong_runtime_spec = replace(right_spec, frame_microbatch=8)
    wrong_runtime = provenance.build_run_provenance(
        right_cfg,
        encoder_spec=wrong_runtime_spec,
        dataset_identity=dataset,
        trainable_init="i" * 64,
        whitening_payload_fingerprint="t" * 64,
        tracking_identity=None,
    )
    with pytest.raises(ValueError, match="encoder-independent fields"):
        provenance.compare_run_provenance(left, wrong_runtime)


def test_fc_only_provenance_records_explicit_trainability_contract(tmp_path):
    """A run artifact states the frozen modules, sole trainable, and stopped EMA."""
    import provenance
    from config import Config

    cfg = Config()
    cfg.train.optimization_scope = "fc_only"
    cfg.encoder.precision = "fp32"
    cfg.encoder.frame_microbatch = 4
    cfg.data.data_root = str(tmp_path / "data")
    root = tmp_path / "data" / cfg.data.dataset
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True)
        (root / split / f"{split}.webm").write_bytes(split.encode())
    dataset = {"dataset": "fake", "fingerprint": "d" * 64}
    frozen_hashes = {"B": "b" * 64, "B_EMA": "e" * 64, "D": "d" * 64}

    resolved = provenance.build_run_provenance(
        cfg,
        encoder_spec=_spec(),
        dataset_identity=dataset,
        trainable_init="i" * 64,
        whitening_payload_fingerprint=None,
        frozen_state_hashes=frozen_hashes,
    )

    assert resolved["optimization_contract"] == {
        "scope": "fc_only",
        "trainable_modules": ["F_c"],
        "frozen_modules": ["B", "B_EMA", "D"],
        "ema_updates": False,
        "optimized_objective_terms": ["L_flow"],
    }
    assert resolved["frozen_state_hashes"] == frozen_hashes
    assert resolved["common"]["frozen_state_hashes"] == frozen_hashes


def test_checkpoint_provenance_dataset_transfer_drops_only_dataset_fields():
    """The explicit transfer policy permits data changes but retains every other guard."""
    import copy

    import provenance

    source = {
        "schema": "hjepa-run-provenance-v1",
        "common_identity": "source",
        "common": {
            "dataset_fingerprint": "a" * 64,
            "data_order": {"train": "source"},
            "config": {"data": {"dataset": "ssv2"}, "seed": 42},
            "encoder_runtime_contract": {"precision": "bf16"},
            "runtime": {"torch": "test"},
            "seed_streams": {"base": 42},
            "trainable_init_hash": "i" * 64,
        },
    }
    target = copy.deepcopy(source)
    target["common_identity"] = "target"
    target["common"]["dataset_fingerprint"] = "b" * 64
    target["common"]["data_order"] = {"train": "target"}
    target["common"]["config"]["data"] = {"dataset": "ego4d"}

    with pytest.raises(ValueError, match="provenance"):
        provenance.compare_checkpoint_provenance(source, target)
    provenance.compare_checkpoint_provenance(
        source,
        target,
        allow_dataset_transfer=True,
    )

    target["common"]["config"]["seed"] = 7
    with pytest.raises(ValueError, match="non-dataset"):
        provenance.compare_checkpoint_provenance(
            source,
            target,
            allow_dataset_transfer=True,
        )


def test_temporal_target_provenance_allows_only_predict_residual():
    """Paired full-prediction arms may differ in exactly one target-mode boolean."""
    import copy

    import provenance

    full = {
        "schema": "hjepa-run-provenance-v1",
        "common_identity": "full",
        "common": {
            "config": {"train": {"predict_residual": False, "lambda_cov": 0.01}},
            "warm_start": {"checkpoint_sha256": "a" * 64},
        },
        "resolved_config": {
            "checkpoint_dir": "/workspace/ckpt/full",
            "train": {"predict_residual": False, "lambda_cov": 0.01},
        },
        "encoder_spec": {"feature_fingerprint": "e" * 64},
        "dataset_identity": {"fingerprint": "d" * 64},
        "warm_start": {"checkpoint_sha256": "a" * 64},
        "tracking_identity": {"name": "full"},
        "resource_preflight": {"metrics": {"L_flow": 1.0}},
    }
    residual = copy.deepcopy(full)
    residual["common_identity"] = "residual"
    residual["common"]["config"]["train"]["predict_residual"] = True
    residual["resolved_config"]["train"]["predict_residual"] = True
    residual["resolved_config"]["checkpoint_dir"] = "/workspace/ckpt/residual"
    residual["tracking_identity"]["name"] = "residual"
    residual["resource_preflight"]["metrics"]["L_flow"] = 2.0

    provenance.compare_temporal_target_provenance(full, residual)

    residual["resolved_config"]["train"]["lambda_cov"] = 0.0
    with pytest.raises(ValueError, match="beyond"):
        provenance.compare_temporal_target_provenance(full, residual)
