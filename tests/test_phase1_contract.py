import importlib
import json
import os
from pathlib import Path


def test_config_exposes_locked_phase1_constants(monkeypatch):
    """Config exposes Phase 1 constants and the single data-root override."""
    monkeypatch.setenv("JEPA_DATA_ROOT", "/tmp/jepa-data")
    config = importlib.import_module("config")

    cfg = config.Config()

    assert cfg.model.t_ctx == 8
    assert cfg.model.h == 256
    assert cfg.model.w == 256
    assert cfg.model.n_ctx == 1024
    assert cfg.model.n_tgt == 1024
    assert cfg.model.n_c == 32
    assert cfg.model.d_e == 1024
    assert cfg.model.d_c == 256
    assert cfg.model.encoder_repo == "facebook/vjepa2-vitl-fpc64-256"
    assert cfg.encoder.alias == "vjepa2_vitl16"
    assert cfg.encoder.revision is None
    assert cfg.encoder.input_frames == 8
    assert cfg.encoder.input_height == cfg.encoder.input_width == 256
    assert cfg.encoder.hf_cache_dir == cfg.hf_cache_dir
    assert cfg.train.stage1_steps == 15_000
    assert cfg.train.total_latent_steps == 105_000
    assert cfg.train.lambda_var == 0.10
    assert cfg.train.sigreg_warmup_steps == 2_000
    assert cfg.train.recon_loss_mode == "cosine"
    assert cfg.train.present_recon_only is False
    assert cfg.train.var_floor_std_target == 1.0
    assert cfg.data.data_root == "/tmp/jepa-data"
    assert cfg.data.full_root == "/tmp/jepa-data/ssv2"
    assert cfg.data.tiny_root == "/tmp/jepa-data/ssv2_tiny"
    assert cfg.data.ego4d_root == "/tmp/jepa-data/ego4d"
    assert cfg.data.ego4d_tiny_root == "/tmp/jepa-data/ego4d_tiny"
    assert cfg.checkpoint_dir == "/workspace/checkpoints"

    custom = config.Config(hf_cache_dir="/tmp/legacy-cache-constructor")
    assert custom.hf_cache_dir == "/tmp/legacy-cache-constructor"
    assert custom.encoder.hf_cache_dir == custom.hf_cache_dir
    custom.hf_cache_dir = "/tmp/legacy-cache-mutation"
    assert custom.encoder.hf_cache_dir == custom.hf_cache_dir


def test_dataset_root_selects_all_four_datasets(monkeypatch):
    """dataset_root() resolves every --data choice and rejects unknown names."""
    monkeypatch.setenv("JEPA_DATA_ROOT", "/tmp/jepa-data")
    config = importlib.import_module("config")

    expected = {
        "ssv2": "/tmp/jepa-data/ssv2",
        "ssv2_tiny": "/tmp/jepa-data/ssv2_tiny",
        "ego4d": "/tmp/jepa-data/ego4d",
        "ego4d_tiny": "/tmp/jepa-data/ego4d_tiny",
    }
    for name, root in expected.items():
        cfg = config.Config()
        cfg.data.dataset = name
        assert cfg.data.dataset_root() == root, name

    cfg = config.Config()
    cfg.data.dataset = "not-a-dataset"
    try:
        cfg.data.dataset_root()
    except ValueError:
        pass
    else:
        raise AssertionError("dataset_root() accepted an unknown dataset name")


def test_dataset_indexes_webm_and_mp4_deterministically(tmp_path):
    """The video glob indexes mixed .webm/.mp4 directories in one sorted order."""
    import pytest

    pytest.importorskip("torch")
    config = importlib.import_module("config")
    data = importlib.import_module("data")

    root = tmp_path / "mixed"
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True)
    for name in ("c_chunk.mp4", "a_video.webm", "b_chunk.mp4"):
        (root / "train" / name).write_bytes(b"fake")
    (root / "validation" / "d_video.webm").write_bytes(b"fake")

    dataset = data.SSV2Dataset(root, "train", config.Config())
    assert [p.name for p in dataset.paths] == ["a_video.webm", "b_chunk.mp4", "c_chunk.mp4"]
    assert len(dataset) == 3

    empty_root = tmp_path / "empty"
    (empty_root / "train").mkdir(parents=True)
    try:
        data.SSV2Dataset(empty_root, "train", config.Config())
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("SSV2Dataset accepted an empty split directory")


def test_make_subset_creates_stratified_symlinks_and_manifest(tmp_path, monkeypatch):
    """make_subset creates deterministic symlinks plus a manifest."""
    make_subset = importlib.import_module("make_subset")
    data_root = tmp_path / "data"
    full = data_root / "ssv2"
    raw = tmp_path / "raw"
    for split in ("train", "validation"):
        (full / split).mkdir(parents=True)
    raw.mkdir()

    labels = {}
    for cls in ("putting something", "moving something"):
        for idx in range(3):
            video_id = f"{cls.split()[0]}_{idx}"
            labels[video_id] = cls
            target = raw / f"{video_id}.webm"
            target.write_bytes(b"fake")
            for split in ("train", "validation"):
                os.symlink(target, full / split / f"{video_id}.webm")

    (full / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    manifest = make_subset.create_subset(
        data_root=data_root,
        train_per_class=2,
        val_per_class=1,
        seed=123,
    )

    tiny = data_root / "ssv2_tiny"
    assert manifest["seed"] == 123
    assert len(list((tiny / "train").iterdir())) == 4
    assert len(list((tiny / "validation").iterdir())) == 2
    assert (tiny / "manifest.json").exists()
    assert all(path.is_symlink() for path in (tiny / "train").iterdir())


def test_phase1_scaffolding_files_are_present():
    """Phase 1 deliverable files exist at the flat repository root."""
    root = Path(__file__).resolve().parents[1]
    for name in (
        "config.py",
        "make_subset.py",
        "data.py",
        "models.py",
        "encoders.py",
        "losses.py",
        "diagnostics.py",
        "train.py",
        "requirements.txt",
        "pyproject.toml",
        ".pre-commit-config.yaml",
    ):
        assert (root / name).exists(), name
