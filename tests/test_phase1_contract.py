import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


def test_cli_sets_bottleneck_internal_width_before_stage0(monkeypatch):
    """The public CLI exposes the internal-width sweep without source edits."""
    train = importlib.import_module("train")
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--stage0-only", "--bottleneck-mixer-dim", "512"],
    )
    monkeypatch.setattr(train, "run_stage0", lambda cfg: captured.setdefault("cfg", cfg))

    train.main()

    assert captured["cfg"].model.bottleneck_mixer_dim == 512


@pytest.mark.parametrize("width", [0, 510])
def test_finalize_rejects_invalid_bottleneck_internal_width(width):
    """Internal width must be positive and split evenly across attention heads."""
    config = importlib.import_module("config")
    train = importlib.import_module("train")
    cfg = config.Config()
    cfg.model.bottleneck_mixer_dim = width

    with pytest.raises(ValueError, match="bottleneck_mixer_dim"):
        train.finalize_training_config(cfg)


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
    assert cfg.model.bottleneck_mixer_dim == 256
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


def test_dataset_emits_raw_context_only_or_shared_full_clip(monkeypatch, tmp_path):
    """Context-only decoding never reads a target and both modes stay raw `[0,1]`."""
    import pytest

    pytest.importorskip("torch")
    config = importlib.import_module("config")
    data = importlib.import_module("data")

    root = tmp_path / "clips"
    (root / "validation").mkdir(parents=True)
    (root / "validation" / "clip.webm").write_bytes(b"fake")
    decoded: list[list[int]] = []

    class FakeReader:
        def __len__(self):
            return 48

    def fake_decode(_reader, indices):
        decoded.append(list(indices))
        return np.full((len(indices), 256, 256, 3), 128, dtype=np.uint8)

    monkeypatch.setattr(data, "_open_video_reader", lambda _path: FakeReader())
    monkeypatch.setattr(data, "_decode_frames", fake_decode)
    cfg = config.Config()

    context_only = data.SSV2Dataset(root, "validation", cfg, needs_target=False)[0]
    assert context_only.target is None
    assert context_only.sample_id == "validation/clip.webm"
    assert len(decoded[-1]) == 8
    assert 0.0 <= float(context_only.context.min()) <= float(context_only.context.max()) <= 1.0

    full = data.SSV2Dataset(root, "validation", cfg, needs_target=True)[0]
    assert full.target is not None
    assert len(decoded[-1]) == 16
    assert 0.0 <= float(full.context.min()) <= float(full.target.max()) <= 1.0
    assert data.TRANSFORM_VERSION

    context_dataset = data.SSV2Dataset(root, "validation", cfg, needs_target=False)
    monkeypatch.setattr(
        context_dataset,
        "_window_indices",
        lambda *_args: (_ for _ in ()).throw(AssertionError("target indices were computed")),
    )
    assert context_dataset[0].target is None
    with pytest.raises(ValueError, match="at least one frame"):
        context_dataset._context_indices(0)


def test_stats_loader_can_keep_the_final_partial_training_batch(tmp_path):
    """Training drops incomplete batches; deterministic offline stats may retain them."""
    import pytest

    pytest.importorskip("torch")
    config = importlib.import_module("config")
    data = importlib.import_module("data")

    root = tmp_path / "data" / "ssv2_tiny"
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True)
    for index in range(5):
        (root / "train" / f"{index}.webm").write_bytes(b"fake")
    (root / "validation" / "0.webm").write_bytes(b"fake")
    cfg = config.Config()
    cfg.data.data_root = str(tmp_path / "data")
    cfg.data.num_workers = 0

    training = data.build_dataloader(cfg, "train", batch_size=4, needs_target=False)
    stats = data.build_dataloader(
        cfg,
        "train",
        batch_size=4,
        needs_target=False,
        drop_last=False,
    )
    assert training.drop_last is True
    assert stats.drop_last is False
    assert len(training) == 1
    assert len(stats) == 2


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
