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
    assert cfg.train.stage1_steps == 15_000
    assert cfg.train.total_latent_steps == 105_000
    assert cfg.train.lambda_var == 0.10
    assert cfg.train.sigreg_warmup_steps == 2_000
    assert cfg.train.var_floor_std_target == 1.0
    assert cfg.data.data_root == "/tmp/jepa-data"
    assert cfg.data.full_root == "/tmp/jepa-data/ssv2"
    assert cfg.data.tiny_root == "/tmp/jepa-data/ssv2_tiny"
    assert cfg.checkpoint_dir == "/workspace/checkpoints"


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
        "losses.py",
        "diagnostics.py",
        "train.py",
        "requirements.txt",
        "pyproject.toml",
        ".pre-commit-config.yaml",
    ):
        assert (root / name).exists(), name
