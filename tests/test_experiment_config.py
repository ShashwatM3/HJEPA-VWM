import sys
from pathlib import Path

import pytest


def test_yaml_controls_recipe_hyperparameters_and_runtime(tmp_path: Path) -> None:
    """Load non-sweep settings from one strict experiment YAML file."""
    from config import load_experiment_config

    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
model:
  decoder_dim: 512
  decoder_blocks: 4
train:
  max_steps: 12000
  lambda_var: 0.5
  lambda_cov: 0.01
  present_recon_only: true
  lambda_recon: 1.0
runtime:
  mode: stage0
  require_wandb: true
wandb:
  project: yaml-project
  group: yaml-group
""",
        encoding="utf-8",
    )

    experiment = load_experiment_config(path)

    assert experiment.config.model.decoder_dim == 512
    assert experiment.config.model.decoder_blocks == 4
    assert experiment.config.train.max_steps == 12000
    assert experiment.config.train.lambda_var == 0.5
    assert experiment.config.train.lambda_cov == 0.01
    assert experiment.config.train.present_recon_only is True
    assert experiment.config.train.lambda_recon == 1.0
    assert experiment.runtime.mode == "stage0"
    assert experiment.runtime.require_wandb is True
    assert experiment.wandb.project == "yaml-project"
    assert experiment.wandb.group == "yaml-group"


def test_yaml_rejects_unknown_keys_before_training(tmp_path: Path) -> None:
    """Reject misspelled recipe fields instead of silently falling back to defaults."""
    from config import load_experiment_config

    path = tmp_path / "typo.yaml"
    path.write_text(
        """
train:
  lamdba_var: 0.5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"config\.train.*lamdba_var"):
        load_experiment_config(path)


def test_yaml_rejects_duplicate_keys_before_training(tmp_path: Path) -> None:
    """Do not silently let the last spelling of a paid-run setting win."""
    from config import load_experiment_config

    path = tmp_path / "duplicate.yaml"
    path.write_text(
        """
train:
  lambda_var: 0.1
  lambda_var: 0.5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Duplicate YAML key.*lambda_var"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "body",
    [
        "model:\n  encoder_frozen: false\n",
        "experiment_config_sha256: forged\n",
        "hf_cache_dir: /tmp/legacy-cache\n",
    ],
)
def test_yaml_rejects_legacy_ignored_and_audit_fields(tmp_path: Path, body: str) -> None:
    """A recipe may not claim to change fields that production training ignores or owns."""
    from config import load_experiment_config

    path = tmp_path / "forbidden.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=r"not configurable from experiment YAML"):
        load_experiment_config(path)


def test_yaml_encoder_cache_path_stays_synchronized(tmp_path: Path) -> None:
    """The canonical nested cache field must update the retained legacy mirror."""
    from config import load_experiment_config

    path = tmp_path / "cache.yaml"
    path.write_text("encoder:\n  hf_cache_dir: /tmp/custom-cache\n", encoding="utf-8")

    experiment = load_experiment_config(path)

    assert experiment.config.encoder.hf_cache_dir == "/tmp/custom-cache"
    assert experiment.config.hf_cache_dir == experiment.config.encoder.hf_cache_dir


def test_cli_keeps_only_five_scientific_sweep_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose only the five axes repeatedly varied in the latest investigations."""
    import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/train.yaml",
            "--data",
            "ego4d",
            "--encoder",
            "dinov3_vitb16",
            "--n-c",
            "16",
            "--d-c",
            "512",
            "--bottleneck-mixer-dim",
            "512",
        ],
    )

    args = train.parse_args()

    assert args.config == "configs/train.yaml"
    assert args.data == "ego4d"
    assert args.encoder == "dinov3_vitb16"
    assert args.n_c == 16
    assert args.d_c == 512
    assert args.bottleneck_mixer_dim == 512

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/train.yaml", "--lambda-var", "0.5"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_scientific_cli_group_contains_exactly_five_options() -> None:
    """Make CLI-surface growth an explicit reviewed test change."""
    import train

    parser = train.build_arg_parser()
    scientific_group = next(
        group for group in parser._action_groups if group.title == "scientific hot overrides"
    )
    option_strings = {
        option
        for action in scientific_group._group_actions
        for option in action.option_strings
        if option.startswith("--")
    }

    assert option_strings == {
        "--data",
        "--encoder",
        "--n-c",
        "--d-c",
        "--bottleneck-mixer-dim",
    }


def test_main_loads_yaml_then_applies_hot_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve the complete recipe from YAML before applying the five CLI axes."""
    import train

    path = tmp_path / "stage0.yaml"
    path.write_text(
        """
model:
  n_c: 64
  d_c: 256
  bottleneck_mixer_dim: 512
  decoder_dim: 512
encoder:
  alias: siglip2_vitb16
data:
  dataset: ego4d
train:
  lambda_var: 0.5
runtime:
  mode: stage0
""",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(path),
            "--encoder",
            "dinov3_vitb16",
            "--n-c",
            "16",
        ],
    )
    monkeypatch.setattr(train, "run_stage0", lambda cfg: captured.setdefault("cfg", cfg))

    train.main()

    cfg = captured["cfg"]
    assert cfg.data.dataset == "ego4d"
    assert cfg.encoder.alias == "dinov3_vitb16"
    assert cfg.model.n_c == 16
    assert cfg.model.d_c == 256
    assert cfg.model.bottleneck_mixer_dim == 512
    assert cfg.model.decoder_dim == 512
    assert cfg.train.lambda_var == 0.5


def test_encoder_override_rejects_a_revision_pinned_for_another_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not carry a repository-specific SHA across a hot encoder override."""
    import train

    path = tmp_path / "pinned.yaml"
    path.write_text(
        """
encoder:
  alias: vjepa2_vitl16
  revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
runtime:
  mode: stage0
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(path), "--encoder", "dinov3_vitb16"],
    )

    with pytest.raises(ValueError, match=r"--encoder.*encoder\.revision"):
        train.main()


def test_shipped_yaml_is_a_complete_runnable_default_recipe() -> None:
    """Ship one versioned recipe so training never depends on a long ad-hoc command."""
    from config import load_experiment_config

    path = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    experiment = load_experiment_config(path)

    assert experiment.config.data.dataset == "ssv2_tiny"
    assert experiment.config.encoder.alias == "vjepa2_vitl16"
    assert experiment.config.model.n_c == 32
    assert experiment.config.model.d_c == 256
    assert experiment.config.model.bottleneck_mixer_dim == 256
    assert experiment.config.train.max_steps == 15000
    assert experiment.config.train.lambda_var == 0.10
    assert experiment.config.train.lambda_recon == 0.0
    assert experiment.runtime.mode == "train"
    assert experiment.wandb.project == "hjepa-vwm"
    assert experiment.config.experiment_config_path == str(path)
    assert len(experiment.config.experiment_config_sha256) == 64


def test_yaml_rejects_wrong_value_types_before_training(tmp_path: Path) -> None:
    """Reject quoted numeric values instead of failing deep inside a paid run."""
    from config import load_experiment_config

    path = tmp_path / "wrong-type.yaml"
    path.write_text(
        """
train:
  global_batch: "64"
""",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"config\.train\.global_batch.*integer"):
        load_experiment_config(path)


def test_yaml_rejects_unknown_runtime_modes(tmp_path: Path) -> None:
    """Reject a misspelled execution mode before any model or dataset work."""
    from config import load_experiment_config

    path = tmp_path / "wrong-mode.yaml"
    path.write_text(
        """
runtime:
  mode: paid-trian
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"runtime\.mode"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("name", "encoder_alias", "frame_microbatch"),
    [
        ("inv017_latent_shape.yaml", "vjepa2_vitl16", 8),
        ("inv017_dinov3_latent_shape.yaml", "dinov3_vitb16", 32),
    ],
)
def test_active_latent_shape_recipes_are_valid(
    name: str, encoder_alias: str, frame_microbatch: int
) -> None:
    """Keep the active Investigation-17 launcher backgrounds valid after CLI simplification."""
    from config import load_experiment_config
    from train import finalize_training_config

    path = Path(__file__).resolve().parents[1] / "configs" / name
    experiment = load_experiment_config(path)
    finalize_training_config(experiment.config)

    assert experiment.config.train.present_recon_only is True
    assert experiment.config.train.lambda_recon == 1.0
    assert experiment.config.train.lambda_var == 0.5
    assert experiment.config.train.lambda_cov == 0.01
    assert experiment.config.encoder.alias == encoder_alias
    assert experiment.config.encoder.frame_microbatch == frame_microbatch
