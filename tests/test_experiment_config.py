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


def test_yaml_loads_fc_only_optimization_scope(tmp_path: Path) -> None:
    """Select the complete frozen-representation contract with one scientific field."""
    from config import load_experiment_config

    path = tmp_path / "fc_only.yaml"
    path.write_text(
        """
train:
  optimization_scope: fc_only
""",
        encoding="utf-8",
    )

    experiment = load_experiment_config(path)

    assert experiment.config.train.optimization_scope == "fc_only"


def test_yaml_rejects_unknown_optimization_scope(tmp_path: Path) -> None:
    """Reject an invalid trainability enum during strict YAML loading."""
    from config import load_experiment_config

    path = tmp_path / "invalid_scope.yaml"
    path.write_text("train:\n  optimization_scope: flow_only\n", encoding="utf-8")

    with pytest.raises(ValueError, match="optimization_scope.*joint.*fc_only"):
        load_experiment_config(path)


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
        "model:\n  f_c_dim: 512\n",
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


def test_cli_keeps_only_seven_scientific_sweep_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose only the seven explicitly reviewed scientific hot overrides."""
    import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
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

    assert args.config == train.EXPERIMENT_CONFIG_PATH
    assert args.data == "ego4d"
    assert args.encoder == "dinov3_vitb16"
    assert args.n_c == 16
    assert args.d_c == 512
    assert args.bottleneck_mixer_dim == 512

    monkeypatch.setattr(sys, "argv", ["train.py", "--lambda-var", "0.0"])
    assert train.parse_args().lambda_var == 0.0
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/another.yaml"],
    )
    assert train.parse_args().config == Path("configs/another.yaml")


def test_scientific_cli_group_contains_exactly_seven_options() -> None:
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
        "--temporal-target",
        "--optimization-scope",
    }


def test_main_loads_yaml_then_applies_hot_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve the complete recipe from YAML before applying the seven CLI axes."""
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
            "--encoder",
            "dinov3_vitb16",
            "--n-c",
            "16",
        ],
    )
    monkeypatch.setattr(train, "EXPERIMENT_CONFIG_PATH", path)
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


@pytest.mark.parametrize(
    ("temporal_target", "expected"),
    [("residual", True), ("full_latent", False)],
)
def test_temporal_target_override_changes_only_predict_residual(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    temporal_target: str,
    expected: bool,
) -> None:
    """The paired CLI selector maps directly to the existing target-mode boolean."""
    import train
    from config import load_experiment_config

    path = tmp_path / "stage0.yaml"
    path.write_text(
        """
train:
  present_recon_only: false
runtime:
  mode: stage0
""",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--temporal-target", temporal_target],
    )
    monkeypatch.setattr(train, "EXPERIMENT_CONFIG_PATH", path)
    monkeypatch.setattr(train, "run_stage0", lambda cfg: captured.setdefault("cfg", cfg))

    train.main()

    cfg = captured["cfg"]
    baseline = load_experiment_config(path).config
    train.finalize_training_config(baseline)
    assert cfg.train.predict_residual is expected
    cfg.train.predict_residual = baseline.train.predict_residual
    assert cfg == baseline


def test_optimization_scope_override_changes_only_optimization_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map the single Fc-only selector directly to the resolved training contract."""
    import train
    from config import load_experiment_config

    path = tmp_path / "train.yaml"
    path.write_text("runtime:\n  mode: train\n", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--optimization-scope",
            "fc_only",
            "--warm-start-from",
            "source.pt",
        ],
    )
    monkeypatch.setattr(train, "EXPERIMENT_CONFIG_PATH", path)
    monkeypatch.setattr(
        train,
        "run_training",
        lambda cfg, *_args, **_kwargs: captured.setdefault("cfg", cfg),
    )

    train.main()

    cfg = captured["cfg"]
    baseline = load_experiment_config(path).config
    train.finalize_training_config(baseline)
    assert cfg.train.optimization_scope == "fc_only"
    cfg.train.optimization_scope = baseline.train.optimization_scope
    assert cfg == baseline


def test_finalize_rejects_unknown_optimization_scope() -> None:
    """Fail before paid work when the trainability contract is misspelled."""
    import train
    from config import Config

    cfg = Config()
    cfg.train.optimization_scope = "flow_only"

    with pytest.raises(ValueError, match="optimization_scope"):
        train.finalize_training_config(cfg)


def test_finalize_rejects_present_only_fc_only_combination() -> None:
    """Fc-only mode must retain the prediction path that supplies F_c gradients."""
    import train
    from config import Config

    cfg = Config()
    cfg.train.optimization_scope = "fc_only"
    cfg.train.present_recon_only = True
    cfg.train.lambda_recon = 1.0

    with pytest.raises(ValueError, match="present_recon_only"):
        train.finalize_training_config(cfg)


def test_finalize_rejects_prediction_reconstruction_in_fc_only() -> None:
    """The fixed-coordinate experiment optimizes the flow objective and nothing else."""
    import train
    from config import Config

    cfg = Config()
    cfg.train.optimization_scope = "fc_only"
    cfg.train.lambda_recon_pred = 1.0

    with pytest.raises(ValueError, match="lambda_recon_pred"):
        train.finalize_training_config(cfg)


def test_main_rejects_fc_only_without_warm_start_or_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Never spend a run training F_c against a randomly frozen representation."""
    import train

    path = tmp_path / "fc_only.yaml"
    path.write_text(
        """
train:
  optimization_scope: fc_only
runtime:
  mode: train
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train.py"])
    monkeypatch.setattr(train, "EXPERIMENT_CONFIG_PATH", path)
    monkeypatch.setattr(
        train,
        "run_training",
        lambda *_args, **_kwargs: pytest.fail("training started without fixed source state"),
    )

    with pytest.raises(ValueError, match="warm start or resume"):
        train.main()


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
        ["train.py", "--encoder", "dinov3_vitb16"],
    )
    monkeypatch.setattr(train, "EXPERIMENT_CONFIG_PATH", path)

    with pytest.raises(ValueError, match=r"--encoder.*encoder\.revision"):
        train.main()


def test_shipped_yaml_is_the_only_complete_runnable_recipe() -> None:
    """Keep exactly one recipe, always read by train.py."""
    from config import load_experiment_config

    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "train.yaml"
    assert sorted(candidate.name for candidate in path.parent.glob("*.yaml")) == ["train.yaml"]
    experiment = load_experiment_config(path)

    assert experiment.config.data.dataset == "ssv2_tiny"
    assert experiment.config.encoder.alias == "vjepa2_vitl16"
    assert experiment.config.model.n_c == 32
    assert experiment.config.model.d_c == 256
    assert experiment.config.model.bottleneck_mixer_dim == 512
    assert experiment.config.model.decoder_dim == 512
    assert experiment.config.model.decoder_blocks == 4
    assert experiment.config.train.max_steps == 15000
    assert experiment.config.train.horizon_k == 12
    assert experiment.config.train.lambda_var == 0.5
    assert experiment.config.train.lambda_cov == 0.01
    assert experiment.config.train.lambda_sigreg == 0.0
    assert experiment.config.train.lambda_slot == 0.0
    assert experiment.config.train.lambda_recon == 1.0
    assert experiment.config.train.present_recon_only is False
    assert experiment.config.train.whiten_features is False
    assert experiment.runtime.mode == "train"
    assert experiment.wandb.project == "hjepa-vwm"
    assert experiment.config.experiment_config_path == str(path)
    assert len(experiment.config.experiment_config_sha256) == 64


@pytest.mark.parametrize(
    ("encoder_alias", "expected_frame_microbatch"),
    [
        ("vjepa2_vitl16", 8),
        ("siglip2_vitb16", 8),
        ("dinov3_vitb16", 32),
    ],
)
def test_single_yaml_resolves_tested_microbatch_for_each_encoder(
    encoder_alias: str, expected_frame_microbatch: int
) -> None:
    """Preserve each encoder lane's tested execution identity from one shared YAML."""
    from config import load_experiment_config
    from train import finalize_training_config

    path = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    cfg = load_experiment_config(path).config
    cfg.encoder.alias = encoder_alias

    finalize_training_config(cfg)

    assert cfg.encoder.frame_microbatch == expected_frame_microbatch


@pytest.mark.parametrize(
    "body",
    [
        "encoder:\n  alias: typo_encoder\n",
        "encoder:\n  precision: fp16\n",
        "encoder:\n  attention_implementation: flash_attention_2\n",
        "train:\n  precision: fp16\n",
        "train:\n  recon_loss_mode: mse\n",
        "data:\n  dataset: typo_dataset\n",
    ],
)
def test_yaml_rejects_values_outside_documented_categories(tmp_path: Path, body: str) -> None:
    """Make categorical inline comments an enforced pre-launch contract."""
    from config import load_experiment_config

    path = tmp_path / "invalid-category.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=r"must be one of"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("field_path", "invalid_value", "message"),
    [
        ("model.condition_dropout", 1.1, "condition_dropout"),
        ("train.ema_m_start", 1.0, "ema_m_start"),
        ("train.ema_m_end", 0.9, "ema_m_end"),
        ("train.warmup_steps", 15_000, "warmup_steps"),
        ("train.max_steps", 15_001, "max_steps"),
        ("train.adam_betas", (0.9, 1.0), "adam_betas"),
        ("train.log_every", 0, "log_every"),
    ],
)
def test_finalizer_rejects_invalid_hard_yaml_constraints(
    field_path: str, invalid_value: object, message: str
) -> None:
    """Reject hard range/order violations before model, data, or W&B construction."""
    from config import Config
    from train import finalize_training_config

    cfg = Config()
    target_name, field_name = field_path.split(".")
    setattr(getattr(cfg, target_name), field_name, invalid_value)

    with pytest.raises(ValueError, match=message):
        finalize_training_config(cfg)


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


@pytest.mark.parametrize("yaml_number", [".nan", ".inf", "-.inf"])
def test_yaml_rejects_non_finite_numbers_before_training(tmp_path: Path, yaml_number: str) -> None:
    """Never allow NaN or infinity to evade range checks and reach an optimizer."""
    from config import load_experiment_config

    path = tmp_path / "non-finite.yaml"
    path.write_text(
        f"train:\n  lr_bottleneck: {yaml_number}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"config\.train\.lr_bottleneck.*finite"):
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


def test_every_shipped_yaml_value_has_an_inline_allowed_value_comment() -> None:
    """Keep the single recipe self-documenting at every editable leaf."""
    path = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    missing_comments = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.endswith(":"):
            continue
        if "  # " not in line:
            missing_comments.append((line_number, stripped))

    assert missing_comments == []


def test_single_yaml_has_explicit_category_titles() -> None:
    """Segment the long recipe into scannable named hyperparameter categories."""
    path = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    category_titles = {
        line.strip().removeprefix("# CATEGORY: ").strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("# CATEGORY: ")
    }

    assert category_titles == {
        "Reproducibility and outputs",
        "Frozen encoder and input contract",
        "Abstract latent shape",
        "Bottleneck architecture",
        "Coarse flow architecture",
        "Reconstruction decoder architecture",
        "Batch and training duration",
        "Optimizer and learning-rate schedule",
        "Gradient stability",
        "EMA target schedule",
        "Latent geometry objectives",
        "Reconstruction objectives",
        "Offline feature whitening",
        "Temporal prediction and logging",
        "Dataset and loader",
        "Runtime controls",
        "Weights & Biases identity",
    }
