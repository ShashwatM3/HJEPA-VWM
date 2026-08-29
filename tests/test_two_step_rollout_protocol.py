"""Locked-pair, evaluator, W&B, and preflight contracts for Investigation 23."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    directory = root / "configs" / "experiments"
    return directory / "two_step_rollout_control.yaml", directory / "two_step_rollout_treatment.yaml"


def _scientific_payload(experiment) -> dict:
    payload = json.loads(json.dumps(experiment.config, default=lambda value: value.__dict__))
    for key in ("checkpoint_dir", "experiment_config_path", "experiment_config_sha256"):
        payload.pop(key, None)
    return payload


def test_paired_configs_differ_only_in_rollout_weight() -> None:
    from config import load_experiment_config

    control_path, treatment_path = _paths()
    control = load_experiment_config(control_path)
    treatment = load_experiment_config(treatment_path)
    left = _scientific_payload(control)
    right = _scientific_payload(treatment)
    assert left["train"].pop("lambda_rollout") == 0.0
    assert right["train"].pop("lambda_rollout") == 0.1
    assert left == right
    assert control.wandb.group == treatment.wandb.group
    assert control.wandb.name != treatment.wandb.name
    assert control.config.checkpoint_dir != treatment.config.checkpoint_dir
    assert control.protocol == treatment.protocol


def _valid_protocol_config(tmp_path: Path):
    from config import Config, ProtocolConfig

    checkpoint = tmp_path / "run60.pt"
    checkpoint.write_bytes(b"locked-run-60")
    cfg = Config()
    cfg.seed = 42
    cfg.encoder.alias = "dinov3_vitb16"
    cfg.encoder.precision = "bf16"
    cfg.model.n_c = 64
    cfg.model.d_c = 512
    cfg.model.condition_dropout = 0.0
    cfg.train.optimization_scope = "fc_only"
    cfg.train.flow_source = "present"
    cfg.train.flow_bottleneck_checkpoint = str(checkpoint)
    cfg.train.global_batch = 64
    cfg.train.max_steps = 5_000
    cfg.train.checkpoint_every = 2_500
    cfg.train.horizon_k = 16
    cfg.train.frame_stride = 2
    cfg.train.precision = "bf16"
    cfg.train.rollout_ramp_steps = 1_500
    cfg.train.lambda_var = 0.0
    cfg.train.lambda_cov = 0.0
    cfg.train.lambda_slot = 0.0
    cfg.train.lambda_sigreg = 0.0
    cfg.train.lambda_recon = 0.0
    cfg.train.lambda_recon_pred = 0.0
    cfg.train.lambda_rollout = 0.0
    cfg.data.dataset = "ego4d"
    protocol = ProtocolConfig(
        name="two_step_rollout_v1",
        fixed_bottleneck_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    return cfg, protocol


def test_matched_control_passes_locked_validation(tmp_path: Path) -> None:
    from train import validate_experiment_protocol

    cfg, protocol = _valid_protocol_config(tmp_path)
    validate_experiment_protocol(cfg, protocol)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("train.optimization_scope", "joint", "train.optimization_scope='fc_only'"),
        ("train.flow_source", "noise", "train.flow_source='present'"),
        ("model.condition_dropout", 0.1, "model.condition_dropout=0.0"),
        ("train.lambda_recon", 1.0, "train.lambda_recon=0.0"),
        ("train.lambda_var", 0.5, "train.lambda_var=0.0"),
        ("train.horizon_k", 12, "train.horizon_k=16"),
        ("model.n_c", 32, "model.n_c=64"),
        ("model.d_c", 256, "model.d_c=512"),
    ],
)
def test_treatment_lock_identifies_exact_incompatible_field(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    from train import validate_experiment_protocol

    cfg, protocol = _valid_protocol_config(tmp_path)
    cfg.train.lambda_rollout = 0.1
    section, name = field.split(".")
    setattr(getattr(cfg, section), name, value)
    with pytest.raises(ValueError, match=message):
        validate_experiment_protocol(cfg, protocol)


def test_locked_validation_rejects_mismatched_bottleneck(tmp_path: Path) -> None:
    from train import validate_experiment_protocol

    cfg, protocol = _valid_protocol_config(tmp_path)
    protocol.fixed_bottleneck_sha256 = "0" * 64
    with pytest.raises(ValueError, match="flow_bottleneck_checkpoint sha256"):
        validate_experiment_protocol(cfg, protocol)


def test_checkpoint_selection_is_exact() -> None:
    from evaluate_two_step_rollout import validate_checkpoint_steps

    validate_checkpoint_steps((2_500, 5_000))
    with pytest.raises(ValueError, match="2500, 5000"):
        validate_checkpoint_steps((2_500, 4_999))


class DeterministicFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.calls = 0

    def forward(self, state, tau, condition, *, condition_drop):
        self.calls += 1
        return self.scale * state + 0.5 * condition + tau[:, None, None]


def test_fixed_batch_evaluation_is_deterministic_and_has_solver_schema() -> None:
    from evaluate_two_step_rollout import evaluate_latents

    present = torch.randn(4, 3, 2, generator=torch.Generator().manual_seed(5))
    future = torch.randn(4, 3, 2, generator=torch.Generator().manual_seed(6))
    tau = torch.linspace(0.1, 0.9, 4)
    first = evaluate_latents(DeterministicFlow(), present, future, tau)
    second = evaluate_latents(DeterministicFlow(), present, future, tau)
    assert first == second
    assert set(first) == {"rollout", "teacher_forced"}
    assert set(first["rollout"]["solver_evaluations"]) == {"1", "2", "4", "8"}
    expected = {
        "endpoint_mse",
        "endpoint_copy_ratio",
        "endpoint_batch_mean_ratio",
        "displacement_cosine",
        "displacement_norm_ratio",
        "correct_vs_shuffled_degradation",
        "shuffled_endpoint_mse",
        "zero_endpoint_mse",
    }
    assert all(set(row) == expected for row in first["rollout"]["solver_evaluations"].values())


def test_structured_artifact_identity_fields(tmp_path: Path) -> None:
    from data import ClipBatch
    from evaluate_two_step_rollout import (
        SCHEMA,
        build_evaluation_artifact,
        fixed_batch_identity,
    )

    batch = ClipBatch(torch.zeros(2, 1), torch.ones(2, 1), ("a.mp4", "b.mp4"))
    identity = fixed_batch_identity(batch, "dataset-hash")
    assert SCHEMA == "hjepa-two-step-rollout-evaluation-v1"
    assert identity["dataset_fingerprint"] == "dataset-hash"
    assert identity["sample_ids"] == ["a.mp4", "b.mp4"]
    assert len(identity["sha256"]) == 64
    checkpoint = {
        "step": 2_500,
        "checkpoint_path": "/workspace/ckpt/step2500.pt",
        "checkpoint_sha256": "a" * 64,
        "config_hash": "b" * 64,
        "model_hash": "c" * 64,
        "metrics": {"rollout": {}, "teacher_forced": {}},
    }
    artifact = build_evaluation_artifact(
        config_path=tmp_path / "arm.yaml",
        config_sha256="d" * 64,
        arm="control",
        fixed_bottleneck_sha256="e" * 64,
        batch_identity=identity,
        checkpoints=[checkpoint],
    )
    assert set(artifact) == {
        "schema",
        "config_path",
        "config_sha256",
        "arm",
        "fixed_bottleneck_sha256",
        "fixed_batch",
        "solver_grid",
        "checkpoints",
    }
    assert artifact["solver_grid"] == [1, 2, 4, 8]
    assert set(artifact["checkpoints"][0]) == set(checkpoint)


def test_wandb_allowlist_has_only_stable_primary_rollout_evaluation_keys() -> None:
    from config import Config
    from train import _select_wandb_metrics

    cfg = Config()
    metrics = {
        "loss": 1.0,
        "eval/rollout_4_copy_ratio": 0.8,
        "eval/rollout_8_copy_ratio": 0.9,
        "eval/rollout_4_condition_shuffle_degradation": 0.2,
        "eval/rollout_8_condition_shuffle_degradation": 0.1,
        "rollout_1step_normal_endpoint_mse": 1.2,
    }
    selected = _select_wandb_metrics(metrics, cfg)
    assert set(selected) == set(metrics) - {"rollout_1step_normal_endpoint_mse"}


def test_preflight_helper_changes_no_parameters_and_has_no_side_effect_api() -> None:
    from config import Config
    from preflight_two_step_rollout import treatment_graph_loss

    cfg = Config()
    cfg.train.lambda_rollout = 0.1
    flow = DeterministicFlow()
    present = torch.randn(2, 3, 4)
    future = torch.randn_like(present)
    before = copy.deepcopy(flow.state_dict())
    treatment_graph_loss(flow, present, future, cfg).backward()
    assert flow.calls == 3
    assert all(torch.equal(value, before[name]) for name, value in flow.state_dict().items())
    assert "optimizer" not in treatment_graph_loss.__code__.co_names
    assert "save_checkpoint" not in treatment_graph_loss.__code__.co_names


def test_confirmation_configs_preserve_science_and_lock_verified_sources() -> None:
    from config import load_experiment_config
    from train import validate_confirmation_resume, validate_experiment_protocol

    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    control = load_experiment_config(root / "two_step_rollout_confirmation_control.yaml")
    treatment = load_experiment_config(root / "two_step_rollout_confirmation_treatment.yaml")
    left = _scientific_payload(control)
    right = _scientific_payload(treatment)
    assert left["train"].pop("lambda_rollout") == 0.0
    assert right["train"].pop("lambda_rollout") == 0.1
    assert left == right
    for experiment in (control, treatment):
        validate_experiment_protocol(experiment.config, experiment.protocol)
        assert validate_confirmation_resume(
            experiment.config,
            experiment.protocol,
            resume=experiment.runtime.resume,
            resume_wandb_run=experiment.runtime.resume_wandb_run,
        ) == {
            "approved": True,
            "source_checkpoint_commit": "017a972780c09974742137696cca78e7b2b47538",
            "allowed_config_changes": ["checkpoint_steps", "max_steps"],
        }
        assert experiment.runtime.resume_wandb_run is False
        assert experiment.config.checkpoint_dir not in {
            "/workspace/ckpt/inv023_two_step_rollout_control",
            "/workspace/ckpt/inv023_two_step_rollout_treatment",
        }


def test_confirmation_provenance_allows_only_preregistered_train_fields() -> None:
    from provenance import compare_checkpoint_provenance

    saved = {
        "schema": "hjepa-run-provenance-v1",
        "common_identity": "old",
        "common": {
            "config": {"train": {"max_steps": 5000, "checkpoint_steps": [], "horizon_k": 16}},
            "runtime": {"git_commit": "old", "git_dirty": False, "torch": "2.4.1"},
            "dataset_fingerprint": "dataset",
        },
    }
    expected = copy.deepcopy(saved)
    expected["common_identity"] = "new"
    expected["common"]["config"]["train"]["max_steps"] = 15000
    expected["common"]["config"]["train"]["checkpoint_steps"] = [5500, 7500, 10000, 15000]
    expected["common"]["runtime"]["git_commit"] = "new"
    expected["common"]["runtime"]["git_dirty"] = False
    allowed = frozenset({"max_steps", "checkpoint_steps"})
    migration = {
        "approved": True,
        "source_checkpoint_commit": "old",
        "continuation_implementation_commit": "new",
        "allowed_config_changes": ["checkpoint_steps", "max_steps"],
    }
    compare_checkpoint_provenance(
        saved, expected, allowed_config_changes=allowed, protocol_migration=migration
    )
    expected["common"]["runtime"]["git_dirty"] = True
    with pytest.raises(ValueError, match="clean commit"):
        compare_checkpoint_provenance(
            saved, expected, allowed_config_changes=allowed, protocol_migration=migration
        )
    expected["common"]["runtime"]["git_dirty"] = False
    expected["common"]["config"]["train"]["horizon_k"] = 12
    with pytest.raises(ValueError, match="beyond allowed continuation fields"):
        compare_checkpoint_provenance(
            saved, expected, allowed_config_changes=allowed, protocol_migration=migration
        )


def test_explicit_checkpoint_schedule_is_sorted_unique_and_replaces_cadence() -> None:
    from config import Config
    from train import finalize_training_config, should_save_checkpoint

    cfg = Config()
    cfg.train.max_steps = 15_000
    cfg.train.checkpoint_steps = (5_500, 7_500, 10_000, 15_000)
    finalize_training_config(cfg)
    assert [step for step in range(5_000, 15_001) if should_save_checkpoint(step, cfg)] == [
        5_500,
        7_500,
        10_000,
        15_000,
    ]
    cfg.train.checkpoint_steps = (5_500, 5_500)
    with pytest.raises(ValueError, match="sorted and unique"):
        finalize_training_config(cfg)


def test_confirmation_resume_creates_new_wandb_run_and_preserves_step_continuity(tmp_path) -> None:
    from train import prepare_wandb_init_options, validate_wandb_step

    checkpoint = tmp_path / "source.pt"
    torch.save({"wandb_run_id": "completed-original"}, checkpoint)
    fresh = prepare_wandb_init_options(str(checkpoint), False, {"name": "confirmation", "id": None})
    assert fresh == {"name": "confirmation"}
    reopened = prepare_wandb_init_options(str(checkpoint), True, {"name": "legacy", "id": None})
    assert reopened["id"] == "completed-original"
    assert reopened["resume"] == "must"
    with pytest.raises(ValueError, match="cannot specify an existing run ID"):
        prepare_wandb_init_options(str(checkpoint), False, {"id": "completed-original"})
    validate_wandb_step(5_000, 5_000)
    validate_wandb_step(5_050, 5_000)
    with pytest.raises(ValueError, match="precedes continuation start"):
        validate_wandb_step(4_999, 5_000)


def test_checkpoint_writer_remains_atomic() -> None:
    import inspect

    import train

    source = inspect.getsource(train.save_checkpoint)
    assert "atomic_torch_save(payload, path)" in source
    assert "torch.save(" not in source
