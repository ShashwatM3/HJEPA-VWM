"""Strict run identity, checkpoint, and CLI contracts for encoder comparisons."""

from __future__ import annotations

import importlib
import sys
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
        feature_dim=16,
        layout=FeatureLayout(8, 2, 2, "time_y_x", "frame", 1, 1),
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


def _small_modules(spec):
    from config import Config
    from models import build_phase1_modules

    cfg = Config()
    cfg.model.n_c = 4
    cfg.model.d_c = 8
    cfg.model.bottleneck_mixer_dim = 8
    cfg.model.bottleneck_cross_attn_heads = 2
    cfg.model.bottleneck_latent_blocks = 1
    cfg.model.f_c_blocks = 1
    cfg.model.f_c_heads = 2
    cfg.model.decoder_dim = 8
    cfg.model.decoder_blocks = 1
    cfg.model.decoder_heads = 2
    torch.manual_seed(1042)
    modules = build_phase1_modules(cfg, load_encoder=False, encoder_spec=spec)
    return cfg, modules


def test_trainable_init_hash_is_encoder_name_independent_for_equal_geometry():
    import provenance

    spec_a = _spec("offline/frame-a")
    spec_b = replace(spec_a, repo_id="offline/frame-b")
    _, modules_a = _small_modules(spec_a)
    _, modules_b = _small_modules(spec_b)

    assert provenance.trainable_state_hash(modules_a[1:]) == provenance.trainable_state_hash(
        modules_b[1:]
    )


def test_checkpoint_validates_encoder_before_mutating_and_resumes_next_step(tmp_path):
    import train

    class SpecOnlyEncoder(torch.nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.spec = spec

    spec = _spec()
    cfg, built = _small_modules(spec)
    modules = (SpecOnlyEncoder(spec), *built[1:])
    optimizer = train.make_optimizer(modules[1], modules[3], modules[4], cfg)
    dataset = {"fingerprint": "d" * 64, "dataset": "fake"}
    path = tmp_path / "checkpoint.pt"
    train.save_checkpoint(
        path,
        8,
        modules,
        optimizer,
        cfg,
        dataset_identity=dataset,
        trainable_init_hash="i" * 64,
    )
    assert not list(tmp_path.glob("*.tmp"))
    assert (
        train.load_checkpoint(
            path,
            modules,
            optimizer,
            expected_encoder_spec=spec,
            expected_dataset_identity=dataset,
        )
        == 8
    )

    before = {name: value.clone() for name, value in modules[1].state_dict().items()}
    with pytest.raises(ValueError, match="encoder fingerprint"):
        train.load_checkpoint(
            path,
            modules,
            expected_encoder_spec=replace(spec, repo_id="offline/frame-b"),
            expected_dataset_identity=dataset,
        )
    assert all(torch.equal(before[name], value) for name, value in modules[1].state_dict().items())


def test_checkpoint_restores_exact_saved_dataloader_position(tmp_path):
    import train
    from data import build_dataloader

    class SpecOnlyEncoder(torch.nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.spec = spec

    spec = _spec()
    cfg, built = _small_modules(spec)
    cfg.train.global_batch = 2
    cfg.data.data_root = str(tmp_path / "data")
    cfg.data.dataset = "ssv2_tiny"
    cfg.data.num_workers = 0
    root = tmp_path / "data" / "ssv2_tiny"
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True)
    for index in range(10):
        (root / "train" / f"{index:02d}.webm").write_bytes(b"fake")
    (root / "validation" / "00.webm").write_bytes(b"fake")

    modules = (SpecOnlyEncoder(spec), *built[1:])
    optimizer = train.make_optimizer(modules[1], modules[3], modules[4], cfg)
    dataset = {
        "fingerprint": "d" * 64,
        "dataset": "fake",
        "splits": {"train": {"count": 10}},
    }
    path = tmp_path / "checkpoint.pt"
    train.save_checkpoint(path, 3, modules, optimizer, cfg, dataset_identity=dataset)

    restored: dict[str, int] = {}
    assert (
        train.load_checkpoint(
            path,
            modules,
            optimizer,
            expected_encoder_spec=spec,
            expected_dataset_identity=dataset,
            sampler_state_out=restored,
        )
        == 3
    )
    assert restored == {"epoch": 0, "batch_offset": 6}

    uninterrupted = build_dataloader(cfg, "train", epoch=0, start_offset=0)
    resumed = build_dataloader(cfg, "train", epoch=restored["epoch"], start_offset=6)
    assert list(resumed.sampler) == list(uninterrupted.sampler)[6:]

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["sampler_state"]["batch_offset"] = 4
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="sampler state"):
        train.load_checkpoint(
            path,
            modules,
            expected_encoder_spec=spec,
            expected_dataset_identity=dataset,
            sampler_state_out={},
        )

    checkpoint["sampler_state"] = None
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="missing the sampler state"):
        train.load_checkpoint(
            path,
            modules,
            expected_encoder_spec=spec,
            expected_dataset_identity=dataset,
            sampler_state_out={},
        )


def test_resource_throughput_counts_examples_frames_and_encoder_tokens():
    import train

    rates = train._throughput_rates(
        seconds=2.0,
        batch_size=4,
        input_frames=8,
        detailed_tokens_per_clip=2048,
        encoder_passes=2,
    )

    assert rates == {
        "examples_per_second": 2.0,
        "frames_per_second": 32.0,
        "detailed_tokens_per_second": 8192.0,
    }


def test_resource_throughput_rejects_nonpositive_measurements():
    import train

    with pytest.raises(ValueError, match="seconds"):
        train._throughput_rates(0.0, 4, 8, 2048, 2)
    with pytest.raises(ValueError, match="positive integers"):
        train._throughput_rates(1.0, 0, 8, 2048, 2)


def test_checkpoint_validates_auxiliary_state_before_mutating_models(tmp_path):
    import train
    from models import FeatureMeanTracker

    class SpecOnlyEncoder(torch.nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.spec = spec

    spec = _spec()
    cfg, built = _small_modules(spec)
    modules = (SpecOnlyEncoder(spec), *built[1:])
    optimizer = train.make_optimizer(modules[1], modules[3], modules[4], cfg)
    tracker = FeatureMeanTracker(spec.layout.n_tokens, spec.feature_dim, 0.99)
    dataset = {"fingerprint": "d" * 64, "dataset": "fake"}
    path = tmp_path / "bad-tracker.pt"
    train.save_checkpoint(
        path,
        1,
        modules,
        optimizer,
        cfg,
        mean_tracker=tracker,
        dataset_identity=dataset,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["recon_feature_mean"]["mean"] = torch.zeros(1, 1)
    torch.save(checkpoint, path)

    first_parameter = next(modules[1].parameters())
    with torch.no_grad():
        first_parameter.add_(7.0)
    before = {name: value.clone() for name, value in modules[1].state_dict().items()}
    with pytest.raises(RuntimeError, match="recon_feature_mean"):
        train.load_checkpoint(
            path,
            modules,
            optimizer,
            mean_tracker=tracker,
            expected_encoder_spec=spec,
            expected_dataset_identity=dataset,
        )
    assert all(torch.equal(before[name], value) for name, value in modules[1].state_dict().items())


def test_checkpoint_rejects_optimizer_tensor_shape_before_mutating_models(tmp_path):
    import train

    class SpecOnlyEncoder(torch.nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.spec = spec

    spec = _spec()
    cfg, built = _small_modules(spec)
    modules = (SpecOnlyEncoder(spec), *built[1:])
    optimizer = train.make_optimizer(modules[1], modules[3], modules[4], cfg)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    dataset = {"fingerprint": "d" * 64, "dataset": "fake"}
    path = tmp_path / "bad-optimizer.pt"
    train.save_checkpoint(path, 1, modules, optimizer, cfg, dataset_identity=dataset)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    first_state = next(iter(checkpoint["optimizer"]["state"].values()))
    first_state["exp_avg"] = torch.zeros(1)
    torch.save(checkpoint, path)

    first_parameter = next(modules[1].parameters())
    with torch.no_grad():
        first_parameter.add_(7.0)
    before = {name: value.clone() for name, value in modules[1].state_dict().items()}
    with pytest.raises(RuntimeError, match="optimizer state tensor"):
        train.load_checkpoint(
            path,
            modules,
            optimizer,
            expected_encoder_spec=spec,
            expected_dataset_identity=dataset,
        )
    assert all(torch.equal(before[name], value) for name, value in modules[1].state_dict().items())


def test_training_cli_exposes_complete_encoder_and_operator_surface(monkeypatch):
    train = importlib.import_module("train")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--encoder",
            "siglip2_vitb16",
            "--encoder-precision",
            "fp32",
            "--encoder-frame-microbatch",
            "2",
            "--batch-size",
            "4",
            "--lr-decoder",
            "2e-4",
            "--preflight-only",
            "--provenance-out",
            "/tmp/run.json",
            "--require-wandb",
        ],
    )
    args = train.parse_args()
    assert args.encoder == "siglip2_vitb16"
    assert args.encoder_precision == "fp32"
    assert args.encoder_frame_microbatch == 2
    assert args.batch_size == 4
    assert args.lr_decoder == pytest.approx(2e-4)
    assert args.preflight_only and args.require_wandb


def test_training_rejects_batch_one_because_shuffled_video_probe_needs_a_peer():
    """A one-sample validation batch makes shuffled-c an identity operation."""
    import train
    from config import Config

    cfg = Config()
    cfg.train.global_batch = 1
    with pytest.raises(ValueError, match="shuffled-c"):
        train.finalize_training_config(cfg)


def test_checkpoint_dataset_transfer_flag_preserves_non_dataset_provenance_guards(tmp_path):
    """The explicit transfer flag works without disabling architecture/runtime checks."""
    import copy

    import train

    class SpecOnlyEncoder(torch.nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.spec = spec

    spec = _spec()
    cfg, built = _small_modules(spec)
    modules = (SpecOnlyEncoder(spec), *built[1:])
    optimizer = train.make_optimizer(modules[1], modules[3], modules[4], cfg)
    source_dataset = {"fingerprint": "a" * 64, "dataset": "ssv2"}
    target_dataset = {"fingerprint": "b" * 64, "dataset": "ego4d"}
    source_provenance = {
        "schema": "hjepa-run-provenance-v1",
        "common_identity": "source",
        "common": {
            "dataset_fingerprint": source_dataset["fingerprint"],
            "data_order": {"train": "source"},
            "config": {"data": {"dataset": "ssv2"}, "seed": 42},
            "encoder_runtime_contract": {"precision": "fp32"},
            "runtime": {"torch": "test"},
            "seed_streams": {"base": 42},
            "trainable_init_hash": "i" * 64,
        },
    }
    target_provenance = copy.deepcopy(source_provenance)
    target_provenance["common_identity"] = "target"
    target_provenance["common"]["dataset_fingerprint"] = target_dataset["fingerprint"]
    target_provenance["common"]["data_order"] = {"train": "target"}
    target_provenance["common"]["config"]["data"] = {"dataset": "ego4d"}
    path = tmp_path / "transfer.pt"
    train.save_checkpoint(
        path,
        3,
        modules,
        optimizer,
        cfg,
        dataset_identity=source_dataset,
        run_provenance=source_provenance,
    )

    with pytest.raises(ValueError, match="dataset fingerprint"):
        train.load_checkpoint(
            path,
            modules,
            expected_encoder_spec=spec,
            expected_dataset_identity=target_dataset,
            expected_run_provenance=target_provenance,
        )
    assert (
        train.load_checkpoint(
            path,
            modules,
            expected_encoder_spec=spec,
            expected_dataset_identity=target_dataset,
            expected_run_provenance=target_provenance,
            allow_dataset_transfer=True,
        )
        == 3
    )


def test_diagnostic_rng_stream_cannot_change_future_training_draws(monkeypatch):
    import train
    from config import Config

    expected_state = torch.Generator().manual_seed(77).get_state()
    torch.random.set_rng_state(expected_state)
    monkeypatch.setattr(
        train,
        "_run_diagnostics_impl",
        lambda *_args, **_kwargs: {"draw": float(torch.rand(()))},
    )
    train.run_diagnostics(None, None, Config(), torch.device("cpu"))
    assert torch.equal(torch.random.get_rng_state(), expected_state)


def test_interrupted_resume_matches_uninterrupted_next_update(tmp_path):
    """Checkpoint next_step, optimizer, EMA, parameters, and metrics continue exactly."""
    import train

    spec = _spec()

    class DeterministicEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec

        def forward(self, clips):
            value = clips.mean(dim=(1, 2, 3, 4))[:, None, None]
            token = torch.linspace(0.0, 0.1, spec.feature_dim)[None, None]
            return (value + token).expand(-1, spec.layout.n_tokens, -1)

    def build():
        cfg, modules = _small_modules(spec)
        cfg.train.present_recon_only = True
        cfg.train.lambda_var = 0.1
        cfg.train.lambda_recon = 0.0
        full = (DeterministicEncoder(), *modules[1:])
        optimizer = train.make_optimizer(full[1], full[3], full[4], cfg)
        return cfg, full, optimizer

    generator = torch.Generator().manual_seed(9)
    batches = [(torch.rand(2, 8, 3, 4, 4, generator=generator), None) for _ in range(2)]

    cfg_a, modules_a, optimizer_a = build()
    train.train_step(batches[0], modules_a, optimizer_a, 0, cfg_a, torch.device("cpu"))
    metrics_a = train.train_step(batches[1], modules_a, optimizer_a, 1, cfg_a, torch.device("cpu"))

    cfg_b, modules_b, optimizer_b = build()
    train.train_step(batches[0], modules_b, optimizer_b, 0, cfg_b, torch.device("cpu"))
    path = tmp_path / "resume.pt"
    dataset = {"dataset": "fake", "fingerprint": "d" * 64}
    train.save_checkpoint(path, 1, modules_b, optimizer_b, cfg_b, dataset_identity=dataset)

    cfg_c, modules_c, optimizer_c = build()
    assert (
        train.load_checkpoint(
            path,
            modules_c,
            optimizer_c,
            expected_encoder_spec=spec,
            expected_dataset_identity=dataset,
        )
        == 1
    )
    metrics_c = train.train_step(batches[1], modules_c, optimizer_c, 1, cfg_c, torch.device("cpu"))

    for module_a, module_c in zip(modules_a[1:], modules_c[1:], strict=True):
        for name, value in module_a.state_dict().items():
            assert torch.equal(value, module_c.state_dict()[name]), name

    def assert_nested_equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                assert_nested_equal(left_item, right_item)
        else:
            assert left == right

    assert_nested_equal(optimizer_a.state_dict(), optimizer_c.state_dict())
    assert metrics_a == metrics_c
