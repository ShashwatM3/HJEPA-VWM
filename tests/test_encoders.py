"""Offline contracts for the encoder-independent frozen feature seam."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


_REVISION_A = "a" * 40
_REVISION_B = "b" * 40
_VJEPA2_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
_SIGLIP2_REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"


class _FakeTubeletBackend(nn.Module):
    """Video backend returning an exact 4x16x16 time-major lattice."""

    frame_based = False

    def __init__(self, *, resolved_revision: str = _REVISION_A) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.resolved_revision = resolved_revision
        self.seen: list[torch.Tensor] = []

    def forward_units(self, clips: torch.Tensor) -> torch.Tensor:
        self.seen.append(clips.detach().clone())
        token = torch.arange(1024, device=clips.device, dtype=clips.dtype).view(1, 1024, 1)
        signal = clips.mean(dim=(1, 2, 3, 4)).view(-1, 1, 1)
        return (token + signal).expand(clips.shape[0], 1024, 1024)

    @staticmethod
    def select_dense_tokens(output: torch.Tensor) -> torch.Tensor:
        return output


class _FakeFrameBackend(nn.Module):
    """Image backend whose private selection rule drops one synthetic special token."""

    frame_based = True

    def __init__(self, *, resolved_revision: str = _REVISION_A) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.resolved_revision = resolved_revision
        self.microbatch_sizes: list[int] = []

    def forward_units(self, frames: torch.Tensor) -> torch.Tensor:
        self.microbatch_sizes.append(frames.shape[0])
        frame_ids = frames[:, 0, 0, 0].view(-1, 1, 1) * 1_000
        patches = torch.arange(256, device=frames.device, dtype=frames.dtype).view(1, 256, 1)
        patches = (frame_ids + patches).expand(frames.shape[0], 256, 768)
        special = torch.full(
            (frames.shape[0], 1, 768), -999.0, device=frames.device, dtype=frames.dtype
        )
        return torch.cat((special, patches), dim=1)

    @staticmethod
    def select_dense_tokens(output: torch.Tensor) -> torch.Tensor:
        return output[:, 1:]


def _tubelet_spec(**changes):
    from encoders import EncoderSpec, FeatureLayout

    values = {
        "family": "fake_tubelet",
        "repo_id": "offline/fake-tubelet",
        "requested_revision": _REVISION_A,
        "resolved_revision": _REVISION_A,
        "input_frames": 8,
        "input_height": 256,
        "input_width": 256,
        "feature_dim": 1024,
        "layout": FeatureLayout(
            temporal=4,
            height=16,
            width=16,
            order="time_y_x",
            temporal_unit="tubelet",
            temporal_stride_frames=2,
            temporal_support_frames=2,
        ),
        "normalization_id": "test_imagenet_once",
        "normalization_mean": (0.0, 0.0, 0.0),
        "normalization_std": (1.0, 1.0, 1.0),
        "preprocess_version": "test-v1",
        "inference_precision": "fp32",
        "frame_microbatch": 3,
        "attention_implementation": "sdpa",
        "cache_dir": "/tmp/hf-cache-a",
        "parameter_count": 1,
    }
    values.update(changes)
    return EncoderSpec(**values)


def _frame_spec(**changes):
    from encoders import FeatureLayout

    return _tubelet_spec(
        family="fake_frame",
        repo_id="offline/fake-frame",
        feature_dim=768,
        layout=FeatureLayout(
            temporal=8,
            height=16,
            width=16,
            order="time_y_x",
            temporal_unit="frame",
            temporal_stride_frames=1,
            temporal_support_frames=1,
        ),
        **changes,
    )


def _injected_encoder(backend: nn.Module, spec, *, mean=(0.0, 0.0, 0.0), std=(1, 1, 1)):
    from encoders import FrozenEncoder

    spec = replace(
        spec,
        normalization_mean=tuple(float(value) for value in mean),
        normalization_std=tuple(float(value) for value in std),
    )
    return FrozenEncoder(
        _backend=backend,
        spec=spec,
    )


def test_public_encoder_surface_is_small():
    import encoders

    assert encoders.__all__ == [
        "FeatureLayout",
        "EncoderSpec",
        "FrozenEncoder",
        "build_frozen_encoder",
    ]


def test_feature_layouts_cover_tubelet_and_time_major_frame_contracts():
    tubelet = _tubelet_spec()
    frame = _frame_spec()

    assert tubelet.layout.n_tokens == 4 * 16 * 16 == 1024
    assert tubelet.layout.temporal_unit == "tubelet"
    assert frame.layout.n_tokens == 8 * 16 * 16 == 2048
    assert frame.layout.temporal_unit == "frame"
    assert frame.feature_dim == 768


def test_normalization_happens_once_in_fp32_and_output_is_dense():
    backend = _FakeTubeletBackend()
    encoder = _injected_encoder(
        backend,
        _tubelet_spec(),
        mean=(0.5, 0.25, 0.75),
        std=(0.25, 0.5, 0.125),
    )
    raw = torch.empty(1, 8, 3, 256, 256, dtype=torch.float16)
    raw[:, :, 0].fill_(0.5)
    raw[:, :, 1].fill_(0.25)
    raw[:, :, 2].fill_(0.75)

    tokens = encoder(raw)

    assert backend.seen[0].dtype == torch.float32
    assert torch.count_nonzero(backend.seen[0]) == 0
    assert tokens.shape == (1, 1024, 1024)
    assert tokens.requires_grad is False


def test_fp32_precision_disables_a_callers_outer_autocast():
    class AutocastSensitiveBackend(_FakeTubeletBackend):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(1, 1, bias=False)

        def forward_units(self, clips):
            pooled = clips.mean(dim=(1, 2, 3, 4)).view(-1, 1)
            value = self.projection(pooled).view(-1, 1, 1)
            return value.expand(clips.shape[0], 1024, 1024)

    backend = AutocastSensitiveBackend()
    parameter_count = sum(parameter.numel() for parameter in backend.parameters())
    encoder = _injected_encoder(backend, _tubelet_spec(parameter_count=parameter_count))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens = encoder(torch.full((1, 8, 3, 256, 256), 0.5))

    assert tokens.dtype == torch.float32


def test_bf16_precision_rejects_non_cuda_execution():
    encoder = _injected_encoder(_FakeTubeletBackend(), _tubelet_spec(inference_precision="bf16"))

    with pytest.raises(RuntimeError, match="only on CUDA"):
        encoder(torch.zeros(1, 8, 3, 256, 256))


def test_frame_token_selection_order_and_microbatch_equivalence():
    raw = torch.zeros(1, 8, 3, 256, 256)
    for frame_index in range(8):
        raw[:, frame_index].fill_(frame_index / 10)

    backend_three = _FakeFrameBackend()
    encoder_three = _injected_encoder(backend_three, _frame_spec(frame_microbatch=3))
    output_three = encoder_three(raw)

    backend_eight = _FakeFrameBackend()
    encoder_eight = _injected_encoder(backend_eight, _frame_spec(frame_microbatch=8))
    output_eight = encoder_eight(raw)

    assert backend_three.microbatch_sizes == [3, 3, 2]
    assert backend_eight.microbatch_sizes == [8]
    assert output_three.shape == (1, 2048, 768)
    assert torch.equal(output_three, output_eight)
    # Time-major: all 256 patches for frame t precede all patches for frame t+1.
    assert output_three[0, 0, 0].item() == 0
    assert output_three[0, 255, 0].item() == 255
    assert output_three[0, 256, 0].item() == pytest.approx(100)
    assert output_three[0, 511, 0].item() == pytest.approx(355)
    assert torch.count_nonzero(output_three == -999.0) == 0


def test_feature_fingerprint_is_stable_and_all_identity_fields_matter():
    baseline = _tubelet_spec()
    assert baseline.fingerprint == _tubelet_spec().fingerprint
    assert len(baseline.fingerprint) == 64

    variants = (
        replace(baseline, requested_revision=_REVISION_B, resolved_revision=_REVISION_B),
        replace(baseline, cache_dir="/tmp/hf-cache-b"),
        replace(baseline, family="different-family"),
        replace(baseline, repo_id="offline/different-repo"),
        replace(baseline, feature_dim=512),
        replace(
            baseline,
            layout=replace(baseline.layout, temporal_support_frames=3),
        ),
        replace(baseline, normalization_id="different-normalization"),
        replace(baseline, normalization_mean=(0.1, 0.2, 0.3)),
        replace(baseline, normalization_std=(0.9, 0.8, 0.7)),
        replace(baseline, preprocess_version="test-v2"),
        replace(baseline, inference_precision="bf16"),
        replace(baseline, frame_microbatch=7),
        replace(baseline, attention_implementation="eager"),
        replace(baseline, parameter_count=2),
    )
    assert all(variant.fingerprint != baseline.fingerprint for variant in variants)


def test_eval_and_freeze_are_sticky():
    backend = _FakeTubeletBackend()
    encoder = _injected_encoder(backend, _tubelet_spec())

    encoder.train(True)
    encoder.requires_grad_(True)

    assert encoder.training is False
    assert backend.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    parent = nn.Sequential(encoder)
    parent.train(True)
    assert encoder.training is False
    assert backend.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_dino_registry_has_pinned_default_revision():
    from encoders import _ADAPTER_REGISTRY

    assert (
        _ADAPTER_REGISTRY["dinov3_vitb16"].default_revision
        == "5931719e67bbdb9737e363e781fb0c67687896bc"
    )


def test_dino_public_factory_uses_pinned_default_revision(monkeypatch):
    """Selecting DINO alone must load the reviewed immutable checkpoint."""
    import transformers

    from config import EncoderConfig
    from encoders import build_frozen_encoder

    pinned_revision = "5931719e67bbdb9737e363e781fb0c67687896bc"
    captured = {}

    class FakeDINOModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(
                _commit_hash=pinned_revision,
                num_register_tokens=4,
            )

    def fake_from_pretrained(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return FakeDINOModel()

    monkeypatch.setattr(transformers.DINOv3ViTModel, "from_pretrained", fake_from_pretrained)

    encoder = build_frozen_encoder(
        EncoderConfig(
            alias="dinov3_vitb16",
            revision=None,
            precision="fp32",
            hf_cache_dir="/tmp/dino-default-cache",
        )
    )

    assert captured["repo_id"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert captured["revision"] == pinned_revision
    assert encoder.spec.requested_revision == pinned_revision
    assert encoder.spec.resolved_revision == pinned_revision


def test_transformers_pin_exposes_both_planned_frame_encoder_architectures():
    """The shared wheel must contain DINOv3 ViT and SigLIP vision classes offline."""
    import transformers

    assert transformers.__version__ == "4.57.6"
    assert transformers.DINOv3ViTModel.__name__ == "DINOv3ViTModel"
    assert transformers.SiglipVisionModel.__name__ == "SiglipVisionModel"


def test_dino_registry_loads_explicit_revision_and_strips_special_tokens(monkeypatch):
    """The public factory exposes 8x16x16 DINO patch tokens when a revision is pinned."""
    import transformers

    from config import ENCODER_IMAGE_MEAN, ENCODER_IMAGE_STD, EncoderConfig
    from encoders import build_frozen_encoder

    captured = {}

    class FakeDINOModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(11))
            self.config = SimpleNamespace(_commit_hash=_REVISION_A, num_register_tokens=4)

        def forward(self, *, pixel_values, return_dict):
            assert return_dict is True
            frame_ids = pixel_values[:, 0, 0, 0].view(-1, 1, 1) * 1_000
            specials = torch.full(
                (pixel_values.shape[0], 5, 768),
                -999.0,
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            patches = torch.arange(256, device=pixel_values.device, dtype=pixel_values.dtype).view(
                1, 256, 1
            )
            tokens = (frame_ids + patches).expand(pixel_values.shape[0], 256, 768)
            return SimpleNamespace(last_hidden_state=torch.cat((specials, tokens), dim=1))

    def fake_from_pretrained(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return FakeDINOModel()

    monkeypatch.setattr(transformers.DINOv3ViTModel, "from_pretrained", fake_from_pretrained)
    cfg = EncoderConfig(
        alias="dinov3_vitb16",
        revision=_REVISION_A,
        precision="fp32",
        frame_microbatch=3,
        hf_cache_dir="/tmp/dino-cache",
    )
    raw = torch.zeros(1, 8, 3, 256, 256)
    for frame_index in range(8):
        raw[:, frame_index].fill_(0.5 + frame_index / 100)

    encoder = build_frozen_encoder(cfg)
    tokens = encoder(raw)

    assert captured == {
        "repo_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "revision": _REVISION_A,
        "cache_dir": "/tmp/dino-cache",
        "attn_implementation": "sdpa",
    }
    assert encoder.spec.family == "dinov3"
    assert encoder.spec.normalization_mean == ENCODER_IMAGE_MEAN
    assert encoder.spec.normalization_std == ENCODER_IMAGE_STD
    assert encoder.spec.layout.n_tokens == 2048
    assert encoder.spec.feature_dim == 768
    assert encoder.spec.parameter_count == 11
    assert tokens.shape == (1, 2048, 768)
    assert torch.count_nonzero(tokens == -999.0) == 0

    mean = torch.tensor(ENCODER_IMAGE_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(ENCODER_IMAGE_STD).view(1, 1, 3, 1, 1)
    normalized = (raw - mean) / std
    frame_signal = normalized[0, :, 0, 0, 0] * 1_000
    assert tokens[0, 0, 0].item() == pytest.approx(frame_signal[0].item())
    assert tokens[0, 255, 0].item() == pytest.approx(frame_signal[0].item() + 255)
    assert tokens[0, 256, 0].item() == pytest.approx(frame_signal[1].item())
    assert tokens.dtype == torch.float32
    assert torch.isfinite(tokens).all()
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    # The adapter is frame-native: identical images have identical patch blocks,
    # while a temporal permutation only permutes those complete blocks.
    repeated = torch.full_like(raw, 0.6)
    repeated_tokens = encoder(repeated).reshape(1, 8, 256, 768)
    assert all(
        torch.equal(repeated_tokens[:, 0], repeated_tokens[:, index]) for index in range(1, 8)
    )

    permutation = torch.tensor([7, 2, 5, 0, 6, 1, 4, 3])
    permuted_tokens = encoder(raw[:, permutation]).reshape(1, 8, 256, 768)
    assert torch.equal(permuted_tokens, tokens.reshape(1, 8, 256, 768)[:, permutation])

    # Chunking flattened frames is an execution detail, not feature semantics.
    encoder_one = build_frozen_encoder(replace(cfg, frame_microbatch=1))
    encoder_eight = build_frozen_encoder(replace(cfg, frame_microbatch=8))
    assert torch.equal(encoder_one(raw), tokens)
    assert torch.equal(encoder_eight(raw), tokens)

    encoder.train(True)
    encoder.requires_grad_(True)
    assert encoder.training is False
    assert encoder._backend.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


@pytest.mark.parametrize(
    ("register_tokens", "sequence_length", "width", "message"),
    [
        (3, 261, 768, "four register tokens"),
        (4, 260, 768, "256 patch tokens"),
        (4, 261, 512, "patch width changed"),
    ],
)
def test_dino_registry_rejects_bad_dense_contracts(
    monkeypatch, register_tokens, sequence_length, width, message
):
    import transformers

    from config import EncoderConfig
    from encoders import build_frozen_encoder

    class FakeDINOModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(
                _commit_hash=_REVISION_A, num_register_tokens=register_tokens
            )

        def forward(self, *, pixel_values, return_dict):
            assert return_dict is True
            return SimpleNamespace(
                last_hidden_state=torch.zeros(
                    pixel_values.shape[0], sequence_length, width, device=pixel_values.device
                )
            )

    monkeypatch.setattr(
        transformers.DINOv3ViTModel,
        "from_pretrained",
        lambda repo_id, **kwargs: FakeDINOModel(),
    )
    cfg = EncoderConfig(alias="dinov3_vitb16", revision=_REVISION_A, precision="fp32")

    with pytest.raises(RuntimeError, match=message):
        encoder = build_frozen_encoder(cfg)
        encoder(torch.zeros(1, 8, 3, 256, 256))


def test_dino_registry_requires_last_hidden_state(monkeypatch):
    import transformers

    from config import EncoderConfig
    from encoders import build_frozen_encoder

    class FakeDINOModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(_commit_hash=_REVISION_A, num_register_tokens=4)

        def forward(self, *, pixel_values, return_dict):
            assert return_dict is True
            return SimpleNamespace(pooler_output=torch.zeros(pixel_values.shape[0], 768))

    monkeypatch.setattr(
        transformers.DINOv3ViTModel,
        "from_pretrained",
        lambda repo_id, **kwargs: FakeDINOModel(),
    )
    encoder = build_frozen_encoder(
        EncoderConfig(alias="dinov3_vitb16", revision=_REVISION_A, precision="fp32")
    )

    with pytest.raises(TypeError, match="last_hidden_state"):
        encoder(torch.zeros(1, 8, 3, 256, 256))


def test_siglip_registry_loads_only_vision_tower_and_preserves_patch_order(monkeypatch):
    """The public factory exposes 8x16x16 patch tokens without loading a text tower."""
    import transformers

    from config import EncoderConfig
    from encoders import build_frozen_encoder

    captured = {}

    class FakeVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(7))
            self.config = SimpleNamespace(_commit_hash=_SIGLIP2_REVISION)
            self.vision_model = nn.Module()
            self.vision_model.head = nn.Linear(1, 1)
            self.vision_model.use_head = True

        def forward(self, *, pixel_values, return_dict):
            assert return_dict is True
            frame_ids = pixel_values[:, 0, 0, 0].view(-1, 1, 1) * 1_000
            patches = torch.arange(256, device=pixel_values.device, dtype=pixel_values.dtype).view(
                1, 256, 1
            )
            tokens = (frame_ids + patches).expand(pixel_values.shape[0], 256, 768)
            return SimpleNamespace(last_hidden_state=tokens)

    def fake_from_pretrained(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return FakeVisionModel()

    monkeypatch.setattr(transformers.SiglipVisionModel, "from_pretrained", fake_from_pretrained)
    cfg = EncoderConfig(
        alias="siglip2_vitb16",
        precision="fp32",
        frame_microbatch=3,
        hf_cache_dir="/tmp/siglip-cache",
    )
    raw = torch.zeros(1, 8, 3, 256, 256)
    for frame_index in range(8):
        # SigLIP normalization maps raw 0.5 to zero, so encode an identity-bearing value.
        raw[:, frame_index].fill_(0.5 + frame_index / 100)

    encoder = build_frozen_encoder(cfg)
    tokens = encoder(raw)

    assert captured == {
        "repo_id": "google/siglip2-base-patch16-256",
        "revision": _SIGLIP2_REVISION,
        "cache_dir": "/tmp/siglip-cache",
        "attn_implementation": "sdpa",
    }
    assert encoder.spec.family == "siglip2"
    assert encoder.spec.normalization_mean == (0.5, 0.5, 0.5)
    assert encoder.spec.normalization_std == (0.5, 0.5, 0.5)
    assert encoder.spec.layout.n_tokens == 2048
    assert encoder.spec.feature_dim == 768
    assert encoder.spec.parameter_count == 7
    assert encoder._backend.model.vision_model.head is None
    assert encoder._backend.model.vision_model.use_head is False
    assert tokens.shape == (1, 2048, 768)
    assert tokens[0, 0, 0].item() == pytest.approx(0.0)
    assert tokens[0, 255, 0].item() == pytest.approx(255.0)
    assert tokens[0, 256, 0].item() == pytest.approx(20.0)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_unknown_alias_and_mutable_revision_fail_before_loading():
    from config import EncoderConfig
    from encoders import build_frozen_encoder

    with pytest.raises(ValueError, match="Unknown encoder alias"):
        build_frozen_encoder(EncoderConfig(alias="not-an-encoder"))
    with pytest.raises(ValueError, match="40-character.*commit SHA"):
        build_frozen_encoder(EncoderConfig(revision="main"))
    with pytest.raises(ValueError, match="40-character.*commit SHA"):
        build_frozen_encoder(EncoderConfig(revision="abc123"))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"input_frames": 4}, "exactly 8 frames"),
        ({"input_height": 224, "input_width": 224}, "256x256"),
        ({"input_height": 256, "input_width": 224}, "square"),
        ({"precision": "fp16"}, "precision"),
        ({"frame_microbatch": 0}, "frame_microbatch"),
        ({"attention_implementation": ""}, "attention_implementation"),
        ({"hf_cache_dir": ""}, "hf_cache_dir"),
    ],
)
def test_encoder_config_contract_errors(changes, message):
    from config import EncoderConfig
    from encoders import build_frozen_encoder

    with pytest.raises(ValueError, match=message):
        build_frozen_encoder(EncoderConfig(**changes))


def test_vjepa_registry_load_is_pinned_cached_and_preserves_legacy_layout(monkeypatch):
    import transformers

    from config import EncoderConfig
    from encoders import build_frozen_encoder

    captured = {}

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(5))
            self.config = SimpleNamespace(_commit_hash=_VJEPA2_REVISION)

        def get_vision_features(self, clips):
            base = torch.arange(1024, device=clips.device, dtype=clips.dtype)
            signal = clips.mean(dim=(1, 2, 3, 4)).view(-1, 1, 1)
            return (base.view(1, 1024, 1) + signal).expand(clips.shape[0], 1024, 1024)

    def fake_from_pretrained(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_from_pretrained)
    cfg = EncoderConfig(precision="fp32", hf_cache_dir="/tmp/explicit-hf-cache")

    encoder = build_frozen_encoder(cfg)
    tokens = encoder(torch.full((1, 8, 3, 256, 256), 0.5))

    assert captured == {
        "repo_id": "facebook/vjepa2-vitl-fpc64-256",
        "revision": _VJEPA2_REVISION,
        "cache_dir": "/tmp/explicit-hf-cache",
        "attn_implementation": "sdpa",
    }
    assert encoder.spec.requested_revision == _VJEPA2_REVISION
    assert encoder.spec.resolved_revision == _VJEPA2_REVISION
    assert encoder.spec.layout.temporal == 4
    assert encoder.spec.layout.n_tokens == 1024
    assert encoder.spec.feature_dim == 1024
    assert encoder.spec.parameter_count == 5
    assert tokens.shape == (1, 1024, 1024)

    # Exact regression: the wrapper applies the historical V-JEPA normalization once.
    from config import ENCODER_IMAGE_MEAN, ENCODER_IMAGE_STD

    mean = torch.tensor(ENCODER_IMAGE_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(ENCODER_IMAGE_STD).view(1, 1, 3, 1, 1)
    raw = torch.full((1, 8, 3, 256, 256), 0.5)
    normalized_signal = ((raw - mean) / std).mean()
    expected = (torch.arange(1024, dtype=raw.dtype).view(1, 1024, 1) + normalized_signal).expand(
        1, 1024, 1024
    )
    assert torch.equal(encoder(raw), expected)


@pytest.mark.parametrize(
    ("raw", "error", "message"),
    [
        (torch.zeros(1, 8, 3, 256, 256, dtype=torch.int64), TypeError, "floating point"),
        (torch.zeros(1, 7, 3, 256, 256), ValueError, "shape"),
        (torch.full((1, 8, 3, 256, 256), -0.01), ValueError, r"\[0, 1\]"),
        (torch.full((1, 8, 3, 256, 256), 1.01), ValueError, r"\[0, 1\]"),
        (torch.full((1, 8, 3, 256, 256), float("nan")), ValueError, "finite"),
    ],
)
def test_input_error_paths(raw, error, message):
    encoder = _injected_encoder(_FakeTubeletBackend(), _tubelet_spec())

    with pytest.raises(error, match=message):
        encoder(raw)


@pytest.mark.parametrize("failure", ["shape", "dimension", "nonfinite", "integer"])
def test_output_error_paths(failure):
    class BadBackend(_FakeTubeletBackend):
        def forward_units(self, clips):
            if failure == "shape":
                return torch.zeros(clips.shape[0], 1023, 1024)
            if failure == "dimension":
                return torch.zeros(clips.shape[0], 1024, 10)
            if failure == "integer":
                return torch.zeros(clips.shape[0], 1024, 1024, dtype=torch.int64)
            output = torch.zeros(clips.shape[0], 1024, 1024)
            output[0, 0, 0] = torch.nan
            return output

    encoder = _injected_encoder(BadBackend(), _tubelet_spec())
    with pytest.raises((TypeError, ValueError), match="output"):
        encoder(torch.zeros(1, 8, 3, 256, 256))


def test_invalid_layout_and_spec_fail_early():
    from encoders import FeatureLayout

    with pytest.raises(ValueError, match="positive"):
        FeatureLayout(0, 16, 16, "time_y_x", "frame", 1, 1)
    with pytest.raises(ValueError, match="order"):
        FeatureLayout(8, 16, 16, "x_y_time", "frame", 1, 1)
    with pytest.raises(ValueError, match="temporal_unit"):
        FeatureLayout(8, 16, 16, "time_y_x", "second", 1, 1)
    with pytest.raises(ValueError, match="40-character"):
        _tubelet_spec(resolved_revision="main")
