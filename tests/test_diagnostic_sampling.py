"""Tests for deterministic source-diverse fixed diagnostic batches."""

from pathlib import Path

import pytest
import torch

import data
from config import Config


def test_ego4d_selector_skips_later_chunks_from_seen_source_videos():
    sample_ids = [
        "validation/alpha_uid_00000.mp4",
        "validation/alpha_uid_00001.mp4",
        "validation/bravo_uid_00000.mp4",
        "validation/bravo_uid_00001.mp4",
        "validation/charlie_uid_00000.mp4",
    ]

    selected = data.select_unique_source_indices(sample_ids, "ego4d", 3)

    assert selected == [0, 2, 4]
    assert [sample_ids[index] for index in selected] == [
        "validation/alpha_uid_00000.mp4",
        "validation/bravo_uid_00000.mp4",
        "validation/charlie_uid_00000.mp4",
    ]


def test_ego4d_selector_is_deterministic_and_returns_unique_source_ids():
    sample_ids = [
        "validation/source_a_00000.mp4",
        "validation/source_a_00001.mp4",
        "validation/source_b_00000.mp4",
        "validation/source_c_00000.mp4",
    ]

    first = data.select_unique_source_indices(sample_ids, "ego4d_tiny", 3)
    second = data.select_unique_source_indices(sample_ids, "ego4d_tiny", 3)
    source_ids = [data.source_video_id(sample_ids[index], "ego4d_tiny") for index in first]

    assert first == second == [0, 2, 3]
    assert len(source_ids) == len(set(source_ids)) == 3


def test_ssv2_selector_preserves_first_n_behavior():
    sample_ids = [
        "validation/00001.webm",
        "validation/00002.webm",
        "validation/00003.webm",
    ]

    assert data.select_unique_source_indices(sample_ids, "ssv2", 2) == [0, 1]


def test_selector_warns_and_does_not_refill_when_unique_sources_are_insufficient():
    sample_ids = [
        "validation/source_a_00000.mp4",
        "validation/source_a_00001.mp4",
        "validation/source_b_00000.mp4",
    ]

    with pytest.warns(RuntimeWarning, match="only found 2"):
        selected = data.select_unique_source_indices(sample_ids, "ego4d", 4)

    assert selected == [0, 2]


def test_selector_rejects_batches_with_fewer_than_two_unique_sources():
    sample_ids = [
        "validation/source_a_00000.mp4",
        "validation/source_a_00001.mp4",
    ]

    with pytest.warns(RuntimeWarning, match="only found 1"):
        with pytest.raises(RuntimeError, match="at least two distinct source videos"):
            data.select_unique_source_indices(sample_ids, "ego4d", 4)


def test_ego4d_source_id_requires_authoritative_chunk_name():
    with pytest.raises(ValueError, match="index:05d"):
        data.source_video_id("validation/not_an_ego4d_chunk.mp4", "ego4d")


def test_diagnostic_dataloader_contains_unique_source_ids(monkeypatch, tmp_path):
    sample_ids = (
        "validation/source_a_00000.mp4",
        "validation/source_a_00001.mp4",
        "validation/source_b_00000.mp4",
        "validation/source_c_00000.mp4",
    )

    class FakeDataset:
        def __init__(self, root, split, cfg, *, needs_target, epoch):
            del root, split, cfg, needs_target, epoch

        def __len__(self):
            return len(sample_ids)

        def sample_id_at(self, index):
            return sample_ids[index]

        def __getitem__(self, index):
            return data.ClipSample(
                context=torch.full((8, 3, 2, 2), float(index)),
                target=None,
                sample_id=sample_ids[index],
            )

    monkeypatch.setattr(data, "SSV2Dataset", FakeDataset)
    cfg = Config()
    cfg.data.data_root = str(Path(tmp_path))
    cfg.data.dataset = "ego4d"
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False

    batch = data.build_fixed_diagnostic_batch(
        cfg,
        batch_size=3,
        needs_target=False,
    )
    source_ids = [data.source_video_id(sample_id, "ego4d") for sample_id in batch.sample_ids]

    assert batch.sample_ids == (
        "validation/source_a_00000.mp4",
        "validation/source_b_00000.mp4",
        "validation/source_c_00000.mp4",
    )
    assert len(source_ids) == len(set(source_ids)) == 3
