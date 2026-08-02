"""Contract tests for selecting downloadable EGO4D source videos."""

import argparse

from select_ego4d_uids import file_sha256, filter_videos, load_downloadable_uids, write_outputs


def test_filter_videos_drops_v21_grouped_video_uids():
    """Exclude grouped Goal-Step videos because video_540ss has no files for them."""
    videos = [
        {"video_uid": "normal-video", "duration_sec": 120.0, "fps": 30.0},
        {
            "video_uid": "grp-719d9e89-4eb2-49ea-be14-dc2637dc303f",
            "duration_sec": 120.0,
            "fps": 30.0,
        },
    ]

    kept, drops = filter_videos(videos)

    assert [video["video_uid"] for video in kept] == ["normal-video"]
    assert drops["grouped_video_not_in_video_540ss"] == 1


def test_download_manifest_filters_metadata_only_uids(tmp_path):
    """Select only source UIDs represented by the requested download tier."""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "video_uid,canonical_s3_location\n" "downloadable,s3://example/downloadable.mp4\n"
    )
    videos = [
        {"video_uid": "downloadable", "duration_sec": 120.0, "fps": 30.0},
        {"video_uid": "metadata-only", "duration_sec": 120.0, "fps": 30.0},
    ]

    downloadable_uids = load_downloadable_uids(manifest_path)
    kept, drops = filter_videos(videos, downloadable_uids=downloadable_uids)

    assert [video["video_uid"] for video in kept] == ["downloadable"]
    assert drops["not_in_video_540ss_manifest"] == 1


def test_selection_manifest_binds_authoritative_source_files(tmp_path):
    """Selection provenance includes both metadata and downloadable-tier identities."""
    metadata = tmp_path / "ego4d.json"
    download_manifest = tmp_path / "manifest.csv"
    metadata.write_text('{"videos": []}')
    download_manifest.write_text("video_uid,canonical_s3_location\n")
    args = argparse.Namespace(
        seed=42,
        target_hours=1.0,
        val_fraction=0.1,
        batches=1,
        metadata=str(metadata),
        download_manifest=str(download_manifest),
    )

    manifest = write_outputs(tmp_path / "out", [], {}, {}, {}, {}, args)

    assert manifest["source_fingerprints"] == {
        "metadata_sha256": file_sha256(metadata),
        "video_540ss_manifest_sha256": file_sha256(download_manifest),
    }
