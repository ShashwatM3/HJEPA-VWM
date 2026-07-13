"""Contract tests for selecting downloadable EGO4D source videos."""

from select_ego4d_uids import filter_videos, load_downloadable_uids


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
        "video_uid,canonical_s3_location\n"
        "downloadable,s3://example/downloadable.mp4\n"
    )
    videos = [
        {"video_uid": "downloadable", "duration_sec": 120.0, "fps": 30.0},
        {"video_uid": "metadata-only", "duration_sec": 120.0, "fps": 30.0},
    ]

    downloadable_uids = load_downloadable_uids(manifest_path)
    kept, drops = filter_videos(videos, downloadable_uids=downloadable_uids)

    assert [video["video_uid"] for video in kept] == ["downloadable"]
    assert drops["not_in_video_540ss_manifest"] == 1
