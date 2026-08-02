"""Contract tests for EGO4D chunking prerequisites."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import chunk_ego4d


def test_require_ffmpeg_fails_before_chunking(monkeypatch):
    """Report the missing system dependency before worker processes are created."""
    monkeypatch.setattr(chunk_ego4d.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match=r"apt-get install -y ffmpeg"):
        chunk_ego4d.require_ffmpeg()


def test_default_workers_are_capped(monkeypatch):
    """Avoid multiplying many process workers by ffmpeg's own thread pools."""
    monkeypatch.setattr(chunk_ego4d.os, "cpu_count", lambda: 128)

    assert chunk_ego4d.default_workers() == 4


def test_encode_window_limits_ffmpeg_to_one_thread(monkeypatch, tmp_path):
    """Keep concurrent encoders within the pod's process and thread limits."""
    source = tmp_path / "source.mp4"
    output = tmp_path / "chunk.mp4"
    source.touch()
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"encoded")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(chunk_ego4d.subprocess, "run", fake_run)

    assert chunk_ego4d.encode_window(source, output, 0.0, 4.0, 12, 256, 27)
    thread_indexes = [i for i, arg in enumerate(captured["cmd"]) if arg == "-threads"]
    assert len(thread_indexes) == 2
    assert all(captured["cmd"][i + 1] == "1" for i in thread_indexes)
    assert captured["cmd"][-1].endswith(".part.mp4")
    assert output.read_bytes() == b"encoded"
    assert not (tmp_path / "chunk.part.mp4").exists()


def test_failed_encodes_make_the_batch_fail():
    """Never permit raw deletion after one or more windows failed to encode."""
    with pytest.raises(RuntimeError, match=r"7 ffmpeg window encodes failed"):
        chunk_ego4d.raise_for_failed_encodes({"failed": 7})
