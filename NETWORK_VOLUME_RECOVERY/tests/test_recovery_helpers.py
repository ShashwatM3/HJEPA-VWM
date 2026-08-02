from __future__ import annotations

import csv
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_script(name: str, *args: object, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / name), *(str(arg) for arg in args)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != expected:
        raise AssertionError(
            f"{name} returned {result.returncode}, expected {expected}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


class RecoveryHelperTests(unittest.TestCase):
    def test_rebuild_directories_round_trip_and_skip_service_internal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            inventory = Path(temporary) / "directories.csv"
            with inventory.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["empty root/"])
                writer.writerow(["nested/empty/"])
                writer.writerow([".s3compat_uploads/incomplete/"])
            dry = run_script(
                "rebuild_directories_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
                "--dry-run",
            )
            self.assertIn("would_create: 2", dry.stdout)
            self.assertIn("service_internal_skipped: 1", dry.stdout)
            run_script(
                "rebuild_directories_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
            )
            self.assertTrue((workspace / "empty root").is_dir())
            self.assertTrue((workspace / "nested/empty").is_dir())
            self.assertFalse((workspace / ".s3compat_uploads").exists())
            run_script(
                "rebuild_directories_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
                "--check",
            )

    def test_rebuild_directories_refuses_symlink_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            outside = Path(temporary) / "outside"
            workspace.mkdir()
            outside.mkdir()
            (workspace / "escape").symlink_to(outside, target_is_directory=True)
            inventory = Path(temporary) / "directories.csv"
            inventory.write_text("escape/new-directory/\n")
            result = run_script(
                "rebuild_directories_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
                expected=1,
            )
            self.assertIn("symlink collision", result.stderr)
            self.assertFalse((outside / "new-directory").exists())

    def test_rebuild_directories_refuses_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            inventory = Path(temporary) / "directories.csv"
            inventory.write_text("../escape/\n")
            result = run_script(
                "rebuild_directories_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
                expected=2,
            )
            self.assertIn("unsafe/empty directory path", result.stderr)
            self.assertFalse((Path(temporary) / "escape").exists())

    def test_rebuild_links_from_inventory_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            ssv2_target = workspace / "ssv2_raw/20bn-something-something-v2/1.webm"
            ego_target = workspace / "data/ego4d/train/uid_00001.mp4"
            ssv2_target.parent.mkdir(parents=True)
            ego_target.parent.mkdir(parents=True)
            ssv2_target.write_bytes(b"ssv2")
            ego_target.write_bytes(b"ego4d")
            inventory = Path(temporary) / "source.csv"
            with inventory.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([4, "data/ssv2/train/1.webm"])
                writer.writerow([5, "data/ego4d_tiny/train/uid_00001.mp4"])
                writer.writerow([3, "unrelated/file.bin"])

            dry = run_script(
                "rebuild_links_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
                "--dry-run",
            )
            self.assertIn("would_create: 2", dry.stdout)
            run_script(
                "rebuild_links_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
            )
            ssv2_link = workspace / "data/ssv2/train/1.webm"
            ego_link = workspace / "data/ego4d_tiny/train/uid_00001.mp4"
            self.assertTrue(ssv2_link.is_symlink())
            self.assertEqual(ssv2_link.resolve(), ssv2_target.resolve())
            self.assertTrue(ego_link.is_symlink())
            self.assertEqual(ego_link.resolve(), ego_target.resolve())
            checked = run_script(
                "rebuild_links_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
                "--check",
            )
            self.assertIn("already_correct: 2", checked.stdout)

    def test_rebuild_links_refuses_regular_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            target = workspace / "ssv2_raw/20bn-something-something-v2/1.webm"
            collision = workspace / "data/ssv2/train/1.webm"
            target.parent.mkdir(parents=True)
            collision.parent.mkdir(parents=True)
            target.write_bytes(b"video")
            collision.write_bytes(b"video")
            inventory = Path(temporary) / "source.csv"
            inventory.write_text("5,data/ssv2/train/1.webm\n")
            result = run_script(
                "rebuild_links_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
                expected=1,
            )
            self.assertIn("existing non-symlink collision", result.stderr)
            self.assertFalse(collision.is_symlink())

    def test_rebuild_links_refuses_symlinked_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            outside = Path(temporary) / "outside"
            target = workspace / "ssv2_raw/20bn-something-something-v2/1.webm"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"video")
            outside.mkdir()
            (workspace / "data").symlink_to(outside, target_is_directory=True)
            inventory = Path(temporary) / "source.csv"
            inventory.write_text("5,data/ssv2/train/1.webm\n")
            result = run_script(
                "rebuild_links_from_inventory.py",
                "--inventory",
                inventory,
                "--workspace",
                workspace,
                expected=1,
            )
            self.assertIn("symlink parent collision", result.stderr)
            self.assertFalse((outside / "ssv2/train/1.webm").exists())

    def test_rebuild_ssv2_from_split_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            raw = workspace / "ssv2_raw/20bn-something-something-v2"
            raw.mkdir(parents=True)
            for name in ("1.webm", "2.webm", "3.webm"):
                (raw / name).write_bytes(name.encode())
            train = Path(temporary) / "train.json"
            validation = Path(temporary) / "validation.json"
            train.write_text(
                json.dumps([{"id": "1", "template": "a"}, {"id": "2", "template": "b"}])
            )
            validation.write_text(json.dumps([{"id": "3", "template": "c"}]))
            run_script(
                "rebuild_ssv2_from_splits.py",
                "--train-json",
                train,
                "--validation-json",
                validation,
                "--workspace",
                workspace,
            )
            self.assertEqual(
                (workspace / "data/ssv2/train/1.webm").resolve(), (raw / "1.webm").resolve()
            )
            self.assertEqual(
                (workspace / "data/ssv2/validation/3.webm").resolve(), (raw / "3.webm").resolve()
            )
            labels = json.loads((workspace / "data/ssv2/labels.json").read_text())
            self.assertEqual(labels, {"1": "a", "2": "b", "3": "c"})
            (workspace / "data/ssv2/labels.json").write_text(
                json.dumps({"3": "c", "1": "a", "2": "b"}, separators=(",", ":"))
            )
            run_script(
                "rebuild_ssv2_from_splits.py",
                "--train-json",
                train,
                "--validation-json",
                validation,
                "--workspace",
                workspace,
                "--check",
            )

    def test_rebuild_ssv2_refuses_symlinked_data_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            outside = Path(temporary) / "outside"
            raw = workspace / "ssv2_raw/20bn-something-something-v2"
            raw.mkdir(parents=True)
            (raw / "1.webm").write_bytes(b"video")
            outside.mkdir()
            (workspace / "data").symlink_to(outside, target_is_directory=True)
            train = Path(temporary) / "train.json"
            validation = Path(temporary) / "validation.json"
            train.write_text(json.dumps([{"id": "1", "template": "a"}]))
            validation.write_text("[]")
            result = run_script(
                "rebuild_ssv2_from_splits.py",
                "--train-json",
                train,
                "--validation-json",
                validation,
                "--workspace",
                workspace,
                expected=2,
            )
            self.assertIn("symlink parent collision", result.stderr)
            self.assertFalse((outside / "ssv2/train/1.webm").exists())

    def test_rebuild_tiny_links_from_both_manifest_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            ssv2_target = workspace / "ssv2_raw/20bn-something-something-v2/1.webm"
            ego_target = workspace / "data/ego4d/validation/uid_00001.mp4"
            ssv2_target.parent.mkdir(parents=True)
            ego_target.parent.mkdir(parents=True)
            ssv2_target.write_bytes(b"ssv2")
            ego_target.write_bytes(b"ego4d")
            ssv2_manifest = Path(temporary) / "ssv2-manifest.json"
            ssv2_manifest.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {"count": 1, "per_class": {"class": ["1"]}},
                            "validation": {"count": 0, "per_class": {"class": []}},
                        }
                    }
                )
            )
            ego_manifest = Path(temporary) / "ego-manifest.json"
            ego_manifest.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {"count": 0, "per_video": {}},
                            "validation": {
                                "count": 1,
                                "per_video": {"uid": ["uid_00001.mp4"]},
                            },
                        }
                    }
                )
            )
            run_script(
                "rebuild_tiny_from_manifest.py",
                "--dataset",
                "ssv2",
                "--manifest",
                ssv2_manifest,
                "--workspace",
                workspace,
            )
            run_script(
                "rebuild_tiny_from_manifest.py",
                "--dataset",
                "ego4d",
                "--manifest",
                ego_manifest,
                "--workspace",
                workspace,
            )
            self.assertEqual(
                (workspace / "data/ssv2_tiny/train/1.webm").resolve(),
                ssv2_target.resolve(),
            )
            self.assertEqual(
                (workspace / "data/ego4d_tiny/validation/uid_00001.mp4").resolve(),
                ego_target.resolve(),
            )

    def test_rebuild_tiny_rejects_unsafe_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            manifest = Path(temporary) / "unsafe.json"
            manifest.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {"count": 1, "per_class": {"class": ["../escape"]}},
                            "validation": {"count": 0, "per_class": {}},
                        }
                    }
                )
            )
            result = run_script(
                "rebuild_tiny_from_manifest.py",
                "--dataset",
                "ssv2",
                "--manifest",
                manifest,
                "--workspace",
                workspace,
                expected=2,
            )
            self.assertIn("unsafe/empty manifest filename", result.stderr)
            self.assertFalse((Path(temporary) / "escape.webm").exists())

    def test_rebuild_tiny_refuses_symlinked_tiny_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            outside = Path(temporary) / "outside"
            target = workspace / "ssv2_raw/20bn-something-something-v2/1.webm"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"video")
            (workspace / "data").mkdir()
            outside.mkdir()
            (workspace / "data/ssv2_tiny").symlink_to(outside, target_is_directory=True)
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {"count": 1, "per_class": {"class": ["1"]}},
                            "validation": {"count": 0, "per_class": {}},
                        }
                    }
                )
            )
            result = run_script(
                "rebuild_tiny_from_manifest.py",
                "--dataset",
                "ssv2",
                "--manifest",
                manifest,
                "--workspace",
                workspace,
                expected=2,
            )
            self.assertIn("symlink parent collision", result.stderr)
            self.assertFalse((outside / "train/1.webm").exists())

    def test_restore_git_modes_uses_index_without_changing_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            executable = repo / "run.sh"
            plain = repo / "notes.txt"
            executable.write_text("#!/bin/sh\nexit 0\n")
            plain.write_text("notes\n")
            executable.chmod(0o755)
            plain.chmod(0o644)
            subprocess.run(["git", "-C", str(repo), "add", "run.sh", "notes.txt"], check=True)
            executable.chmod(0o644)
            plain.chmod(0o755)
            before_executable = executable.read_bytes()
            before_plain = plain.read_bytes()
            run_script("restore_git_modes.py", "--repo", repo)
            self.assertTrue(stat.S_IMODE(executable.stat().st_mode) & stat.S_IXUSR)
            self.assertFalse(stat.S_IMODE(plain.stat().st_mode) & stat.S_IXUSR)
            self.assertEqual(executable.read_bytes(), before_executable)
            self.assertEqual(plain.read_bytes(), before_plain)

    def test_compare_inventory_detects_and_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            destination = Path(temporary) / "destination.csv"
            digest = "d41d8cd98f00b204e9800998ecf8427e"
            source.write_text(f"0,{digest},empty\n3,,file\n")
            destination.write_text(f"0,{digest},empty\n3,,file\n")
            run_script("compare_inventory.py", source, destination)
            destination.write_text(f"0,{digest},empty\n4,,file\n")
            mismatch = run_script("compare_inventory.py", source, destination, expected=1)
            self.assertIn('"size_differences": 1', mismatch.stdout)

    def test_audit_writes_json_even_when_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            output = Path(temporary) / "audit.json"
            run_script(
                "audit_restored_volume.py",
                "--workspace",
                workspace,
                "--json-out",
                output,
                expected=1,
            )
            payload = json.loads(output.read_text())
            self.assertTrue(payload["errors"])
            self.assertEqual(payload["workspace"], str(workspace.resolve()))


if __name__ == "__main__":
    unittest.main()
