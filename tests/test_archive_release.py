from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "build_public_archive",
    REPOSITORY_ROOT / "scripts" / "build_public_archive.py",
)
verifier = load_module(
    "verify_archive_release",
    REPOSITORY_ROOT / "scripts" / "verify_archive_release.py",
)
public_verifier = load_module(
    "verify_public_release",
    REPOSITORY_ROOT / "scripts" / "verify_public_release.py",
)


class ArchiveReleaseTests(unittest.TestCase):
    def test_archive_contract_uses_frozen_tag_and_rejects_bad_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                return subprocess.run(
                    ["git", *args], cwd=root, check=True, capture_output=True
                )

            git("init", "-q")
            old = b"frozen dependency list\n"
            (root / "pyproject.toml").write_bytes(old)
            git("add", "pyproject.toml")
            git(
                "-c",
                "user.name=Archive Test",
                "-c",
                "user.email=archive@example.invalid",
                "commit",
                "-qm",
                "freeze archive",
            )
            git("tag", "season-2026-archive-v1")
            # Current main is allowed to develop after the frozen archive.
            (root / "pyproject.toml").write_text(
                "new dependency list\n", encoding="utf-8"
            )
            (root / "README.md").write_text(
                "暂时休眠；没有在正式活动环境中完成过一整局；actionable=false",
                encoding="utf-8",
            )
            (root / "STATUS.md").write_text("", encoding="utf-8")
            descriptor = {
                "archive_id": "season-2026-archive-v1",
                "project_status": "hibernating",
                "event_status": "offline",
                "canonical_source_ref": "season-2026-archive-v1",
                "capability_boundary": {
                    key: False
                    for key in (
                        "formal_full_match_end_to_end_completed",
                        "fresh_prospective_validation_completed",
                        "production_enabled",
                        "automatic_bid_execution_in_public_project",
                        "all_assistant_outputs_actionable",
                    )
                },
                "component_manifests": [
                    {
                        "path": "pyproject.toml",
                        "bytes": len(old),
                        "sha256": hashlib.sha256(old).hexdigest(),
                    }
                ],
            }
            path = root / "archive.json"
            path.write_text(json.dumps(descriptor), encoding="utf-8")
            with (
                patch.object(public_verifier, "REPOSITORY_ROOT", root),
                patch.object(public_verifier, "ARCHIVE_DESCRIPTOR", path),
            ):
                errors = []
                public_verifier.verify_archive_contract(errors)
                self.assertEqual(errors, [])
                descriptor["component_manifests"][0]["sha256"] = "0" * 64
                path.write_text(json.dumps(descriptor), encoding="utf-8")
                errors = []
                public_verifier.verify_archive_contract(errors)
                self.assertIn(
                    "archive component sha256 mismatch: pyproject.toml", errors
                )
                git("tag", "-d", "season-2026-archive-v1")
                errors = []
                public_verifier.verify_archive_contract(errors)
                self.assertIn(
                    "archive source tag unavailable; fetch tags before verification",
                    errors,
                )

    def test_deterministic_zip_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("alpha\n", encoding="utf-8")
            second.write_text("beta\n", encoding="utf-8")
            entries = [(second, "b/second.txt"), (first, "a/first.txt")]
            one = root / "one.zip"
            two = root / "two.zip"
            builder.write_deterministic_zip(one, entries)
            builder.write_deterministic_zip(two, reversed(entries))
            self.assertEqual(one.read_bytes(), two.read_bytes())
            with zipfile.ZipFile(one) as archive:
                self.assertEqual(archive.namelist(), ["a/first.txt", "b/second.txt"])

    def test_verifier_rejects_promoted_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in verifier.REQUIRED_FILES - {
                verifier.RELEASE_MANIFEST,
                verifier.CHECKSUMS,
            }:
                (root / filename).write_bytes(filename.encode("utf-8"))
            payload_path = root / "STATUS.md"
            manifest = {
                "archive_id": verifier.ARCHIVE_ID,
                "project_status": "hibernating",
                "source_commit": "0" * 40,
                "capability_boundary": {
                    "formal_full_match_end_to_end_completed": True,
                    "fresh_prospective_validation_completed": False,
                    "production_enabled": False,
                    "automatic_bid_execution_in_public_project": False,
                    "all_assistant_outputs_actionable": False,
                },
                "payloads": [
                    {
                        "filename": payload_path.name,
                        "bytes": payload_path.stat().st_size,
                        "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                    }
                ],
            }
            (root / verifier.RELEASE_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            checksummed = [
                path for path in root.iterdir() if path.name != verifier.CHECKSUMS
            ]
            (root / verifier.CHECKSUMS).write_text(
                "".join(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                    for path in sorted(checksummed)
                ),
                encoding="utf-8",
            )
            errors = verifier.verify_bundle(root)
            self.assertIn(
                "capability boundary must keep formal_full_match_end_to_end_completed=false",
                errors,
            )


if __name__ == "__main__":
    unittest.main()
