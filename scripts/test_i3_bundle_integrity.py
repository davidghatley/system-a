"""Persistent tiny fixtures for fail-closed distribution integrity checks."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release/i3"))
from bundle_integrity import verify_tree  # noqa: E402

FIXTURES = ROOT / "artifacts/release_i3/integrity_fixtures"


class BundleIntegrityTest(unittest.TestCase):
    def test_valid_and_tampered_files(self):
        for name, contents in (("valid", "verified"), ("tampered", "altered"), ("extra", "verified"), ("missing", None)):
            directory = FIXTURES / name
            directory.mkdir(parents=True, exist_ok=True)
            manifest = {"schema_version": "system-a-i3-bundle-sha256-v1", "files": {"config.json": hashlib.sha256(b"verified").hexdigest()}}
            (directory / "MANIFEST.sha256.json").write_text(json.dumps(manifest) + "\n")
            if contents is not None:
                (directory / "config.json").write_text(contents)
            if name == "extra":
                (directory / "unexpected.txt").write_text("untracked")
        verify_tree(FIXTURES / "valid")
        for name in ("tampered", "extra", "missing"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "bundle (hash|file set) mismatch"):
                verify_tree(FIXTURES / name)

    def test_rejects_top_level_bytecode(self):
        directory = FIXTURES / "bytecode"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "system-a-i3-bundle-sha256-v1",
            "files": {"config.json": hashlib.sha256(b"verified").hexdigest()},
        }
        (directory / "MANIFEST.sha256.json").write_text(json.dumps(manifest) + "\n")
        (directory / "config.json").write_bytes(b"verified")
        (directory / "shadow.pyc").write_bytes(b"untrusted top-level bytecode")
        with self.assertRaisesRegex(ValueError, "top-level bundle bytecode forbidden"):
            verify_tree(directory)


if __name__ == "__main__":
    unittest.main()
