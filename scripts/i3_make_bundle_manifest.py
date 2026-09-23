#!/usr/bin/env python3
"""Generate reviewable SHA-256 manifest for the full inference distribution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "release/i3"
files = {}
for path in ROOT.rglob("*"):
    if path.is_symlink():
        raise ValueError(f"bundle symlink forbidden: {path}")
    if not path.is_file():
        continue
    relative = path.relative_to(ROOT).as_posix()
    if relative == "MANIFEST.sha256.json" or "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    files[relative] = digest.hexdigest()
manifest = {"schema_version": "system-a-i3-bundle-sha256-v1", "files": dict(sorted(files.items()))}
(ROOT / "MANIFEST.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"manifested {len(files)} files")
