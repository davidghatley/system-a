"""Fail-closed SHA-256 verification of every distribution file before model load."""
import hashlib
import json
from pathlib import Path


def verify_tree(root: Path) -> None:
    root = Path(root).resolve()
    manifest_path = root / "MANIFEST.sha256.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "system-a-i3-bundle-sha256-v1":
        raise ValueError("unsupported bundle integrity manifest")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("empty or malformed bundle integrity manifest")
    actual = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"bundle symlink forbidden: {path.relative_to(root)}")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and len(Path(relative).parts) == 1 and path.suffix in (".pyc", ".pyo"):
            raise ValueError(f"top-level bundle bytecode forbidden: {relative}")
        if path.is_dir():
            continue
        if relative == "MANIFEST.sha256.json" or "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual[relative] = digest.hexdigest()
    if actual.keys() != expected.keys():
        raise ValueError(f"bundle file set mismatch: missing={sorted(expected.keys() - actual.keys())}, extra={sorted(actual.keys() - expected.keys())}")
    if actual != expected:
        raise ValueError(f"bundle hash mismatch: {sorted(k for k in actual if actual[k] != expected[k])}")


if __name__ == "__main__":
    verify_tree(Path(__file__).parent)
    print("bundle integrity verified")
