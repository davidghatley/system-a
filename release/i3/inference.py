"""Local, fixed-schema Laya next-action classifier and output contract."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import math
import os
import stat
import sys
import threading
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from .bundle_integrity import verify_tree
except ImportError:  # direct execution from a copied standalone directory
    from bundle_integrity import verify_tree

LABELS = ("read", "search", "edit", "execute", "other_tool", "respond_or_finish")
CHOICE_ORDER = ("edit", "execute", "other_tool", "read", "respond_or_finish", "search")
FROZEN_INSTRUCTIONS = "Choose the next observable agent action type."
FROZEN_CRITERIA = {
    "edit": "Create or modify files or structured content.",
    "execute": "Execute a shell command, program, or test.",
    "other_tool": "Call another tool not covered by the named action types.",
    "read": "Read a known file or resource.",
    "respond_or_finish": "Respond without an executable tool call or finish the task.",
    "search": "Search or list files, symbols, or resources.",
}


class InputError(ValueError):
    pass


# The token and registry are deliberately process-local.  A module found under
# one of our generated names is reusable only when this process created it and
# still has the same object in sys.modules.  A caller-supplied object with a
# convincing-looking __file__ is therefore not treated as trusted code.
_PRIVATE_TOKEN = object()
_PRIVATE_LOADED: dict[tuple[str, str], types.ModuleType] = {}
_PRIVATE_LOAD_LOCK = threading.RLock()
_PRIVATE_TOKEN_ATTR = "__system_a_i3_private_token__"
_PRIVATE_NAMESPACE_ATTR = "__system_a_i3_namespace__"
_PRIVATE_BUNDLE_ATTR = "__system_a_i3_bundle_path__"
_PRIVATE_KIND_ATTR = "__system_a_i3_private_kind__"
_PRIVATE_SOURCE_PATH_ATTR = "__system_a_i3_source_path__"
_PRIVATE_SOURCE_REL_ATTR = "__system_a_i3_source_relative__"
_PRIVATE_SOURCE_DIGEST_ATTR = "__system_a_i3_source_digest__"
_PRIVATE_LOADER_TOKEN_ATTR = "__system_a_i3_private_loader_token__"


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _module_origin_is_within(module: object, root: Path) -> bool:
    origins: list[object] = []
    module_dict = getattr(module, "__dict__", {})
    file_name = module_dict.get("__file__")
    if file_name is not None:
        origins.append(file_name)
    spec = module_dict.get("__spec__")
    origin = getattr(spec, "origin", None)
    if origin is not None:
        origins.append(origin)
    search_paths = module_dict.get("__path__", ()) or ()
    if isinstance(search_paths, (str, bytes, os.PathLike)):
        origins.append(search_paths)
    else:
        try:
            origins.extend(search_paths)
        except TypeError:
            pass
    for value in origins:
        if not isinstance(value, (str, bytes, os.PathLike)):
            continue
        if isinstance(value, bytes):
            value = os.fsdecode(value)
        if isinstance(value, str) and value.startswith("<"):
            continue
        try:
            if _path_is_within(Path(value), root):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _is_allowed_release_module(name: str, module: object) -> bool:
    if module is sys.modules.get(__name__):
        return True
    if name in {"__main__", "release", "release.i3", "test_inference"}:
        return True
    if name.startswith("release.i3."):
        return True
    if name == "bundle_integrity" or name.endswith(".bundle_integrity"):
        return True
    return isinstance(name, str) and name.startswith("_system_a_i3_laya_")


@contextlib.contextmanager
def _trusted_dependency_imports(bundle_root: Path):
    """Prevent release-root modules from satisfying third-party imports.

    The release directory is commonly first on ``sys.path`` in quickstart and
    verification scripts.  Temporarily remove that directory (and descendants)
    while dependencies are imported, and reject any already-loaded module whose
    origin is inside it.  The original path list is restored on exit; predictor
    initialization and prediction both execute within this boundary.
    """
    root = Path(bundle_root).resolve()
    original = list(sys.path)
    for name, module in tuple(sys.modules.items()):
        if _module_origin_is_within(module, root) and not _is_allowed_release_module(name, module):
            raise ImportError(f"untrusted module was imported from the release bundle: {name}")

    filtered = []
    for entry in original:
        try:
            if not entry:
                candidate = Path.cwd()
            elif isinstance(entry, bytes):
                candidate = Path(os.fsdecode(entry))
            else:
                candidate = Path(entry)
        except (TypeError, ValueError):
            filtered.append(entry)
            continue
        if _path_is_within(candidate, root):
            continue
        filtered.append(entry)
    sys.path[:] = filtered
    try:
        yield
    finally:
        sys.path[:] = original


def _read_manifest_snapshot(bundle: Path) -> tuple[bytes, dict[str, str], str]:
    """Read one manifest snapshot and fingerprint the expected source set.

    The caller passes this snapshot into the loader after ``verify_tree``.  A
    source or manifest change racing the verification therefore cannot silently
    change the digest used by the loader between verification and execution.
    """
    bundle = Path(bundle).resolve()
    root = bundle.parent
    manifest_path = root / "MANIFEST.sha256.json"
    if manifest_path.is_symlink():
        raise ValueError("bundle manifest symlink forbidden")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "system-a-i3-bundle-sha256-v1":
        raise ValueError("unsupported bundle integrity manifest")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("empty or malformed bundle integrity manifest")

    checked: dict[str, str] = {}
    for relative, digest in expected.items():
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("malformed bundle integrity path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"malformed bundle integrity path: {relative}")
        if (not isinstance(digest, str) or len(digest) != 64
                or digest != digest.lower()
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"malformed bundle integrity digest: {relative}")
        checked[relative] = digest

    source_prefix = (bundle / "upstream").relative_to(root).as_posix().rstrip("/") + "/"
    common_relative = (bundle / "upstream/laya/common.py").relative_to(root).as_posix()
    if common_relative not in checked:
        raise FileNotFoundError("bundled laya source is missing from the integrity manifest")
    source_entries = {
        relative: digest for relative, digest in checked.items()
        if relative.startswith(source_prefix)
    }
    source_fingerprint = hashlib.sha256(
        json.dumps(source_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return raw, checked, source_fingerprint


def _private_namespace(bundle: Path, source_fingerprint: str) -> str:
    """Return a stable, bundle-and-source-specific private package name."""
    identity = (str(Path(bundle).resolve()) + ":" + source_fingerprint).encode("utf-8")
    return "_system_a_i3_laya_" + hashlib.sha256(identity).hexdigest()[:24]


def _legacy_private_namespace(bundle: Path, common_digest: str) -> str:
    """Recognize the pre-source-set namespace so old entries cannot be reused."""
    identity = (str(Path(bundle).resolve()) + ":" + common_digest).encode("utf-8")
    return "_system_a_i3_laya_" + hashlib.sha256(identity).hexdigest()[:24]


def _private_namespace_for_bundle(bundle: Path) -> str:
    """Private helper used by isolation tests to identify the exact namespace."""
    bundle = Path(bundle).resolve()
    _, _, source_fingerprint = _read_manifest_snapshot(bundle)
    return _private_namespace(bundle, source_fingerprint)


def _assert_manifest_unchanged(bundle: Path, snapshot: tuple[bytes, dict[str, str], str]) -> None:
    """Fail if the manifest changed while tree verification was running."""
    manifest_path = Path(bundle).resolve().parent / "MANIFEST.sha256.json"
    if manifest_path.is_symlink() or manifest_path.read_bytes() != snapshot[0]:
        raise ValueError("bundle manifest changed during verification")


def _expected_asset_digest(expected: Mapping[str, str], relative: str) -> str:
    digest = expected.get(relative)
    if digest is None:
        raise ValueError(f"bundle asset is absent from captured manifest: {relative}")
    return digest


def _read_authenticated_asset(root: Path, relative: str, expected: str) -> bytes:
    """Read one regular file below *root* and authenticate those exact bytes.

    The descriptor is opened with ``O_NOFOLLOW`` while walking each directory
    component from the already-resolved bundle root.  The bytes are read once
    and hashed after the read; callers receive the same immutable ``bytes``
    object that was authenticated, never a pathname to reopen later.
    """
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("authenticated release loading requires O_NOFOLLOW/O_DIRECTORY")
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ValueError(f"malformed authenticated release asset path: {relative}")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"malformed authenticated release asset digest: {relative}")

    root = Path(root).resolve()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    parts = tuple(relative.split("/"))
    root_parts = root.parts
    if not root.is_absolute() or not root_parts or root_parts[0] != root.anchor:
        raise ValueError("authenticated release root must be absolute")
    try:
        # Walk from the filesystem anchor rather than trusting any absolute
        # path component supplied by the caller or replaced concurrently.
        current_fd = os.open(root.anchor, directory_flags)
        try:
            for part in (*root_parts[1:], *parts[:-1]):
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            fd = os.open(parts[-1], file_flags, dir_fd=current_fd)
        finally:
            os.close(current_fd)
    except OSError as exc:
        raise ValueError(f"unable to open authenticated release asset: {relative}") from exc

    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"authenticated release asset is not a regular file: {relative}")
        data = bytearray()
        while True:
            try:
                chunk = os.read(fd, 1024 * 1024)
            except OSError as exc:
                raise ValueError(f"unable to read authenticated release asset: {relative}") from exc
            if not chunk:
                break
            data.extend(chunk)
        immutable = bytes(data)
    finally:
        os.close(fd)

    if hashlib.sha256(immutable).hexdigest() != expected:
        raise ValueError(f"authenticated release asset hash mismatch: {relative}")
    return immutable


def _authenticated_json(root: Path, relative: str, expected: str):
    data = _read_authenticated_asset(root, relative, expected)
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"authenticated release asset is not valid UTF-8 JSON: {relative}") from exc


class _PrivateBundleContext:
    """Filesystem and provenance policy for one generated private package."""

    def __init__(self, bundle: Path, expected: dict[str, str], source_fingerprint: str):
        self.bundle = Path(bundle).resolve()
        self.root = self.bundle.parent
        self.expected = expected
        self.laya_root = (self.bundle / "upstream" / "laya").resolve()
        self.namespace = _private_namespace(self.bundle, source_fingerprint)
        self.package_name = self.namespace + ".laya"
        self.common_name = self.package_name + ".common"
        self.new_modules: list[str] = []

    def is_private_name(self, name: str) -> bool:
        return isinstance(name, str) and (
            name == self.namespace or name == self.package_name or name.startswith(self.namespace + ".")
        )

    def source_info(self, fullname: str) -> tuple[Path, str, str, bool] | None:
        """Map one private module name to a manifest-covered source file."""
        prefix = self.package_name + "."
        if not fullname.startswith(prefix):
            if fullname.startswith(self.namespace + "."):
                raise ImportError(f"private bundle module is outside the vendored laya package: {fullname}")
            return None
        suffix = fullname[len(prefix):]
        if not suffix:
            return None
        parts = suffix.split(".")
        if any(not part or not part.isidentifier() for part in parts):
            raise ImportError(f"invalid private bundle module name: {fullname}")

        base = self.laya_root.joinpath(*parts)
        package_source = base / "__init__.py"
        module_source = base.with_suffix(".py")
        package_exists = package_source.exists() or package_source.is_symlink()
        module_exists = module_source.exists() or module_source.is_symlink()
        if package_exists:
            source = package_source
            is_package = True
        elif module_exists:
            source = module_source
            is_package = False
        else:
            raise ImportError(f"private bundle source is missing: {fullname}")
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"private bundle source is not a regular file: {source}")
        try:
            resolved = source.resolve(strict=True)
            resolved.relative_to(self.laya_root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"private bundle source escapes resolved bundle: {source}") from exc
        relative = source.relative_to(self.root).as_posix()
        expected = self.expected.get(relative)
        if expected is None:
            raise ImportError(f"private bundle source is absent from integrity manifest: {relative}")
        return source, relative, expected, is_package

    def read_verified_source(self, source: Path, relative: str, expected: str) -> bytes:
        """Read source exactly once, then authenticate and execute those bytes."""
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"private bundle source is not a regular file: {relative}")
        try:
            resolved = source.resolve(strict=True)
            resolved.relative_to(self.laya_root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"unable to resolve private bundle source: {relative}") from exc
        return _read_authenticated_asset(self.root, relative, expected)

    def _registry_key(self, name: str) -> tuple[str, str]:
        return self.namespace, name

    def _module_is_registered(self, name: str, module: object) -> bool:
        return module is not None and _PRIVATE_LOADED.get(self._registry_key(name)) is module

    def _bind(self, module: types.ModuleType, name: str, kind: str, *,
              source: Path | None = None, relative: str | None = None,
              digest: str | None = None) -> None:
        setattr(module, _PRIVATE_TOKEN_ATTR, _PRIVATE_TOKEN)
        setattr(module, _PRIVATE_NAMESPACE_ATTR, self.namespace)
        setattr(module, _PRIVATE_BUNDLE_ATTR, str(self.bundle))
        setattr(module, _PRIVATE_KIND_ATTR, kind)
        if source is not None:
            module.__file__ = str(source)
            setattr(module, _PRIVATE_SOURCE_PATH_ATTR, str(source))
            setattr(module, _PRIVATE_SOURCE_REL_ATTR, relative)
            setattr(module, _PRIVATE_SOURCE_DIGEST_ATTR, digest)
        else:
            module.__package__ = name

    def bind_source(self, module: types.ModuleType, name: str, source: Path,
                    relative: str, digest: str) -> None:
        module.__package__ = name.rpartition(".")[0]
        self._bind(module, name, "source", source=source, relative=relative, digest=digest)

    def remember(self, module: types.ModuleType) -> None:
        name = module.__name__
        key = self._registry_key(name)
        previous = _PRIVATE_LOADED.get(key)
        if previous is not None and previous is not module and sys.modules.get(name) is previous:
            raise ImportError(f"private bundle module was replaced concurrently: {name}")
        if previous is None or previous is not module:
            self.new_modules.append(name)
        _PRIVATE_LOADED[key] = module

    def _module_matches(self, name: str, module: object, *,
                        kind: str, source: Path | None = None,
                        relative: str | None = None, digest: str | None = None) -> bool:
        if not isinstance(module, types.ModuleType):
            return False
        if getattr(module, _PRIVATE_TOKEN_ATTR, None) is not _PRIVATE_TOKEN:
            return False
        if getattr(module, "__name__", None) != name:
            return False
        if getattr(module, _PRIVATE_NAMESPACE_ATTR, None) != self.namespace:
            return False
        if getattr(module, _PRIVATE_BUNDLE_ATTR, None) != str(self.bundle):
            return False
        if getattr(module, _PRIVATE_KIND_ATTR, None) != kind:
            return False
        expected_package = name if kind in ("root", "package") else name.rpartition(".")[0]
        if getattr(module, "__package__", None) != expected_package:
            return False
        if not self._module_is_registered(name, module):
            return False
        if kind == "source":
            if (getattr(module, _PRIVATE_SOURCE_PATH_ATTR, None) != str(source)
                    or getattr(module, _PRIVATE_SOURCE_REL_ATTR, None) != relative
                    or getattr(module, _PRIVATE_SOURCE_DIGEST_ATTR, None) != digest):
                return False
            loader = getattr(module, "__loader__", None)
            if getattr(loader, _PRIVATE_LOADER_TOKEN_ATTR, None) is not _PRIVATE_TOKEN:
                return False
            try:
                origin = getattr(getattr(module, "__spec__", None), "origin", None)
                if origin is not None and Path(origin).resolve() != source.resolve():
                    return False
                if Path(getattr(module, "__file__", "")).resolve() != source.resolve():
                    return False
            except (OSError, TypeError, ValueError):
                return False
        elif (
            list(getattr(module, "__path__", ())) != []
            or (kind == "package"
                and getattr(module, "__file__", None) != str(self.laya_root / "__init__.py"))
        ):
            return False
        return True

    def validate_existing(self, name: str) -> types.ModuleType:
        """Validate a pre-existing private module or fail closed."""
        module = sys.modules.get(name)
        if name == self.namespace:
            if not self._module_matches(name, module, kind="root"):
                raise ImportError(f"unverified private module is already loaded: {name}")
            return module
        if name == self.package_name:
            if not self._module_matches(name, module, kind="package"):
                raise ImportError(f"unverified private module is already loaded: {name}")
            return module
        info = self.source_info(name)
        if info is None:
            raise ImportError(f"unverified private module is already loaded: {name}")
        source, relative, expected, _ = info
        if not self._module_matches(name, module, kind="source", source=source,
                                    relative=relative, digest=expected):
            raise ImportError(f"unverified private module is already loaded: {name}")
        # A verified module may be reused, but only while its source still
        # matches the manifest snapshot taken before verify_tree.
        self.read_verified_source(source, relative, expected)
        return module

    def reject_unverified_existing(self) -> None:
        for name in tuple(sys.modules):
            if self.is_private_name(name):
                self.validate_existing(name)

    def install_root_and_package(self) -> None:
        for name, kind in ((self.namespace, "root"), (self.package_name, "package")):
            if name in sys.modules:
                self.validate_existing(name)
                continue
            module = types.ModuleType(name)
            module.__path__ = []
            module.__package__ = name
            if kind == "package":
                module.__file__ = str(self.laya_root / "__init__.py")
            self._bind(module, name, kind)
            self.remember(module)
            sys.modules[name] = module

    def is_registered_common(self, module: object) -> bool:
        return (sys.modules.get(self.common_name) is module
                and self._module_is_registered(self.common_name, module))

    def cleanup_new_modules(self) -> None:
        for name in reversed(tuple(self.new_modules)):
            module = sys.modules.get(name)
            if module is not None and self._module_is_registered(name, module):
                del sys.modules[name]
            if self._module_is_registered(name, _PRIVATE_LOADED.get(self._registry_key(name))):
                del _PRIVATE_LOADED[self._registry_key(name)]


class _PrivateSourceLoader(importlib.abc.Loader):
    """Loader that authenticates bytes immediately before executing them."""

    def __init__(self, context: _PrivateBundleContext, fullname: str,
                 source: Path, relative: str, expected: str, is_package: bool):
        self.context = context
        self.fullname = fullname
        self.source = source
        self.relative = relative
        self.expected = expected
        self.is_package_flag = is_package
        setattr(self, _PRIVATE_LOADER_TOKEN_ATTR, _PRIVATE_TOKEN)

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__file__ = str(self.source)
        module.__package__ = spec.parent
        if self.is_package_flag:
            module.__path__ = []
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        if sys.modules.get(self.fullname) is not module:
            raise ImportError(f"private bundle module changed during import: {self.fullname}")
        data = self.context.read_verified_source(self.source, self.relative, self.expected)
        code = compile(data, str(self.source), "exec", dont_inherit=True)
        self.context.bind_source(module, self.fullname, self.source, self.relative, self.expected)
        exec(code, module.__dict__)
        self.context.remember(module)

    def get_filename(self, fullname: str | None = None) -> str:
        if fullname not in (None, self.fullname):
            raise ImportError(f"loader is scoped to {self.fullname}")
        return str(self.source)

    def is_package(self, fullname: str) -> bool:
        if fullname != self.fullname:
            raise ImportError(f"loader is scoped to {self.fullname}")
        return self.is_package_flag


class _PrivateBundleFinder(importlib.abc.MetaPathFinder):
    """Resolve only the active private package while it is on sys.meta_path."""

    def __init__(self, context: _PrivateBundleContext):
        self.context = context

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname in (self.context.namespace, self.context.package_name):
            return None
        info = self.context.source_info(fullname)
        if info is None:
            return None
        source, relative, expected, is_package = info
        loader = _PrivateSourceLoader(
            self.context, fullname, source, relative, expected, is_package
        )
        return importlib.util.spec_from_loader(
            fullname, loader, origin=str(source), is_package=is_package
        )


def _remove_finder(finder: _PrivateBundleFinder) -> bool:
    for index, candidate in enumerate(tuple(sys.meta_path)):
        if candidate is finder:
            del sys.meta_path[index]
            return True
    return False


def _load_bundled_laya(bundle: Path, *, manifest_snapshot: tuple[bytes, dict[str, str], str] | None = None):
    """Load one manifest-authenticated bundle without changing public imports."""
    bundle = Path(bundle).resolve()
    if manifest_snapshot is None:
        manifest_snapshot = _read_manifest_snapshot(bundle)
    _, expected, source_fingerprint = manifest_snapshot
    source = bundle / "upstream" / "laya" / "common.py"
    if not source.is_file():
        raise FileNotFoundError("bundled laya source is missing")
    context = _PrivateBundleContext(bundle, expected, source_fingerprint)
    finder = None
    succeeded = False
    with _PRIVATE_LOAD_LOCK:
        try:
            # The earlier implementation used only the common-file digest in
            # this name.  Treat any entry left at that deterministic legacy
            # name as untrusted too, rather than allowing a name collision to
            # bypass the source-set namespace below.
            common_relative = (bundle / "upstream/laya/common.py").relative_to(bundle.parent).as_posix()
            legacy_namespace = _legacy_private_namespace(bundle, expected[common_relative])
            if legacy_namespace != context.namespace:
                for name in tuple(sys.modules):
                    if isinstance(name, str) and (
                        name == legacy_namespace or name.startswith(legacy_namespace + ".")
                    ):
                        raise ImportError(f"unverified private module is already loaded: {name}")
            # This check happens before creating or importing anything.  It is
            # what makes a pre-populated exact private name fail closed.
            context.reject_unverified_existing()
            if context.common_name in sys.modules:
                module = context.validate_existing(context.common_name)
                succeeded = True
                return module

            context.install_root_and_package()
            finder = _PrivateBundleFinder(context)
            sys.meta_path.insert(0, finder)
            try:
                module = importlib.import_module(context.common_name)
            finally:
                _remove_finder(finder)
                finder = None
            if not context.is_registered_common(module):
                raise ImportError("private bundle common module was not verified")
            succeeded = True
            return module
        finally:
            if finder is not None:
                _remove_finder(finder)
            if not succeeded:
                context.cleanup_new_modules()


def _load_authenticated_tokenizer(root: Path, expected: Mapping[str, str]) -> object:
    """Build the pinned fast tokenizer from authenticated bytes only."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    tokenizer_data = _read_authenticated_asset(
        root,
        "bundle/tokenizer/tokenizer.json",
        _expected_asset_digest(expected, "bundle/tokenizer/tokenizer.json"),
    )
    tokenizer_config = _authenticated_json(
        root,
        "bundle/tokenizer/tokenizer_config.json",
        _expected_asset_digest(expected, "bundle/tokenizer/tokenizer_config.json"),
    )
    if not isinstance(tokenizer_config, dict):
        raise ValueError("tokenizer configuration must be a JSON object")
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    if tokenizer_class not in (None, "PreTrainedTokenizerFast"):
        raise ValueError(f"unsupported release tokenizer class: {tokenizer_class!r}")
    path_fields = {
        "vocab_file",
        "merges_file",
        "tokenizer_file",
        "added_tokens_file",
        "special_tokens_map_file",
    }
    if path_fields.intersection(tokenizer_config):
        raise ValueError("tokenizer configuration may not redirect authenticated loading to paths")
    tokenizer_config = dict(tokenizer_config)
    tokenizer_config.pop("tokenizer_class", None)
    try:
        backend = Tokenizer.from_str(tokenizer_data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("release tokenizer JSON is not valid UTF-8") from exc
    return PreTrainedTokenizerFast(tokenizer_object=backend, **tokenizer_config)


def _build_authenticated_model(laya, cfg: dict, root: Path, expected: Mapping[str, str]):
    """Construct the pinned Laya model without reopening encoder paths.

    This mirrors the pinned ``common.build_model`` construction while replacing
    its ``AutoConfig.from_pretrained(encoder_dir)`` read with the authenticated
    config object below.
    """
    from transformers import AutoConfig, AutoModel

    encoder_data = _authenticated_json(
        root,
        "bundle/encoder/config.json",
        _expected_asset_digest(expected, "bundle/encoder/config.json"),
    )
    if not isinstance(encoder_data, dict) or not isinstance(encoder_data.get("model_type"), str):
        raise ValueError("release encoder configuration is malformed")
    encoder_config = dict(encoder_data)
    model_type = encoder_config.pop("model_type")
    encoder = AutoModel.from_config(
        AutoConfig.for_model(model_type, **encoder_config),
        attn_implementation="sdpa",
    )
    try:
        model_class = laya.DecisionModel
    except AttributeError as exc:
        raise ValueError("bundled Laya model constructor is unavailable") from exc
    return model_class(
        encoder,
        cfg.get("head_layers", 2),
        len(cfg.get("act_costs", {})) + 1,
    )


def validate_input(record: object) -> dict:
    """Validate deployment input (state + one next_action choice question).

    Evaluation-only fields (gold/metadata) are neither needed nor accepted.
    """
    if not isinstance(record, Mapping) or set(record) != {"state", "questions"}:
        raise InputError("input must contain exactly state and questions")
    if not isinstance(record["state"], str) or not record["state"].strip():
        raise InputError("state must be a non-empty string")
    questions = record["questions"]
    if not isinstance(questions, Mapping) or set(questions) != {"next_action"}:
        raise InputError("questions must contain exactly next_action")
    question = questions["next_action"]
    if (not isinstance(question, Mapping) or not {"type", "criteria"} <= set(question)
            or set(question) - {"type", "criteria", "instructions"}
            or question.get("type") != "choice"
            or not isinstance(question.get("criteria"), Mapping)
            or tuple(question["criteria"].keys()) != CHOICE_ORDER):
        raise InputError("next_action must be choice with the frozen six-label option order")
    if question.get("instructions") != FROZEN_INSTRUCTIONS or dict(question["criteria"]) != FROZEN_CRITERIA:
        raise InputError("next_action instructions and six descriptions must match the frozen question exactly")
    return dict(record)


def validate_probabilities(values: Sequence[float]) -> dict[str, float]:
    if isinstance(values, (str, bytes)) or len(values) != len(LABELS):
        raise InputError("model must return six probabilities in LABELS order")
    probs = tuple(values)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in probs):
        raise InputError("probabilities must be finite, non-negative numbers")
    if not math.isclose(sum(probs), 1.0, abs_tol=2e-6, rel_tol=0):
        raise InputError("probabilities must sum to one")
    return dict(zip(LABELS, map(float, probs)))


def predict(record: object, predictor, *, calibrator=None) -> dict[str, float]:
    """Invoke supplied local predictor. Calibration is optional and external."""
    item = validate_input(record)
    raw = validate_probabilities(predictor(item))
    if calibrator is None:
        return raw
    return validate_probabilities(calibrator(tuple(raw[label] for label in LABELS)))


class LocalLayaPredictor:
    """Load the bundled seed-42 checkpoint using only bundle-relative assets."""
    def __init__(self, bundle: Path | None = None, device: str = "cpu"):
        self.bundle = (bundle or Path(__file__).parent / "bundle").resolve()
        if device != "cpu":
            raise ValueError("this release interface is CPU-only")
        with _trusted_dependency_imports(self.bundle.parent):
            self._initialize()

    def _initialize(self) -> None:
        import torch
        from safetensors.torch import load as load_safetensors

        # Capture the manifest before verification, then use this same
        # snapshot for all source and asset authentication.  Consumers receive
        # the bytes read below, never a later pathname read from the bundle.
        manifest_snapshot = _read_manifest_snapshot(self.bundle)
        verify_tree(self.bundle.parent)
        _assert_manifest_unchanged(self.bundle, manifest_snapshot)
        root = self.bundle.parent
        expected = manifest_snapshot[1]
        laya = _load_bundled_laya(self.bundle, manifest_snapshot=manifest_snapshot)
        QTYPES, build_sequence, render_options = (
            laya.QTYPES, laya.build_sequence, laya.render_options
        )

        self.torch = torch
        self.build_sequence = build_sequence
        self.render_options = render_options
        self.qtypes = QTYPES
        cfg = _authenticated_json(
            root,
            "bundle/rl_agent_config.json",
            _expected_asset_digest(expected, "bundle/rl_agent_config.json"),
        )
        exp = _authenticated_json(
            root,
            "bundle/experiment_config.json",
            _expected_asset_digest(expected, "bundle/experiment_config.json"),
        )
        calibration = _authenticated_json(
            root,
            "bundle/calibration.json",
            _expected_asset_digest(expected, "bundle/calibration.json"),
        )
        if not isinstance(cfg, dict) or not isinstance(exp, dict) or not isinstance(calibration, dict):
            raise ValueError("release configuration assets must contain JSON objects")
        cfg.update(max_len=exp["model"]["max_len"], head_max_len=exp["model"]["head_max_len"])
        weights = _read_authenticated_asset(
            root,
            "bundle/model/model.safetensors",
            _expected_asset_digest(expected, "bundle/model/model.safetensors"),
        )
        self.model = _build_authenticated_model(laya, cfg, root, expected)
        self.model.load_state_dict(load_safetensors(weights), strict=True)
        del weights
        self.model.encoder.config.reference_compile = False
        self.model.eval()
        self.tokenizer = _load_authenticated_tokenizer(root, expected)
        self.max_len = exp["model"]["max_len"]
        self.head_max_len = exp["model"]["head_max_len"]
        self.truncate_left = exp["model"]["truncate_left"]
        self.temperature = calibration["temperature"]
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("calibration temperature must be positive and finite")

    def __call__(self, record: Mapping) -> list[float]:
        raw, _ = self.predict_pair(record)
        return [raw[label] for label in LABELS]

    def predict_pair(self, record: Mapping) -> tuple[dict[str, float], dict[str, float]]:
        with _trusted_dependency_imports(self.bundle.parent):
            return self._predict_pair(record)

    def _predict_pair(self, record: Mapping) -> tuple[dict[str, float], dict[str, float]]:
        """One forward pass; raw and dev-temperature-scaled six-class outputs."""
        item = validate_input(record)
        q = item["questions"]["next_action"]
        internal = {"t": q["type"], "ins": q.get("instructions", ""), "crit": q["criteria"]}
        ids, markers = self.build_sequence(self.tokenizer, item["state"], internal,
                                            self.max_len, self.head_max_len,
                                            truncate_left=self.truncate_left)
        if len(markers) != len(self.render_options(internal)):
            raise InputError("choice markers exceed model head budget")
        # Single-record equivalent of Laya collate_items; no training/evaluation fields.
        t = self.torch
        input_ids = t.tensor([ids], dtype=t.long)
        attention = t.ones_like(input_ids)
        marker_pos = t.tensor([markers], dtype=t.long)
        marker_mask = t.ones_like(marker_pos, dtype=t.bool)
        qtype = t.tensor([self.qtypes["choice"]], dtype=t.long)
        with t.inference_mode():
            logits, _ = self.model(input_ids, attention, marker_pos, marker_mask, qtype)
            # Training criterion order is frozen CHOICE_ORDER.
            choice_logits = logits[0, :len(CHOICE_ORDER)].float()
            probs = t.softmax(choice_logits, dim=-1).tolist()
            calibrated = t.softmax(choice_logits / self.temperature, dim=-1).tolist()
        # Public mapping is fixed LABELS order, not choice criterion order.
        by_choice = dict(zip(CHOICE_ORDER, probs))
        by_calibrated_choice = dict(zip(CHOICE_ORDER, calibrated))
        return (validate_probabilities([by_choice[label] for label in LABELS]),
                validate_probabilities([by_calibrated_choice[label] for label in LABELS]))
