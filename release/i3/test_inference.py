import contextlib
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    from . import inference as inference_module
    from .inference import (
        FROZEN_CRITERIA,
        FROZEN_INSTRUCTIONS,
        InputError,
        LABELS,
        predict,
        validate_input,
    )
except ImportError:  # direct execution from a copied standalone directory
    import inference as inference_module
    from inference import (
        FROZEN_CRITERIA,
        FROZEN_INSTRUCTIONS,
        InputError,
        LABELS,
        predict,
        validate_input,
    )


FIXTURE_ROOT = Path(__file__).parents[2] / "artifacts/review_60fd/import_fixtures"
_COMMON_TEMPLATE = """import builtins
from . import helper as _helper
from .helper import marker
builtins._i3_test_exec.append({label!r} + ':' + marker)
QTYPES={{'choice': {qtype}}}
def build_model(*a, **k): return builtins._i3_test_model()
def build_sequence(*a, **k): pass
def render_options(*a, **k): return []
"""


def item():
    return {"state": "context", "questions": {"next_action": {
        "type": "choice", "instructions": FROZEN_INSTRUCTIONS,
        "criteria": dict(FROZEN_CRITERIA)}}}


class InputContractTests(unittest.TestCase):
    def test_valid_input_and_label_mapping(self):
        self.assertEqual(validate_input(item())["state"], "context")
        result = predict(item(), lambda _: [1 / 6] * 6)
        self.assertEqual(tuple(result), LABELS)

    def test_rejects_extra_evaluation_fields(self):
        invalid = item() | {"gold": {}}
        with self.assertRaises(InputError):
            validate_input(invalid)

    def test_rejects_wrong_choice_order(self):
        bad = item()
        bad["questions"]["next_action"]["criteria"] = {"read": ""}
        with self.assertRaises(InputError):
            validate_input(bad)

    def test_rejects_nested_gold_and_nontext_descriptions(self):
        bad = item()
        bad["questions"]["next_action"]["gold"] = "edit"
        with self.assertRaises(InputError):
            validate_input(bad)
        bad = item()
        bad["questions"]["next_action"]["criteria"]["edit"] = "edit any unrelated thing"
        with self.assertRaises(InputError):
            validate_input(bad)
        bad = item()
        bad["questions"]["next_action"]["criteria"]["edit"] = {"metadata": 1}
        with self.assertRaises(InputError):
            validate_input(bad)

    def test_rejects_invalid_output_and_bad_calibration(self):
        with self.assertRaises(InputError):
            predict(item(), lambda _: [0.2] * 6)
        with self.assertRaises(InputError):
            predict(item(), lambda _: [1 / 6] * 6, calibrator=lambda _: [float("nan")] * 6)


class BundleImportIsolationTests(unittest.TestCase):
    """Exercise only synthetic files; no model weights or GPU are loaded."""

    def setUp(self):
        self.bundles = [FIXTURE_ROOT / name for name in ("bundle_one", "bundle_two")]
        self._write_fixture(self.bundles[0], "one-v1", 0, "first")
        self._write_fixture(self.bundles[1], "two", 1, "second")
        self._restore_after_test = []
        for root in self.bundles:
            snapshot = {
                path: path.read_bytes()
                for path in root.rglob("*")
                if path.is_file() and path.name != "MANIFEST.sha256.json"
            }
            manifest = (root / "MANIFEST.sha256.json").read_bytes()
            self._restore_after_test.append((root, snapshot, manifest))

    def tearDown(self):
        # Source and manifest patches are restored, while the durable ignored
        # fixture directory itself is intentionally retained.
        for root, snapshot, manifest in reversed(self._restore_after_test):
            for path, data in snapshot.items():
                path.write_bytes(data)
            (root / "MANIFEST.sha256.json").write_bytes(manifest)

    def test_rejects_prepopulated_exact_private_common_namespace(self):
        executions = []
        bundle = self.bundles[0] / "bundle"
        namespace = inference_module._private_namespace_for_bundle(bundle)
        private_name = namespace + ".laya.common"
        sentinel = types.ModuleType(private_name)
        sentinel.__file__ = str(FIXTURE_ROOT / "foreign_common_must_not_execute.py")
        sentinel.__system_a_i3_source_digest__ = json.loads(
            (self.bundles[0] / "MANIFEST.sha256.json").read_text(encoding="utf-8")
        )["files"]["bundle/upstream/laya/common.py"]
        sentinel.QTYPES = {"choice": 999}
        sentinel.build_model = lambda *a, **k: self.fail("sentinel common was executed")
        sentinel.build_sequence = lambda *a, **k: []
        sentinel.render_options = lambda *a, **k: []

        with self._synthetic_runtime(executions) as public_modules:
            sys.modules[private_name] = sentinel
            with self.assertRaises(ImportError):
                inference_module.LocalLayaPredictor(bundle)
            self.assertIs(sys.modules[private_name], sentinel)
            self.assertEqual(executions, [])
            self._assert_public_modules_unchanged(public_modules)

    def test_rejects_prepopulated_legacy_private_common_namespace(self):
        executions = []
        bundle = self.bundles[0] / "bundle"
        common_digest = hashlib.sha256(
            (bundle / "upstream/laya/common.py").read_bytes()
        ).hexdigest()
        legacy_namespace = "_system_a_i3_laya_" + hashlib.sha256(
            (str(bundle.resolve()) + ":" + common_digest).encode("utf-8")
        ).hexdigest()[:24]
        private_name = legacy_namespace + ".laya.common"
        sentinel = types.ModuleType(private_name)
        sentinel.__file__ = str(FIXTURE_ROOT / "foreign_legacy_common_must_not_execute.py")
        sentinel.QTYPES = {"choice": 999}
        sentinel.build_model = lambda *a, **k: self.fail("legacy sentinel common was executed")

        with self._synthetic_runtime(executions) as public_modules:
            sys.modules[private_name] = sentinel
            with self.assertRaises(ImportError):
                inference_module.LocalLayaPredictor(bundle)
            self.assertIs(sys.modules[private_name], sentinel)
            self.assertEqual(executions, [])
            self._assert_public_modules_unchanged(public_modules)

    def test_rejects_prepopulated_private_sibling_namespace(self):
        executions = []
        bundle = self.bundles[0] / "bundle"
        namespace = inference_module._private_namespace_for_bundle(bundle)
        private_name = namespace + ".laya.helper"
        sentinel = types.ModuleType(private_name)
        sentinel.__file__ = str(FIXTURE_ROOT / "foreign_helper_must_not_execute.py")
        sentinel.__system_a_i3_source_digest__ = json.loads(
            (self.bundles[0] / "MANIFEST.sha256.json").read_text(encoding="utf-8")
        )["files"]["bundle/upstream/laya/helper.py"]
        sentinel.marker = "foreign-sibling"

        with self._synthetic_runtime(executions) as public_modules:
            sys.modules[private_name] = sentinel
            with self.assertRaises(ImportError):
                inference_module.LocalLayaPredictor(bundle)
            self.assertIs(sys.modules[private_name], sentinel)
            self.assertEqual(executions, [])
            self._assert_public_modules_unchanged(public_modules)

    def test_rejects_post_verify_relative_helper_tamper_without_manifest_update(self):
        executions = []
        bundle = self.bundles[0] / "bundle"
        helper = bundle / "upstream/laya/helper.py"
        original_manifest = (self.bundles[0] / "MANIFEST.sha256.json").read_bytes()
        original_verify_tree = inference_module.verify_tree

        def verify_then_tamper(root):
            original_verify_tree(root)
            helper.write_text(
                "import builtins\n"
                "builtins._i3_test_exec.append('tampered-helper-executed')\n"
                "marker = 'tampered'\n",
                encoding="utf-8",
            )

        with self._synthetic_runtime(executions) as public_modules:
            meta_path_before = tuple(sys.meta_path)
            sys_path_before = tuple(sys.path)
            with patch.object(inference_module, "verify_tree", side_effect=verify_then_tamper):
                with self.assertRaises(ValueError):
                    inference_module.LocalLayaPredictor(bundle)
            self.assertEqual(executions, [])
            self.assertEqual((self.bundles[0] / "MANIFEST.sha256.json").read_bytes(), original_manifest)
            self.assertEqual(tuple(sys.meta_path), meta_path_before)
            self.assertEqual(tuple(sys.path), sys_path_before)
            self._assert_public_modules_unchanged(public_modules)

    def test_rejects_post_verify_common_tamper_without_manifest_update(self):
        executions = []
        bundle = self.bundles[0] / "bundle"
        common = bundle / "upstream/laya/common.py"
        original_manifest = (self.bundles[0] / "MANIFEST.sha256.json").read_bytes()
        original_verify_tree = inference_module.verify_tree

        def verify_then_tamper(root):
            original_verify_tree(root)
            common.write_text(
                "import builtins\n"
                "builtins._i3_test_exec.append('tampered-common-executed')\n",
                encoding="utf-8",
            )

        with self._synthetic_runtime(executions) as public_modules:
            with patch.object(inference_module, "verify_tree", side_effect=verify_then_tamper):
                with self.assertRaises(ValueError):
                    inference_module.LocalLayaPredictor(bundle)
            self.assertEqual(executions, [])
            self.assertEqual((self.bundles[0] / "MANIFEST.sha256.json").read_bytes(), original_manifest)
            self._assert_public_modules_unchanged(public_modules)

    def test_two_bundles_repeat_and_regenerated_source_changes_load_separately(self):
        executions = []
        with self._synthetic_runtime(executions) as public_modules:
            meta_path_before = tuple(sys.meta_path)
            sys_path_before = tuple(sys.path)
            first = inference_module.LocalLayaPredictor(self.bundles[0] / "bundle")
            first_common = sys.modules[
                inference_module._private_namespace_for_bundle(self.bundles[0] / "bundle") + ".laya.common"
            ]
            expected_common = json.loads(
                (self.bundles[0] / "MANIFEST.sha256.json").read_text(encoding="utf-8")
            )["files"]["bundle/upstream/laya/common.py"]
            self.assertEqual(
                getattr(first_common, "__system_a_i3_source_digest__"), expected_common
            )
            self.assertEqual(
                Path(first_common.__file__),
                (self.bundles[0] / "bundle/upstream/laya/common.py").resolve(),
            )
            first_again = inference_module.LocalLayaPredictor(self.bundles[0] / "bundle")
            self.assertIs(first_again.build_sequence, first.build_sequence)
            self.assertEqual(executions, ["one-v1:first"])
            second = inference_module.LocalLayaPredictor(self.bundles[1] / "bundle")
            self.assertEqual(executions, ["one-v1:first", "two:second"])
            self.assertEqual(first.qtypes["choice"], 0)
            self.assertEqual(second.qtypes["choice"], 1)
            self._assert_public_modules_unchanged(public_modules)

            common = self.bundles[0] / "bundle/upstream/laya/common.py"
            common.write_text(
                _COMMON_TEMPLATE.format(label="one-v2", qtype=2),
                encoding="utf-8",
            )
            self._write_manifest(self.bundles[0])
            changed = inference_module.LocalLayaPredictor(self.bundles[0] / "bundle")
            changed_namespace = inference_module._private_namespace_for_bundle(self.bundles[0] / "bundle")
            self.assertNotEqual(changed_namespace + ".laya.common", first_common.__name__)
            self.assertIs(sys.modules[first_common.__name__], first_common)
            self.assertEqual(executions[-1], "one-v2:first")
            self.assertEqual(changed.qtypes["choice"], 2)

            helper = self.bundles[0] / "bundle/upstream/laya/helper.py"
            helper.write_text("marker = 'changed-sibling'\n", encoding="utf-8")
            self._write_manifest(self.bundles[0])
            sibling_changed = inference_module.LocalLayaPredictor(self.bundles[0] / "bundle")
            sibling_namespace = inference_module._private_namespace_for_bundle(self.bundles[0] / "bundle")
            self.assertNotEqual(sibling_namespace, changed_namespace)
            self.assertEqual(executions[-1], "one-v2:changed-sibling")
            self.assertIsNotNone(sibling_changed)
            self.assertEqual(tuple(sys.meta_path), meta_path_before)
            self.assertEqual(tuple(sys.path), sys_path_before)
            self._assert_public_modules_unchanged(public_modules)

    @contextlib.contextmanager
    def _synthetic_runtime(self, executions):
        import builtins

        model = Mock()
        model.encoder.config = SimpleNamespace()

        torch_stub = types.ModuleType("torch")
        safetensors_stub = types.ModuleType("safetensors")
        safetensors_torch_stub = types.ModuleType("safetensors.torch")
        safetensors_torch_stub.load_file = lambda *a, **k: {}
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoTokenizer = SimpleNamespace(
            from_pretrained=lambda *a, **k: object()
        )

        foreign_common = types.ModuleType("laya.common")
        foreign_common.__file__ = str(FIXTURE_ROOT / "foreign_common_must_not_execute.py")
        foreign_common.QTYPES = {"choice": 999}
        foreign_common.build_model = lambda *a, **k: self.fail("public laya.common was executed")
        foreign_common.build_sequence = lambda *a, **k: []
        foreign_common.render_options = lambda *a, **k: []
        laya_sentinel = types.ModuleType("laya")
        foreign_helper = types.ModuleType("laya.helper")
        foreign_helper.marker = "public-foreign"

        public_modules = {
            "laya": laya_sentinel,
            "laya.common": foreign_common,
            "laya.helper": foreign_helper,
            "torch": torch_stub,
            "safetensors": safetensors_stub,
            "safetensors.torch": safetensors_torch_stub,
            "transformers": transformers_stub,
        }
        private_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name.startswith("_system_a_i3_laya_")
        }
        private_registry = dict(inference_module._PRIVATE_LOADED)
        try:
            with patch.dict(sys.modules, public_modules, clear=False):
                for name in private_modules:
                    sys.modules.pop(name, None)
                inference_module._PRIVATE_LOADED.clear()
                with patch.object(builtins, "_i3_test_exec", executions, create=True), \
                     patch.object(builtins, "_i3_test_model", lambda: model, create=True):
                    yield public_modules
        finally:
            for name in tuple(sys.modules):
                if name.startswith("_system_a_i3_laya_"):
                    sys.modules.pop(name, None)
            sys.modules.update(private_modules)
            inference_module._PRIVATE_LOADED.clear()
            inference_module._PRIVATE_LOADED.update(private_registry)

    @staticmethod
    def _assert_public_modules_unchanged(public_modules):
        for name, module in public_modules.items():
            # The runtime patch owns these objects; checking identity catches
            # accidental replacement or deletion without depending on attributes.
            if sys.modules.get(name) is not module:
                raise AssertionError(f"public module changed: {name}")

    @classmethod
    def _write_fixture(cls, root, label, qtype, marker):
        bundle = root / "bundle"
        (bundle / "upstream/laya").mkdir(parents=True, exist_ok=True)
        (bundle / "model").mkdir(parents=True, exist_ok=True)
        (bundle / "encoder").mkdir(parents=True, exist_ok=True)
        (bundle / "tokenizer").mkdir(parents=True, exist_ok=True)
        (bundle / "upstream/laya/common.py").write_text(
            _COMMON_TEMPLATE.format(label=label, qtype=qtype), encoding="utf-8"
        )
        (bundle / "upstream/laya/helper.py").write_text(
            f"marker = {marker!r}\n", encoding="utf-8"
        )
        (bundle / "model/model.safetensors").write_bytes(b"synthetic weights")
        (bundle / "rl_agent_config.json").write_text("{}", encoding="utf-8")
        (bundle / "experiment_config.json").write_text(
            json.dumps({"model": {"max_len": 8, "head_max_len": 4, "truncate_left": True}}),
            encoding="utf-8",
        )
        (bundle / "calibration.json").write_text('{"temperature":1.0}', encoding="utf-8")
        cls._write_manifest(root)

    @staticmethod
    def _write_manifest(root):
        files = {}
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and path.name != "MANIFEST.sha256.json"
                and "__pycache__" not in path.parts
                and path.suffix not in (".pyc", ".pyo")
            ):
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "MANIFEST.sha256.json").write_text(
            json.dumps(
                {"schema_version": "system-a-i3-bundle-sha256-v1", "files": files},
                sort_keys=True,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
