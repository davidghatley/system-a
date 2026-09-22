# Reflex Replay Final Verification

Date: 2026-09-22

## Scope

Independent CPU/offline review of `apps/reflex_replay/`, `REVIEW.md`, and
`REPAIR_1.md`. No application implementation was edited. No model was loaded,
no network or GPU was used, and no final-test evaluation was performed. Only
the three frozen `dev.jsonl` cases were replayed.

## Findings

- **Verified:** The accepted stored result exactly matches
  `artifacts/iteration_2/integration_result.json` for record ID, observed
  reference, model ID/revision, all six probabilities, selected label, and
  recorded load-plus-inference latency. The evidence file's measured SHA-256 is
  the stored provenance hash.
- **Verified:** The suite binds each case to a raw source line SHA-256, record
  ID, observed label, and canonical compact-state SHA-256. Altering the frozen
  source row fails during suite loading. Altering any bound artifact-identity
  field fails before prediction evaluation.
- **Verified:** The envelope digest covers the complete prediction payload,
  including record/reference, model, probabilities, latency and scope, and
  provenance. Stale-digest changes to each category fail closed. Missing or
  invalid model, distribution, latency, reference, identity, algorithm, or
  digest produces `ReplayError`; CLI input errors exit 2.
- **Verified:** Eight tests pass. The repaired regression tests directly cover
  probabilities, model, reference, latency, and compact-source identity.
- **Integrity, not authenticity:** The current checked-in files form a
  consistent hash-bound snapshot and detect edits made without updating the
  digest. The SHA-256 envelope is unkeyed and stored beside its payload. A
  capable editor can substitute a schema-valid model/probability/latency payload
  and recompute the digest; replay accepts it. The mechanism therefore does not
  identify the producer, prove that inference occurred, or protect against an
  editor able to rewrite both payload and digest. Acceptance here relies on the
  independently measured checked-in Iteration 2 evidence hash, not on a
  signature or external trust anchor.
- **Unknown:** physical source execution, teacher identity, action optimality,
  task success, source/model overlap, calibration, and broader predictive
  quality remain unverified. The three selected dev cases are diagnostic only.

## Commands And Results

All commands ran from repository root with CPU/offline environment controls.

```bash
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay python -m unittest discover -s apps/reflex_replay/tests -v
```

Result: exit 0; `Ran 8 tests in 0.150s`; `OK`.

```bash
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay python -m reflex_replay --case accepted-integration
```

Result: exit 0; one frozen dev case. Majority predicted `execute`; specialist
`convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`
predicted `other_tool`; observed reference was `search`; both were wrong. The
specialist margin was `0.0216` and stored elapsed value was `6.338613s` with
scope `model load plus inference in accepted Iteration 2 integration run`.
Replay elapsed was `0.003493s`.

```bash
sha256sum "artifacts/iteration_2/integration_result.json" "artifacts/trace2decision_i2/output/dev.jsonl" "apps/reflex_replay/examples/stored_predictions.jsonl" "apps/reflex_replay/examples/frozen_suite.json"
```

Result:

```text
07f4f1efad39116a13687965800b718045ab598105d844e8fd7f81457e6b42fb  artifacts/iteration_2/integration_result.json
eed09d564f1ae17cabb2ef47bec8ae94aaa08c41ccd6da692149f14b0b47a540  artifacts/trace2decision_i2/output/dev.jsonl
f4a3ecb6f4b807f2ab7d50b6c076e78424c1c71284dcf145857ba49e1ef83024  apps/reflex_replay/examples/stored_predictions.jsonl
2c1be9ecf9d12c46df159dc033e9d0ae58169e52f1054542cd9be7d71542c74e  apps/reflex_replay/examples/frozen_suite.json
```

```bash
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay python - <<'PY'
import copy
import hashlib
import json
import pathlib
from unittest.mock import patch

from reflex_replay.core import ReplayError, _canonical, _prediction_body, load_predictions, load_suite, replay

root = pathlib.Path.cwd()
app = root / "apps/reflex_replay"
suite_path = app / "examples/frozen_suite.json"
pred_path = app / "examples/stored_predictions.jsonl"
evidence_path = root / "artifacts/iteration_2/integration_result.json"
cases = load_suite(root, suite_path)
stored = load_predictions(pred_path)
base = stored[0]
evidence_bytes = evidence_path.read_bytes()
evidence = json.loads(evidence_bytes)
reflex = evidence["reflex"]
assert hashlib.sha256(evidence_bytes).hexdigest() == base["provenance"]["source_sha256"]
assert base["record_id"] == evidence["stable_metadata"]["id"]
assert base["reference"] == reflex["gold_label"] == cases[0]["observed_label"]
assert base["model"] == reflex["checkpoint"]
assert base["probabilities"] == reflex["probabilities"]
assert base["elapsed_seconds"] == reflex["load_and_inference_seconds"]
assert replay(cases, stored)[0]["predictions"][1]["selected"] == reflex["selected"]
print("accepted-evidence binding: PASS")

mutations = {
    "record_id": lambda p: p.__setitem__("record_id", cases[1]["record_id"]),
    "reference": lambda p: p.__setitem__("reference", "execute"),
    "model.id": lambda p: p["model"].__setitem__("id", "substitute"),
    "model.revision": lambda p: p["model"].__setitem__("revision", "substitute"),
    "probabilities": lambda p: p["probabilities"].__setitem__("search", p["probabilities"]["search"] + 0.001),
    "elapsed_seconds": lambda p: p.__setitem__("elapsed_seconds", 99.0),
    "elapsed_scope": lambda p: p.__setitem__("elapsed_scope", "substitute"),
    "provenance.source": lambda p: p["provenance"].__setitem__("source", "substitute.json"),
    "provenance.source_sha256": lambda p: p["provenance"].__setitem__("source_sha256", "0" * 64),
    "artifact.source": lambda p: p["integrity"]["artifact"].__setitem__("source", "substitute.jsonl"),
    "artifact.source_line": lambda p: p["integrity"]["artifact"].__setitem__("source_line", 2),
    "artifact.source_line_sha256": lambda p: p["integrity"]["artifact"].__setitem__("source_line_sha256", "0" * 64),
    "artifact.compact_source_sha256": lambda p: p["integrity"]["artifact"].__setitem__("compact_source_sha256", "0" * 64),
    "artifact.record_id": lambda p: p["integrity"]["artifact"].__setitem__("record_id", "substitute"),
    "artifact.observed_label": lambda p: p["integrity"]["artifact"].__setitem__("observed_label", "execute"),
}
for name, mutate in mutations.items():
    item = copy.deepcopy(base)
    mutate(item)
    try:
        replay(cases, [item])
    except ReplayError:
        print(f"stale-digest tamper {name}: REJECTED")
    else:
        raise AssertionError(f"tamper accepted: {name}")

original_read_text = pathlib.Path.read_text
def altered_read_text(path, *args, **kwargs):
    text = original_read_text(path, *args, **kwargs)
    if path == root / cases[0]["source"]:
        lines = text.splitlines()
        row = json.loads(lines[0])
        row["state"] += " TAMPER"
        lines[0] = json.dumps(row, separators=(",", ":"))
        return "\n".join(lines) + "\n"
    return text
with patch.object(pathlib.Path, "read_text", altered_read_text):
    try:
        load_suite(root, suite_path)
    except ReplayError:
        print("frozen source-row tamper: REJECTED")
    else:
        raise AssertionError("source-row tamper accepted")

for name, mutate in {
    "invalid model": lambda p: p["model"].__setitem__("revision", ""),
    "invalid probability": lambda p: p["probabilities"].update({"edit": -0.1, "execute": p["probabilities"]["execute"] + 0.2685168516851685}),
    "invalid latency": lambda p: p.__setitem__("elapsed_seconds", -1),
    "wrong reference": lambda p: p.__setitem__("reference", "execute"),
}.items():
    item = copy.deepcopy(base)
    mutate(item)
    artifact = item["integrity"]["artifact"]
    item["integrity"]["sha256"] = hashlib.sha256(_canonical(_prediction_body(item, artifact))).hexdigest()
    try:
        replay(cases, [item])
    except ReplayError:
        print(f"recomputed-digest {name}: REJECTED")
    else:
        raise AssertionError(f"invalid recomputed payload accepted: {name}")

forged = copy.deepcopy(base)
forged["model"]["revision"] = "attacker-substitute"
forged["probabilities"] = {"edit": 1.0, "execute": 0.0, "other_tool": 0.0, "read": 0.0, "respond_or_finish": 0.0, "search": 0.0}
forged["elapsed_seconds"] = 0.0
artifact = forged["integrity"]["artifact"]
forged["integrity"]["sha256"] = hashlib.sha256(_canonical(_prediction_body(forged, artifact))).hexdigest()
result = replay(cases, [forged])[0]["predictions"][1]
assert result["model"]["revision"] == "attacker-substitute" and result["selected"] == "edit"
print("valid forged payload with recomputed unkeyed digest: ACCEPTED (no authenticity)")
PY
```

Result: exit 0. Accepted-evidence binding passed; all 15 stale-digest field
tamper cases, the frozen source-row tamper, and four schema/reference attacks
with recomputed digests were rejected. A schema-valid substituted payload with
a recomputed unkeyed digest was accepted, confirming the stated authenticity
boundary.

```bash
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=apps/reflex_replay python - <<'PY'
import copy
import pathlib
from reflex_replay.core import ReplayError, load_predictions, load_suite, replay
root = pathlib.Path.cwd()
cases = load_suite(root, root / 'apps/reflex_replay/examples/frozen_suite.json')
base = load_predictions(root / 'apps/reflex_replay/examples/stored_predictions.jsonl')[0]
for name, mutate in {
    'missing integrity': lambda p: p.pop('integrity'),
    'algorithm': lambda p: p['integrity'].__setitem__('algorithm', 'MD5'),
    'canonicalization': lambda p: p['integrity'].__setitem__('canonicalization', 'other'),
    'digest': lambda p: p['integrity'].__setitem__('sha256', '0' * 64),
}.items():
    item = copy.deepcopy(base)
    mutate(item)
    try:
        replay(cases, [item])
    except ReplayError:
        print(f'{name}: REJECTED')
    else:
        raise AssertionError(f'{name}: accepted')
PY
```

Result: exit 0; missing integrity envelope, unsupported algorithm, unsupported
canonicalization, and incorrect digest were each rejected.

```bash
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 python -m compileall -q apps/reflex_replay
git diff --check
```

Result: both exited 0 with no output.

ACCEPTED
