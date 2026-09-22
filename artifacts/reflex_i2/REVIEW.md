# Reflex Iteration 2 Independent Review

Date: 2026-09-21

## Overall status

**REPAIR**. The pinned specialist was actually exercised and the bounded runtime gates pass, but the shared validator has a material optional-gold validation hole, and the implementation explicitly retains a parallel Iteration-1 gold schema. The README also presents base-checkpoint measurements as the default specialist example.

## Scope and reproducibility

Reviewed `research/27_iteration_2_acceptance.md`, the working-tree v1/v2 diffs for Reflex and `shared/typed_decisions`, the Reflex implementation/tests/README, and all files under `artifacts/reflex_i2/`. The implementation revision under review is the uncommitted working tree; the last committed baseline is `9539fc7` (`docs: record iteration 1 outcomes`).

Relevant pinned revisions and evidence:

- Specialist: `convaiinnovations/laya-typed-decisions@f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`.
- Explicit base comparison: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Laya source: `data/laya` at `d113dca2512fb3eaca313534bc54c7162d87c1d4`; the checkout has only ignored generated `__pycache__` entries.
- Specialist config reports `model_name=laya-typed-decisions`, ModernBERT-large, `max_len=1024`; specialist weight SHA-256 is `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.
- Unchanged example SHA-256 matches `HEAD`: `f84e75f5eab11b8e254c9173f8eef1916ac1c8b4f1a7275328dbf1d306d412ad`.
- Primary evidence: `artifacts/reflex_i2/COMMANDS_RESULTS.md`, `IMPLEMENTATION_REPORT.md`, `base_output.json`, `specialist_output.json`, `specialist_benchmark.json`, `tests.txt`, and `compile.txt`.

Commands run by this review:

```text
git status --short
git log --oneline -10
git diff -- apps/reflex/reflex/core.py apps/reflex/tests/test_reflex.py shared/typed_decisions/schema.py apps/reflex/README.md
git diff -- apps/reflex/examples/agent_state.json
git show HEAD:apps/reflex/examples/agent_state.json | sha256sum
sha256sum apps/reflex/examples/agent_state.json
git -C data/laya rev-parse HEAD
git -C data/laya status --short --ignored --untracked-files=all
PYTHONPATH=apps/reflex .venv/bin/python -m unittest discover -s apps/reflex/tests -v
.venv/bin/python -m compileall -q apps/reflex/reflex shared/typed_decisions
PYTHONPATH=apps/reflex .venv/bin/python - <<'PY' ... parse two records with invalid questions and no gold ... PY
```

## Verified gates

- **Pin/source compatibility: VERIFIED for this environment.** `Reflex.from_checkpoint` resolves the exact specialist revision, loads it through the pinned Laya `laya.agent.Agent` path, and the recorded specialist inference exits 0 offline on `cuda:0`. The source commit and import-payload integrity gate were inspected. Cross-environment compatibility remains INFERRED/UNKNOWN.
- **One shared native contract: NOT VERIFIED.** Reflex delegates validation to `shared.typed_decisions`, but that shared validator deliberately accepts and preserves the bare Iteration-1 distribution form as a second gold representation. The unchanged example still uses that bare form. This is direct source evidence, not an inference.
- **Unchanged example semantics: VERIFIED.** The example hash is unchanged; both base and specialist consumed the same recorded input and both selected `publish` and `true`, with 449 input tokens. This demonstrates retained execution semantics, not accuracy.
- **Metadata exclusion: VERIFIED at the tested boundary.** The unit test records the exact state passed to `Agent.system_one` and confirms top-level `metadata` is absent; only state and questions are passed.
- **Same-input outputs: VERIFIED.** `base_output.json` and `specialist_output.json` both report `cuda:0`, 449 input tokens, and the same selected answers, while carrying their distinct pinned checkpoint IDs/revisions.
- **Warnings/docs/benchmark/errors/tests: PARTLY VERIFIED.** The runtime warning is present and stderr artifacts are empty; actionable-error tests, 18 no-weight tests, compileall, offline specialist inference, and the RTX 3060 benchmark pass. The benchmark records 3 warmups/10 samples and 40.150 ms min, 41.093 ms median, 41.893 ms p95/max excluding model load. README expected probabilities are the base output despite the default now being specialist, so documentation is not fully consistent.
- **Intended model: YES for the specialist runtime gate.** The specialist output and benchmark identify `convaiinnovations/laya-typed-decisions` at the exact requested revision, and the local snapshot checksum/config match the recorded specialist evidence. The unchanged example is a smoke/semantic test only; it is not an accuracy evaluation, and no positive-control claim is supported.

## Findings (severity ordered)

### Finding 1 — HIGH: optional gold bypasses question validation

#### FINDING

`shared/typed_decisions/schema.py:65-66` returns immediately when `gold` is absent, before validating question IDs, question objects, instructions, types, and criteria. The review command supplied records with invalid questions and no gold; both were **ACCEPTED**. Therefore `gold`-optional inference can pass malformed questions into the model, contrary to the shared validation-contract intent.

#### WHY

This is a runtime input-validation defect affecting the required inference path. The existing test `test_gold_is_optional_for_inference` checks only that gold is absent from the parsed result; it does not check that the questions remain validated. The 18 passing tests therefore do not cover this failure mode.

#### REQUIRED CHANGE

Validate every question independently of whether gold is present. Validate optional gold afterward as a subset of question IDs. Add a regression test for malformed type/instructions/criteria with no gold and ensure `Reflex` raises `ReflexInputError` before model invocation.

#### ACCEPTANCE TEST

With `gold` omitted, records containing an unsupported question type, empty instructions, and invalid criteria each fail validation; a valid gold-less record still reaches the fake agent. Run the full Reflex test suite and retain the command/output in the Reflex artifact record.

### Finding 2 — HIGH: parallel legacy gold schema remains in the shared native contract

#### FINDING

The shared validator explicitly retains a bare mapping such as `{"inspect": 1.0, "act": 0.0}` in `schema.py:77-81`, while also accepting native `{type, label, probabilities}`. The unchanged example and `valid_record()` use the bare form, and `test_accepts_native_contract` actually tests that legacy form. This is direct evidence that Reflex does not yet use one native gold shape exclusively.

#### WHY

Acceptance item 15 asks to remove or minimize private schema duplication and accept the shared native gold shape. Supporting two gold schemas preserves the old parallel representation and makes contract interoperability ambiguous. It also makes the test name and README claim (“native gold”) misleading for the example path.

#### REQUIRED CHANGE

Choose and document one policy: migrate the example/fixtures to native gold and reject the bare form, or explicitly define a shared, versioned compatibility adapter outside the native contract with a removal boundary. For this iteration’s “one shared native gold contract” criterion, the required repair is native `{type,label,probabilities}` in Reflex-facing records, with tests proving the bare form is rejected (or clearly quarantined as a separately named migration input).

#### ACCEPTANCE TEST

The unchanged-example requirement must be reconciled with the selected policy. A native-gold example must preserve the same state/questions and produce the same semantic reference labels; tests must cover native acceptance and bare-form behavior according to the documented policy. `grep`/diff review must show no Reflex-private alternate gold validator.

### Finding 3 — MEDIUM: README default-output example is for the base, not the default specialist

#### FINDING

The CLI default is `--checkpoint specialist`, but README lines 29–45 show base probabilities (`true=0.7523`, `publish=0.3424`). The recorded default specialist output is different (`true=0.5804`, `publish=0.3226`) in `artifacts/reflex_i2/specialist_output.json`.

#### WHY

Following the documented quickstart and comparing its expected shape to the default output gives a false result. This weakens reproducibility and obscures which model is being demonstrated, even though the actual specialist command and artifact are correct.

#### REQUIRED CHANGE

Label the README block as `--checkpoint base`, or replace it with the specialist output and separately label the base comparison. Keep checkpoint ID/revision in the documented expected output.

#### ACCEPTANCE TEST

Run the README quickstart offline with the default specialist and compare its selected answers/probabilities to the documented block; run the explicit base command and compare it to the separately labeled base block. Both must identify their exact revisions.

## Unknowns and limitations

- Accuracy, calibration, representative-domain performance, CPU latency, and long-state behavior were not measured; the general example is not a specialist evaluation.
- Cross-hardware/library bitwise stability and compatibility with source revisions other than `d113dca...` are UNKNOWN.
- No paid API, training, broad dataset search, or Iteration 3 work was evidenced in the reviewed Reflex artifacts.

## Reviewer conclusion

The model pin, source checkout, metadata boundary, unchanged-input smoke runs, warning/error behavior, tests, and bounded benchmark are credible and reproducible for the recorded environment. The review remains **REPAIR**, not ACCEPTED, until optional-gold records validate their questions, the native-vs-legacy gold policy is resolved, and the default-model documentation is corrected.
