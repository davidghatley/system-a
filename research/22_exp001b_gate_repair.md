# exp_001b: minimal source-level gate repair

Date: 2026-09-20. **Static review only; no implementation or execution.** Inspected `experiments/exp_001b_hermes_preprocessing/protocol.json`, its `scripts/build_dataset.py`, and root `test_gates.py` / `test_laya_fidelity.py` using file reads. No shell commands, external research, tokenizer/model execution, or measured dataset results. References below are builder line numbers unless stated otherwise.

## Verified defects

- Canary checks (266–300) append text to rendered state or the included prior call, then count unchanged truncated tokens as failures. This tests neither excluded-source-field isolation nor reasoning removal; `\nreason` is not a structural reasoning span.
- Sequence construction truncates and slices to 512 (85–89), making the later `len > 512` check ineffective (263). Reported budget exclusions and resulting bias gates cannot establish the frozen no-truncation requirement.
- Regex extraction can ignore incomplete call syntax and classify a malformed target as the empty set (245–252). Prior linkage compares names/counts but does not establish complete wrapper coverage or payload validity; missing names can compare equal (230–244).
- `test_gates.py` ends with an unterminated string (98), uses external temporary-file defaults, and prints rather than asserts. `test_laya_fidelity.py` tests a copied implementation with one short state and print-only comparisons; its suffix slice contradicts its keep-start comment (52). Neither establishes production fail-closed behavior.

## Minimal source-level contract and renderer allowlist

Test the actual source-row → validation/stripping → candidate projection → rendering → six-sequence path. Keep the fixture's candidate position fixed; do not inject canaries after rendering.

**Explicit model-input allowlist:** source `task`; the immediately preceding validated adjacent `gpt` call message's reasoning-stripped `value`; the adjacent `tool` result message's reasoning-stripped `value`; fixed ordered tool names `patch, process, read_file, search_files, terminal, write_file`; fixed section headings, six noul questions, no/yes option wording/order, and pinned tokenizer special-token framing. This retains the current full stripped prior-message representation (171–174, 255), not a new payload-only projection. Roles and source tool definitions validate eligibility; no arbitrary row/dictionary serialization is allowed.

**Target isolation:** mutate valid tool-call names/set in the next source assistant message. Labels must change to the expected six-bit bundle (duplicates collapsed; control mode derived from emptiness), while rendered state, all six token arrays, option-marker positions, lengths, and token-state hash remain exactly equal. Valid changes to target prose/arguments must likewise leave inputs equal. Malformed or unavailable target calls must be rejected, never converted to negative labels.

**Excluded metadata:** source ID, category, provenance/attribution, auxiliary annotations and extra fields, derived split/group bookkeeping, labels/control mode, target-turn content, and nonselected conversation content are excluded from model inputs. Mutate actual metadata fields independently, including adding unknown fields; inputs and labels must remain equal for otherwise valid fixtures. IDs/category and bookkeeping may legitimately change audit records. Tool definitions and roles remain validation inputs; task remains both a rendering and grouping input, so neither is an unconditional metadata-invariance mutation. Nonselected-message changes must still satisfy whole-trajectory reasoning validation.

**Structural reasoning invariance:** insert or replace balanced, nonnested `<think>…</think>` spans in source conversation values before stripping, preserving all outside bytes. Include target and prior-pair fixtures and fake tool-call markup inside the span. Valid fixtures must retain identical eligibility, labels, state and six sequences. Unclosed, orphan-closing or nested tags reject the entire trajectory; residual supported marker forms in rendered content block acceptance. The source `task` is not currently stripped: a marker there must fail the residual-marker gate rather than silently pass.

## Required fail-closed tests; preserve science

- Assertions must exercise production helpers/pipeline and exit nonzero on mismatch. Included-field positive controls must visibly change state/tokens; otherwise an inert renderer could pass every isolation test.
- Reject incomplete/nested/stray call/result wrappers, unparsable or invalid payloads, unknown target tools, count/name mismatch, and nonadjacent pairs. Revalidate retained pairs; no malformed syntax may masquerade as a valid no-call target. Current linkage gate also fails on any counted linkage exclusion: retain that stricter behavior unless explicitly revised.
- Measure complete untruncated sequences for all six questions. Accept exactly 512 only with all required content intact; exclude any candidate exceeding 512 for any question. Assert state/token hashes, markers, complete content and pinned Laya representation parity, including long-state fixtures that expose hidden truncation.
- Preserve source checksum, zero leakage/markers, exact token deduplication with deterministic first-row selection, trajectory/proxy isolation, **≥10 positive trajectories per tool per split**, **≤10 percentage-point split budget-rate difference**, and **≤20 percentage-point per-tool positive/negative difference**. Missing support, empty denominators or empty output must not yield a vacuous PASS. Recompute bias/support only after genuine budget exclusion. No tool removal, threshold relaxation, truncation, training, or official evaluation; scientific changes require an explicit protocol revision.

## Stale-output risk and acceptance evidence

PASS-only split files and ID manifests (388–399) survive a subsequent FAIL; early exceptions can leave an old PASS profile/checksums. Sequential overwrites can mix runs, and `distribution([])` can abort before fresh disposition is written. These are code-established risks; existing artifact freshness was not inspected.

A future repair should retain evidence in distinct durable run directories and make downstream consumption require a completed, all-gates-PASS manifest binding every output checksum to the source, protocol, builder and tokenizer revisions. Interrupted/failed runs must never fall back to prior artifacts. Record exclusion reasons, explicit trajectory/proxy split manifests and per-canary assertions; a log or existing split filename alone is insufficient.

Pinned provenance in reviewed files: Hermes revision `b92885e4f0161d4b2536512710e004d4892cac6e`, shard SHA-256 `d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02`; Laya tokenizer revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. These values were read, not independently verified. **Subset viability and repaired gate outcomes remain unknown.**
