# exp_001b Build Dataset Review

**Protocol**: `experiments/exp_001b_hermes_preprocessing/protocol.json` (FROZEN_FOR_PREPROCESSING_ONLY, 2026-09-20)
**Laya Source**: `data/laya` @ `d113dca2512fb3eaca313534bc54c7162d87c1d4`
**Build Script**: `experiments/exp_001b_hermes_preprocessing/scripts/build_dataset.py`

---

## 1. Laya Representation Fidelity

**Status: RECOMMENDED (falsified)**

The build script's `render()` + `tokenizer.encode(state, add_special_tokens=True)` **does not match** Laya's `build_sequence()` for six noul questions.

### Laya `build_sequence()` (data/laya/laya/common.py:49-86)
```python
# Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
for o in opt_ids:
    markers.append(len(ids))
    ids.extend(o)          # each opt_ids[i] = [MASK] + " " + opt_text[:48]
ids.append(tok.sep_token_id)
room = max(0, max_len - len(ids) - 1)
st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
ids = ids + st + [tok.sep_token_id]
```
- Creates **one sequence per question** (6 sequences for 6 noul questions)
- Each sequence: `[CLS] noul question: <ins> [SEP] [MASK] false: ... [MASK] true: ... [SEP] state [SEP]`
- Uses `[MASK]` token (50284) as option delimiters
- Truncates state to fit `max_len` (512) with `head_max_len` (192) budget for question+options

### Build Script `render()` (build_dataset.py:109-111, 191)
```python
def render(task, prior_call, prior_result):
    return (f"Task:\n{task}\nAvailable tools: {', '.join(TOOLS)}\n"
            f"Previous assistant call:\n{prior_call}\nLatest tool response:\n{prior_result}\n{QUESTION_BLOCK}")
# QUESTION_BLOCK = "Question call_patch: no | yes\nQuestion call_process: no | yes\n..."
token_ids = tokenizer.encode(state, add_special_tokens=True)  # only adds [CLS]...[SEP]
```
- Creates **one combined sequence** for all 6 questions
- No `[MASK]` tokens between options
- No per-question structure; questions rendered as plain text lines
- `add_special_tokens=True` only wraps with `[CLS]` + `[SEP]`

**Conclusion**: The representation is fundamentally incompatible with Laya's expected input format. The build script produces a single flat sequence; Laya expects six structured sequences with mask-delimited options.

---

## 2. Gzip Bug (`gzip.open(..., mtime=0)`)

**Status: VERIFIED (bug confirmed)**

`gzip.open()` in Python's standard library **does not accept `mtime` parameter**.

```python
# build_dataset.py:286, 294
with gzip.open(output, "wt", encoding="utf-8", newline="\n", mtime=0) as handle:
```

```bash
$ python3 -c "import gzip; gzip.open('/tmp/test.gz', 'wt', mtime=0)"
TypeError: open() got an unexpected keyword argument 'mtime'
```

The `mtime` parameter exists on `gzip.GzipFile` constructor, not `gzip.open()`. All four `gzip.open` calls (lines 286, 294, and two more for model splits) will raise `TypeError` and crash the script.

---

## 3. Canary Implementation (lines 193-200)

**Status: RECOMMENDED (implementation does not test claimed property)**

```python
canary_record = dict(source_row)
for field in ("id", "category", "subcategory", "tools"):
    canary_record[field] = f"EXCLUDED_METADATA_CANARY_{field}"
canary_labels = ["TARGET_CANARY"]
canary_state = render(source_row["task"], prior_call["value"], prior_result["value"])
if tokenizer.encode(canary_state, add_special_tokens=True) != token_ids or canary_labels == labels:
    canary_failures += 1
```

**Issues**:
1. `canary_state` is rendered from **identical inputs** (`task`, `prior_call`, `prior_result`) as the real state → `tokenizer.encode(canary_state)` **always equals** `token_ids` (first condition always `False`)
2. `canary_labels = ["TARGET_CANARY"]` is compared to actual `labels` (e.g., `["patch", "terminal"]`) → second condition `canary_labels == labels` is `False` unless labels happen to be exactly `["TARGET_CANARY"]`
3. The canary **never mutates rendered output** because excluded metadata fields are never passed to `render()`
4. The test validates that metadata doesn't affect output (true by construction), but **does not test** whether target labels or metadata changes would leak into tokens

**Protocol requirement** (stop_gate #3): "any target or excluded-metadata canary reaches rendered tokens" — the canary as written cannot detect this.

---

## 4. Stop Gate Behavior on FAIL

**Status: VERIFIED (writes artifacts and exits 0 on FAIL)**

```python
# Lines 283-293: Model-ready splits written UNCONDITIONALLY
for split in SPLITS:
    output = DATA_OUT / f"{split}.jsonl.gz"
    with gzip.open(output, "wt", encoding="utf-8", newline="\n", mtime=0) as handle:
        for item in retained:
            if item["split"] == split:
                handle.write(...)
    model_paths[split] = output
    ids = sorted({item["trajectory_id"] for item in retained if item["split"] == split})
    (RESULTS / f"{split}_ids.txt").write_text(...)

# Lines 294-337: Ledger, profile, bias, checksums, logs ALL written
# Line 341: Script exits normally (no sys.exit(1))
```

**Disposition** (line 281): `disposition = "PASS" if all(gates.values()) else "FAIL"` — but no gating on artifact emission.

**Protocol expectation**: FAIL should mean "Hermes candidate is rejected or the subset/protocol must become a new explicit revision" — not "write model-ready splits anyway."

---

## 5. All 12 Stop Gates from protocol.json

**Status: PARTIAL (8/12 implemented as hard assertions; 4 missing or incorrect)**

| # | Protocol Stop Gate | Implementation (gates dict, lines 268-280) | Status |
|---|---------------------|--------------------------------------------|--------|
| 1 | source checksum mismatch | `"source_checksum": source_sha == SOURCE_SHA256` | ✅ VERIFIED |
| 2 | any residual reasoning marker | `"zero_residual_reasoning_markers": marker_rows == 0` | ✅ VERIFIED |
| 3 | any target/excluded-metadata canary reaches tokens | `"target_and_metadata_canary_invariance": canary_failures == 0` | ❌ **BROKEN** (canary never mutates output) |
| 4 | incomplete/nonadjacent/unparsable/mismatched call-result | `"complete_adjacent_linkage": counts["linkage_exclusions"] == 0` | ✅ VERIFIED |
| 5 | unavailable/malformed retained target label | `"valid_retained_labels": all(set(item["labels"]) <= set(TOOLS) ...)` | ✅ VERIFIED |
| 6 | retained state exceeds 512 tokens or loses content | `"no_truncation_and_max_512": all(item["token_count"] <= 512 and tokenizer.encode(...) == item["token_ids"])` | ✅ VERIFIED |
| 7 | duplicate retained tokenized state | `"exact_token_state_unique": len(seen) == len(retained)` | ✅ VERIFIED |
| 8 | trajectory or proxy group crosses splits | `"split_isolation": all(len(value) == 1 for value in trajectory_splits.values()) and ...` | ✅ VERIFIED |
| 9 | any split has <10 positive trajectories for a tool | `"minimum_positive_trajectory_support": all(support[split][tool]["positive_trajectories"] >= 10 ...)` | ✅ VERIFIED |
| 10 | over-budget rate diff >10pp between splits | `"split_over_budget_difference_lte_10pp": max(split_rates) - min(split_rates) <= 0.10` | ✅ VERIFIED |
| 11 | over-budget rate diff >20pp between pos/neg for any tool | `"label_over_budget_difference_lte_20pp": all(value <= 0.20 for value in label_rate_differences.values())` | ✅ VERIFIED |
| 12 | (implicit) no candidate with unknown tool in target | Checked at line 182: `value.get("name") not in TOOLS` → exclusion | ✅ VERIFIED |

**Missing gate**: Protocol stop_gate #3 (canary) is implemented but **non-functional** (see item 3).

---

## 6. Exact Source SHA-256 Verification

**Status: VERIFIED**

```python
SOURCE_SHA256 = "d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02"
# Line 128-130:
source_sha = sha256_file(SOURCE)
if source_sha != SOURCE_SHA256:
    raise RuntimeError(f"STOP: source checksum mismatch: {source_sha}")
```
Matches protocol.json line 12: `"shard_sha256": "d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02"`

---

## 7. All Reasoning Markers Removed

**Status: VERIFIED**

```python
# Lines 71-95: strip_reasoning() removes <think>...</think> tags, validates no nesting
# Line 148-151: Applied to every message; malformed trajectories excluded
# Line 267: Final scan: marker_rows = sum(bool(REASONING_MARKER_RE.search(item["state"])) for item in retained)
# Gate: "zero_residual_reasoning_markers": marker_rows == 0
```
`REASONING_MARKER_RE` (line 33) catches `<think>`, ``, `<|begin_of_thought|>`, `<|end_of_thought|>`, and `reasoning:` / `analysis:` prefixes.

---

## 8. Complete Adjacent Call/Result Linkage

**Status: VERIFIED**

```python
# Lines 161-179: Prior call (message_index-2) must be from "gpt", prior result (message_index-1) from "tool"
# CALL_RE / RESPONSE_RE extract <tool_call>...</tool_call>