# Laya preprocessing fidelity — scoped review

2026-09-20. **Finding: the current builder reproduces the tested upstream sequence format, but its post-truncation budget gate does not enforce the frozen no-truncation protocol.** Review limited to local `data/laya`, exp_001b `scripts/build_dataset.py`, and `protocol.json`; tokenizer-only boundary probes, no dataset rebuild, model construction, inference, or GPU initialization requested.

## Source and canonical contract

`git -C data/laya rev-parse HEAD` verified `d113dca2512fb3eaca313534bc54c7162d87c1d4`; `git -C data/laya status --short` was empty.

- `laya/common.py:15–18`: `serialize_state` passes strings through; dict/list states use `json.dumps(..., ensure_ascii=False)`.
- `laya/agent.py:229–238`: `Agent._to_internal` converts public question definitions to internal `{"t", "ins", "crit"}` fields. This static method requires no Agent instance.
- `laya/common.py:33–46`: `render_options` fixes noul label order to **false (0), true (1)**. With absent criteria, exact option text is `false: no, the statement does not hold` and `true: yes, the statement holds`. Protocol `no/yes` are semantic labels, not literal upstream option strings.
- `laya/common.py:49–86`: `build_sequence(tok, state, q, max_len=512, head_max_len=192, option_order=None, truncate_left=False)` is the canonical constructor. It sanitizes literal mask tokens to spaces, prepends `noul question: ` to instructions, tokenizes options separately (leading space, up to 48 text tokens each), and emits `[CLS] head [SEP] [MASK] false-option [MASK] true-option [SEP] state [SEP]`. Marker positions point to the two option masks. It **truncates state internally**; its returned length alone is not a fit test.

The six experiment-specific questions use upstream's public structure (these tool questions are not built-in upstream presets):

```python
tools = ("patch", "process", "read_file", "search_files", "terminal", "write_file")
questions = {
    tool: {"type": "noul",
           "instructions": f"Does the next assistant turn call the {tool} tool?"}
    for tool in tools
}
# Each internal question:
# {"t": "noul", "ins": questions[tool]["instructions"], "crit": None}
```

`Agent.system_one` constructs one sequence per question (`agent.py:254–264`); the limit is per sequence, not the sum of six sequences. If consuming outputs directly through the upstream collator, items also require `qtype=QTYPES["noul"]=2`; builder rows currently store `tool`, `ids`, `markers`, and `length`, so an adapter is required.

## Exact room and complete-state fit

Let `E_q = build_sequence(tok, "", q, 512, 192)[0]`. This includes the final separator and no state tokens. Thus **`room_q = 512 - len(E_q)`**, equivalently upstream `512 - len(prefix_before_state) - 1`. Do not subtract a second final-separator allowance from the empty sequence length.

Verified locally with the builder's pinned tokenizer snapshot `1c5edc17a7acd8701df6fc341c0d179f1c62c982`:

| Tool | Empty sequence length | State room | Mask markers (zero-based) |
|---|---:|---:|---|
| patch | 36 | 476 | 16, 26 |
| process | 36 | 476 | 16, 26 |
| read_file | 38 | 474 | 18, 28 |
| search_files | 38 | 474 | 18, 28 |
| terminal | 36 | 476 | 16, 26 |
| write_file | 38 | 474 | 18, 28 |

The complete-state test must tokenize **without truncation**, using upstream serialization and mask replacement:

```python
full = tok(serialize_state(state).replace(tok.mask_token, " "),
           add_special_tokens=False, truncation=False)["input_ids"]
fits = all(len(full) <= 512 - len(build_sequence(tok, "", q, 512, 192)[0])
           for q in internal_questions)
```

For these fixed questions, all six fit iff `len(full) <= 474`. A retained sequence can additionally be checked against `empty[:-1] + full + [tok.sep_token_id]`, with unchanged markers. This preserves the complete *upstream-sanitized* state; literal mask-token replacement is itself canonical behavior.

**Measured boundary probes:** for every question, `state = " x" * room_q` produced exactly `room_q` state tokens, a 512-token sequence, and an intact state slice. Adding one repetition yielded `room_q + 1` state tokens but still a 512-token sequence, losing the final state token. The local builder's IDs and markers matched upstream in all 12 boundary cases. This demonstrates default/right-truncation parity on the probes, not general equivalence or corpus-wide preservation.

## Current discrepancies and consequences

1. **Vacuous no-truncation gate.** Builder lines 84–89 truncate during tokenization and cap the result; lines 261–264 then test `len(seq_ids) > 512`, which cannot detect lost content. The retained-row gate at line 379 inherits this problem. Protocol lines 20–21 and 56 require exclusion of states exceeding the full budget or losing required content. Over-budget counts and selection-bias comparisons consequently cannot measure the intended exclusions.
2. **Wrong room inference for canaries.** Lines 269–278 inspect a populated `patch` sequence: the state span is the number of tokens retained, not capacity when the state is short; `patch` also has two more available tokens than three other questions. Scanning separator IDs is ambiguous if the state contains separator tokens; the fallback `400` has no upstream basis. Empty-state construction avoids all three issues.
3. **Canaries do not test isolation.** Lines 282–300 mutate already-rendered state or an included prior call, rather than excluded source target/metadata fields followed by rerendering. Equality is counted as failure, whereas excluded-field changes should leave rendered tokens invariant. The reasoning probe appends `reason` after stripping, not a removable reasoning span. Tail truncation can hide these mutations. The named invariance gate therefore does not establish protocol lines 43 and 53.
4. **API drift beyond the tested default.** The local copy ignores its `truncate_left` argument and relies on tokenizer `truncation_side` (locally verified `right`); upstream explicitly slices left/right itself. It hardcodes default noul options rather than using `render_options` and accepts a tool name rather than an internal question. Those restrictions match the present six questions but are not a general upstream implementation.
5. **Truncated-token deduplication.** Lines 315–329 deduplicate the six already-truncated sequences. Distinct complete states differing only in discarded suffixes can collapse. This follows from code inspection; affected corpus counts were not measured.
6. **Linkage gate is stricter than retained-only protocol wording.** Line 377 requires zero linkage exclusions across all candidates, although protocol line 54 prohibits invalid *retained* pairs. Rejected candidates alone can fail this gate.

## Local verification and reproduction

Normal `from laya.common import build_sequence, serialize_state, render_options` succeeded after prepending `data/laya` to `sys.path`; so did importing the builder without calling `main`. `common.py` eagerly imports NumPy and PyTorch (including `torch.nn`), even for sequence-only use. The builder additionally eagerly imports PyArrow and `AutoTokenizer`. Verified versions: NumPy `2.5.3`, PyTorch `2.11.0+cu128`, Transformers `5.17.0`, PyArrow `25.0.1`. These imports did not require model loading; no Agent was instantiated. No dependency installation was needed.

The probes ran via `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -B -c '<Python below>'`, from the repository root. The following reproduces their substantive checks without writing files:

```python
import sys, importlib.util
sys.path.insert(0, "data/laya")
from laya.common import build_sequence, serialize_state
from laya.agent import Agent
from transformers import AutoTokenizer
spec = importlib.util.spec_from_file_location(
    "builder", "experiments/exp_001b_hermes_preprocessing/scripts/build_dataset.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
tok = AutoTokenizer.from_pretrained(b.TOKENIZER_PATH, local_files_only=True)
for tool in b.TOOLS:
    q = Agent._to_internal({"type": "noul", "instructions": b.render_noul_instruction(tool)})
    empty, markers = build_sequence(tok, "", q, 512, 192)
    room = 512 - len(empty)
    print(tool, len(empty), room, markers)
    for n in (room, room + 1):
        state = " x" * n
        full = tok(serialize_state(state).replace(tok.mask_token, " "),
                   add_special_tokens=False, truncation=False)["input_ids"]
        seq, m = build_sequence(tok, state, q, 512, 192)
        print(len(full), len(seq), len(full) <= room,
              seq[len(empty)-1:-1] == full,
              b.build_laya_sequence(tok, state, tool) == (seq, m))
```

`sha256sum` recorded reviewed-file identity:

| File | SHA-256 |
|---|---|
| exp_001b `scripts/build_dataset.py` | `514f98e368150e33f9d49ed54720621fa7c1994874c3476bdc1f5a1baf0f40bc` |
| exp_001b `protocol.json` | `ac22306f90480e4ac8da3bff15beb5d24298ff714bf7fe1f78ed770577844aac` |
| upstream `laya/common.py` | `f948ee606abe2ed2463f830c051f1a60dccc1b9f5ca1fdc15635cfcbe0cff7b2` |
| upstream `laya/agent.py` | `025522e4fd212d805703dfe1b93919c6cfae7a3c12dc72a984f88de02f2eb9c0` |

**Unknown:** true complete-state retention rate, corrected split/label exclusion bias, and corrected protocol disposition. None can be inferred from the post-truncation gate or these synthetic probes.
