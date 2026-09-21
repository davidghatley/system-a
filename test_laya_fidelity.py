#!/usr/bin/env python3
"""Test Laya representation fidelity between build_dataset.py and rl_common.py"""

import sys
import json
sys.path.insert(0, '/home/davidhatley/Projects/research/system_a/data/cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982')
from rl_common import build_sequence, serialize_state
from transformers import AutoTokenizer

TOKENIZER_PATH = '/home/davidhatley/Projects/research/system_a/data/cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982/tokenizer'
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# From build_dataset.py
TOOLS = ("patch", "process", "read_file", "search_files", "terminal", "write_file")

def render_noul_instruction(tool_name: str) -> str:
    return f"Does the next assistant turn call the {tool_name} tool?"

def serialize_state_ds(state):
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)

def build_laya_sequence_ds(tok, state: str, tool_name: str, max_len: int = 512, head_max_len: int = 192) -> tuple[list[int], list[int]]:
    mask_tok = tok.mask_token
    opts = [
        "false: no, the statement does not hold",
        "true: yes, the statement holds"
    ]
    ins = render_noul_instruction(tool_name).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % ("noul", ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for opt in opts:
        opt_ids.append(
            [tok.mask_token_id]
            + tok(" " + opt.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state_ds(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[-room:]  # truncate_left=False means right truncation (keep start of state)
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]

# Test state
state = """Task:
test task
Available tools: patch, process, read_file, search_files, terminal, write_file
Previous assistant call:
{"name": "patch", "arguments": {}}
Latest tool response:
{"name": "patch", "result": "ok"}"""

# Test all 6 tools
print("Testing all 6 noul questions:")
for tool in TOOLS:
    # Laya build_sequence
    q = {"t": "noul", "ins": render_noul_instruction(tool), "crit": {}}
    laya_ids, laya_markers = build_sequence(tok, state, q, max_len=512, head_max_len=192, truncate_left=False)
    
    # build_dataset build_laya_sequence
    ds_ids, ds_markers = build_laya_sequence_ds(tok, state, tool)
    
    match = laya_ids == ds_ids and laya_markers == ds_markers
    print(f"  {tool}: {'MATCH' if match else 'MISMATCH'} (Laya len={len(laya_ids)}, DS len={len(ds_ids)})")
    if not match:
        # Find first difference
        for i, (a, b) in enumerate(zip(laya_ids, ds_ids)):
            if a != b:
                print(f"    First diff at index {i}: Laya={a}, DS={b}")
                break
        if len(laya_ids) != len(ds_ids):
            print(f"    Length diff: Laya={len(laya_ids)}, DS={len(ds_ids)}")
        if laya_markers != ds_markers:
            print(f"    Markers diff: Laya={laya_markers}, DS={ds_markers}")