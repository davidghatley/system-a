#!/usr/bin/env python3
"""Deterministic Trace2Decision v2 conversion with an actual pinned Laya builder."""
from __future__ import annotations

import argparse, collections, hashlib, json, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DEFAULT = ROOT / "artifacts/trace2decision_i1/source/traces.jsonl"
TOKENIZER_DEFAULT = ROOT / "data/cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982/tokenizer"
LAYA_DEFAULT = ROOT / "data/laya"
sys.path.insert(0, str(LAYA_DEFAULT))
from laya.common import build_sequence, serialize_state  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ONTOLOGY = ("read", "search", "edit", "execute", "other_tool", "respond_or_finish")
QUESTION = {"type": "choice", "instructions": "Choose the next observable agent action type.", "criteria": {
    "read": "Read a known file or resource.", "search": "Search or list files, symbols, or resources.",
    "edit": "Create or modify files or structured content.", "execute": "Execute a shell command, program, or test.",
    "other_tool": "Call another tool not covered by the named action types.",
    "respond_or_finish": "Respond without an executable tool call or finish the task.",
}}
LayaQ = {"t": "choice", "ins": QUESTION["instructions"], "crit": QUESTION["criteria"]}
SPLIT_SEED = "trace2decision-i1-20260921"
MAX_LEN, HEAD_MAX_LEN = 512, 192
READ = {"read", "read_file", "cat", "view", "open_file"}
SEARCH = {"ls", "list", "glob", "grep", "find", "search", "search_files", "ripgrep"}
EDIT = {"edit", "write", "write_file", "apply_patch", "patch", "replace"}
EXEC = {"bash", "shell", "terminal", "exec", "execute", "execute_code", "run", "process"}

def canon(x: Any) -> str: return json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
def sha(x: bytes) -> str: return hashlib.sha256(x).hexdigest()
def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()
def bounded(s: str, n: int) -> str:
    if len(s) <= n: return s
    marker = f"\n...[TRUNCATED {len(s)-n} CHARS]...\n"; room = n-len(marker)
    left = room//2
    return s[:left] + marker + s[-(room-left):]
def split_for(group: str) -> str:
    b = int(sha(f"{SPLIT_SEED}\n{group}".encode())[:16], 16) % 10000
    return "train" if b < 8000 else "dev" if b < 9000 else "test"
def action(name: str) -> str:
    n = name.lower().strip().replace("-", "_")
    if n in READ or n.startswith("read_"): return "read"
    if n in SEARCH or any(x in n for x in ("search", "grep", "glob", "find")): return "search"
    if n in EDIT or any(x in n for x in ("edit", "write", "patch")): return "edit"
    if n in EXEC or any(x in n for x in ("shell", "terminal", "execute")): return "execute"
    return "other_tool"
def render(m: dict[str, Any], limit: int = 3500) -> str:
    bits = []
    if m.get("tool_call_id"): bits.append(f"TOOL_RESULT id={m['tool_call_id']}")
    if m.get("name"): bits.append(f"name={m['name']}")
    if m.get("content"): bits.append(bounded(str(m["content"]), limit))
    for c in m.get("tool_calls") or []:
        fn = c.get("function", {})
        bits.append("TOOL_CALL " + canon({"id": c.get("id", ""), "name": fn.get("name", ""), "arguments": fn.get("arguments", "")}))
    return f"[{str(m.get('role', 'unknown')).upper()}]\n" + ("\n".join(bits) or "[EMPTY]")
def initial(prefix):
    system = next((str(m.get("content") or "") for m in prefix if m.get("role") == "system"), "")
    task_i = next((i for i,m in enumerate(prefix) if m.get("role") == "user"), -1)
    task = str(prefix[task_i].get("content") or "") if task_i >= 0 else ""
    return system, task, task_i
def token_count(tok, state):
    return len(tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False, truncation=False)["input_ids"])
def distribution(values):
    total = len(values)
    return {"min": min(values), "max": max(values), "mean": round(sum(values) / total, 2),
            "exceeding": {str(b): {"count": sum(v > b for v in values),
                                     "percentage": round(100 * sum(v > b for v in values) / total, 2)}
                           for b in (512, 768, 1024)}}
def state_for(prefix, tok, rooms):
    system, task, task_i = initial(prefix)
    obs_i = max((i for i,m in enumerate(prefix) if m.get("role") == "tool"), default=-1)
    if obs_i < 0:
        obs_i = max((i for i,m in enumerate(prefix) if i != task_i and m.get("role") != "system"), default=-1)
    # Protected fields are rendered first. Only optional history is dropped by the budget loop.
    limits = {"task": 6000, "system": 3000, "observation": 7000}
    def make(t, s, o, history):
        return ("TASK\n" + t + "\n\nSYSTEM CONSTRAINTS\n" + s +
                "\n\nAVAILABLE ACTIONS\n" + ", ".join(ONTOLOGY) +
                "\n\nLATEST RELEVANT OBSERVATION\n" + o +
                "\n\nRECENT HISTORY\n" + ("\n\n".join(history) if history else "[NO ADDITIONAL HISTORY]"))
    candidates = [(i, render(m)) for i,m in enumerate(prefix) if i not in {task_i, obs_i} and m.get("role") != "system"]
    # This is deliberately side-effect-free: it is the complete bounded source
    # candidate, before the 512-token policy selects optional history.
    full_history = [item for _, item in candidates]
    pre_state = make(bounded(task, limits["task"]), bounded(system, limits["system"]),
                     bounded(render(prefix[obs_i]) if obs_i >= 0 else "[NO OBSERVATION]", limits["observation"]),
                     full_history)
    pre_tokens = 97 + token_count(tok, pre_state)  # pinned empty Laya envelope + serialized state
    history = []
    while True:
        state = make(bounded(task, limits["task"]), bounded(system, limits["system"]),
                     bounded(render(prefix[obs_i]) if obs_i >= 0 else "[NO OBSERVATION]", limits["observation"]), history)
        if token_count(tok, state) <= rooms: break
        if history:
            history.pop(0)
            continue
        # Deterministic protected-field reductions, preserving both ends and every section.
        reduced = False
        for key, floor in (("system", 100), ("observation", 300), ("task", 300)):
            if limits[key] > floor:
                limits[key] = max(floor, limits[key] - 400); reduced = True; break
        if not reduced:
            raise ValueError("protected task/system/observation cannot fit pinned Laya room")
    # Add newest complete history messages while retaining the protected state.
    selected = []
    for item in reversed(candidates):
        trial = selected + [item]
        candidate_state = make(bounded(task, limits["task"]), bounded(system, limits["system"]),
                               bounded(render(prefix[obs_i]) if obs_i >= 0 else "[NO OBSERVATION]", limits["observation"]),
                               [rendered for _, rendered in trial])
        if token_count(tok, candidate_state) <= rooms: selected = trial
        else: break
    # Selection is newest-first, but model-facing conversational history is
    # restored to source chronology after selection.
    history = [item for _, item in sorted(selected, key=lambda item: item[0])]
    state = make(bounded(task, limits["task"]), bounded(system, limits["system"]),
                 bounded(render(prefix[obs_i]) if obs_i >= 0 else "[NO OBSERVATION]", limits["observation"]), history)
    selected_indexes = {i for i, _ in selected}
    omitted = [{"source_index": i, "role": prefix[i].get("role"), "reason": "optional_history_budget"}
               for i, rendered in candidates if i not in selected_indexes]
    return state, {"history_candidates": len(candidates), "history_retained": len(history), "history_omitted": len(omitted),
                    "omitted_history": omitted, "latest_observation_source_index": obs_i,
                    "history_source_indices": [i for i, rendered in sorted(selected, key=lambda item: item[0])],
                    "pre_policy_token_length": pre_tokens,
                   "protected_char_limits": limits, "task_chars_rendered": len(bounded(task, limits["task"])),
                   "observation_chars_rendered": len(bounded(render(prefix[obs_i]) if obs_i >= 0 else "[NO OBSERVATION]", limits["observation"]))}

def convert(source: Path, output: Path, tokenizer_path: Path = TOKENIZER_DEFAULT) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    empty, markers = build_sequence(tok, "", LayaQ, MAX_LEN, HEAD_MAX_LEN)
    room = MAX_LEN - len(empty)
    rows, malformed, unresolved = [], [], 0
    source_rows = 0
    for line, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip(): continue
        source_rows += 1; row = json.loads(raw); messages = row.get("messages", []); target_i = row.get("target_message_index")
        if not isinstance(messages, list) or not messages or target_i != len(messages)-1 or messages[-1].get("role") != "assistant":
            malformed.append({"line": line, "reason": "invalid cumulative target boundary"}); continue
        prefix, target = messages[:-1], messages[-1]
        system, task, task_i = initial(prefix)
        if not task: malformed.append({"line": line, "reason": "missing task"}); continue
        prior_calls = {str(c.get("id")) for m in prefix for c in (m.get("tool_calls") or []) if c.get("id")}
        prior_results = {str(m.get("tool_call_id")) for m in prefix if m.get("role") == "tool" and m.get("tool_call_id")}
        if prior_calls - prior_results: unresolved += 1; continue
        state, info = state_for(prefix, tok, room)
        full = tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False, truncation=False)["input_ids"]
        seq, got_markers = build_sequence(tok, state, LayaQ, MAX_LEN, HEAD_MAX_LEN)
        if len(full) > room or seq != empty[:-1] + full + [tok.sep_token_id] or got_markers != markers:
            raise AssertionError("v2 construction lost protected state or markers")
        names = [str(c.get("function", {}).get("name", "")) for c in target.get("tool_calls") or []]
        label = action(names[0]) if names else "respond_or_finish"
        group = sha(" ".join(task.lower().split()).encode())
        trajectory = str(row.get("source_trajectory_sha256"))
        step = int(row.get("assistant_step")); ident = "t2d-i2-" + sha(f"{trajectory}\n{step}\n{group}".encode())[:32]
        probs = {x: float(x == label) for x in ONTOLOGY}
        rec = {"state": state, "questions": {"next_action": QUESTION}, "gold": {"next_action": {"type": "choice", "label": label, "probabilities": probs}},
               "metadata": {"id": ident, "trajectory_id": trajectory, "task_group": group, "source_step": step, "source_line": line,
                            "source_split": row.get("split"), "raw_tool_names": names, "laya_sequence_length": len(seq),
                            "laya_state_tokens": len(full), "laya_room_512": room, **info}}
        rows.append((group, trajectory, step, ident, rec))
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    counts = collections.Counter(); trajectories = collections.defaultdict(set); groups = collections.defaultdict(set); lengths = []; hist = []; pre_lengths = []
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for group, trajectory, step, ident, rec in rows:
                sp = split_for(group)
                if sp != split: continue
                f.write(canon(rec) + "\n"); counts[split] += 1; trajectories[split].add(trajectory); groups[split].add(group)
                lengths.append(rec["metadata"]["laya_sequence_length"]); hist.append(rec["metadata"]["history_omitted"])
                pre_lengths.append(rec["metadata"]["pre_policy_token_length"])
    samples = sorted((r for _,_,_,_,r in rows), key=lambda r: r["metadata"]["id"])[:12]
    (output / "samples.jsonl").write_text("".join(canon(x)+"\n" for x in samples), encoding="utf-8")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": file_sha(p), "rows": sum(1 for _ in p.open(encoding="utf-8"))} for p in [output/f"{s}.jsonl" for s in ("train","dev","test")] + [output/"samples.jsonl"]}
    report = {"schema_version": "trace2decision-i2-v2", "source": {"dataset": "11-47/glm-5.2-coding-and-debugging-traces", "revision": "1371ed38f8890d0520a53bc7ad850308eb4d7a22", "file": "artifacts/trace2decision_i1/source/traces.jsonl", "sha256": file_sha(source), "rows": source_rows},
              "pins": {"laya_source_revision": "d113dca2512fb3eaca313534bc54c7162d87c1d4", "tokenizer_revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982", "tokenizer_path": str(tokenizer_path), "max_len": 512, "head_max_len": 192},
              "contract": {"required": ["state", "questions", "gold"], "optional_metadata": ["id", "trajectory_id", "task_group", "source_step"], "metadata_outside_model_state": True, "labels": list(ONTOLOGY)},
              "policy": {"protected_order": ["TASK", "SYSTEM CONSTRAINTS", "AVAILABLE ACTIONS", "LATEST RELEVANT OBSERVATION"], "history": "newest complete rendered messages that fit after protected fields", "no_summaries": True, "loss": "only optional older history messages; target/reasoning/IDs never enter state"},
              "conversion": {"source_rows": source_rows, "malformed_rows": len(malformed), "unresolved_prefix_rows": unresolved, "retained_rows": len(rows), "malformed_examples": malformed[:20]},
              "split": {"seed": SPLIT_SEED, "group": "sha256(normalized initial user task text)", "rows": dict(counts), "trajectories": {s: len(trajectories[s]) for s in ("train","dev","test")}, "groups": {s: len(groups[s]) for s in ("train","dev","test")}},
               "token_audit": {"empty_sequence_512": len(empty), "state_room_512": room, "markers": markers,
                               "pre_policy": distribution(pre_lengths),
                               "post_policy": {"512": distribution(lengths),
                                                "768": distribution([len(build_sequence(tok, rec["state"], LayaQ, 768, HEAD_MAX_LEN)[0]) for _,_,_,_,rec in rows]),
                                                "1024": distribution([len(build_sequence(tok, rec["state"], LayaQ, 1024, HEAD_MAX_LEN)[0]) for _,_,_,_,rec in rows])},
                               "history_omitted": {"max": max(hist), "rows_with_omissions": sum(x>0 for x in hist)}}, "files": files}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return report

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,default=SOURCE_DEFAULT); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--tokenizer",type=Path,default=TOKENIZER_DEFAULT)
    print(canon({"rows": convert(ap.parse_args().source, ap.parse_args().output_dir, ap.parse_args().tokenizer)["conversion"]["retained_rows"]}))
