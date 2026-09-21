#!/usr/bin/env python3
"""Small local tokenizer-only regression checks; retained evidence stays in artifacts."""
from copy import deepcopy
from unittest.mock import patch
import build_dataset as b
import laya.common as upstream


def main():
  tok = b.AutoTokenizer.from_pretrained(b.TOKENIZER_PATH, local_files_only=True)
  assert b.build_sequence is upstream.build_sequence
  inputs = b.LayaInputs(tok)
  assert min(inputs.rooms.values()) == 474
  for text in ("short complete state", " x" * 474, "literal [MASK] and [SEP]"):
    seqs, length = inputs.build(text)
    assert seqs is not None and length <= 512
    for seq in seqs:
      ids, markers = upstream.build_sequence(tok, text, b.QUESTIONS[seq["tool"]], 512, 192)
      assert (seq["ids"], seq["markers"]) == (ids, markers)
  with patch.object(b, "build_sequence", wraps=upstream.build_sequence) as actual:
    assert inputs.build("short")[0] is not None
    assert actual.call_count == 6
    actual.reset_mock()
    assert inputs.build(" x" * 475) == (None, 513)
    assert actual.call_count == 0, "over-budget must be rejected before production construction"

  source = {"id": "one", "category": "fixture", "task": "Inspect file", "tools": "[]",
            "conversations": [
              {"from": "gpt", "value": '<think>original secret</think><tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'},
              {"from": "tool", "value": '<tool_response>{"name":"read_file","result":"ok"}</tool_response>'},
              {"from": "gpt", "value": '<tool_call>{"name":"terminal","arguments":{}}</tool_call>'}]}
  state = b.render_source(source, 2)
  seqs, _ = inputs.build(state)
  b.canary_checks(source, 2, inputs, state, seqs)
  changed = deepcopy(source)
  changed["conversations"][1]["value"] = changed["conversations"][1]["value"].replace("ok", "different observable result")
  assert b.render_source(changed, 2) != state
  assert b.input_key(inputs.build(b.render_source(changed, 2))[0]) != b.input_key(seqs)
  for malformed in ("<think>unclosed", "</think>orphan", "<think>outer<think>inner</think></think>"):
    assert not b.strip_reasoning(malformed)[1]
    changed = deepcopy(source)
    changed["conversations"][0]["value"] = malformed
    try:
      b.render_source(changed, 2)
    except AssertionError:
      pass
    else:
      raise AssertionError("malformed reasoning accepted")
  for malformed in ('<tool_call>{"name":"patch"}', '</tool_call>', '<tool_call>{}</tool_call>',
                    '<tool_call><tool_call>{"name":"patch"}</tool_call></tool_call>'):
    assert b.payloads(malformed, "tool_call") is None
  assert b.payloads("done", "tool_call") == []
  evidence = b.ROOT / "artifacts/exp_001b/fixture_checks"
  evidence.mkdir(parents=True, exist_ok=True)
  denied = evidence / "must_not_be_model_ready"
  assert b.write_model_outputs([], {"injected_integrity_failure": False}, denied) == {}
  assert not denied.exists()
  for name in ("gzip_a.gz", "gzip_b.gz"):
    b.deterministic_gzip_write(evidence / name, "retained fixture evidence\n")
  assert (evidence / "gzip_a.gz").read_bytes() == (evidence / "gzip_b.gz").read_bytes()
  print("PASS: fit/boundary, upstream invocation/fidelity, source canaries, observable positive control, malformed reasoning/calls, fail-closed output, deterministic gzip")


if __name__ == "__main__":
  assert not __import__("sys").flags.optimize
  main()
