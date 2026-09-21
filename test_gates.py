#!/usr/bin/env python3
"""Test remaining gates"""

import gzip
import hashlib
import json
import tempfile
from pathlib import Path

# 2. Gzip bug test
print("2. Gzip mtime=0 test:")
content = "test content for deterministic gzip"
with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
    path = Path(f.name)

# Write with mtime=0
with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=path.open("wb")) as handle:
    handle.write(content.encode("utf-8"))

# Read back
with gzip.open(path, "rb") as f:
    read_content = f.read().decode("utf-8")

print(f"  Content matches: {read_content == content}")

# Check mtime is actually 0 in the gzip header
import struct
with path.open("rb") as f:
    header = f.read(10)
    # gzip header: ID1, ID2, CM, FLG, MTIME (4 bytes), XFL, OS
    mtime_bytes = header[4:8]
    mtime = struct.unpack("<I", mtime_bytes)[0]
    print(f"  MTIME in header: {mtime} (should be 0)")

path.unlink()

# 7. Reasoning markers removed - test strip_reasoning
print("\n7. Reasoning marker removal test:")
THINK_TAG_RE = __import__('re').compile(r"<think\b[^>]*>|</think\s*>", __import__('re').I)
REASONING_MARKER_RE = __import__('re').compile(r"</?think\b|<\|(?:begin|end)_of_thought\|>|(?:^|\n)\s*(?:reasoning|analysis)\s*:", __import__('re').I)

def strip_reasoning(value: str) -> tuple[str, bool]:
    output = []
    cursor = 0
    inside = False
    valid = True
    for match in THINK_TAG_RE.finditer(value):
        closing = match.group(0).lower().startswith("</")
        if closing:
            if not inside:
                valid = False
            else:
                inside = False
        else:
            if inside:
                valid = False
            else:
                output.append(value[cursor:match.start()])
                inside = True
        cursor = match.end()
    if inside:
        valid = False
    else:
        output.append(value[cursor:])
    stripped = "".join(output)
    return stripped, valid and not THINK_TAG_RE.search(stripped)

test_cases = [
    ("Hello <think>reason world", "Hello  world"),
    ("<think>nested <think>inner", ""),
    ("No tags here", "No tags here"),
    ("<think>unclosed", ""),
    ("un", "unclosed"),
]

for inp, expected in test_cases:
    out, valid = strip_reasoning(inp)
    print(f"  Input: {repr(inp)[:50]} -> Output: {repr(out)[:50]}, Valid: {valid}, Expected: {repr(expected)}")

# Check REASONING_MARKER_RE catches remaining markers
print("\n  REASONING_MARKER_RE tests:")
marker_tests = [
    "<think>content</think>",
    "<|begin_of_thought|>content<|end_of_thought|>",
    "reasoning: this is analysis",
    "analysis: something",
    "No markers here",
]
for t in marker_tests:
    match = REASONING_MARKER_RE.search(t)
    print(f"    {repr(t)[:40]} -> {'MATCH' if match else 'NO MATCH'}")

# 8. Adjacent call/result linkage
print("\n8. Adjacent call/result linkage test:")
CALL_RE = __import__('re').compile(r"<tool_call\b[^>]*>(.*?)</tool_call\s*>", __import__('re').I | __import__('re').S)
RESPONSE_RE = __import__('re').compile(r"<tool_response\b[^>]*>(.*?)</tool_response\s*>", __import__('re').I | __import__('re').S)

prior_call = '<tool_call name="patch">{"arg": 1}</tool_call>