#!/usr/bin/env python3
"""Freeze B2 source-order JSONL subsets from the existing pinned Parquets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "training/laya_local/b2_config.json"
TRAIN_SOURCE = ROOT / "artifacts/train_i1/source_train.parquet"
DEV_SOURCE = ROOT / "artifacts/train_i1/source_test.parquet"
DEFAULT_OUTPUT = ROOT / "artifacts/train_i2/b2_subset"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_types(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"choice": 0, "score": 0, "noul": 0}
    for record in records:
        for question in record["questions"].values():
            counts[question["type"]] += 1
    return counts


def dump_ordered_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    from shared.typed_decisions import parse_record

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            parsed = parse_record(record)
            handle.write(json.dumps(parsed, ensure_ascii=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = (ROOT / "artifacts/train_i2").resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError("B2 subset output must remain under artifacts/train_i2")

    cfg = json.loads(args.config.read_text())
    sources = {
        "train": (TRAIN_SOURCE, cfg["train_rows"], cfg["train_source_sha256"]),
        "dev": (DEV_SOURCE, cfg["dev_rows"], cfg["dev_source_sha256"]),
    }
    for split, (source, _, expected_hash) in sources.items():
        if sha256(source) != expected_hash:
            raise RuntimeError(f"{split} source SHA-256 mismatch")

    import pyarrow.parquet as pq
    from shared.typed_decisions import parse_record

    output.mkdir(parents=True, exist_ok=True)
    manifest_splits = {}
    for split, (source, count, _) in sources.items():
        table = pq.read_table(source)
        if split == "dev" and table.num_rows != count:
            raise RuntimeError(f"dev must retain all {count} rows, source has {table.num_rows}")
        rows = table.slice(0, count).to_pylist()
        records = [parse_record(row) for row in rows]
        if len(records) != count:
            raise RuntimeError(f"{split}: expected {count} rows, found {len(records)}")
        target = output / f"{split}.jsonl"
        dump_ordered_jsonl(records, target)
        types = count_types(records)
        manifest_splits[split] = {
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "output": str(target.relative_to(ROOT)),
            "output_sha256": sha256(target),
            "rows": len(records),
            "record_ids": [record.get("id") for record in records],
            "question_count": sum(types.values()),
            "question_types": types,
        }

    if manifest_splits["train"]["question_count"] != cfg["expected_items_per_train_epoch"]:
        raise RuntimeError("train question count does not match frozen sequence count")
    if manifest_splits["dev"]["question_count"] != cfg["expected_dev_items"]:
        raise RuntimeError("dev question count does not match frozen sequence count")
    manifest = {
        "dataset_id": cfg["dataset_id"],
        "dataset_revision": cfg["dataset_revision"],
        "dataset_config": cfg["dataset_config"],
        "selection_rule": cfg["subset_rule"],
        "order_preservation": "row, question, criterion, and probability insertion order preserved",
        "splits": manifest_splits,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
