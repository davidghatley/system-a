#!/usr/bin/env python3
"""Freeze small source-order train/dev subsets from pinned public Parquet files."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.typed_decisions.schema import dump_jsonl, parse_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training/laya_local/config.json")
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--dev-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/train_i1/subsets")
    args = parser.parse_args()
    args.train_parquet = args.train_parquet.resolve()
    args.dev_parquet = args.dev_parquet.resolve()
    args.output_dir = args.output_dir.resolve()
    cfg = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for split, source, count in (
        ("train", args.train_parquet, cfg["train_rows"]),
        ("dev", args.dev_parquet, cfg["dev_rows"]),
    ):
        rows = pq.read_table(source).slice(0, count).to_pylist()
        records = [parse_record(row) for row in rows]
        if len(records) != count:
            raise RuntimeError(f"{split}: expected {count} rows, found {len(records)}")
        output = args.output_dir / f"{split}.jsonl"
        dump_jsonl(records, output)
        outputs[split] = {
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "output": str(output.relative_to(ROOT)),
            "output_sha256": sha256(output),
            "rows": count,
            "record_ids": [record.get("id") for record in records],
            "question_count": sum(len(record["questions"]) for record in records),
            "choice_question_count": sum(
                question["type"] == "choice" for record in records for question in record["questions"].values()
            ),
        }
    manifest = {
        "frozen_before_baseline": True,
        "dataset_id": cfg["dataset_id"],
        "dataset_revision": cfg["dataset_revision"],
        "dataset_config": cfg["dataset_config"],
        "selection_rule": cfg["subset_rule"],
        "splits": outputs,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
