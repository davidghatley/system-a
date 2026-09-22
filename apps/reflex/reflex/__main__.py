"""Command line interface for Reflex."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

from .core import Reflex, ReflexError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observe typed decisions over a JSON agent state with pinned Laya")
    parser.add_argument("input", type=pathlib.Path, help="JSON file containing state, questions, and optional gold")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("data/cache/huggingface/hub"))
    parser.add_argument("--laya-source", type=pathlib.Path, default=pathlib.Path("data/laya"))
    parser.add_argument("--checkpoint", choices=("specialist", "base"), default="specialist")
    parser.add_argument("--device", help="torch device, for example cuda:0 or cpu (default: auto)")
    parser.add_argument("--offline", action="store_true", help="forbid checkpoint network access")
    parser.add_argument("--benchmark", action="store_true", help="measure repeated inference after model load")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.benchmark and (args.warmup < 0 or args.samples < 1):
            raise ReflexError("benchmark requires --warmup >= 0 and --samples >= 1")
        try:
            record = json.loads(args.input.read_text())
        except FileNotFoundError as exc:
            raise ReflexError(f"input file not found: {args.input}") from exc
        except json.JSONDecodeError as exc:
            raise ReflexError(f"invalid JSON in {args.input}: line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc

        reflex = Reflex.from_checkpoint(
            cache_dir=args.cache_dir,
            laya_source=args.laya_source,
            device=args.device,
            offline=args.offline,
            checkpoint=args.checkpoint,
        )
        result = reflex.decide(record)
        if args.benchmark:
            for _ in range(args.warmup):
                reflex.decide(record)
            durations = []
            for _ in range(args.samples):
                started = time.perf_counter()
                reflex.decide(record)
                durations.append((time.perf_counter() - started) * 1000.0)
            ordered = sorted(durations)
            p95_index = min(len(ordered) - 1, math_ceil(0.95 * len(ordered)) - 1)
            result["benchmark"] = {
                "warmup": args.warmup,
                "samples": args.samples,
                "milliseconds": {
                    "min": round(min(durations), 3),
                    "median": round(statistics.median(durations), 3),
                    "p95_nearest_rank": round(ordered[p95_index], 3),
                    "max": round(max(durations), 3),
                },
                "scope": "inference plus tokenization; excludes model load",
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ReflexError as exc:
        print(f"reflex: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"reflex: could not read input: {exc}", file=sys.stderr)
        return 2


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if integer == value else integer + 1


if __name__ == "__main__":
    raise SystemExit(main())
