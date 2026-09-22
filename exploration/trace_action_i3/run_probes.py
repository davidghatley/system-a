#!/usr/bin/env python3
"""Deterministic, standard-library probes over Trace2Decision v2 train/dev."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import resource
import time
from pathlib import Path


LABELS = ("read", "search", "edit", "execute", "other_tool", "respond_or_finish")
SECTIONS = (
    "TASK",
    "SYSTEM CONSTRAINTS",
    "AVAILABLE ACTIONS",
    "LATEST RELEVANT OBSERVATION",
    "RECENT HISTORY",
)
TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label(row: dict) -> str:
    return row["gold"]["next_action"]["label"]


def majority(values: list[str], fallback: str | None = None) -> str:
    counts = collections.Counter(values)
    if not counts:
        if fallback is None:
            raise ValueError("empty majority without fallback")
        return fallback
    return min(counts, key=lambda item: (-counts[item], LABELS.index(item)))


def position_bin(step: int) -> str:
    if step <= 3:
        return str(step)
    if step <= 5:
        return "4-5"
    if step <= 10:
        return "6-10"
    return "11+"


def predecessor_features(rows: list[dict]) -> tuple[list[str], list[bool]]:
    latest: dict[str, tuple[int, str]] = {}
    previous, contiguous = [], []
    for row in rows:
        meta = row["metadata"]
        trajectory, step = meta["trajectory_id"], int(meta["source_step"])
        prior = latest.get(trajectory)
        is_contiguous = prior is not None and prior[0] == step - 1
        previous.append(prior[1] if is_contiguous else "START")
        contiguous.append(is_contiguous)
        latest[trajectory] = (step, label(row))
    return previous, contiguous


def fit_table(keys: list[object], values: list[str]) -> dict[object, str]:
    grouped: dict[object, list[str]] = collections.defaultdict(list)
    for key, value in zip(keys, values):
        grouped[key].append(value)
    return {key: majority(items) for key, items in grouped.items()}


def metrics(y_true: list[str], y_pred: list[str], probabilities=None) -> dict:
    support = collections.Counter(y_true)
    correct = collections.Counter(t for t, p in zip(y_true, y_pred) if t == p)
    recalls = {item: correct[item] / support[item] if support[item] else 0.0 for item in LABELS}
    result = {
        "rows": len(y_true),
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true),
        "macro_f1": 0.0,
        "per_label": {},
    }
    f1s = []
    for item in LABELS:
        tp = sum(t == item and p == item for t, p in zip(y_true, y_pred))
        fp = sum(t != item and p == item for t, p in zip(y_true, y_pred))
        fn = sum(t == item and p != item for t, p in zip(y_true, y_pred))
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1s.append(f1)
        result["per_label"][item] = {"support": support[item], "recall": recalls[item], "f1": f1}
    result["macro_f1"] = sum(f1s) / len(LABELS)
    if probabilities is not None:
        result["nll"] = -sum(math.log(max(prob[label], 1e-300)) for label, prob in zip(y_true, probabilities)) / len(y_true)
        result["brier"] = sum(
            sum((prob[item] - float(item == truth)) ** 2 for item in LABELS)
            for truth, prob in zip(y_true, probabilities)
        ) / len(y_true)
    return result


def paired_bootstrap(rows: list[dict], truth: list[str], candidate: list[str], reference: list[str], seed: int) -> dict:
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["metadata"]["trajectory_id"]].append(index)
    names = sorted(groups)
    rng = random.Random(seed)
    differences = {"accuracy": [], "macro_f1": []}
    for _ in range(5000):
        sampled = [rng.choice(names) for _ in names]
        indexes = [index for name in sampled for index in groups[name]]
        sampled_truth = [truth[index] for index in indexes]
        for metric_name in differences:
            candidate_value = metrics(sampled_truth, [candidate[index] for index in indexes])[metric_name]
            reference_value = metrics(sampled_truth, [reference[index] for index in indexes])[metric_name]
            differences[metric_name].append(candidate_value - reference_value)
    output = {"groups": len(names), "replicates": 5000, "seed": seed}
    for metric_name, values in differences.items():
        ordered = sorted(values)
        output[metric_name] = {
            "difference": metrics(truth, candidate)[metric_name] - metrics(truth, reference)[metric_name],
            "percentile_95": [ordered[124], ordered[4874]],
        }
    return output


def parse_sections(state: str) -> dict[str, str]:
    pattern = re.compile(r"^(" + "|".join(re.escape(item) for item in SECTIONS) + r")\n", re.MULTILINE)
    matches = list(pattern.finditer(state))
    if [match.group(1) for match in matches] != list(SECTIONS):
        raise ValueError("unexpected state section structure")
    return {
        match.group(1): state[match.end() : matches[index + 1].start()].rstrip("\n")
        if index + 1 < len(matches)
        else state[match.end() :]
        for index, match in enumerate(matches)
    }


def features(text: str) -> set[str]:
    tokens = TOKEN_RE.findall(text.lower())
    output = {"u=" + token for token in tokens}
    output.update("b=" + left + "_" + right for left, right in zip(tokens, tokens[1:]))
    return output


def fit_nb(texts: list[str], labels: list[str]) -> dict:
    documents = [features(text) for text in texts]
    document_frequency = collections.Counter(feature for document in documents for feature in document)
    vocabulary = {feature for feature, count in document_frequency.items() if count >= 2}
    class_docs = collections.Counter(labels)
    feature_counts = {item: collections.Counter() for item in LABELS}
    totals = collections.Counter()
    for document, item in zip(documents, labels):
        selected = document & vocabulary
        feature_counts[item].update(selected)
        totals[item] += len(selected)
    return {
        "vocabulary": vocabulary,
        "class_docs": class_docs,
        "feature_counts": feature_counts,
        "totals": totals,
        "rows": len(labels),
    }


def predict_nb(model: dict, texts: list[str]) -> tuple[list[str], list[dict[str, float]]]:
    vocabulary = model["vocabulary"]
    size = len(vocabulary)
    predictions, probabilities = [], []
    for text in texts:
        selected = features(text) & vocabulary
        scores = {}
        for item in LABELS:
            score = math.log((model["class_docs"][item] + 1) / (model["rows"] + len(LABELS)))
            denominator = model["totals"][item] + size
            score += sum(math.log((model["feature_counts"][item][feature] + 1) / denominator) for feature in selected)
            scores[item] = score
        maximum = max(scores.values())
        normalizer = sum(math.exp(value - maximum) for value in scores.values())
        probability = {item: math.exp(scores[item] - maximum) / normalizer for item in LABELS}
        probabilities.append(probability)
        predictions.append(min(LABELS, key=lambda item: (-probability[item], LABELS.index(item))))
    return predictions, probabilities


def main() -> None:
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train, dev = read_rows(args.train), read_rows(args.dev)
    train_y, dev_y = [label(row) for row in train], [label(row) for row in dev]
    train_previous, train_contiguous = predecessor_features(train)
    dev_previous, dev_contiguous = predecessor_features(dev)
    global_majority = majority(train_y)
    majority_predictions = [global_majority] * len(dev)

    train_steps = [int(row["metadata"]["source_step"]) for row in train]
    dev_steps = [int(row["metadata"]["source_step"]) for row in dev]
    train_bins, dev_bins = [position_bin(step) for step in train_steps], [position_bin(step) for step in dev_steps]
    step_table = fit_table(train_steps, train_y)
    bin_table = fit_table(train_bins, train_y)
    transition_table = fit_table(train_previous, train_y)
    joint_table = fit_table(list(zip(train_bins, train_previous)), train_y)
    step_predictions = [step_table.get(step, global_majority) for step in dev_steps]
    bin_predictions = [bin_table.get(item, global_majority) for item in dev_bins]
    transition_predictions = [transition_table.get(item, global_majority) for item in dev_previous]
    joint_predictions = [
        joint_table.get((pos, previous), transition_table.get(previous, global_majority))
        for pos, previous in zip(dev_bins, dev_previous)
    ]
    workflow_predictions = {
        "majority": majority_predictions,
        "exact_step": step_predictions,
        "position_bin": bin_predictions,
        "previous_action": transition_predictions,
        "position_plus_previous": joint_predictions,
    }
    workflow = {
        "predecessor_definition": "previous retained label only when source_step is contiguous; otherwise START",
        "train_contiguous_predecessor": {"rows": sum(train_contiguous), "fraction": sum(train_contiguous) / len(train)},
        "dev_contiguous_predecessor": {"rows": sum(dev_contiguous), "fraction": sum(dev_contiguous) / len(dev)},
        "methods": {},
    }
    for name, predictions in workflow_predictions.items():
        workflow["methods"][name] = {"metrics": metrics(dev_y, predictions)}
        if name != "majority":
            workflow["methods"][name]["paired_vs_majority"] = paired_bootstrap(
                dev, dev_y, predictions, majority_predictions, 20260922
            )

    train_sections = [parse_sections(row["state"]) for row in train]
    dev_sections = [parse_sections(row["state"]) for row in dev]
    text_variants = {
        "task": (
            [sections["TASK"] for sections in train_sections],
            [sections["TASK"] for sections in dev_sections],
        ),
        "task_plus_latest_observation": (
            [sections["TASK"] + "\nOBSERVATION\n" + sections["LATEST RELEVANT OBSERVATION"] for sections in train_sections],
            [sections["TASK"] + "\nOBSERVATION\n" + sections["LATEST RELEVANT OBSERVATION"] for sections in dev_sections],
        ),
    }
    sparse = {"settings": {"model": "multinomial_naive_bayes", "alpha": 1.0, "features": "lowercase word unigrams+bigrams", "min_train_document_frequency": 2}, "methods": {}}
    sparse_predictions = {}
    for name, (train_texts, dev_texts) in text_variants.items():
        model = fit_nb(train_texts, train_y)
        predictions, probabilities = predict_nb(model, dev_texts)
        sparse_predictions[name] = predictions
        sparse["methods"][name] = {
            "vocabulary_size": len(model["vocabulary"]),
            "metrics": metrics(dev_y, predictions, probabilities),
        }
    sparse["task_plus_observation_vs_task"] = paired_bootstrap(
        dev,
        dev_y,
        sparse_predictions["task_plus_latest_observation"],
        sparse_predictions["task"],
        20260922,
    )

    result = {
        "status": "exploratory",
        "starting_commit": "2d4408de85587a046c3f14fec1835dd9e69940c9",
        "warning": "V2 labels may be affected by the known first-tool-call target bug; mixed-category targets were not excluded.",
        "inputs": {
            "train": {"path": str(args.train), "rows": len(train), "sha256": file_sha256(args.train)},
            "dev": {"path": str(args.dev), "rows": len(dev), "sha256": file_sha256(args.dev)},
            "test_read": False,
        },
        "label_support": {"train": dict(collections.Counter(train_y)), "dev": dict(collections.Counter(dev_y))},
        "probe_1_workflow": workflow,
        "probe_2_latest_observation": sparse,
        "resource_use": {
            "cpu_seconds": time.process_time() - started_cpu,
            "wall_seconds": time.monotonic() - started_wall,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "configured_threads": 1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
