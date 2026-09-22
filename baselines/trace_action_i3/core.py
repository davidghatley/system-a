"""Deterministic, dependency-free baselines over compact Trace2Decision state text."""

from __future__ import annotations

import collections
import gzip
import hashlib
import io
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LABELS = ("read", "search", "edit", "execute", "other_tool", "respond_or_finish")
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
NAME_RE = re.compile(r"(?:name=|\"name\"\s*:\s*\")([A-Za-z0-9_.-]+)")
READ = {"read", "read_file", "cat", "view", "open_file"}
SEARCH = {"ls", "list", "glob", "grep", "find", "search", "search_files", "ripgrep"}
EDIT = {"edit", "write", "write_file", "apply_patch", "patch", "replace"}
EXECUTE = {"bash", "shell", "terminal", "exec", "execute", "execute_code", "run", "process"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path, labels: tuple[str, ...], question_id: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                state = row["state"]
                answer = row["gold"][question_id]
                metadata = row["metadata"]
                label = answer["label"]
                if not isinstance(state, str) or label not in labels:
                    raise ValueError("invalid state or fixed-list label")
                trajectory = metadata["trajectory_id"]
                step = metadata["source_step"]
                ident = metadata["id"]
                if not isinstance(trajectory, str) or not isinstance(step, int) or not isinstance(ident, str):
                    raise ValueError("invalid trajectory metadata")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid baseline record: {exc}") from exc
            records.append({"id": ident, "state": state, "label": label, "trajectory_id": trajectory, "step": step})
    if not records:
        raise ValueError(f"{path}: no records")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError(f"{path}: duplicate record id")
    return records


def smooth_counts(counts: collections.Counter[str], labels: tuple[str, ...], alpha: float) -> list[float]:
    denominator = sum(counts.values()) + alpha * len(labels)
    return [(counts[label] + alpha) / denominator for label in labels]


def majority_model(train: list[dict[str, Any]], labels: tuple[str, ...], alpha: float) -> dict[str, Any]:
    counts = collections.Counter(row["label"] for row in train)
    probabilities = smooth_counts(counts, labels, alpha)
    # Fixed label order is the declared tie-break everywhere.
    prediction = labels[max(range(len(labels)), key=lambda i: probabilities[i])]
    return {"counts": dict(counts), "probabilities": probabilities, "prediction": prediction}


def tool_action(name: str) -> str:
    normalized = name.lower().strip().replace("-", "_")
    if normalized in READ or normalized.startswith("read_"):
        return "read"
    if normalized in SEARCH or any(part in normalized for part in ("search", "grep", "glob", "find")):
        return "search"
    if normalized in EDIT or any(part in normalized for part in ("edit", "write", "patch")):
        return "edit"
    if normalized in EXECUTE or any(part in normalized for part in ("shell", "terminal", "execute")):
        return "execute"
    return "other_tool"


def previous_action_from_state(state: str) -> str | None:
    observation_marker = "LATEST RELEVANT OBSERVATION\n"
    history_marker = "\n\nRECENT HISTORY\n"
    if observation_marker not in state:
        return None
    remainder = state.split(observation_marker, 1)[1]
    observation, _, history = remainder.partition(history_marker)
    # History precedes the latest observation in source chronology. Restrict
    # extraction to these evidence sections so task text cannot become a feature.
    names = NAME_RE.findall(history) + NAME_RE.findall(observation)
    return tool_action(names[-1]) if names else None


def transition_model(
    train: list[dict[str, Any]], labels: tuple[str, ...], alpha: float, fallback: list[float]
) -> dict[str, Any]:
    by_trajectory: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in train:
        by_trajectory[row["trajectory_id"]].append(row)
    counts = {label: collections.Counter() for label in labels}
    transitions = 0
    skipped_step_gaps = 0
    for rows in by_trajectory.values():
        ordered = sorted(rows, key=lambda row: (row["step"], row["id"]))
        for previous, current in zip(ordered, ordered[1:]):
            if current["step"] != previous["step"] + 1:
                skipped_step_gaps += 1
                continue
            counts[previous["label"]][current["label"]] += 1
            transitions += 1
    probabilities = {
        previous: smooth_counts(counts[previous], labels, alpha) if counts[previous] else list(fallback)
        for previous in labels
    }
    return {
        "counts": {key: dict(value) for key, value in counts.items()},
        "probabilities": probabilities,
        "fallback": list(fallback),
        "fallback_policy": "training-majority distribution when compact state has no mapped previous action",
        "fit_trajectories": len(by_trajectory),
        "fit_transitions": transitions,
        "skipped_step_gaps": skipped_step_gaps,
    }


def transition_predict(model: dict[str, Any], states: Iterable[str]) -> list[list[float]]:
    return [
        model["probabilities"].get(previous, model["fallback"]) if previous else model["fallback"]
        for previous in (previous_action_from_state(state) for state in states)
    ]


def word_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    result = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        result.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return result


@dataclass
class SparseLogistic:
    labels: tuple[str, ...]
    vocabulary: dict[str, int]
    idf: list[float]
    weights: list[list[float]]
    bias: list[float]
    config: dict[str, Any]

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        maximum = max(scores)
        values = [math.exp(score - maximum) for score in scores]
        total = sum(values)
        return [value / total for value in values]

    def transform_one(self, text: str) -> dict[int, float]:
        counts = collections.Counter(word_ngrams(text, tuple(self.config["ngram_range"])))
        values = {self.vocabulary[token]: (1.0 + math.log(count)) * self.idf[self.vocabulary[token]]
                  for token, count in counts.items() if token in self.vocabulary}
        norm = math.sqrt(sum(value * value for value in values.values()))
        return {index: value / norm for index, value in values.items()} if norm else {}

    def predict_proba(self, texts: Iterable[str]) -> list[list[float]]:
        output = []
        for text in texts:
            vector = self.transform_one(text)
            scores = [self.bias[k] + sum(self.weights[k][j] * value for j, value in vector.items())
                      for k in range(len(self.labels))]
            output.append(self._softmax(scores))
        return output

    def serializable(self) -> dict[str, Any]:
        return {"format": "trace-action-sparse-logistic-v1", "labels": list(self.labels),
                "vocabulary": self.vocabulary, "idf": self.idf, "weights": self.weights,
                "bias": self.bias, "config": self.config}

    @classmethod
    def from_serializable(cls, value: dict[str, Any]) -> "SparseLogistic":
        if value.get("format") != "trace-action-sparse-logistic-v1":
            raise ValueError("unsupported sparse logistic format")
        return cls(tuple(value["labels"]), value["vocabulary"], value["idf"], value["weights"],
                   value["bias"], value["config"])


def load_model(path: Path) -> SparseLogistic:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return SparseLogistic.from_serializable(json.load(handle))


def fit_logistic(
    texts: list[str], targets: list[str], labels: tuple[str, ...], candidate: dict[str, Any], seed: int
) -> SparseLogistic:
    ngram_range = tuple(candidate["ngram_range"])
    doc_frequency: collections.Counter[str] = collections.Counter()
    term_counts = []
    for text in texts:
        counts = collections.Counter(word_ngrams(text, ngram_range))
        term_counts.append(counts)
        doc_frequency.update(counts.keys())
    eligible = [(frequency, token) for token, frequency in doc_frequency.items()
                if frequency >= candidate["min_df"]]
    eligible.sort(key=lambda item: (-item[0], item[1]))
    vocabulary = {token: index for index, (_, token) in enumerate(eligible[: candidate["max_features"]])}
    if not vocabulary:
        raise ValueError("TF-IDF candidate has empty vocabulary")
    idf = [0.0] * len(vocabulary)
    for token, index in vocabulary.items():
        idf[index] = math.log((1.0 + len(texts)) / (1.0 + doc_frequency[token])) + 1.0
    vectors = []
    for counts in term_counts:
        vector = {vocabulary[token]: (1.0 + math.log(count)) * idf[vocabulary[token]]
                  for token, count in counts.items() if token in vocabulary}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        vectors.append({index: value / norm for index, value in vector.items()} if norm else {})
    classes = {label: index for index, label in enumerate(labels)}
    y = [classes[target] for target in targets]
    weights = [[0.0] * len(vocabulary) for _ in labels]
    bias = [0.0] * len(labels)
    rng = random.Random(seed)
    order = list(range(len(texts)))
    regularization = 1.0 / candidate["C"]
    step = 0
    for epoch in range(candidate["epochs"]):
        rng.shuffle(order)
        learning_rate = candidate["learning_rate"] / math.sqrt(epoch + 1.0)
        # Accumulate the per-example L2 shrink once per epoch so sparse updates
        # do not become dense operations.
        shrink = max(0.0, 1.0 - learning_rate * regularization)
        for k in range(len(labels)):
            weights[k] = [value * shrink for value in weights[k]]
        for index in order:
            step += 1
            vector = vectors[index]
            scores = [bias[k] + sum(weights[k][j] * value for j, value in vector.items())
                      for k in range(len(labels))]
            probabilities = SparseLogistic._softmax(scores)
            for k in range(len(labels)):
                error = probabilities[k] - float(k == y[index])
                bias[k] -= learning_rate * error
                for j, value in vector.items():
                    weights[k][j] -= learning_rate * error * value
    model_config = dict(candidate)
    model_config.update({"optimizer": "deterministic shuffled SGD", "seed": seed,
                         "vocabulary_size": len(vocabulary), "updates": step})
    return SparseLogistic(labels, vocabulary, idf, weights, bias, model_config)


def hard_predictions(probabilities: list[list[float]], labels: tuple[str, ...]) -> list[str]:
    return [labels[max(range(len(labels)), key=lambda i: row[i])] for row in probabilities]


def evaluate_predictions(
    gold: list[str], probabilities: list[list[float]], labels: tuple[str, ...] = LABELS
) -> dict[str, Any]:
    if len(gold) != len(probabilities) or not gold:
        raise ValueError("gold/probability rows must have equal nonzero length")
    predictions = hard_predictions(probabilities, labels)
    confusion = [[0 for _ in labels] for _ in labels]
    label_index = {label: index for index, label in enumerate(labels)}
    nll = 0.0
    brier = 0.0
    for truth, predicted, distribution in zip(gold, predictions, probabilities):
        if truth not in label_index or len(distribution) != len(labels):
            raise ValueError("prediction uses incompatible fixed label list")
        if any((not math.isfinite(value) or value <= 0.0) for value in distribution):
            raise ValueError("probabilities must be finite and strictly positive after smoothing")
        if not math.isclose(sum(distribution), 1.0, abs_tol=1e-9):
            raise ValueError("probabilities must sum to one")
        truth_index = label_index[truth]
        confusion[truth_index][label_index[predicted]] += 1
        nll -= math.log(distribution[truth_index])
        brier += sum((value - float(index == truth_index)) ** 2 for index, value in enumerate(distribution))
    per_class = {}
    f1_values = []
    for index, label in enumerate(labels):
        support = sum(confusion[index])
        true_positive = confusion[index][index]
        predicted_count = sum(row[index] for row in confusion)
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {"support": support, "recall": recall, "precision": precision, "f1": f1}
    return {"macro_f1": sum(f1_values) / len(labels),
            "accuracy": sum(confusion[i][i] for i in range(len(labels))) / len(gold),
            "per_class": per_class, "confusion": {"labels": list(labels), "matrix": confusion},
            "nll": nll / len(gold), "brier": brier / len(gold), "rows": len(gold)}


def bootstrap_difference(
    gold: list[str], probabilities_a: list[list[float]], probabilities_b: list[list[float]],
    groups: list[str], labels: tuple[str, ...] = LABELS, *, replicates: int = 1000, seed: int = 0
) -> dict[str, Any]:
    if not (len(gold) == len(probabilities_a) == len(probabilities_b) == len(groups)):
        raise ValueError("paired bootstrap inputs must align")
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    keys = sorted(grouped)
    if not keys or replicates < 1:
        raise ValueError("bootstrap needs groups and positive replicates")
    rng = random.Random(seed)
    differences = []
    for _ in range(replicates):
        indices = []
        for _ in keys:
            indices.extend(grouped[rng.choice(keys)])
        metric_a = evaluate_predictions([gold[i] for i in indices], [probabilities_a[i] for i in indices], labels)
        metric_b = evaluate_predictions([gold[i] for i in indices], [probabilities_b[i] for i in indices], labels)
        differences.append(metric_a["macro_f1"] - metric_b["macro_f1"])
    differences.sort()
    lower = differences[max(0, math.floor(0.025 * replicates))]
    upper = differences[min(replicates - 1, math.ceil(0.975 * replicates) - 1)]
    observed = (evaluate_predictions(gold, probabilities_a, labels)["macro_f1"] -
                evaluate_predictions(gold, probabilities_b, labels)["macro_f1"])
    return {"metric": "macro_f1_difference_a_minus_b", "observed": observed,
            "interval_percentile_95": [lower, upper], "replicates": replicates,
            "seed": seed, "resampling_unit": "trajectory_id", "independent_groups": len(keys)}


def validate_manifest(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    if manifest.get("schema_version") != "trace-action-baseline-manifest-v1":
        raise ValueError("unsupported manifest schema_version")
    if tuple(manifest.get("labels", ())) != LABELS:
        raise ValueError(f"labels must exactly equal {list(LABELS)}")
    data = manifest.get("data", {})
    if set(data) != {"train", "dev"}:
        raise ValueError("data must contain exactly train and dev; held-out inputs are prohibited")
    paths = {key: (root / value).resolve() for key, value in data.items()}
    if paths["train"] == paths["dev"] or any("test" in path.name.lower() or "heldout" in path.name.lower() for path in paths.values()):
        raise ValueError("test/held-out paths are prohibited")
    output = (root / manifest["output_dir"]).resolve()
    allowed = (root / "artifacts/baseline_i3/preflight").resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError("output_dir must remain under artifacts/baseline_i3/preflight")
    candidates = manifest.get("logistic_candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
        raise ValueError("logistic_candidates must explicitly contain 1 to 4 candidates")
    ids = [candidate.get("id") for candidate in candidates]
    if any(not isinstance(ident, str) for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique strings")
    return {"train": paths["train"], "dev": paths["dev"], "output": output}


def run_manifest(manifest_path: Path, root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = validate_manifest(manifest, root)
    labels = tuple(manifest["labels"])
    question_id = manifest.get("question_id", "next_action")
    alpha = float(manifest["probability_smoothing_alpha"])
    if alpha <= 0:
        raise ValueError("probability_smoothing_alpha must be positive")
    train = load_records(paths["train"], labels, question_id)
    dev = load_records(paths["dev"], labels, question_id)
    train_groups = {row["trajectory_id"] for row in train}
    dev_groups = {row["trajectory_id"] for row in dev}
    if train_groups & dev_groups:
        raise ValueError("trajectory leakage between train and dev")
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    gold = [row["label"] for row in dev]
    states = [row["state"] for row in dev]
    majority = majority_model(train, labels, alpha)
    majority_probabilities = [majority["probabilities"] for _ in dev]
    transition = transition_model(train, labels, alpha, majority["probabilities"])
    transition_probabilities = transition_predict(transition, states)
    transition_evidence_rows = sum(previous_action_from_state(state) is not None for state in states)
    methods = {
        "majority": {"metrics": evaluate_predictions(gold, majority_probabilities, labels),
                     "config": {"tie_break": "first in fixed label list", "smoothing_alpha": alpha}},
        "transition": {"metrics": evaluate_predictions(gold, transition_probabilities, labels),
                        "config": {"state_evidence": "mapped tool name in RECENT HISTORY or LATEST RELEVANT OBSERVATION",
                                   "fallback": transition["fallback_policy"], "smoothing_alpha": alpha,
                                  "fit_trajectories": transition["fit_trajectories"],
                                  "fit_transitions": transition["fit_transitions"],
                                   "skipped_step_gaps": transition["skipped_step_gaps"],
                                   "dev_evidence_rows": transition_evidence_rows,
                                   "dev_fallback_rows": len(dev) - transition_evidence_rows,
                                   "dev_previous_action_invalid_rows": len(dev) - transition_evidence_rows}},
    }
    candidate_results = []
    fitted: dict[str, SparseLogistic] = {}
    seed = int(manifest["seed"])
    for candidate in manifest["logistic_candidates"]:
        started = time.monotonic()
        model = fit_logistic([row["state"] for row in train], [row["label"] for row in train], labels, candidate, seed)
        probabilities = model.predict_proba(states)
        metrics = evaluate_predictions(gold, probabilities, labels)
        elapsed = time.monotonic() - started
        fitted[candidate["id"]] = model
        candidate_results.append({"id": candidate["id"], "config": model.config, "metrics": metrics,
                                  "fit_and_dev_seconds": elapsed})
    # Highest macro-F1 wins; lexicographically smallest ID is the declared exact-score tie-break.
    selected_result = sorted(candidate_results, key=lambda item: (-item["metrics"]["macro_f1"], item["id"]))[0]
    selected = fitted[selected_result["id"]]
    selected_probabilities = selected.predict_proba(states)
    methods["tfidf_logistic"] = {"metrics": selected_result["metrics"], "config": selected.config,
                                 "selection": {"metric": "dev_macro_f1", "tie_break": "candidate id ascending",
                                               "selected_id": selected_result["id"],
                                               "candidate_budget": len(candidate_results)}}
    probabilities_by_method = {"majority": majority_probabilities, "transition": transition_probabilities,
                               "tfidf_logistic": selected_probabilities}
    bootstrap_config = manifest["bootstrap"]
    bootstrap = {}
    for baseline in ("majority", "transition"):
        bootstrap[f"tfidf_logistic_minus_{baseline}"] = bootstrap_difference(
            gold, selected_probabilities, probabilities_by_method[baseline],
            [row["trajectory_id"] for row in dev], labels,
            replicates=int(bootstrap_config["replicates"]), seed=int(bootstrap_config["seed"]),
        )
    predictions_path = output / "dev_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(dev):
            methods_row = {}
            for name, probabilities in probabilities_by_method.items():
                methods_row[name] = {"prediction": hard_predictions([probabilities[index]], labels)[0],
                                     "probabilities": {label: probabilities[index][j] for j, label in enumerate(labels)}}
            handle.write(canonical({"id": row["id"], "trajectory_id": row["trajectory_id"],
                                    "gold": row["label"], "methods": methods_row}) + "\n")
    model_path = output / "selected_model.json.gz"
    with model_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                handle.write(canonical(selected.serializable()) + "\n")
    selected_config = {"selected_id": selected_result["id"], "selection_metric": "dev_macro_f1",
                       "tie_break": "candidate id ascending", "candidate_budget": len(candidate_results),
                       "config": selected.config, "labels": list(labels), "input_field": "state"}
    (output / "selected_config.json").write_text(json.dumps(selected_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": "trace-action-baseline-result-v1",
        "status": ("corrected_v3_dev_only" if "corrected_v3" in manifest_path.name
                    else "v2_interface_smoke_not_final_evidence"),
        "starting_commit": "2d4408d", "manifest": str(manifest_path.relative_to(root)),
        "data": {"train": {"path": str(paths["train"].relative_to(root)), "sha256": file_sha256(paths["train"]),
                            "rows": len(train), "trajectories": len(train_groups)},
                 "dev": {"path": str(paths["dev"].relative_to(root)), "sha256": file_sha256(paths["dev"]),
                          "rows": len(dev), "trajectories": len(dev_groups)}},
        "labels": list(labels), "primary_metric": "macro_f1", "methods": methods,
        "logistic_candidates": candidate_results, "bootstrap": bootstrap,
        "probability_metrics": {"nll": "mean negative log probability of gold label",
                                "brier": "mean multiclass sum of squared probability errors",
                                "count_smoothing": f"symmetric additive alpha={alpha}; logistic softmax needs no smoothing"},
        "artifacts": {"predictions": predictions_path.name, "selected_model": model_path.name,
                      "selected_config": "selected_config.json"},
        "limitations": (["corrected v3 result is train/dev evidence only; no held-out split was read",
                          "candidate selection used dev macro-F1 and is subject to dev-selection optimism",
                          "transition marks rows without recoverable previous action invalid and uses only the declared training-majority fallback"]
                        if "corrected_v3" in manifest_path.name else
                        ["v2 train/dev are interface-only and labels precede corrected v3 target construction",
                         "dev selected the TF-IDF candidate and is not held-out final evidence",
                         "transition uses only previous-action evidence recoverable from compact state text"]),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if "corrected_v3" in manifest_path.name:
        (root / "artifacts/baseline_i3/result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result
