"""Local, fixed-schema Laya next-action classifier and output contract."""
from __future__ import annotations

import math
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from .bundle_integrity import verify_tree
except ImportError:  # direct execution from a copied standalone directory
    from bundle_integrity import verify_tree

LABELS = ("read", "search", "edit", "execute", "other_tool", "respond_or_finish")
CHOICE_ORDER = ("edit", "execute", "other_tool", "read", "respond_or_finish", "search")
FROZEN_INSTRUCTIONS = "Choose the next observable agent action type."
FROZEN_CRITERIA = {
    "edit": "Create or modify files or structured content.",
    "execute": "Execute a shell command, program, or test.",
    "other_tool": "Call another tool not covered by the named action types.",
    "read": "Read a known file or resource.",
    "respond_or_finish": "Respond without an executable tool call or finish the task.",
    "search": "Search or list files, symbols, or resources.",
}


class InputError(ValueError):
    pass


def validate_input(record: object) -> dict:
    """Validate deployment input (state + one next_action choice question).

    Evaluation-only fields (gold/metadata) are neither needed nor accepted.
    """
    if not isinstance(record, Mapping) or set(record) != {"state", "questions"}:
        raise InputError("input must contain exactly state and questions")
    if not isinstance(record["state"], str) or not record["state"].strip():
        raise InputError("state must be a non-empty string")
    questions = record["questions"]
    if not isinstance(questions, Mapping) or set(questions) != {"next_action"}:
        raise InputError("questions must contain exactly next_action")
    question = questions["next_action"]
    if (not isinstance(question, Mapping) or not {"type", "criteria"} <= set(question)
            or set(question) - {"type", "criteria", "instructions"}
            or question.get("type") != "choice"
            or not isinstance(question.get("criteria"), Mapping)
            or tuple(question["criteria"].keys()) != CHOICE_ORDER):
        raise InputError("next_action must be choice with the frozen six-label option order")
    if question.get("instructions") != FROZEN_INSTRUCTIONS or dict(question["criteria"]) != FROZEN_CRITERIA:
        raise InputError("next_action instructions and six descriptions must match the frozen question exactly")
    return dict(record)


def validate_probabilities(values: Sequence[float]) -> dict[str, float]:
    if isinstance(values, (str, bytes)) or len(values) != len(LABELS):
        raise InputError("model must return six probabilities in LABELS order")
    probs = tuple(values)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in probs):
        raise InputError("probabilities must be finite, non-negative numbers")
    if not math.isclose(sum(probs), 1.0, abs_tol=2e-6, rel_tol=0):
        raise InputError("probabilities must sum to one")
    return dict(zip(LABELS, map(float, probs)))


def predict(record: object, predictor, *, calibrator=None) -> dict[str, float]:
    """Invoke supplied local predictor. Calibration is optional and external."""
    item = validate_input(record)
    raw = validate_probabilities(predictor(item))
    if calibrator is None:
        return raw
    return validate_probabilities(calibrator(tuple(raw[label] for label in LABELS)))


class LocalLayaPredictor:
    """Load the bundled seed-42 checkpoint using only bundle-relative assets."""
    def __init__(self, bundle: Path | None = None, device: str = "cpu"):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        self.bundle = (bundle or Path(__file__).parent / "bundle").resolve()
        if device != "cpu":
            raise ValueError("this release interface is CPU-only")
        verify_tree(self.bundle.parent)
        if not (self.bundle / "model/model.safetensors").is_file():
            raise FileNotFoundError("bundle model weights are missing")
        sys.path.insert(0, str(self.bundle / "upstream"))
        from laya.common import QTYPES, build_model, build_sequence, render_options

        self.torch = torch
        self.build_sequence = build_sequence
        self.render_options = render_options
        self.qtypes = QTYPES
        cfg = json.loads((self.bundle / "rl_agent_config.json").read_text())
        exp = json.loads((self.bundle / "experiment_config.json").read_text())
        cfg.update(max_len=exp["model"]["max_len"], head_max_len=exp["model"]["head_max_len"])
        self.model = build_model(cfg, encoder_dir=str(self.bundle / "encoder"))
        self.model.load_state_dict(load_file(self.bundle / "model/model.safetensors", device="cpu"), strict=True)
        self.model.encoder.config.reference_compile = False
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.bundle / "tokenizer", local_files_only=True)
        self.max_len = exp["model"]["max_len"]
        self.head_max_len = exp["model"]["head_max_len"]
        self.truncate_left = exp["model"]["truncate_left"]
        calibration = json.loads((self.bundle / "calibration.json").read_text())
        self.temperature = calibration["temperature"]
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("calibration temperature must be positive and finite")

    def __call__(self, record: Mapping) -> list[float]:
        raw, _ = self.predict_pair(record)
        return [raw[label] for label in LABELS]

    def predict_pair(self, record: Mapping) -> tuple[dict[str, float], dict[str, float]]:
        """One forward pass; raw and dev-temperature-scaled six-class outputs."""
        item = validate_input(record)
        q = item["questions"]["next_action"]
        internal = {"t": q["type"], "ins": q.get("instructions", ""), "crit": q["criteria"]}
        ids, markers = self.build_sequence(self.tokenizer, item["state"], internal,
                                            self.max_len, self.head_max_len,
                                            truncate_left=self.truncate_left)
        if len(markers) != len(self.render_options(internal)):
            raise InputError("choice markers exceed model head budget")
        # Single-record equivalent of Laya collate_items; no training/evaluation fields.
        t = self.torch
        input_ids = t.tensor([ids], dtype=t.long)
        attention = t.ones_like(input_ids)
        marker_pos = t.tensor([markers], dtype=t.long)
        marker_mask = t.ones_like(marker_pos, dtype=t.bool)
        qtype = t.tensor([self.qtypes["choice"]], dtype=t.long)
        with t.inference_mode():
            logits, _ = self.model(input_ids, attention, marker_pos, marker_mask, qtype)
            # Training criterion order is frozen CHOICE_ORDER.
            choice_logits = logits[0, :len(CHOICE_ORDER)].float()
            probs = t.softmax(choice_logits, dim=-1).tolist()
            calibrated = t.softmax(choice_logits / self.temperature, dim=-1).tolist()
        # Public mapping is fixed LABELS order, not choice criterion order.
        by_choice = dict(zip(CHOICE_ORDER, probs))
        by_calibrated_choice = dict(zip(CHOICE_ORDER, calibrated))
        return (validate_probabilities([by_choice[label] for label in LABELS]),
                validate_probabilities([by_calibrated_choice[label] for label in LABELS]))
