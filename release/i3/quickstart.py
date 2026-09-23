"""One-input CPU example from a self-contained local bundle."""
import json
from pathlib import Path

from inference import FROZEN_CRITERIA, FROZEN_INSTRUCTIONS, LocalLayaPredictor

bundle = Path(__file__).parent / "bundle"
predictor = LocalLayaPredictor(bundle=bundle, device="cpu")
sample = {"state": "The requested change is small and can be applied directly.",
          "questions": {"next_action": {"type": "choice", "instructions": FROZEN_INSTRUCTIONS,
                                      "criteria": FROZEN_CRITERIA}}}
raw, calibrated = predictor.predict_pair(sample)
print(json.dumps({"raw": raw, "dev_temperature_scaled": calibrated}, indent=2))
