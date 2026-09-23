import unittest

try:
    from .inference import FROZEN_CRITERIA, FROZEN_INSTRUCTIONS, InputError, LABELS, predict, validate_input
except ImportError:  # direct execution from a copied standalone directory
    from inference import FROZEN_CRITERIA, FROZEN_INSTRUCTIONS, InputError, LABELS, predict, validate_input


def item():
    return {"state": "context", "questions": {"next_action": {
        "type": "choice", "instructions": FROZEN_INSTRUCTIONS,
        "criteria": dict(FROZEN_CRITERIA)}}}


class InputContractTests(unittest.TestCase):
    def test_valid_input_and_label_mapping(self):
        self.assertEqual(validate_input(item())["state"], "context")
        result = predict(item(), lambda _: [1 / 6] * 6)
        self.assertEqual(tuple(result), LABELS)

    def test_rejects_extra_evaluation_fields(self):
        invalid = item() | {"gold": {}}
        with self.assertRaises(InputError):
            validate_input(invalid)

    def test_rejects_wrong_choice_order(self):
        bad = item()
        bad["questions"]["next_action"]["criteria"] = {"read": ""}
        with self.assertRaises(InputError):
            validate_input(bad)

    def test_rejects_nested_gold_and_nontext_descriptions(self):
        bad = item()
        bad["questions"]["next_action"]["gold"] = "edit"
        with self.assertRaises(InputError):
            validate_input(bad)
        bad = item()
        bad["questions"]["next_action"]["criteria"]["edit"] = "edit any unrelated thing"
        with self.assertRaises(InputError):
            validate_input(bad)
        bad = item()
        bad["questions"]["next_action"]["criteria"]["edit"] = {"metadata": 1}
        with self.assertRaises(InputError):
            validate_input(bad)

    def test_rejects_invalid_output_and_bad_calibration(self):
        with self.assertRaises(InputError):
            predict(item(), lambda _: [0.2] * 6)
        with self.assertRaises(InputError):
            predict(item(), lambda _: [1 / 6] * 6, calibrator=lambda _: [float("nan")] * 6)


if __name__ == "__main__":
    unittest.main()
