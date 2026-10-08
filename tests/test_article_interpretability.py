"""Small checks of portable plot grouping and overwrite/additivity guards."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from article_interpretability import _prepare_beeswarm_arrays, _waterfall_sequence, regenerate


class InterpretabilityTests(unittest.TestCase):
    def test_complete_rows_signed_residual_and_original_colour(self):
        values = np.arange(60, dtype=np.float64).reshape(3, 20) - 24
        features = np.arange(60, dtype=np.float64).reshape(3, 20) / 7
        result = _prepare_beeswarm_arrays(values, features, [f"feature_{index}" for index in range(20)])
        order = np.argsort(np.abs(values).mean(axis=0))[::-1]
        np.testing.assert_array_equal(result["shap_values"][:, :14], values[:, order[:14]])
        np.testing.assert_array_equal(result["shap_values"][:, 14],
                                      np.sum([values[:, index] for index in order[14:]], axis=0))
        np.testing.assert_array_equal(result["feature_values"][:, 14], features[:, order[14]])
        np.testing.assert_allclose(result["shap_values"].sum(axis=1), values.sum(axis=1))
        self.assertEqual(result["feature_names"][-1], "Sum of 6 other features")
        self.assertEqual(result["shap_values"].dtype, values.dtype)

    def test_malformed_plot_inputs_are_rejected(self):
        for values, features in ((np.ones((2, 14)), np.ones((2, 14))),
                                 (np.ones((2, 15)), np.ones((3, 15))),
                                 (np.full((2, 15), np.nan), np.ones((2, 15)))):
            with self.assertRaises(ValueError):
                _prepare_beeswarm_arrays(values, features, [str(index) for index in range(values.shape[1])])

    def test_waterfall_sorts_displayed_features_and_preserves_residual(self):
        record = {"displayed_signed_features": {"a": .2, "b": -.9},
                  "remaining_signed_contribution": .3, "expected_logit": -1., "output_logit": -1.4}
        names, values = _waterfall_sequence(record)
        self.assertEqual(names, ["b", "a", "Remaining features"])
        np.testing.assert_allclose(values, [-.9, .2, .3])
        record["output_logit"] = 2.
        with self.assertRaises(ValueError):
            _waterfall_sequence(record)

    def test_replay_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                regenerate(Path(directory), Path(directory))


if __name__ == "__main__":
    unittest.main()
