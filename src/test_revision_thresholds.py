"""Synthetic unit checks for validation-only revision threshold derivation."""

import json
import unittest

import numpy as np

from revision_thresholds import derive_thresholds, standard_json, verify_primary_threshold


class SavedValidationThresholdTests(unittest.TestCase):
    def test_classical_median_uses_five_distinct_folds(self):
        labels = np.tile([0, 0, 1, 1], 5)
        scores = np.concatenate([np.array([.1, .2, .7, .8]) + index * .01 for index in range(5)])
        folds = np.repeat(np.arange(1, 6), 4)
        result = derive_thresholds(labels, scores, folds, "lgbm")
        np.testing.assert_allclose(result["thresholds_per_fold"]["max_f2"], [.7, .71, .72, .73, .74])
        self.assertAlmostEqual(result["thresholds_median"]["max_f2"], .72)
        self.assertEqual(len(result["validation_folds"]), 5)

    def test_transformer_uses_one_saved_holdout(self):
        result = derive_thresholds(np.array([0, 0, 1, 1]), np.array([.1, .2, .7, .8]),
                                   np.ones(4, dtype=int), "fttransformer")
        self.assertAlmostEqual(result["thresholds_median"]["max_f2"], .7)
        self.assertEqual(result["thresholds_per_fold"]["max_f2"], [.7])

    def test_primary_threshold_mismatch_is_rejected(self):
        verify_primary_threshold(.123456789, .123456789)
        with self.assertRaises(ValueError):
            verify_primary_threshold(.123456789, .123457)

    def test_classical_holdout_substitution_is_rejected(self):
        with self.assertRaises(ValueError):
            derive_thresholds(np.array([0, 1]), np.array([.1, .8]), np.ones(2), "rf")

    def test_non_finite_validation_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            derive_thresholds(np.array([0, 1]), np.array([.1, np.nan]), np.ones(2), "fttransformer")

    def test_unattainable_precision_is_standard_json_null(self):
        value = standard_json({"thresholds": [float("inf"), .5]})
        self.assertEqual(value["thresholds"], [None, .5])
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)


if __name__ == "__main__":
    unittest.main()
