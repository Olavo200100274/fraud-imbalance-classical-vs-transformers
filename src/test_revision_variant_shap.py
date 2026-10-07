"""Synthetic checks for complete-feature corrected transfer interpretation."""

import unittest

import numpy as np

from revision_variant_shap import contribution_matrix, ranked_importance


class VariantTreeShapTests(unittest.TestCase):
    def test_native_expected_value_is_not_a_feature(self):
        values, baseline = contribution_matrix(np.array([[.2, -.3, -4.0], [.4, -.1, -4.0]]), 2)
        np.testing.assert_array_equal(values, [[.2, -.3], [.4, -.1]])
        np.testing.assert_array_equal(baseline, [-4.0, -4.0])

    def test_invalid_native_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            contribution_matrix(np.ones((2, 2)), 2)

    def test_complete_magnitudes_survive_top_k_selection(self):
        top, complete = ranked_importance(["first", "second", "third"], np.array([.1, .8, .2]), 2)
        self.assertEqual(top, ["second", "third"])
        self.assertEqual(list(complete), ["second", "third", "first"])
        self.assertEqual(complete["first"], .1)

    def test_equal_values_have_stable_source_order(self):
        top, unused = ranked_importance(["a", "b", "c"], np.ones(3), 2)
        self.assertEqual(top, ["a", "b"])

    def test_negative_importance_is_rejected(self):
        with self.assertRaises(ValueError):
            ranked_importance(["a", "b"], np.array([.1, -.1]), 2)


if __name__ == "__main__":
    unittest.main()
