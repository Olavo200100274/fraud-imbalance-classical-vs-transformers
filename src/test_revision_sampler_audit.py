import unittest

import numpy as np
from experiment_protocol import sampler_diagnostics
from strategies.balancing import get_sampler

from revision_sampler_audit import compare_stage_records


def record(**updates):
    value = {"stage": "final_dev", "rows_before": 10, "rows_after": 16,
             "class_counts_before": {"0": 8, "1": 2}, "class_counts_after": {"0": 8, "1": 8},
             "X_before_sha256": "before_X", "y_before_sha256": "before_y",
             "X_after_sha256": "after_X", "y_after_sha256": "after_y",
             "cleaner_removed_rows": 0, "rows_after_smote_before_cleaning": 16}
    value.update(updates)
    return value


class SamplerAuditTests(unittest.TestCase):
    def test_zero_removals_and_equal_arrays_are_valid(self):
        report = compare_stage_records([record()], [record()])
        self.assertTrue(report["final_dev"]["resampled_inputs_identical"])

    def test_zero_removals_but_changed_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "preserve"):
            compare_stage_records([record()], [record(X_after_sha256="different")])

    def test_actual_removals_are_reported_not_suppressed(self):
        report = compare_stage_records([record()], [record(rows_after=15, cleaner_removed_rows=1,
                                                           X_after_sha256="different", class_counts_after={"0": 7, "1": 8})])
        self.assertEqual(report["final_dev"]["cleaner_removed_rows"], 1)
        self.assertFalse(report["final_dev"]["resampled_inputs_identical"])

    def test_mismatched_training_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same training"):
            compare_stage_records([record()], [record(X_before_sha256="other")])

    def test_missing_fold_or_duplicate_stage_is_rejected(self):
        for records in ([], [record(), record()]):
            with self.assertRaisesRegex(ValueError, "unique training stages"):
                compare_stage_records([record()], records)

    def test_installed_composite_decomposition_reproduces_its_outputs(self):
        generator = np.random.default_rng(42)
        features = generator.normal(size=(50, 3))
        labels = np.asarray([0] * 40 + [1] * 10)
        composite = get_sampler("smote_tomek")
        expected_X, expected_y = composite.fit_resample(features, labels)
        reconstructed = get_sampler("smote_tomek")
        reconstructed._validate_estimator()
        smote_X, smote_y = reconstructed.smote_.fit_resample(features, labels)
        actual_X, actual_y = reconstructed.tomek_.fit_resample(smote_X, smote_y)
        self.assertTrue(np.array_equal(expected_X, actual_X))
        self.assertTrue(np.array_equal(expected_y, actual_y))
        first = sampler_diagnostics(reconstructed.smote_, features, labels, smote_X, smote_y, "toy")
        second = sampler_diagnostics(reconstructed, features, labels, actual_X, actual_y, "toy")
        report = compare_stage_records([first], [second])
        self.assertEqual(report["toy"]["cleaner_removed_rows"], len(smote_y) - len(actual_y))


if __name__ == "__main__":
    unittest.main()
