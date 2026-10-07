"""Checks for full-precision saved-score groups and explicit TEST-row loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from revision_lgbm_diagnostics import _read_test_rows, score_diagnostics, source_array_hashes
from experiment_protocol import file_sha256


class ScoreGroupingTests(unittest.TestCase):
    def test_exact_groups_reproduce_average_precision_without_tie_breaking(self):
        labels = np.array([1, 0, 1, 0, 1])
        scores = np.array([0.9, 0.9, 0.5, 0.4, 0.2])
        report = score_diagnostics(labels, scores)
        self.assertAlmostEqual(report["average_precision"], 0.5 / 3 + (2 / 3) / 3 + 0.6 / 3)
        group = report["top_exact_score_groups"][0]
        self.assertEqual((group["rows"], group["fraud"], group["non_fraud"]), (2, 1, 1))
        self.assertEqual(group["cumulative_precision"], 0.5)
        self.assertEqual(group["cumulative_recall"], 1 / 3)
        self.assertEqual(report["unique_exact_scores"], 4)
        self.assertEqual(report["exact_one_scores"], 0)
        self.assertEqual(report["exact_zero_scores"], 0)

    def test_near_equal_scores_are_not_promoted_to_exact_ties(self):
        scores = np.array([0.9, 0.9 - 1e-9, 0.5, 0.2])
        report = score_diagnostics(np.array([1, 0, 1, 0]), scores)
        self.assertEqual(report["top_exact_score_groups"][0]["rows"], 1)
        self.assertEqual(report["near_maximum_groups"]["1e-08"]["rows"], 2)
        self.assertEqual(report["unique_exact_scores"], 4)

    def test_invalid_or_misaligned_scores_are_rejected(self):
        for scores in (np.array([0.5]), np.array([0.5, np.nan])):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                score_diagnostics(np.array([0, 1]), scores)


class ExplicitRowLoaderTests(unittest.TestCase):
    def test_source_array_hash_fields_pin_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = (("y_test_scores.npy", "source_scores_sha256", np.array([0.1, 0.9])),
                       ("y_test.npy", "source_labels_sha256", np.array([0, 1])),
                       ("test_row_indices.npy", "source_indices_sha256", np.array([20, 3])))
            for filename, _, values in records:
                np.save(directory / filename, values)
            hashes = source_array_hashes(directory)
            self.assertEqual(set(hashes), {field for _, field, _ in records})
            for filename, field, _ in records:
                self.assertEqual(hashes[field], file_sha256(directory / filename))

    def test_chunked_loading_restores_saved_unsorted_row_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            pd.DataFrame({"Time": np.arange(20), "V1": np.arange(20) * 0.1,
                          "Class": np.arange(20) % 2}).to_csv(path, index=False)
            indices = np.array([17, 3, 8, 14])
            frame = _read_test_rows(path, indices, indices % 2, chunk_rows=6)
            np.testing.assert_array_equal(frame.index.to_numpy(), indices)
            np.testing.assert_array_equal(frame["Time"].to_numpy(), indices)
            self.assertNotIn("Class", frame.columns)
            with self.assertRaisesRegex(ValueError, "labels"):
                _read_test_rows(path, indices, np.ones(4), chunk_rows=6)

    def test_duplicate_original_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            _read_test_rows("not_read.csv", np.array([1, 1]), np.array([0, 0]))


if __name__ == "__main__":
    unittest.main()
