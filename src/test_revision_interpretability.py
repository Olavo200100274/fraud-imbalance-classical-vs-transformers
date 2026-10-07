"""Unit tests for complete and additive interpretability artefact generation."""

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from generate_revision_interpretability import (ROOT, ShapArtefacts, apply_operating_baseline,
                                               group_contributions, validate_output_directory)


class GroupedContributionTests(unittest.TestCase):
    def test_signed_grouping_preserves_additivity(self):
        values = np.array([[.2, .4, -.3], [-.1, -.2, .5]])
        grouped, names = group_contributions(values, ["income", "device_os_linux", "device_os_other"], absolute=False)
        self.assertEqual(names, ["income", "device_os"])
        np.testing.assert_allclose(grouped.sum(axis=1), values.sum(axis=1))
        np.testing.assert_allclose(grouped[:, 1], [.1, .3])

    def test_absolute_grouping_is_not_absolute_signed_grouping(self):
        values = np.array([[.4, -.3]])
        grouped, names = group_contributions(values, ["device_os_linux", "device_os_other"], absolute=True)
        self.assertEqual(names, ["device_os"])
        np.testing.assert_allclose(grouped, [[.7]])

    def test_name_prefix_does_not_absorb_unrelated_feature(self):
        grouped, names = group_contributions(np.array([[1., 2.]]), ["device_fraud_count", "income"], absolute=False)
        self.assertEqual(names, ["device_fraud_count", "income"])
        self.assertEqual(grouped.shape, (1, 2))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            group_contributions(np.ones((2, 3)), ["income"], absolute=True)

    def test_preserved_output_locations_are_rejected(self):
        for location in ("results", "results_thesis/figures/baf", "Overleaf/figures"):
            with self.assertRaises(ValueError):
                validate_output_directory(ROOT / location)
        self.assertEqual(validate_output_directory(ROOT / "results_revision/20261005/derived"),
                         (ROOT / "results_revision/20261005/derived").resolve())


class OperatingBaselineTests(unittest.TestCase):
    def artefact(self):
        return ShapArtefacts("lgbm", Path("historical"), ["income"], np.zeros((3, 1)),
                            np.array([1, 0, 1]), np.array([.8, .8, .2]), .5,
                            ["income"], np.zeros(1), 0.0, 0.0)

    def create_run(self, run, threshold, scores):
        (run / "config.json").write_text(json.dumps({"dataset": "baf_base", "model": "lgbm",
                                                    "strategy": "none", "threshold_exact": threshold}), encoding="utf-8")
        (run / "metrics_test.json").write_text("{}", encoding="utf-8")
        np.save(run / "y_test.npy", [1, 0, 1])
        np.save(run / "y_test_scores.npy", scores)

    def test_recovered_exact_threshold_requires_equivalent_saved_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.create_run(run, .50000001, [.8, .8, .2])
            artefact = self.artefact()
            result = apply_operating_baseline(artefact, run)
            self.assertEqual(artefact.threshold, .50000001)
            self.assertTrue(all(result["selected_case_classes_unchanged"].values()))

    def test_sub_picounit_score_difference_is_reported_without_changing_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.create_run(run, .5, [.8 + 1e-13, .8, .2])
            result = apply_operating_baseline(self.artefact(), run)
            self.assertFalse(result["scores_bitwise_equal"])
            self.assertGreater(result["maximum_absolute_score_difference"], 0)
            self.assertEqual(result["threshold_decisions_changed"], 0)

    def test_small_difference_cannot_change_operating_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.create_run(run, .8 + 5e-14, [.8 + 1e-13, .8, .2])
            with self.assertRaisesRegex(ValueError, "change primary threshold decisions"):
                apply_operating_baseline(self.artefact(), run)

    def test_changed_primary_scores_cannot_reuse_historical_shap(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.create_run(run, .5, [.81, .8, .2])
            with self.assertRaisesRegex(ValueError, "changed primary TEST scores"):
                apply_operating_baseline(self.artefact(), run)

    def test_changed_error_class_is_not_silently_relabelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.create_run(run, .9, [.8, .8, .2])
            with self.assertRaisesRegex(ValueError, "changes error class"):
                apply_operating_baseline(self.artefact(), run)

    def test_recovered_final_model_provenance_must_match_the_explained_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "current"
            historical = root / "historical"
            run.mkdir()
            historical.mkdir()
            self.create_run(run, .5, [.8, .8, .2])
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            config["source_final_model_sha256"] = "0" * 64
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (historical / "model.joblib").write_bytes(b"Synthetic provenance fixture only")
            artefact = self.artefact()
            artefact.run_dir = historical
            with self.assertRaisesRegex(ValueError, "different historical final model"):
                apply_operating_baseline(artefact, run)


if __name__ == "__main__":
    unittest.main()
