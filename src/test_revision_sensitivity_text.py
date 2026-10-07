"""Synthetic checks for isolated, source-grounded sensitivity table export."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.metrics import compute_all_metrics
from experiment_protocol import PROTOCOL_VERSION, array_sha256, file_sha256
from revision_paired_analysis import MODELS, PAIRED_SOURCE_FILES, paired_difference
import revision_sensitivity_text as text


class SensitivityTextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        self.indices = np.arange(100, 108, dtype=np.int64)
        self.first_scores = np.array([.9, .8, .7, .6, .5, .4, .3, .2])
        self.second_scores = np.array([.1, .9, .2, .8, .3, .7, .4, .6])
        self.primary, self.sensitivity = {"runs": {}, "historical_runs": {}}, {"runs": {}}
        self.report = {"status": "complete", "missingness": {}, "categorical_controls": {},
                       "source_manifest_sha256": "earlier training manifest, intentionally not current",
                       "sensitivity_manifest_sha256": "earlier sensitivity manifest"}
        for model in MODELS:
            first = self.make_run(f"preserve_{model}", model, "none", "preserve", self.first_scores)
            second = self.make_run(f"indicators_{model}", model, "none", "nan_indicators", self.second_scores)
            self.pin(self.primary, model, "none", first)
            self.pin(self.sensitivity, model, "none", second)
            self.report["missingness"][model] = self.standard_pair(first, second)
        first = self.make_run("ft_smote", "fttransformer", "smote", "preserve", self.first_scores)
        second = self.make_run("ft_smotenc", "fttransformer", "smotenc_control", "preserve", self.second_scores)
        self.pin(self.primary, "fttransformer", "smote", first)
        self.pin(self.primary, "fttransformer", "smotenc_control", second)
        self.report["categorical_controls"]["fttransformer"] = self.standard_pair(first, second)
        first = self.make_run("historical_cb_smote", "catboost", "smote", "preserve", self.first_scores, historical=True)
        second = self.make_run("cb_smotenc", "catboost", "smotenc_control", "preserve", self.second_scores)
        self.pin(self.primary, "catboost", "smotenc_control", second)
        self.primary["historical_runs"]["baf_base/catboost/smote"] = {
            "run_dir": str(first), "config_sha256": file_sha256(first / "config.json")}
        generic = self.standard_pair(first, second, historical=True)
        comparison = {
            "source_run": str(first), "source_config_sha256": file_sha256(first / "config.json"),
            "source_artefacts_sha256": {name: file_sha256(first / name) for name in
                                       ("config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy")},
            "raw_file_sha256": "synthetic raw-file identity", "shared_test_row_indices_sha256": array_sha256(self.indices),
            "test_labels_exactly_equal": True, "historical_test_indices_explicitly_saved": False,
            "primary_threshold": .5, "primary_threshold_precision": "historical saved six-decimal threshold; confusion counts reproduced",
            "control_threshold_full_precision": .5,
            "primary_metrics": self.read(first / "metrics_test.json"), "control_metrics": self.read(second / "metrics_test.json"),
            "difference_full_precision": {old: generic["difference_second_minus_first"][new] for old, new in
                                          (("PR-AUC", "average_precision"), ("F2", "F2"), ("ROC-AUC", "roc_auc"))},
            "bootstrap_iterations": 1000, "bootstrap_seed": 42,
            "test_used_for_policy_or_hyperparameter_selection": False,
            "paired_difference_bootstrap": {old: {"ci_95_percent": generic["paired_95_percentile_intervals"][new],
                                                   "valid_replicates": 1000} for old, new in
                                            (("PR-AUC", "average_precision"), ("F2", "F2"), ("ROC-AUC", "roc_auc"))}}
        self.write(second / "primary_smote_comparison.json", comparison)
        self.report["categorical_controls"]["catboost"] = comparison
        self.report["catboost_control_source"] = {"run_dir": str(second), "artefacts_sha256": {
            name: file_sha256(second / name) for name in (*PAIRED_SOURCE_FILES, "primary_smote_comparison.json")}}
        self.report_path, self.primary_path, self.sensitivity_path = (self.root / name for name in
                                                                   ("paired_analysis.json", "primary.json", "sensitivity.json"))
        self.persist()

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def persist(self):
        for path, value in ((self.report_path, self.report), (self.primary_path, self.primary),
                            (self.sensitivity_path, self.sensitivity)):
            self.write(path, value)

    def make_run(self, name, model, strategy, missing_policy, scores, historical=False):
        run = self.root / name
        run.mkdir()
        config = {"dataset": "baf_base", "model": model,
                  "strategy": "n/a" if model == "ocsvm" and strategy == "none" else strategy,
                  "missing_policy": missing_policy, "dataset_hash_sha256": "synthetic raw-file identity",
                  "best_params": {"classifier__depth": 3}, "split_seed": 42}
        if not historical:
            config.update(protocol_version=PROTOCOL_VERSION, threshold_exact=.5)
            self.write(run / "completed.json", {"status": "complete", "protocol_version": PROTOCOL_VERSION})
            np.save(run / "test_row_indices.npy", self.indices)
        self.write(run / "config.json", config)
        self.write(run / "metrics_test.json", compute_all_metrics(self.labels, scores, .5))
        np.save(run / "y_test.npy", self.labels)
        np.save(run / "y_test_scores.npy", scores)
        return run

    @staticmethod
    def pin(manifest, model, strategy, run):
        manifest["runs"][f"baf_base/{model}/{strategy}"] = {
            "run_dir": str(run), "config_sha256": file_sha256(run / "config.json")}

    def standard_pair(self, first, second, historical=False):
        # The fixture supplies synthetic interval metadata; no experimental
        # bootstrap is run or represented by this test construction.
        evidence = paired_difference(self.labels, np.load(first / "y_test_scores.npy"), .5,
                                     np.load(second / "y_test_scores.npy"), .5, iterations=0)
        evidence.update(bootstrap_requested=1000, bootstrap_valid=1000,
                        paired_95_percentile_intervals={name: [max(-1.0, value - .02), min(1.0, value + .02)]
                                                       for name, value in evidence["difference_second_minus_first"].items()},
                        first_run=str(first), second_run=str(second), first_threshold=.5, second_threshold=.5,
                        rows=len(self.labels), fraud_rows=int(self.labels.sum()), policy_selected_using_TEST=False)
        if not historical:
            for role, run in (("first", first), ("second", second)):
                evidence[f"{role}_artefacts_sha256"] = {name: file_sha256(run / name) for name in PAIRED_SOURCE_FILES}
                evidence[f"{role}_config_sha256"] = file_sha256(run / "config.json")
        return evidence

    def generate(self):
        return text.generate(self.report_path, self.primary_path, self.sensitivity_path, self.root / "new_text")

    def test_real_verifier_and_export_are_source_grounded(self):
        qa = self.generate()
        fragment = Path(qa["fragment_path"]).read_text(encoding="utf-8")
        self.assertEqual(qa["status"], "complete")
        self.assertEqual(fragment.count(r"\begin{table}"), 2)
        self.assertEqual(set(qa["tables"]["missingness"]), set(MODELS))
        self.assertEqual(set(qa["tables"]["categorical_controls"]), {"catboost", "fttransformer"})
        self.assertLess(qa["table_layout"]["calculated_table_width_mm"], 150)
        self.assertIn("separate from the primary factorial matrix", fragment)
        self.assertIn("does not establish equivalence", fragment)
        self.assertIn("archived six-decimal precision", fragment)
        self.assertIn("did not save explicit TEST row indices", fragment)
        self.assertFalse(qa["model_fitting"])
        self.assertFalse(qa["bootstrap_recomputed"])
        self.assertEqual(file_sha256(Path(qa["fragment_path"])), qa["fragment_sha256"])

    def test_missingness_difference_is_second_minus_first(self):
        normalised = text.normalise_report(self.report)
        pair = normalised["missingness"]["lgbm"]
        self.assertGreater(pair["difference_second_minus_first"]["average_precision"], 0)
        self.assertGreater(pair["difference_second_minus_first"]["F2"], 0)
        self.assertEqual(pair["second"]["points"]["F2"], 1)
        self.assertAlmostEqual(pair["first"]["points"]["F2"], 10 / 21)

    def test_wrong_difference_direction_is_rejected_before_writing(self):
        self.report["missingness"]["lgbm"]["difference_second_minus_first"]["F2"] *= -1
        self.persist()
        with self.assertRaisesRegex(ValueError, "incorrect value or direction"):
            self.generate()
        self.assertFalse((self.root / "new_text").exists())

    def test_changed_score_hash_is_rejected_before_writing(self):
        run = Path(self.report["missingness"]["lgbm"]["first_run"])
        np.save(run / "y_test_scores.npy", self.second_scores)
        with self.assertRaisesRegex(ValueError, "source artefact changed"):
            self.generate()
        self.assertFalse((self.root / "new_text").exists())

    def test_declared_point_cannot_replace_saved_score_evidence(self):
        self.report["missingness"]["rf"]["first_metrics"]["average_precision"] = .123
        self.persist()
        with self.assertRaisesRegex(ValueError, "point does not reproduce"):
            self.generate()

    def test_bootstrap_budget_is_not_silently_reduced(self):
        self.report["missingness"]["logreg"]["bootstrap_requested"] = 999
        self.persist()
        with self.assertRaisesRegex(ValueError, "1,000 resamples"):
            self.generate()

    def test_missing_or_reversed_intervals_are_rejected(self):
        for interval in (None, [.5, -.5]):
            report = copy.deepcopy(self.report)
            report["missingness"]["logreg"]["paired_95_percentile_intervals"]["F2"] = interval
            with self.assertRaises(ValueError):
                text.normalise_report(report)

    def test_catboost_schema_does_not_invent_a_finer_historical_threshold(self):
        report = copy.deepcopy(self.report)
        report["categorical_controls"]["catboost"]["primary_threshold"] = .5000001
        with self.assertRaisesRegex(ValueError, "declared saved precision"):
            text.normalise_report(report)

    def test_catboost_rounded_threshold_cannot_be_claimed_full_precision(self):
        report = copy.deepcopy(self.report)
        report["categorical_controls"]["catboost"]["primary_threshold_precision"] = "full precision"
        with self.assertRaisesRegex(ValueError, "cannot be described as full precision"):
            text.normalise_report(report)

    def test_null_historical_exact_threshold_does_not_invent_precision(self):
        source = Path(self.report["categorical_controls"]["catboost"]["source_run"])
        config = self.read(source / "config.json")
        config["threshold_exact"] = None
        self.write(source / "config.json", config)
        pair = text.normalise_report(self.report)["categorical_controls"]["catboost"]
        self.assertTrue(pair["first"]["threshold_precision"].startswith("historical saved six-decimal"))

    def test_no_output_directory_is_reused(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            text.generate(self.report_path, self.primary_path, self.sensitivity_path, output)
        self.assertEqual(list(output.iterdir()), [])

    def test_primary_archive_is_rejected_before_reading_inputs(self):
        from generate_revision_interpretability import ROOT
        with self.assertRaises(ValueError):
            text.generate(self.root / "absent.json", self.primary_path, self.sensitivity_path, ROOT / "results_thesis")

    def test_test_policy_selection_is_not_accepted(self):
        report = copy.deepcopy(self.report)
        report["categorical_controls"]["fttransformer"]["policy_selected_using_TEST"] = True
        with self.assertRaisesRegex(ValueError, "cannot be selected using TEST"):
            text.normalise_report(report)

    def test_requested_and_valid_bootstrap_counts_are_not_conflated(self):
        report = copy.deepcopy(self.report)
        report["missingness"]["lgbm"]["bootstrap_valid"] = 999
        normalised = text.normalise_report(report)
        self.assertEqual(normalised["missingness"]["lgbm"]["bootstrap_requested"], 1000)
        self.assertEqual(normalised["missingness"]["lgbm"]["bootstrap_valid"]["F2"], 999)
        self.assertIn("1,000 requested", text.make_fragment(normalised))

    def test_non_finite_or_out_of_range_interval_bounds_are_rejected(self):
        for bounds in ([np.nan, .1], [-1.1, .2], [.1, np.inf]):
            report = copy.deepcopy(self.report)
            report["missingness"]["logreg"]["paired_95_percentile_intervals"]["F2"] = bounds
            with self.assertRaises(ValueError):
                text.normalise_report(report)


if __name__ == "__main__":
    unittest.main()
