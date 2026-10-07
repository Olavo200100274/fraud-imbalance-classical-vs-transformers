import numpy as np
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment_protocol import file_sha256, PROTOCOL_VERSION
from revision_paired_analysis import analyse_pair, paired_difference, PAIRED_SOURCE_FILES, MODELS, verify_paired_report


class PairedAnalysisTests(unittest.TestCase):
    def test_identical_predictions_have_zero_paired_differences(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        report = paired_difference(labels, scores, 0.5, scores.copy(), 0.5, iterations=50)
        assert all(value == 0 for value in report["difference_second_minus_first"].values())
        assert all(interval == [0.0, 0.0] for interval in report["paired_95_percentile_intervals"].values())
        assert report["bootstrap_valid"] > 0


    def test_difference_direction_and_fixed_thresholds(self):
        labels = np.asarray([0, 0, 1, 1])
        first = np.asarray([0.1, 0.2, 0.8, 0.9])
        second = first[::-1]
        report = paired_difference(labels, first, 0.5, second, 0.5, iterations=50)
        assert report["difference_second_minus_first"]["roc_auc"] == -1
        assert report["difference_second_minus_first"]["F2"] == -1
        assert report["difference_second_minus_first"]["alert_rate"] == 0
        assert report["paired_95_percentile_intervals"]["roc_auc"] == [-1.0, -1.0]


    def test_no_bootstrap_is_explicitly_recorded(self):
        report = paired_difference(np.asarray([0, 1]), np.asarray([0.1, 0.9]), 0.5,
                                   np.asarray([0.2, 0.8]), 0.5, iterations=0)
        assert report["paired_95_percentile_intervals"] is None
        assert report["bootstrap_valid"] == 0


    def test_misaligned_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same rows"):
            paired_difference(np.asarray([0, 1]), np.asarray([0.1]), 0.5,
                              np.asarray([0.2, 0.8]), 0.5)

    def test_non_binary_labels_are_not_silently_cast(self):
        for labels in ([0, 1.1], [0, 2], [0, 0]):
            with self.assertRaisesRegex(ValueError, "binary labels"):
                paired_difference(np.asarray(labels), np.asarray([0.1, 0.9]), 0.5,
                                  np.asarray([0.2, 0.8]), 0.5)

    def test_non_finite_scores_or_thresholds_are_rejected(self):
        for scores, threshold in (([0.1, np.nan], 0.5), ([0.1, 0.9], np.inf)):
            with self.assertRaisesRegex(ValueError, "finite"):
                paired_difference(np.asarray([0, 1]), np.asarray(scores), threshold,
                                  np.asarray([0.2, 0.8]), 0.5)

    def test_negative_bootstrap_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            paired_difference(np.asarray([0, 1]), np.asarray([0.1, 0.9]), 0.5,
                              np.asarray([0.2, 0.8]), 0.5, iterations=-1)

    def test_paired_evidence_hashes_both_frozen_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "first", Path(folder) / "second"
            for run in (first, second):
                run.mkdir()
                for name in PAIRED_SOURCE_FILES:
                    (run / name).write_bytes(b"synthetic source")
            arrays = {"y_test": np.asarray([0, 0, 1, 1]),
                      "y_test_scores": np.asarray([0.1, 0.2, 0.8, 0.9]),
                      "test_row_indices": np.arange(4)}
            config = {"dataset_hash_sha256": "fixture", "best_params": {"C": 1.0}, "threshold_exact": 0.5}
            with patch("revision_paired_analysis.load_pin", side_effect=[(first, config, arrays), (second, config, arrays)]):
                report = analyse_pair({}, {}, "logreg", iterations=0)
            for role, run in (("first", first), ("second", second)):
                self.assertEqual(report[f"{role}_artefacts_sha256"],
                                 {name: file_sha256(run / name) for name in PAIRED_SOURCE_FILES})


class PairedSourceVerificationTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        primary, sensitivity = {"runs": {}, "historical_runs": {}}, {"runs": {}}
        report = {"status": "complete", "missingness": {}, "categorical_controls": {},
                  "source_manifest_sha256": "legitimately older training state",
                  "sensitivity_manifest_sha256": "recorded source snapshot"}
        paths = {}

        def make_run(name, model, strategy, policy="preserve", historical=False):
            run = root / name
            run.mkdir(parents=True)
            config = {"dataset": "baf_base", "model": model, "strategy": strategy,
                      "missing_policy": policy, "threshold_exact": 0.5,
                      "dataset_hash_sha256": "synthetic raw file", "best_params": {"fixture": 1}}
            if not historical:
                config["protocol_version"] = PROTOCOL_VERSION
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "completed.json").write_text(json.dumps({"status": "complete", "protocol_version": PROTOCOL_VERSION}), encoding="utf-8")
            (run / "metrics_test.json").write_text("{}", encoding="utf-8")
            for name, values in (("y_test", [0, 1]), ("y_test_scores", [0.1, 0.9]), ("test_row_indices", [4, 5])):
                np.save(run / (name + ".npy"), values)
            return run

        def pin(run):
            return {"run_dir": str(run), "config_sha256": file_sha256(run / "config.json")}

        def hashes(run):
            return {name: file_sha256(run / name) for name in PAIRED_SOURCE_FILES}

        def comparison(first, second):
            return {"first_run": str(first), "second_run": str(second),
                    "first_config_sha256": file_sha256(first / "config.json"),
                    "second_config_sha256": file_sha256(second / "config.json"),
                    "first_artefacts_sha256": hashes(first), "second_artefacts_sha256": hashes(second),
                    "first_threshold": 0.5, "second_threshold": 0.5, "bootstrap_requested": 1000,
                    "paired_95_percentile_intervals": {"F2": [0, 0]}}

        for model in MODELS:
            first = make_run("primary/" + model, model, "n/a" if model == "ocsvm" else "none")
            second = make_run("sensitivity/" + model, model, "n/a" if model == "ocsvm" else "none", "nan_indicators")
            key = f"baf_base/{model}/none"
            primary["runs"][key], sensitivity["runs"][key] = pin(first), pin(second)
            report["missingness"][model] = comparison(first, second)
            paths[model + "/first"], paths[model + "/second"] = first, second
        first, second = make_run("ft_smote", "fttransformer", "smote"), make_run("ft_control", "fttransformer", "smotenc_control")
        primary["runs"]["baf_base/fttransformer/smote"] = pin(first)
        primary["runs"]["baf_base/fttransformer/smotenc_control"] = pin(second)
        report["categorical_controls"]["fttransformer"] = comparison(first, second)
        paths["ft_control/first"], paths["ft_control/second"] = first, second
        historical = make_run("historical_cb_smote", "catboost", "smote", historical=True)
        control = make_run("cb_control", "catboost", "smotenc_control")
        primary["historical_runs"]["baf_base/catboost/smote"] = pin(historical)
        primary["runs"]["baf_base/catboost/smotenc_control"] = pin(control)
        cb = {"source_run": str(historical), "source_config_sha256": file_sha256(historical / "config.json"),
              "source_artefacts_sha256": {name: file_sha256(historical / name) for name in
                  ("config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy")},
              "control_threshold_full_precision": 0.5, "bootstrap_iterations": 1000,
              "paired_difference_bootstrap": {"F2": {"ci_95_percent": [0, 0], "valid_replicates": 1000}}}
        (control / "primary_smote_comparison.json").write_text(json.dumps(cb), encoding="utf-8")
        report["categorical_controls"]["catboost"] = cb
        report["catboost_control_source"] = {"run_dir": str(control), "artefacts_sha256": {
            **hashes(control), "primary_smote_comparison.json": file_sha256(control / "primary_smote_comparison.json")}}
        paths["cb_control"], paths["historical_cb_smote"] = control, historical
        return report, primary, sensitivity, paths

    def test_complete_sources_pass_without_fit_inference_or_bootstrap(self):
        with tempfile.TemporaryDirectory(prefix="fraud_paired_verify_") as folder:
            report, primary, sensitivity, _ = self.fixture(Path(folder))
            with patch("revision_paired_analysis.paired_difference", side_effect=AssertionError("No recomputation")):
                with patch("revision_paired_analysis.load_pin", side_effect=AssertionError("No scoring loader")):
                    self.assertTrue(verify_paired_report(report, primary, sensitivity))

    def test_every_artifact_family_in_each_member_and_control_rejects_later_mutation(self):
        with tempfile.TemporaryDirectory(prefix="fraud_paired_verify_") as folder:
            report, primary, sensitivity, paths = self.fixture(Path(folder))
            for label, run in paths.items():
                names = (("config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy")
                         if label == "historical_cb_smote" else PAIRED_SOURCE_FILES)
                if label == "cb_control":
                    names = (*names, "primary_smote_comparison.json")
                for name in names:
                    with self.subTest(member=label, artefact=name):
                        path = run / name
                        original = path.read_bytes()
                        path.write_bytes(b"mutated synthetic artefact")
                        try:
                            with self.assertRaises(ValueError):
                                verify_paired_report(report, primary, sensitivity)
                        finally:
                            path.write_bytes(original)

    def test_missing_hash_maps_wrong_paths_pins_and_budgets_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="fraud_paired_verify_") as folder:
            report, primary, sensitivity, _ = self.fixture(Path(folder))
            variants = []
            missing = copy.deepcopy(report)
            del missing["missingness"]["rf"]["second_artefacts_sha256"]["y_test_scores.npy"]
            variants.append(missing)
            wrong_path = copy.deepcopy(report)
            wrong_path["categorical_controls"]["fttransformer"]["first_run"] = "another_run"
            variants.append(wrong_path)
            no_control = copy.deepcopy(report)
            del no_control["catboost_control_source"]
            variants.append(no_control)
            unsafe_name = copy.deepcopy(report)
            unsafe_name["catboost_control_source"]["artefacts_sha256"]["../other.json"] = "invalid"
            variants.append(unsafe_name)
            wrong_budget = copy.deepcopy(report)
            wrong_budget["missingness"]["ocsvm"]["bootstrap_requested"] = 0
            variants.append(wrong_budget)
            for candidate in variants:
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    verify_paired_report(candidate, primary, sensitivity)
            changed_pin = copy.deepcopy(primary)
            changed_pin["runs"]["baf_base/catboost/smotenc_control"]["config_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "manifest pin"):
                verify_paired_report(report, changed_pin, sensitivity)

    def test_config_identity_is_checked_even_if_a_changed_hash_is_relabelled(self):
        with tempfile.TemporaryDirectory(prefix="fraud_paired_verify_") as folder:
            report, primary, sensitivity, paths = self.fixture(Path(folder))
            run = paths["cb_control"]
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            config["model"] = "rf"
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            new_hash = file_sha256(run / "config.json")
            primary["runs"]["baf_base/catboost/smotenc_control"]["config_sha256"] = new_hash
            report["catboost_control_source"]["artefacts_sha256"]["config.json"] = new_hash
            with self.assertRaisesRegex(ValueError, "incompatible identity"):
                verify_paired_report(report, primary, sensitivity)


if __name__ == "__main__":
    unittest.main()
