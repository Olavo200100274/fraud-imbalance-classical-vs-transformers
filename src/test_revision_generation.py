"""Synthetic checks that revision generation never guesses or fills absent runs."""

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import generate_results as generator


class PinnedGenerationTests(unittest.TestCase):
    @staticmethod
    def transfer_fixture(directory):
        from evaluation.metrics import compute_all_metrics
        from revision_transfer import _save_array
        from sklearn.metrics import average_precision_score, roc_auc_score
        original_labels = np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.int64)
        original_scores = np.array([.9, .8, .7, .6, .5, .4, .3])
        original_indices = np.arange(100, 107, dtype=np.int64)
        kept = np.array([0, 2, 5, 6])
        labels, scores, indices = original_labels[kept], original_scores[kept], original_indices[kept]
        artefacts = {}
        for name, values in (("y_test.npy", labels), ("y_test_scores.npy", scores),
                             ("test_row_indices.npy", indices), ("original_y_test.npy", original_labels),
                             ("original_y_test_scores.npy", original_scores),
                             ("original_test_row_indices.npy", original_indices)):
            _save_array(directory, name, values, artefacts)
        return {"artefacts": artefacts, "threshold_used": .6,
                "population": {"rows": 4, "positive_rows": 3, "negative_rows": 1},
                "metrics": compute_all_metrics(labels, scores, .6),
                "metrics_full_precision": {"average_precision": average_precision_score(labels, scores),
                                           "roc_auc": roc_auc_score(labels, scores), "threshold": .6}}

    def test_historical_ulb_fallback_is_prohibited(self):
        manifest = {"runs": {}, "historical_runs": {"ulb_2013/lgbm/none": "results/old"}}
        with patch.object(generator, "RUN_MANIFEST", manifest):
            self.assertIsNone(generator.find_latest_run("ulb_2013", "lgbm"))

    def test_preflight_rejects_indices_changed_before_a_new_qa_snapshot(self):
        from experiment_protocol import PROTOCOL_VERSION, array_sha256, file_sha256
        with tempfile.TemporaryDirectory(prefix="fraud_test_index_pin_") as folder:
            root = Path(folder)
            expected = np.array([10, 11, 12], dtype=np.int64)
            labels, scores = np.array([0, 1, 0]), np.array([0.1, 0.9, 0.2])
            manifest = {"results_root": str(root), "runs": {}, "datasets": {}}
            for dataset in ("ulb_2013", "baf_base"):
                run = root / dataset / "lgbm/none/fixture"
                run.mkdir(parents=True)
                provenance = {"test_indices_sha256": array_sha256(expected),
                              "deduplication": {"policy": "exact_predictor_deduplication_before_all_splits"}}
                config = {"dataset": dataset, "model": "lgbm", "strategy": "none",
                          "protocol_version": PROTOCOL_VERSION, "dataset_hash_sha256": "synthetic raw hash",
                          "test_samples": 3, "test_fraud": 1, "data_provenance": provenance}
                for name, content in (("config.json", config), ("metrics_test.json", {}),
                                      ("metrics_cv.json", {}), ("pr_curve_data.json", {}),
                                      ("completed.json", {"status": "complete", "protocol_version": PROTOCOL_VERSION})):
                    (run / name).write_text(json.dumps(content), encoding="utf-8")
                np.save(run / "y_test.npy", labels)
                np.save(run / "y_test_scores.npy", scores)
                np.save(run / "test_row_indices.npy", expected)
                manifest["datasets"][dataset] = provenance
                manifest["runs"][f"{dataset}/lgbm/none"] = {
                    "run_dir": str(run), "config_sha256": file_sha256(run / "config.json")}
                generator.verify_test_index_provenance(run, config, manifest)
                # The content changes before generation, so a newly captured
                # file hash alone would incorrectly bless the altered indices.
                np.save(run / "test_row_indices.npy", expected + 1000)
            with patch.object(generator, "RUN_MANIFEST", manifest), \
                    patch.object(generator, "RESULTS_DIR", root), \
                    patch.object(generator, "MODEL_ORDER", ["lgbm"]), \
                    patch.object(generator, "BALANCE_ORDER", ["none"]), \
                    patch.object(generator, "THRESHOLD_STUDY_ROOT", root / "derived/thresholds"), \
                    patch.object(generator, "CROSS_DOMAIN_ROOT", root / "derived/transfer"):
                with self.assertRaisesRegex(ValueError, "configuration or manifest provenance pin"):
                    generator.preflight_revision()

    def test_historical_transformer_fallback_is_prohibited(self):
        manifest = {"runs": {}, "historical_runs": {"baf_base/fttransformer/none": "results/old"}}
        with patch.object(generator, "RUN_MANIFEST", manifest):
            self.assertIsNone(generator.find_latest_run("baf_base", "fttransformer"))

    def test_historical_baf_classical_requires_explicit_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run_original"
            run.mkdir()
            manifest = {"historical_runs": {"baf_base/lgbm/smote": {"run_dir": str(run)}}}
            with patch.object(generator, "RUN_MANIFEST", manifest):
                self.assertEqual(generator.find_latest_run("baf_base", "lgbm", "smote"), run)
                self.assertIsNone(generator.find_latest_run("baf_base", "lgbm", "ros"))

    def test_baseline_evidence_pin_precedes_historical_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            revision, historical = Path(temporary) / "revision", Path(temporary) / "historical"
            revision.mkdir()
            historical.mkdir()
            manifest = {"baseline_runs": {"baf_base/rf": {"run_dir": str(revision)}},
                        "historical_runs": {"baf_base/rf/none": {"run_dir": str(historical)}}}
            with patch.object(generator, "RUN_MANIFEST", manifest):
                self.assertEqual(generator.find_latest_run("baf_base", "rf"), revision)

    def test_incomplete_revision_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"results_root": str(root), "runs": {}, "historical_runs": {}}
            with patch.object(generator, "RUN_MANIFEST", manifest), patch.object(generator, "RESULTS_DIR", root), \
                    patch.object(generator, "THRESHOLD_STUDY_ROOT", root / "derived" / "thresholds"), \
                    patch.object(generator, "CROSS_DOMAIN_ROOT", root / "transfer" / "evaluation"):
                with self.assertRaisesRegex(FileNotFoundError, "Revision grid is incomplete"):
                    generator.preflight_revision()
                self.assertEqual(list(root.iterdir()), [])

    def test_ambiguous_legacy_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "baf_base" / "lgbm" / "none"
            (base / "run_20260101").mkdir(parents=True)
            (base / "run_20261005").mkdir()
            with patch.object(generator, "RUN_MANIFEST", None), patch.object(generator, "RESULTS_DIR", root):
                with self.assertRaisesRegex(ValueError, "Multiple runs"):
                    generator.find_latest_run("baf_base", "lgbm")

    def test_full_precision_metrics_are_not_six_place_json_values(self):
        from sklearn.metrics import average_precision_score
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            labels = np.array([1, 0, 1, 0, 0, 0, 1])
            scores = np.array([.9, .8, .7, .6, .5, .4, .3])
            source = {"PR-AUC": round(float(average_precision_score(labels, scores)), 6),
                      "ROC-AUC": .5, "TP": 2, "FP": 2, "TN": 2, "FN": 1,
                      "threshold": .6, "k_used": 1}
            from sklearn.metrics import roc_auc_score
            source["ROC-AUC"] = round(float(roc_auc_score(labels, scores)), 6)
            (run / "config.json").write_text(json.dumps({"model": "logreg", "threshold_exact": .6}), encoding="utf-8")
            (run / "metrics_test.json").write_text(json.dumps(source), encoding="utf-8")
            np.save(run / "y_test.npy", labels)
            np.save(run / "y_test_scores.npy", scores)
            metrics = generator.full_precision_metrics(run)
            self.assertEqual(metrics["PR-AUC"], average_precision_score(labels, scores))
            self.assertNotEqual(metrics["PR-AUC"], source["PR-AUC"])
            self.assertEqual(metrics["F2"], 10 / 16)

    def test_missing_roc_interval_is_not_fabricated(self):
        with self.assertRaisesRegex(ValueError, "Missing saved bootstrap interval"):
            generator.validated_bootstrap_interval({"bootstrap_ci": {"PR-AUC_ci": [.1, .2]}}, "ROC-AUC_ci")

    def test_recovery_costs_keep_original_search_and_fit_separate(self):
        config = {"validation_only_recovery": True, "tuning_time_s": 0,
                  "historical_tuning_time_s": 1200, "historical_train_time_s": 16,
                  "historical_cost_source_run": "original", "validation_recovery_time_s": 80,
                  "train_time_source": "historical saved full-DEV fit; no new fit",
                  "train_time_s": 16, "infer_time_s": 1.2}
        costs = generator.computational_cost_basis(config)
        self.assertEqual(costs["tuning_time_s"], 1200)
        self.assertTrue(costs["tuning_historical"])
        self.assertTrue(costs["train_historical"])
        self.assertEqual(costs["validation_recovery_time_s"], 80)

    def test_fixed_transformer_cost_has_no_new_search(self):
        costs = generator.computational_cost_basis({"tuning_time_s": 0, "n_trials": 0,
                                                    "train_time_s": 100, "infer_time_s": 1})
        self.assertFalse(costs["train_historical"])
        self.assertEqual(costs["tuning_source"], "no new search; fixed or untuned")

    def test_resumed_search_cost_is_not_full_trial_budget_wall_time(self):
        costs = generator.computational_cost_basis({"tuning_time_s": 366.53, "train_time_s": 2,
                                                    "infer_time_s": .2, "n_trials": 50,
                                                    "optuna_sampler_checkpoint": {"loaded_from": "saved_TPE"}})
        self.assertTrue(costs["tuning_partial_wall_time"])
        self.assertEqual(costs["scope"]["tuning"], "resumed_invocation_only_not_full_trial_budget")
        self.assertFalse(costs["tuning_historical"])

    def test_transformer_replayed_sampler_is_also_resumed(self):
        costs = generator.computational_cost_basis({"tuning_time_s": 100, "train_time_s": 2,
                                                    "infer_time_s": .2,
                                                    "optuna_sampler_recovery": {"reconstructed_trials": 3}})
        self.assertTrue(costs["tuning_partial_wall_time"])

    def test_ocsvm_brier_is_not_reported_as_probability_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(generator, "TABLES_BASE", Path(temporary)):
                metrics = {"TP": 1, "FP": 1, "FN": 1, "threshold": .5, "brier_score": .1,
                           "PR-AUC": .2, "ROC-AUC": .6, "F1": .5, "F2": .5}
                generator.generate_baseline_table({"ocsvm": {"metrics": metrics}}, "Synthetic", "qa")
                text = (Path(temporary) / "qa" / "baseline.tex").read_text(encoding="utf-8")
                self.assertIn("0.5000 & ---", text)
                self.assertIn("anomaly scores are not probabilities", text)

    def test_stored_shap_verifies_actual_complete_grouped_magnitudes(self):
        values = np.array([[.2, .4, -.3], [-.1, -.2, .5]], dtype=np.float32)
        scores = 1 / (1 + np.exp(-(values.astype(np.float64).sum(axis=1) - 2)))
        report = generator.verify_saved_shap_matrix(
            values, ["income", "device_os_linux", "device_os_other"], ["income", "device_os"],
            {"income": .15, "device_os": .7}, -2, scores)
        self.assertLess(report["maximum_absolute_importance_difference"], 1e-7)

    def test_stored_shap_cannot_validate_placeholder_zeros(self):
        values = np.array([[.2, .4, -.3], [-.1, -.2, .5]], dtype=np.float32)
        scores = 1 / (1 + np.exp(-(values.astype(np.float64).sum(axis=1) - 2)))
        with self.assertRaisesRegex(ValueError, "complete stored attribution matrix"):
            generator.verify_saved_shap_matrix(
                values, ["income", "device_os_linux", "device_os_other"], ["income", "device_os"],
                {"income": 0, "device_os": .7}, -2, scores)

    def test_transfer_uses_complete_metric_precision_and_hash_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.transfer_fixture(Path(temporary))
            metrics, snapshots = generator.verify_transfer_evidence(Path(temporary), record, "lgbm")
            self.assertEqual(metrics["PR-AUC"], record["metrics_full_precision"]["average_precision"])
            self.assertNotEqual(metrics["PR-AUC"], record["metrics"]["PR-AUC"])
            self.assertEqual(len(snapshots), 6)
            self.assertEqual(metrics["F2"], 10 / 14)

    def test_transfer_detects_changed_score_file_before_reporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.transfer_fixture(Path(temporary))
            np.save(Path(temporary) / "y_test_scores.npy", np.array([.9, .7, .4, .8]))
            with self.assertRaisesRegex(ValueError, "Transfer array file hash changed"):
                generator.verify_transfer_evidence(Path(temporary), record, "lgbm")

    def test_transfer_does_not_trust_metric_json_when_arrays_disagree(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.transfer_fixture(Path(temporary))
            record["metrics"]["TP"] = 3
            with self.assertRaisesRegex(ValueError, "does not reproduce its saved scores: TP"):
                generator.verify_transfer_evidence(Path(temporary), record, "lgbm")

    def test_transfer_kept_scores_must_equal_original_observations(self):
        from revision_transfer import _save_array
        with tempfile.TemporaryDirectory() as temporary:
            record = self.transfer_fixture(Path(temporary))
            changed = np.load(Path(temporary) / "original_y_test_scores.npy")
            changed[0] = .85
            (Path(temporary) / "original_y_test_scores.npy").unlink()
            _save_array(Path(temporary), "original_y_test_scores.npy", changed, record["artefacts"])
            with self.assertRaisesRegex(ValueError, "differ from the retained original observations"):
                generator.verify_transfer_evidence(Path(temporary), record, "lgbm")

    def test_paired_reporting_verifies_nested_sources_and_hashes(self):
        from experiment_protocol import file_sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sensitivity_path = root / "sensitivity/revision_manifest.json"
            sensitivity_path.parent.mkdir()
            sensitivity_path.write_text("{}", encoding="utf-8")
            paired_path = root / "derived/paired/paired_analysis.json"
            paired_path.parent.mkdir(parents=True)
            report = {"sensitivity_manifest_sha256": file_sha256(sensitivity_path)}
            paired_path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(generator, "RUN_MANIFEST", {}), \
                    patch("revision_paired_analysis.verify_paired_report") as verify:
                evidence = generator.collect_paired_evidence(root)
                verify.assert_called_once_with(report, {}, {})
            self.assertEqual(evidence["sha256"], file_sha256(paired_path))
            with patch.object(generator, "RUN_MANIFEST", {}), \
                    patch("revision_paired_analysis.verify_paired_report", side_effect=ValueError("nested source changed")):
                with self.assertRaisesRegex(ValueError, "nested source changed"):
                    generator.collect_paired_evidence(root)
            sensitivity_path.write_text('{"changed": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sensitivity manifest changed"):
                generator.collect_paired_evidence(root)

    def test_missing_bootstrap_field_captures_and_freshly_verifies_exact_sidecar(self):
        key = "ulb_2013/fttransformer/none"
        with tempfile.TemporaryDirectory(prefix="fraud_bootstrap_qa_") as folder:
            root = Path(folder)
            (root / "config.json").write_text("{}", encoding="utf-8")
            reference = {"path": str(root / "compatibility.json"), "sha256": "verified report digest"}
            manifest, sources = {"bootstrap_compatibility": {key: reference}}, {key: {"run_dir": str(root)}}
            verify = Mock(return_value=reference)
            module = SimpleNamespace(KEY=key, verify_bootstrap_compatibility=verify)
            with patch.dict("sys.modules", {"revision_bootstrap_compatibility": module}):
                self.assertEqual(generator.collect_bootstrap_compatibility(manifest, sources), {key: reference})
                verify.assert_called_once_with(manifest, root.resolve(), {}, expected_iterations=1000)
                verify.side_effect = ValueError("sidecar bytes changed")
                with self.assertRaisesRegex(ValueError, "sidecar bytes changed"):
                    generator.collect_bootstrap_compatibility(manifest, sources)

    def test_explicit_wrong_bootstrap_budgets_never_use_compatibility_fallback(self):
        key = "ulb_2013/fttransformer/none"
        with tempfile.TemporaryDirectory(prefix="fraud_bootstrap_qa_") as folder:
            root = Path(folder)
            verify = Mock(side_effect=AssertionError("An explicit budget must not use fallback"))
            module = SimpleNamespace(KEY=key, verify_bootstrap_compatibility=verify)
            with patch.dict("sys.modules", {"revision_bootstrap_compatibility": module}):
                for value in (None, 0, 999):
                    (root / "config.json").write_text(json.dumps({"bootstrap_iterations": value}), encoding="utf-8")
                    with self.subTest(value=value), self.assertRaisesRegex(ValueError, "explicitly incompatible"):
                        generator.collect_bootstrap_compatibility({}, {key: {"run_dir": str(root)}})
                (root / "config.json").write_text(json.dumps({"bootstrap_iterations": 1000}), encoding="utf-8")
                self.assertEqual(generator.collect_bootstrap_compatibility({}, {key: {"run_dir": str(root)}}), {})
                verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
