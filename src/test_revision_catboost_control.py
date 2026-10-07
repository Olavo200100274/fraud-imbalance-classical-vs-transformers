"""Isolated tests of the predefined nominal-aware CatBoost control."""

import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from categorical_preprocess import (
    CategoryOneHotRepresentation, MixedCategoryPreprocessor, validate_category_codes,
)
from experiment_protocol import array_sha256, file_sha256
from preprocess import get_preprocessor
import revision_catboost_control as control


def _frame(rows=120):
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "income": rng.uniform(0.1, 0.9, rows),
        "bank_months_count": rng.choice([-1, 0, 4, 9], rows).astype(np.int64),
        "employment_status": np.asarray(["AA", "AB", "AC"] * ((rows + 2) // 3), dtype=object)[:rows],
        "device_os": np.asarray(["linux", "windows"] * ((rows + 1) // 2), dtype=object)[:rows],
    })


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class RepresentationTests(unittest.TestCase):
    def test_no_resampling_representation_is_exactly_the_primary_matrix(self):
        frame = _frame()
        primary = get_preprocessor(frame).fit_transform(frame).to_numpy()
        mixed = MixedCategoryPreprocessor().set_output(transform="default")
        encoded = mixed.fit_transform(frame)
        post = CategoryOneHotRepresentation(mixed.numeric_features_, mixed.categorical_features_,
                                            mixed.categorical_cardinalities_).set_output(transform="default")
        reconstructed = post.fit_transform(encoded)
        np.testing.assert_array_equal(reconstructed, primary)
        self.assertEqual(primary.shape[1], 7)
        self.assertEqual(mixed.categorical_indices_, [2, 3])
        self.assertEqual(mixed.numeric_transformer_.named_steps["scaler"].n_samples_seen_, len(frame))

    def test_validation_category_does_not_enter_the_training_vocabulary(self):
        frame = _frame()
        validation = frame.iloc[:3].copy()
        validation["device_os"] = "unseen"
        mixed = MixedCategoryPreprocessor().set_output(transform="default")
        encoded = mixed.fit_transform(frame)
        post = CategoryOneHotRepresentation(mixed.numeric_features_, mixed.categorical_features_,
                                            mixed.categorical_cardinalities_).set_output(transform="default")
        post.fit(encoded)
        transformed = post.transform(mixed.transform(validation))
        np.testing.assert_array_equal(transformed[:, -2:], np.zeros((3, 2)))
        self.assertNotIn("unseen", mixed.ordinal_encoder_.categories_[1])
        primary = get_preprocessor(frame)
        primary.fit(frame)
        np.testing.assert_array_equal(transformed, primary.transform(validation).to_numpy())

    def test_fractional_and_out_of_vocabulary_codes_are_rejected(self):
        for code in (0.5, -1, 2, np.nan):
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate_category_codes(np.array([[0.1, code]]), [1], [2])
        validate_category_codes(np.array([[0.1, -1]]), [1], [2], allow_unknown=True)

    def test_predictor_order_is_not_silently_changed(self):
        frame = _frame()
        mixed = MixedCategoryPreprocessor().set_output(transform="default").fit(frame)
        with self.assertRaisesRegex(ValueError, "order"):
            mixed.transform(frame[frame.columns[::-1]])


class ResamplingTests(unittest.TestCase):
    def test_smote_nc_has_discrete_one_hot_blocks_and_fixed_classifier_parameters(self):
        class RecordingClassifier:
            def __init__(self, **params):
                self.params = params

            def fit(self, X, y):
                self.training_matrix = X
                self.training_labels = np.asarray(y)
                return self

        frame = _frame()
        labels = np.asarray([0] * 96 + [1] * 24)
        params = {"classifier__iterations": 12, "classifier__depth": 2,
                  "classifier__learning_rate": 0.1, "classifier__l2_leaf_reg": 2}
        model, audit = control.fit_control_model(frame, labels, params, stage="unit_fold",
                                                classifier_factory=RecordingClassifier)
        classifier = model.named_steps["classifier"]
        self.assertEqual(classifier.params["iterations"], 12)
        self.assertEqual(classifier.params["random_state"], 42)
        self.assertNotIn("cat_features", classifier.params)
        self.assertEqual(audit["rows_after"], 192)
        self.assertEqual(np.bincount(classifier.training_labels).tolist(), [96, 96])
        self.assertTrue(audit["categorical_codes_integer_and_in_vocabulary"])
        self.assertEqual(audit["unknown_resampled_category_codes"], 0)
        matrix = classifier.training_matrix
        for block in (matrix[:, 2:5], matrix[:, 5:7]):
            self.assertTrue(np.isin(block, [0, 1]).all())
            np.testing.assert_array_equal(block.sum(axis=1), np.ones(len(matrix)))

    def test_small_fitted_pipeline_can_be_reloaded_and_score_unknown_categories(self):
        frame = _frame()
        labels = np.asarray([0] * 96 + [1] * 24)
        params = {"classifier__iterations": 8, "classifier__depth": 2,
                  "classifier__learning_rate": 0.1, "classifier__l2_leaf_reg": 2}
        model, _ = control.fit_control_model(frame, labels, params, stage="unit_smoke", threads=1)
        validation = frame.iloc[:10].copy()
        validation["employment_status"] = "new"
        scores = model.predict_proba(validation)
        self.assertTrue(np.isfinite(scores).all())
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.joblib"
            joblib.dump(model, destination)
            reloaded = joblib.load(destination)
            np.testing.assert_array_equal(scores, reloaded.predict_proba(validation))


class SourceAndPairingTests(unittest.TestCase):
    @staticmethod
    def _sources(root):
        fixed = {"classifier__iterations": 8, "classifier__depth": 2}
        manifest = {"historical_runs": {}}
        labels = np.asarray([0, 1, 0, 1])
        scores = np.asarray([0.1, 0.8, 0.2, 0.9])
        for role, key in control.SOURCE_KEYS.items():
            run = root / role
            config = {"dataset": "baf_base", "model": "catboost", "strategy": "none" if role == "baseline" else "smote",
                      "best_params": fixed, "dataset_hash_sha256": "raw_hash"}
            _write_json(run / "config.json", config)
            _write_json(run / "metrics_test.json", control.compute_all_metrics(labels, scores, 0.5))
            np.save(run / "y_test.npy", labels)
            np.save(run / "y_test_scores.npy", scores)
            manifest["historical_runs"][key] = {"run_dir": str(run), "config_sha256": file_sha256(run / "config.json")}
        manifest_path = root / "manifest.json"
        _write_json(manifest_path, manifest)
        return manifest_path, labels, scores

    def test_design_only_never_reads_raw_data_or_fits_a_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = self._sources(root)
            with patch.object(control, "load_dataset", side_effect=AssertionError("No raw CSV reading")), \
                    patch.object(control, "fit_control_model", side_effect=AssertionError("No fitting")):
                directory, _ = control.save_design(manifest, root / "output")
            design = json.loads((directory / "design.json").read_text(encoding="utf-8"))
            self.assertEqual(design["status"], "design_validated_not_trained")
            self.assertFalse(design["test_used_for_selection"])

    def test_source_hash_and_hyperparameter_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = self._sources(root)
            source = root / "primary_smote" / "config.json"
            config = json.loads(source.read_text(encoding="utf-8"))
            config["best_params"]["classifier__depth"] = 4
            _write_json(source, config)
            with self.assertRaisesRegex(ValueError, "pin"):
                control.resolve_source_runs(manifest)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["historical_runs"][control.SOURCE_KEYS["primary_smote"]]["config_sha256"] = file_sha256(source)
            _write_json(manifest, value)
            with self.assertRaisesRegex(ValueError, "hyperparameters"):
                control.resolve_source_runs(manifest)

    def test_test_comparison_is_paired_and_checks_original_indices_if_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, labels, scores = self._sources(root)
            sources = control.resolve_source_runs(manifest)
            metadata = {"raw_file_sha256": "raw_hash", "test_indices": np.asarray([6, 8, 2, 7])}
            comparison = control.compare_primary_smote(sources["primary_smote"], labels, scores, 0.5, metadata, iterations=10)
            self.assertEqual(comparison["difference_full_precision"], {"PR-AUC": 0.0, "ROC-AUC": 0.0, "F2": 0.0})
            self.assertEqual(comparison["shared_test_row_indices_sha256"], array_sha256(metadata["test_indices"]))
            self.assertEqual(comparison["paired_difference_bootstrap"]["PR-AUC"]["ci_95_percent"], [0.0, 0.0])
            with self.assertRaisesRegex(ValueError, "labels or row order"):
                control.compare_primary_smote(sources["primary_smote"], labels[::-1], scores, 0.5, metadata)
            np.save(root / "primary_smote" / "test_row_indices.npy", metadata["test_indices"][::-1])
            with self.assertRaisesRegex(ValueError, "row indices"):
                control.compare_primary_smote(sources["primary_smote"], labels, scores, 0.5, metadata)

    def test_runner_saves_out_of_fold_alignment_and_the_full_precision_median(self):
        class FixedScorer:
            def predict_proba(self, frame):
                scores = frame["income"].to_numpy()
                return np.column_stack((1 - scores, scores))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = self._sources(root)
            dev = _frame(120)
            dev.index = np.arange(100, 220)
            labels_dev = pd.Series([0] * 96 + [1] * 24, index=dev.index)
            test = _frame(20)
            test.index = np.arange(300, 320)
            labels_test = pd.Series([0] * 16 + [1] * 4, index=test.index)
            original_scores = FixedScorer().predict_proba(test)[:, 1]
            for role in control.SOURCE_KEYS:
                config_file = root / role / "config.json"
                config = json.loads(config_file.read_text(encoding="utf-8"))
                config.update({"train_samples": 120, "test_samples": 20})
                _write_json(config_file, config)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            for role, key in control.SOURCE_KEYS.items():
                value["historical_runs"][key]["config_sha256"] = file_sha256(root / role / "config.json")
            _write_json(manifest, value)
            np.save(root / "primary_smote" / "y_test.npy", np.asarray(labels_test))
            np.save(root / "primary_smote" / "y_test_scores.npy", original_scores)
            _write_json(root / "primary_smote" / "metrics_test.json",
                        control.compute_all_metrics(labels_test, original_scores, 0.5))
            metadata = {"raw_file_sha256": "raw_hash", "raw_file": "unit_fixture.csv",
                        "dev_indices": dev.index.to_numpy(dtype=np.int64),
                        "test_indices": test.index.to_numpy(dtype=np.int64)}
            destination = root / "saved_run"
            destination.mkdir()
            _write_json(destination / "config.json", {"unit_fixture": True})
            audit = {"rows_after": 200, "classifier_input_columns": 7}
            with patch.object(control, "load_dataset", return_value=(dev, test, labels_dev, labels_test, metadata)), \
                    patch.object(control, "fit_control_model", return_value=(FixedScorer(), audit.copy())) as fitted, \
                    patch.object(control.joblib, "dump"), \
                    patch.object(control, "save_run", return_value=str(destination)) as saved:
                result = control.run_control(manifest, root / "output", bootstrap_iterations=0)
            self.assertEqual(result, destination)
            self.assertEqual(fitted.call_count, 6)
            arguments = saved.call_args.kwargs
            thresholds = [row["threshold_exact"] for row in arguments["metrics_cv"]["per_fold"]]
            self.assertEqual(arguments["config"]["threshold_exact"], float(np.median(thresholds)))
            evidence = arguments["validation_evidence"]
            np.testing.assert_array_equal(evidence["row_indices"], dev.index.to_numpy())
            np.testing.assert_array_equal(evidence["y_val_scores"], FixedScorer().predict_proba(dev)[:, 1])
            self.assertEqual(np.unique(evidence["fold_ids"]).tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(arguments["config"]["bootstrap_iterations"], 0)
            self.assertFalse(arguments["config"]["test_used_for_selection"])

    def test_cli_design_bypasses_lock_but_training_uses_shared_resource(self):
        with patch.object(sys, "argv", ["control", "--design-only"]), \
                patch.object(control, "save_design") as design, \
                patch.object(control, "exclusive_resource") as resource:
            control.main()
            design.assert_called_once()
            resource.assert_not_called()
        with patch.object(sys, "argv", ["control", "--bootstrap-iterations", "0"]), \
                patch.object(control, "run_control") as training, \
                patch.object(control, "exclusive_resource") as resource:
            control.main()
            training.assert_called_once()
            self.assertEqual(resource.call_args.args[1], "baf_training_ram")
            resource.return_value.__enter__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
