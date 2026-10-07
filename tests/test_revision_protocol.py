"""Small synthetic checks for the corrected protocol; never train on full data."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import optuna
import hashlib
import zipfile
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import _split_frame, deduplicate_predictors
from experiment_protocol import (resolve_baseline_run, sampler_diagnostics,
                                 load_sampler_checkpoint, save_sampler_checkpoint, file_sha256)
from experiment_protocol import capture_source_provenance
from main import run_supervised, run_supervised_strategy, run_ocsvm, _load_baseline_params
from preprocess import get_preprocessor
from save_load import save_run
from strategies.balancing import get_sampler


class RevisionProtocolTests(unittest.TestCase):
    def make_frame(self):
        rng = np.random.RandomState(12)
        target = np.concatenate([np.zeros(160, dtype=int), np.ones(40, dtype=int)])
        return pd.DataFrame({
            "first": rng.normal(size=200) + target,
            "second": rng.normal(size=200),
            "Class": target,
        })

    def test_exact_predictor_deduplication(self):
        frame = self.make_frame()
        repeated = pd.concat([frame, frame.iloc[[1, 161]]], ignore_index=True)
        clean, audit = deduplicate_predictors(repeated, "Class")
        self.assertEqual(len(clean), 200)
        self.assertEqual(audit["removed_rows"], 2)
        self.assertEqual(audit["removed_positive_rows"], 1)
        self.assertEqual(clean.index.tolist(), list(range(200)))

    def test_conflicting_labels_fail_before_split(self):
        frame = self.make_frame()
        extra = frame.iloc[[0]].copy()
        extra["Class"] = 1
        repeated = pd.concat([frame, extra], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            deduplicate_predictors(repeated, "Class")

    def test_no_repeated_predictors_cross_corrected_split(self):
        frame = self.make_frame()
        repeated = pd.concat([frame, frame.iloc[:30]], ignore_index=True)
        X_dev, X_test, _, _, metadata = _split_frame(repeated, "Class", deduplicate=True)
        dev_profiles = set(map(tuple, X_dev.to_numpy()))
        test_profiles = set(map(tuple, X_test.to_numpy()))
        self.assertFalse(dev_profiles & test_profiles)
        self.assertEqual(metadata["deduplication"]["removed_rows"], 30)

    def test_baf_split_boundary_is_unchanged(self):
        frame = self.make_frame().rename(columns={"Class": "fraud_bool"})
        actual = _split_frame(frame, "fraud_bool")
        expected = train_test_split(
            frame.drop(columns="fraud_bool"), frame.fraud_bool,
            test_size=0.2, stratify=frame.fraud_bool, random_state=42,
        )
        for got, wanted in zip(actual[:4], expected):
            self.assertTrue(got.equals(wanted))

    def test_sample_fraction_is_validated(self):
        for fraction in (0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                _split_frame(self.make_frame(), "Class", sample=fraction)

    def test_cleaner_counts_are_actual_fitted_counts(self):
        rng = np.random.RandomState(3)
        X = rng.normal(size=(50, 3))
        y = np.concatenate([np.zeros(40, dtype=int), np.ones(10, dtype=int)])
        sampler = get_sampler("smote_tomek")
        X_res, y_res = sampler.fit_resample(X, y)
        audit = sampler_diagnostics(sampler, X, y, X_res, y_res, "unit_test")
        self.assertEqual(audit["rows_after_smote_before_cleaning"], 80)
        self.assertEqual(audit["cleaner_removed_rows"], 80 - len(y_res))
        self.assertEqual(audit["class_counts_before"], {"0": 40, "1": 10})
        self.assertEqual(sum(audit["class_counts_after"].values()), len(y_res))

    def test_original_results_archive_is_protected(self):
        with self.assertRaisesRegex(ValueError, "original results archive"):
            save_run(None, {}, {}, [], [], {}, "unit_test", results_root=PROJECT_ROOT / "results")

    def test_source_snapshot_hashes_match_archived_bytes(self):
        with tempfile.TemporaryDirectory(prefix="fraud_source_snapshot_") as temporary:
            provenance = capture_source_provenance(temporary)
            archive_path = Path(provenance["source_snapshot_path"])
            self.assertEqual(file_sha256(archive_path), provenance["source_snapshot_sha256"])
            with zipfile.ZipFile(archive_path) as archive:
                for name, digest in provenance["source_hashes_at_launch"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)

    def test_baf_validation_recovery_does_not_refit_the_final_classifier(self):
        X_dev, X_test, y_dev, y_test, metadata = _split_frame(self.make_frame(), "Class")
        with tempfile.TemporaryDirectory(prefix="fraud_validation_recovery_") as temporary:
            root = Path(temporary)
            context = {"results_root": root, "manifest_path": root / "revision_manifest.json",
                       "split_metadata": metadata, "dataset_hash": "synthetic",
                       "bootstrap_iterations": 0, "missing_policy": "preserve", "sample_fraction": None}
            run_supervised("logreg", X_dev, X_test, y_dev, y_test, get_preprocessor(X_dev),
                           "synthetic", "baf_base", "synthetic_fixture", n_trials=2, context=context)
            source = resolve_baseline_run("baf_base", "logreg", results_root=root)
            recovered = {**context, "fixed_params_run": source, "recover_validation_only": True}
            original_fit = LogisticRegression.fit
            with patch.object(LogisticRegression, "fit", autospec=True, side_effect=original_fit) as fit:
                run_supervised("logreg", X_dev, X_test, y_dev, y_test, get_preprocessor(X_dev),
                               "synthetic", "baf_base", "synthetic_fixture", context=recovered)
                self.assertEqual(fit.call_count, 5)
            target = resolve_baseline_run("baf_base", "logreg", results_root=root)
            config = json.loads((target / "config.json").read_text())
            self.assertTrue(config["validation_only_recovery"])
            self.assertTrue(config["test_scores_recomputed"])
            self.assertEqual(config["n_trials"], 0)
            self.assertEqual(config["test_scores_max_absolute_difference"], 0)
            np.testing.assert_array_equal(np.load(source / "y_test_scores.npy"),
                                          np.load(target / "y_test_scores.npy"))

    def test_ocsvm_recovery_reuses_validated_test_scores_without_scoring(self):
        X_dev, X_test, y_dev, y_test, metadata = _split_frame(self.make_frame(), "Class")
        with tempfile.TemporaryDirectory(prefix="fraud_ocsvm_recovery_") as temporary:
            root = Path(temporary)
            context = {"results_root": root, "manifest_path": root / "revision_manifest.json",
                       "split_metadata": metadata, "dataset_hash": "synthetic",
                       "bootstrap_iterations": 0, "missing_policy": "preserve", "sample_fraction": None}
            run_ocsvm(X_dev, X_test, y_dev, y_test, get_preprocessor(X_dev), "synthetic",
                      "baf_base", "synthetic_fixture", context=context)
            source = resolve_baseline_run("baf_base", "ocsvm", results_root=root)
            recovered = {**context, "fixed_params_run": source, "recover_validation_only": True}
            original_fit, original_score = OneClassSVM.fit, OneClassSVM.decision_function
            with patch.object(OneClassSVM, "fit", autospec=True, side_effect=original_fit) as fit:
                with patch.object(OneClassSVM, "decision_function", autospec=True,
                                  side_effect=original_score) as score:
                    run_ocsvm(X_dev, X_test, y_dev, y_test, get_preprocessor(X_dev), "synthetic",
                              "baf_base", "synthetic_fixture", context=recovered)
                    self.assertEqual(fit.call_count, 5)
                    self.assertEqual(score.call_count, 5)
            target = resolve_baseline_run("baf_base", "ocsvm", results_root=root)
            config = json.loads((target / "config.json").read_text())
            self.assertFalse(config["test_scores_recomputed"])
            np.testing.assert_array_equal(np.load(source / "y_test_scores.npy"),
                                          np.load(target / "y_test_scores.npy"))

    def test_baseline_selection_has_no_latest_run_fallback(self):
        with tempfile.TemporaryDirectory(prefix="fraud_protocol_test_") as temporary:
            with self.assertRaises(FileNotFoundError):
                resolve_baseline_run("synthetic", "logreg", results_root=temporary)

    def test_sampler_replace_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory(prefix="fraud_sampler_retry_") as temporary:
            path = Path(temporary) / "sampler.joblib"
            study = SimpleNamespace(sampler=optuna.samplers.TPESampler(seed=42), study_name="synthetic")
            import os
            actual_replace = os.replace
            with patch("experiment_protocol.os.replace", side_effect=[PermissionError("lock"), None]) as replace:
                with patch("experiment_protocol.time.sleep"):
                    save_sampler_checkpoint(study, SimpleNamespace(number=2), path)
                self.assertEqual(replace.call_count, 2)
            # The mocked successful replacement does not move the real file.
            actual_replace(path.with_name(path.name + ".tmp"), path)
            sampler, metadata = load_sampler_checkpoint(path)
            self.assertIsInstance(sampler, optuna.samplers.TPESampler)
            self.assertEqual(metadata["completed_trial_number"], 2)

    def test_sampler_recovers_newer_temporary_state_after_committed_trial(self):
        with tempfile.TemporaryDirectory(prefix="fraud_sampler_recover_") as temporary:
            path = Path(temporary) / "sampler.joblib"
            study = SimpleNamespace(sampler=optuna.samplers.TPESampler(seed=42), study_name="synthetic")
            save_sampler_checkpoint(study, SimpleNamespace(number=1), path)
            study.sampler = optuna.samplers.TPESampler(seed=43)
            with patch("experiment_protocol.os.replace", side_effect=PermissionError("persistent lock")):
                with patch("experiment_protocol.time.sleep"):
                    with self.assertRaises(PermissionError):
                        save_sampler_checkpoint(study, SimpleNamespace(number=2), path)
            temporary_hash = file_sha256(path.with_name(path.name + ".tmp"))
            sampler, metadata = load_sampler_checkpoint(path)
            self.assertEqual(metadata["completed_trial_number"], 2)
            self.assertTrue(metadata["recovered_temporary_checkpoint"])
            self.assertEqual(file_sha256(path), temporary_hash)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_exact_threshold_confusion_guard(self):
        with tempfile.TemporaryDirectory(prefix="fraud_threshold_guard_") as temporary:
            # Rounding this threshold down would incorrectly raise an alert.
            with self.assertRaisesRegex(ValueError, "exact decision threshold"):
                save_run(
                    None, {}, {"threshold": 0.123456, "TN": 0, "FP": 1, "FN": 0, "TP": 1},
                    [0, 1], [0.12345645, 0.8], {"threshold_exact": 0.12345647},
                    "synthetic", results_root=temporary,
                )

    def test_ulb_historical_baseline_cannot_supply_revised_parameters(self):
        with tempfile.TemporaryDirectory(prefix="fraud_baseline_guard_") as temporary:
            source = Path(temporary)
            (source / "config.json").write_text(json.dumps({
                "dataset": "ulb_2013", "model": "logreg", "strategy": "none",
                "dataset_hash_sha256": "synthetic", "best_params": {},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate-safe revised baseline"):
                _load_baseline_params("logreg", "ulb_2013", {
                    "baseline_run": source, "dataset_hash": "synthetic",
                })

    def test_synthetic_training_and_resampling_persist_evidence(self):
        X_dev, X_test, y_dev, y_test, metadata = _split_frame(
            self.make_frame(), "Class", deduplicate=True,
        )
        with tempfile.TemporaryDirectory(prefix="fraud_protocol_smoke_") as temporary:
            root = Path(temporary)
            context = {
                "results_root": root, "manifest_path": root / "revision_manifest.json",
                "split_metadata": metadata, "dataset_hash": "synthetic",
                "bootstrap_iterations": 0, "missing_policy": "preserve",
                "sample_fraction": None,
            }
            run_supervised(
                "logreg", X_dev, X_test, y_dev, y_test, get_preprocessor(X_dev),
                "synthetic", "synthetic", "synthetic_fixture", n_trials=2,
                context=context,
            )
            baseline = resolve_baseline_run("synthetic", "logreg", results_root=root)
            config = json.loads((baseline / "config.json").read_text())
            self.assertEqual(config["n_trials"], 2)
            self.assertTrue((baseline / "optuna_trials.json").exists())
            self.assertTrue((baseline / "completed.json").exists())
            self.assertEqual(len(np.load(baseline / "y_val_scores.npy")), len(y_dev))
            self.assertEqual(set(np.load(baseline / "validation_fold_ids.npy")), {1, 2, 3, 4, 5})
            self.assertEqual(config["data_provenance"]["dev_indices_sha256"], metadata["dev_indices_sha256"])
            cv = json.loads((baseline / "metrics_cv.json").read_text())
            self.assertEqual(config["threshold_exact"],
                             np.median([fold["threshold_exact"] for fold in cv["per_fold"]]))
            # Resume must neither replace completed trials nor add to its budget.
            original_trials = json.loads((baseline / "optuna_trials.json").read_text())
            run_supervised(
                "logreg", X_dev, X_test, y_dev, y_test, get_preprocessor(X_dev),
                "synthetic", "synthetic", "synthetic_fixture", n_trials=2,
                context=context,
            )
            resumed_baseline = resolve_baseline_run("synthetic", "logreg", results_root=root)
            resumed_trials = json.loads((resumed_baseline / "optuna_trials.json").read_text())
            self.assertEqual(resumed_trials, original_trials)
            baseline = resumed_baseline
            run_supervised_strategy(
                "logreg", "smote_tomek", X_dev, X_test, y_dev, y_test,
                get_preprocessor(X_dev), "synthetic", "synthetic", "synthetic_fixture",
                context=context,
            )
            manifest = json.loads((root / "revision_manifest.json").read_text())
            intervention = Path(manifest["runs"]["synthetic/logreg/smote_tomek"]["run_dir"])
            diagnostics = json.loads((intervention / "sampler_diagnostics.json").read_text())
            self.assertEqual(len(diagnostics), 6)
            self.assertEqual(diagnostics[-1]["stage"], "final_dev")
            selected = json.loads((intervention / "config.json").read_text())
            self.assertEqual(selected["baseline_run"], str(baseline))


if __name__ == "__main__":
    unittest.main()
