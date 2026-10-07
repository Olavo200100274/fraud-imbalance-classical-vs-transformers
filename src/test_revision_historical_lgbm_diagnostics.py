"""CPU checks for original-split provenance and isolated historical auditing."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from experiment_protocol import PROJECT_ROOT, file_sha256
from revision_historical_lgbm_diagnostics import (
    STRATEGIES, _assert_output_scope, reconstruct_original_test,
    run_historical_diagnostics, select_historical_runs,
)


class FakeBooster:
    def num_trees(self):
        return 3

    def predict(self, values, raw_score=False, pred_leaf=False, num_threads=None):
        probabilities = np.asarray(values)[:, 0]
        if pred_leaf:
            return np.tile((probabilities * 10).astype(int)[:, None], (1, 3))
        if raw_score:
            return np.log(probabilities / (1 - probabilities))
        raise AssertionError("Only read-only margin or leaf inference is expected.")


class FakeClassifier:
    booster_ = FakeBooster()

    def predict_proba(self, values, num_threads=None):
        probabilities = np.asarray(values)[:, 0]
        return np.column_stack((1 - probabilities, probabilities))

    def get_params(self):
        return {"num_leaves": 3, "n_estimators": 3, "class_weight": None}


def make_fixture(directory):
    raw_path = directory / "raw.csv"
    labels = np.array([0, 1] * 50)
    frame = pd.DataFrame({"V1": np.where(np.arange(100) % 3 == 0, 0.9, 0.2),
                          "Time": np.arange(100), "Class": labels})
    frame.to_csv(raw_path, index=False)
    dev, test = train_test_split(np.arange(100), test_size=0.2, stratify=labels, random_state=42)
    scores = frame.loc[test, "V1"].to_numpy()
    test_labels = labels[test]
    manifest = {"historical_runs": {}, "runs": {"must_not_be_selected": "invalid"}}
    for strategy in STRATEGIES:
        run = directory / "sources" / strategy
        run.mkdir(parents=True)
        config = {
            "dataset": "ulb_2013", "model": "lgbm", "strategy": strategy,
            "split_seed": 42, "split_ratio": "80/20 stratified",
            "train_samples": len(dev), "test_samples": len(test),
            "train_fraud": int(labels[dev].sum()), "test_fraud": int(test_labels.sum()),
            "dataset_hash_sha256": file_sha256(raw_path), "best_params": {"classifier__num_leaves": 3},
            "n_trials": 50 if strategy == "none" else 0,
        }
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "model.joblib").write_bytes(b"frozen-model-fixture")
        np.save(run / "y_test.npy", test_labels)
        np.save(run / "y_test_scores.npy", scores)
        (run / "metrics_test.json").write_text(json.dumps({
            "PR-AUC": round(average_precision_score(test_labels, scores), 6),
            "ROC-AUC": round(roc_auc_score(test_labels, scores), 6),
        }), encoding="utf-8")
        manifest["historical_runs"][f"ulb_2013/lgbm/{strategy}"] = {
            "run_dir": str(run), "config_sha256": file_sha256(run / "config.json"),
            "dataset_hash_sha256": file_sha256(raw_path),
        }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return raw_path, manifest_path, manifest, test, test_labels


class HistoricalSelectionTests(unittest.TestCase):
    def test_all_seven_explicit_pins_are_required_without_corrected_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, manifest, _, _ = make_fixture(Path(temporary))
            del manifest["historical_runs"]["ulb_2013/lgbm/weights"]
            with self.assertRaisesRegex(FileNotFoundError, "weights"):
                select_historical_runs(manifest)

    def test_pin_config_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, manifest, _, _ = make_fixture(Path(temporary))
            reference = manifest["historical_runs"]["ulb_2013/lgbm/ros"]
            reference["config_sha256"] = "incorrect"
            with self.assertRaisesRegex(ValueError, "changed"):
                select_historical_runs(manifest)

    def test_raw_split_reproduces_ordered_labels_without_saved_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, _, manifest, expected_test, expected_labels = make_fixture(Path(temporary))
            selected = select_historical_runs(manifest)
            _, test, labels, audit = reconstruct_original_test(raw_path, selected)
            np.testing.assert_array_equal(test, expected_test)
            np.testing.assert_array_equal(labels, expected_labels)
            self.assertTrue(audit["all_saved_label_sequences_match_raw_test"])
            self.assertIn("Not applied", audit["deduplication"])

    def test_reordered_labels_and_wrong_saved_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, _, manifest, test, labels = make_fixture(Path(temporary))
            selected = select_historical_runs(manifest)
            run = selected["smote"][0]
            np.save(run / "y_test.npy", 1 - labels)
            with self.assertRaisesRegex(ValueError, "split/labels/config"):
                reconstruct_original_test(raw_path, selected)
            np.save(run / "y_test.npy", labels)
            np.save(run / "test_row_indices.npy", test[::-1])
            with self.assertRaisesRegex(ValueError, "indices disagree"):
                reconstruct_original_test(raw_path, selected)

    def test_source_results_and_document_output_paths_are_rejected(self):
        for name in ("results", "results_thesis", "Overleaf", "Article 1", "Article 2"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _assert_output_scope(PROJECT_ROOT / name / "new_audit")

    def test_end_to_end_frozen_inference_preserves_sources_and_marks_historical_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw_path, manifest_path, manifest, _, _ = make_fixture(directory)
            source_files = sorted((directory / "sources").rglob("*"))
            before = {str(path): file_sha256(path) for path in source_files if path.is_file()}
            fake_model = SimpleNamespace(named_steps={
                "preprocessor": SimpleNamespace(transform=lambda frame: frame),
                "classifier": FakeClassifier(),
            })
            with patch("revision_historical_lgbm_diagnostics.joblib.load", return_value=fake_model):
                report_path = run_historical_diagnostics(
                    manifest_path, directory / "output", csv_path=raw_path, threads=2,
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["role"], "historical_original_protocol_diagnostic_only")
            self.assertFalse(report["fit_or_resampling_performed"])
            self.assertEqual(set(report["strategies"]), set(STRATEGIES))
            self.assertTrue(report["exact_score_comparisons"]["smote_vs_smote_tomek"]["bitwise_equal"])
            for record in report["strategies"].values():
                self.assertFalse(record["source_indices_artefact_present"])
                self.assertIsNone(record["source_indices_sha256"])
                group = record["top_exact_score_groups"][0]
                self.assertEqual(group["unique_full_leaf_vectors"], 1)
                self.assertEqual(group["raw_margin_unique_values"], 1)
                self.assertEqual(group["leaf_equality_checked_across_trees"], 3)
            after = {str(path): file_sha256(path) for path in source_files if path.is_file()}
            self.assertEqual(before, after)
            with self.assertRaisesRegex(FileExistsError, "must not be replaced"):
                run_historical_diagnostics(manifest_path, directory / "output", csv_path=raw_path)


if __name__ == "__main__":
    unittest.main()
