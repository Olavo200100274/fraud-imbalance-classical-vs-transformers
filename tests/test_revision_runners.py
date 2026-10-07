"""Synthetic operational checks; never launch a scientific training process."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_protocol import PROTOCOL_VERSION, array_sha256, file_sha256
from revision_resources import exclusive_resource, owner_is_stale, process_identity
from run_revision_followups import run_step, step_complete, _run_queue
from run_revision_training import is_complete, tasks


class RevisionRunnerTests(unittest.TestCase):
    def make_run(self, root, model="logreg", dataset="ulb_2013", policy="preserve"):
        run = root / "fixture"
        run.mkdir()
        values = {"dev_row_indices": np.array([0, 1, 2, 3], dtype=np.int64),
                  "test_row_indices": np.array([4, 5], dtype=np.int64),
                  "y_test": np.array([0, 1]), "y_test_scores": np.array([0.2, 0.8]),
                  "y_val": np.array([0, 1, 0, 1]), "y_val_scores": np.array([0.1, 0.9, 0.3, 0.7]),
                  "validation_row_indices": np.array([0, 1, 2, 3], dtype=np.int64),
                  "validation_fold_ids": np.array([1, 2, 1, 2], dtype=np.int64)}
        for name, array in values.items():
            np.save(run / (name + ".npy"), array, allow_pickle=False)
        provenance = {"raw_file_sha256": "synthetic", "feature_columns": ["first"],
                      "dev_indices_sha256": array_sha256(values["dev_row_indices"]),
                      "test_indices_sha256": array_sha256(values["test_row_indices"])}
        config = {"dataset": dataset, "model": model, "strategy": "none",
                  "protocol_version": PROTOCOL_VERSION, "sample_fraction": None,
                  "missing_policy": policy, "split_seed": 42, "split_ratio": "80/20 stratified",
                  "threshold_exact": 0.5, "train_samples": 4, "test_samples": 2,
                  "dataset_hash_sha256": "synthetic", "data_provenance": provenance,
                  "n_trials": 50, "bootstrap_iterations": 1000,
                  "validation_evidence": {"rows": 4,
                                          "row_indices_sha256": array_sha256(values["validation_row_indices"]),
                                          "scores_sha256": array_sha256(values["y_val_scores"]),
                                          "labels_sha256": array_sha256(values["y_val"])}}
        for filename, content in (("config.json", config),
                                  ("completed.json", {"status": "complete", "protocol_version": PROTOCOL_VERSION}),
                                  ("metrics_cv.json", {}),
                                  ("metrics_test.json", {"TP": 1, "FP": 0, "TN": 1, "FN": 0, "bootstrap_ci": {}}),
                                  ("optuna_trials.json", [{"state": "COMPLETE"} for _ in range(50)])):
            (run / filename).write_text(json.dumps(content), encoding="utf-8")
        (run / ("model.pt" if model == "fttransformer" else "model.joblib")).write_bytes(b"synthetic fixture")
        if model == "fttransformer":
            for filename in ("preprocessors.joblib", "validation_model.pt", "validation_preprocessors.joblib"):
                (run / filename).write_bytes(b"synthetic fixture")
            config.update(max_epochs=200, scheduler_horizon=200, best_epoch=12, early_stopping_patience=15)
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        manifest = {"runs": {f"{dataset}/{model}/none": {"run_dir": str(run),
                     "config_sha256": file_sha256(run / "config.json")}}, "datasets": {dataset: provenance}}
        return run, config, manifest

    def rewrite_config(self, run, config, manifest):
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        next(iter(manifest["runs"].values()))["config_sha256"] = file_sha256(run / "config.json")

    def test_missing_final_model_is_not_complete(self):
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            run, _, manifest = self.make_run(Path(directory))
            self.assertTrue(is_complete(manifest, "ulb_2013", "logreg", "none", expected_n_trials=50))
            (run / "model.joblib").unlink()
            self.assertFalse(is_complete(manifest, "ulb_2013", "logreg", "none"))

    def test_tampered_validation_scores_fail(self):
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            run, _, manifest = self.make_run(Path(directory))
            np.save(run / "y_val_scores.npy", np.array([0.1, 0.9, 0.3, 0.6]))
            with self.assertRaisesRegex(ValueError, "validation evidence array"):
                is_complete(manifest, "ulb_2013", "logreg", "none")

    def test_changed_exact_threshold_confusion_fails(self):
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            run, config, manifest = self.make_run(Path(directory))
            config["threshold_exact"] = 0.1
            self.rewrite_config(run, config, manifest)
            with self.assertRaisesRegex(ValueError, "confusion matrix"):
                is_complete(manifest, "ulb_2013", "logreg", "none")

    def test_sensitivity_requires_the_planned_representation(self):
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            _, _, manifest = self.make_run(Path(directory), dataset="baf_base")
            with self.assertRaisesRegex(ValueError, "missing-value policy"):
                is_complete(manifest, "baf_base", "logreg", "none", expected_missing_policy="nan_indicators")

    def test_primary_baseline_requires_all_fifty_trial_records(self):
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            run, _, manifest = self.make_run(Path(directory))
            (run / "optuna_trials.json").write_text(json.dumps([{"state": "COMPLETE"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "terminal Optuna"):
                is_complete(manifest, "ulb_2013", "logreg", "none", expected_n_trials=50)

    def test_shortened_ft_horizon_is_not_primary(self):
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            run, config, manifest = self.make_run(Path(directory), model="fttransformer")
            config["scheduler_horizon"] = 2
            self.rewrite_config(run, config, manifest)
            with self.assertRaisesRegex(ValueError, "shortened FT"):
                is_complete(manifest, "ulb_2013", "fttransformer", "none")

    def test_completed_first_ft_study_does_not_require_a_sqlite_checkpoint(self):
        # The authorised first GPU process loaded its in-memory study before
        # persistence guards existed. Its fifty terminal trials still establish
        # the HPO budget; do not invent a resume guarantee or repeat that fit.
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            run, _, manifest = self.make_run(Path(directory), model="fttransformer")
            self.assertFalse(list(run.rglob("*.sqlite*")))
            self.assertTrue(is_complete(manifest, "ulb_2013", "fttransformer", "none",
                                        expected_n_trials=50, expected_bootstrap_iterations=1000))

    def test_recovery_requires_reusing_the_historical_final_model(self):
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            _, _, manifest = self.make_run(Path(directory), dataset="baf_base")
            with self.assertRaisesRegex(ValueError, "historical final model"):
                is_complete(manifest, "baf_base", "logreg", "none", require_validation_recovery=True)

    def test_missing_bootstrap_metadata_requires_the_independent_verifier(self):
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            run, config, manifest = self.make_run(Path(directory), model="fttransformer")
            config.pop("bootstrap_iterations")
            self.rewrite_config(run, config, manifest)
            with patch("revision_bootstrap_compatibility.verify_bootstrap_compatibility") as verify:
                self.assertTrue(is_complete(manifest, "ulb_2013", "fttransformer", "none",
                                            expected_n_trials=50, expected_bootstrap_iterations=1000))
            verify.assert_called_once_with(manifest, run.resolve(), config, 1000)
            with self.assertRaisesRegex(ValueError, "restricted"):
                is_complete(manifest, "ulb_2013", "fttransformer", "none", expected_bootstrap_iterations=1000)

    def test_explicit_wrong_bootstrap_budget_is_not_a_compatibility_case(self):
        with tempfile.TemporaryDirectory(prefix="fraud_runner_") as directory:
            run, config, manifest = self.make_run(Path(directory), model="fttransformer")
            for value in (None, 0, 999):
                config["bootstrap_iterations"] = value
                self.rewrite_config(run, config, manifest)
                with patch("revision_bootstrap_compatibility.verify_bootstrap_compatibility") as verify:
                    with self.assertRaisesRegex(ValueError, "different bootstrap budget"):
                        is_complete(manifest, "ulb_2013", "fttransformer", "none", expected_bootstrap_iterations=1000)
                verify.assert_not_called()

    def test_baf_lane_uses_fixed_parameters_and_validation_only_flag(self):
        with patch("run_revision_training.read_manifest", return_value={
                "historical_baseline_runs": {"baf_base": {model: model + "_source" for model in
                  ("lgbm", "catboost", "logreg", "rf", "ocsvm", "fttransformer")}}}):
            planned = list(tasks("baf-validation", Path("manifest"), Path("outputs")))
            self.assertEqual(len(planned), 5)
            for _, _, model, strategy, extra, _, _ in planned:
                self.assertEqual(strategy, "none")
                self.assertIn("--recover-validation-only", extra)
                self.assertEqual(extra[extra.index("--fixed-params-run") + 1], model + "_source")

    def test_recorded_step_with_missing_evidence_cannot_be_silently_skipped(self):
        state = {"completed_steps": ["baf_ft_smotenc"]}
        with patch("run_revision_followups.step_complete", return_value=False):
            with patch("run_revision_followups.subprocess.Popen") as popen:
                with self.assertRaisesRegex(ValueError, "cannot silently skip"):
                    run_step("unused.py", [], "baf_ft_smotenc", Path("outputs"), Path("state"), state, Path("manifest"))
                popen.assert_not_called()

    def test_unrecorded_but_complete_step_is_recovered_without_launching(self):
        state = {}
        with patch("run_revision_followups.step_complete", return_value=True):
            with patch("run_revision_followups.write_state"):
                with patch("run_revision_followups.subprocess.Popen") as popen:
                    run_step("unused.py", [], "baf_ft_smotenc", Path("outputs"), Path("state"), state, Path("manifest"))
                    popen.assert_not_called()
                    self.assertEqual(state["completed_steps"], ["baf_ft_smotenc"])

    def test_gpu_queue_excludes_classical_sensitivity_and_passes_attention_dataset(self):
        with tempfile.TemporaryDirectory(prefix="fraud_followup_") as directory:
            root = Path(directory)
            manifest = root / "revision_manifest.json"
            manifest.write_text(json.dumps({"baseline_runs": {"baf_base/fttransformer": {"run_dir": "fixture"}}}), encoding="utf-8")
            args = SimpleNamespace(lane="gpu", manifest=manifest)
            with patch("run_revision_followups.wait_for_pins"):
                with patch("run_revision_followups.run_step") as step:
                    _run_queue(args, root, {"pid": 123, "creation_token": 4})
                    first = step.call_args_list[0].args[1]
                    self.assertEqual(first[first.index("--models") + 1], "fttransformer")
                    attention = step.call_args_list[-1].args[1]
                    self.assertEqual(attention[attention.index("--dataset") + 1], "baf_base")

    def test_win32_liveness_check_does_not_send_a_signal(self):
        with patch("os.kill", side_effect=AssertionError("A process signal must never be used.")):
            identity = process_identity(os.getpid())
            self.assertEqual(identity["status"], "alive")
            self.assertIsNotNone(identity["creation_token"])

    def test_live_owner_is_not_recovered_and_nested_lock_fails(self):
        with tempfile.TemporaryDirectory(prefix="fraud_resource_") as directory:
            root = Path(directory)
            with exclusive_resource(root, "synthetic", "fixture", lock_directory=root) as owner:
                self.assertFalse(owner_is_stale(owner))
                with self.assertRaisesRegex(RuntimeError, "Nested acquisition"):
                    with exclusive_resource(root, "synthetic", "nested", lock_directory=root, wait=False):
                        self.fail("A nested lock must not be acquired.")
            self.assertFalse((root / "synthetic.lock.json").exists())

    def test_pid_reuse_is_positive_evidence_of_stale_ownership(self):
        with patch("revision_resources.process_identity", return_value={"status": "alive", "creation_token": 20}):
            self.assertTrue(owner_is_stale({"pid": 123, "creation_token": 10}))
        with patch("revision_resources.process_identity", return_value={"status": "unknown", "creation_token": None}):
            self.assertFalse(owner_is_stale({"pid": 123, "creation_token": 10}))

    def test_stale_resource_is_recovered_without_signalling(self):
        with tempfile.TemporaryDirectory(prefix="fraud_resource_") as directory:
            root = Path(directory)
            path = root / "synthetic.lock.json"
            path.write_text(json.dumps({"pid": 123, "creation_token": 1, "token": "old", "resource": "synthetic"}), encoding="utf-8")
            with patch("revision_resources.owner_is_stale", return_value=True):
                with exclusive_resource(root, "synthetic", "new", lock_directory=root) as owner:
                    self.assertEqual(json.loads(path.read_text())["token"], owner["token"])
            self.assertFalse(path.exists())

    def test_dead_managed_parent_does_not_release_a_live_scientific_child(self):
        owner = {"pid": 123, "creation_token": 1, "child_pid": 456, "child_creation_token": 9}
        with patch("revision_resources.process_identity", side_effect=[
                {"status": "dead", "creation_token": None},
                {"status": "alive", "creation_token": 9}]):
            self.assertFalse(owner_is_stale(owner))
        with patch("revision_resources.process_identity", side_effect=[
                {"status": "dead", "creation_token": None},
                {"status": "dead", "creation_token": None}]):
            self.assertTrue(owner_is_stale(owner))


if __name__ == "__main__":
    unittest.main()
