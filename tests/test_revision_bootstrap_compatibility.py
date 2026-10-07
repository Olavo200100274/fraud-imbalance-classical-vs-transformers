"""Synthetic sidecar fixtures; never fit a model or alter a real run."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import revision_bootstrap_compatibility as compatibility
from experiment_protocol import PROTOCOL_VERSION, file_sha256


class BootstrapCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bootstrap_compatibility_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "results_revision/fixture" / compatibility.RUN_NAME
        self.run.mkdir(parents=True)
        metrics_source = self.root / "src/evaluation/metrics.py"
        metrics_source.parent.mkdir(parents=True)
        metrics_source.write_bytes(b"synthetic bootstrap source")
        self.metrics_digest = file_sha256(metrics_source)
        self.config = {"dataset": "ulb_2013", "model": "fttransformer", "strategy": "none",
                       "protocol_version": PROTOCOL_VERSION, "n_trials": 50, "sample_fraction": None,
                       "split_seed": 42, "split_ratio": "80/20 stratified", "missing_policy": "preserve",
                       "max_epochs": 200, "scheduler_horizon": 200, "early_stopping_patience": 15,
                       "best_epoch": 25, "threshold_exact": 0.5, "test_samples": 2,
                       "source_files_sha256": {"src\\evaluation\\metrics.py": self.metrics_digest}}
        self.intervals = {key: [0.4, 0.9] for key in compatibility.CI_KEYS}
        for name in compatibility.SOURCE_FILES:
            (self.run / name).write_bytes(b"synthetic saved artefact")
        for name, value in (("config.json", self.config),
                            ("completed.json", {"status": "complete", "protocol_version": PROTOCOL_VERSION}),
                            ("metrics_test.json", {"bootstrap_ci": self.intervals}),
                            ("optuna_trials.json", [{"state": "COMPLETE"} for _ in range(50)])):
            (self.run / name).write_text(json.dumps(value), encoding="utf-8")
        np.save(self.run / "y_test.npy", np.array([0, 1]))
        np.save(self.run / "y_test_scores.npy", np.array([0.1, 0.9]))
        self.config_digest = file_sha256(self.run / "config.json")
        self.manifest = {"runs": {compatibility.KEY: {"run_dir": str(self.run),
                                                     "config_sha256": self.config_digest}}}
        self.report_path = self.run.parent / "audit/verification.json"
        self.report_path.parent.mkdir()
        self.report = {"status": "complete", "role": compatibility.ROLE, "protocol_version": PROTOCOL_VERSION,
                       "key": compatibility.KEY, "run_dir": str(self.run), "config_sha256": self.config_digest,
                       "original_bootstrap_field": "absent", "verified_iterations": 1000, "random_state": 42,
                       "confidence_level": 0.95, "threshold_exact": 0.5,
                       "metrics_source_sha256": self.metrics_digest,
                       "source_artefacts_sha256": compatibility.source_hashes(self.run),
                       "stored_intervals": self.intervals, "recomputed_intervals": self.intervals,
                       "original_launch_snapshot_available": False,
                       "interval_match_proves_original_iteration_count": False,
                       "original_artefacts_modified": False}
        for attribute, value in (("ROOT", self.root), ("CONFIG_SHA256", self.config_digest),
                                 ("METRICS_SHA256", self.metrics_digest)):
            context = patch.object(compatibility, attribute, value)
            context.start()
            self.addCleanup(context.stop)
        self.save_report()

    def save_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        self.manifest["bootstrap_compatibility"] = {compatibility.KEY: {
            "path": str(self.report_path), "sha256": file_sha256(self.report_path)}}

    def verify(self, **kwargs):
        return compatibility.verify_bootstrap_compatibility(self.manifest, self.run, self.config, **kwargs)

    def test_registered_unchanged_sidecar_is_accepted(self):
        self.assertEqual(self.verify(), self.manifest["bootstrap_compatibility"][compatibility.KEY])

    def test_missing_or_unregistered_report_is_rejected(self):
        self.manifest.pop("bootstrap_compatibility")
        with self.assertRaisesRegex(ValueError, "registered"):
            self.verify()

    def test_explicit_none_zero_or_wrong_count_cannot_use_compatibility(self):
        for value in (None, 0, 999, 1000):
            with self.subTest(value=value):
                self.config["bootstrap_iterations"] = value
                with self.assertRaisesRegex(ValueError, "restricted"):
                    self.verify()
        self.config.pop("bootstrap_iterations")
        with self.assertRaisesRegex(ValueError, "restricted"):
            self.verify(expected_iterations=0)

    def test_different_run_or_dataset_is_rejected(self):
        self.config["dataset"] = "baf_base"
        with self.assertRaisesRegex(ValueError, "restricted"):
            self.verify()

    def test_modified_config_or_source_arrays_are_rejected(self):
        (self.run / "y_test_scores.npy").write_bytes(b"changed saved scores")
        with self.assertRaisesRegex(ValueError, "source artefact"):
            self.verify()

    def test_modified_report_is_rejected_even_if_manifest_run_pin_is_unchanged(self):
        self.report_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "report has changed"):
            self.verify()

    def test_incorrect_recomputed_interval_is_rejected(self):
        self.report["recomputed_intervals"] = {key: [0.4, 0.8] for key in compatibility.CI_KEYS}
        self.save_report()
        with self.assertRaisesRegex(ValueError, "reproduce"):
            self.verify()

    def test_original_count_or_launch_snapshot_must_not_be_invented(self):
        for key in ("interval_match_proves_original_iteration_count", "original_launch_snapshot_available"):
            self.report[key] = True
            self.save_report()
            with self.assertRaisesRegex(ValueError, "scope"):
                self.verify()
            self.report[key] = False

    def test_altered_bootstrap_implementation_is_rejected(self):
        (self.root / "src/evaluation/metrics.py").write_bytes(b"changed implementation")
        with self.assertRaisesRegex(ValueError, "implementation"):
            self.verify()

    def test_producer_preserves_all_original_files_and_latest_concurrent_manifest_keys(self):
        self.manifest.pop("bootstrap_compatibility")
        manifest_path = self.run.parent / "revision_manifest.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        output = self.run.parent / "audit/new_independent_verification.json"
        before = compatibility.source_hashes(self.run)

        def update_latest(path, update):
            latest = json.loads(path.read_text(encoding="utf-8"))
            latest["concurrent_pin"] = "must survive"
            update(latest)
            path.write_text(json.dumps(latest), encoding="utf-8")

        with patch("evaluation.metrics.bootstrap_ci", return_value=self.intervals) as bootstrap, \
                patch("revision_audit.update_manifest", side_effect=update_latest):
            reference = compatibility.produce_and_register(manifest_path, output)
        self.assertEqual(compatibility.source_hashes(self.run), before)
        self.assertEqual(json.loads(manifest_path.read_text())["concurrent_pin"], "must survive")
        self.assertEqual(reference["sha256"], file_sha256(output))
        self.assertEqual(bootstrap.call_args.kwargs, {"n_bootstrap": 1000, "ci": 0.95, "random_state": 42})
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            compatibility.produce_and_register(manifest_path, output)


if __name__ == "__main__":
    unittest.main()
