"""Small operational fixtures; no real scientific process is launched."""

import sys
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_revision_overnight as overnight


class OvernightTests(unittest.TestCase):
    def test_live_identity_uses_the_creation_token_without_float_conversion(self):
        token = 134357110998249638
        with patch.object(overnight, "process_identity", return_value={"status": "alive", "creation_token": token}):
            self.assertTrue(overnight.identity_matches({"pid": 14920, "creation_token": str(token)}))
            self.assertFalse(overnight.identity_matches({"pid": 14920, "creation_token": str(token + 1)}))

    def test_unknown_identity_refuses_to_start_duplicate_work(self):
        with patch.object(overnight, "process_identity", return_value={"status": "unknown", "creation_token": None}):
            with self.assertRaisesRegex(RuntimeError, "refuse duplicate work"):
                overnight.identity_matches({"pid": 1, "creation_token": "1"})

    def test_live_queue_is_adopted_without_completion_reads_or_launching(self):
        reference = {"pid": 2, "creation_token": "2"}
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "revision_manifest.json"
            with patch.object(overnight, "identity_matches", side_effect=lambda candidate: bool(candidate)), \
                    patch.object(overnight, "queue_complete") as complete, \
                    patch.object(overnight.subprocess, "Popen") as launch:
                result, process = overnight.attach_or_start("classical", manifest, {"classical": reference}, {}, Path(temporary))
            self.assertTrue(result["adopted"])
            self.assertIsNone(process)
            complete.assert_not_called()
            launch.assert_not_called()

    def test_completed_queue_is_revalidated_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "revision_manifest.json"
            with patch.object(overnight, "identity_matches", return_value=False), \
                    patch.object(overnight, "queue_complete", return_value=True), \
                    patch.object(overnight.subprocess, "Popen") as launch:
                result, process = overnight.attach_or_start("classical", manifest, {}, {}, Path(temporary))
            self.assertEqual(result["status"], "complete")
            self.assertIsNone(process)
            launch.assert_not_called()

    def test_failed_followup_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "revision_manifest.json"
            with patch.object(overnight, "read_manifest", return_value={"status": "failed"}), \
                    patch.object(overnight, "identity_matches", return_value=False), \
                    patch.object(overnight, "queue_complete", return_value=False), \
                    patch.object(overnight.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "previously failed"):
                    overnight.attach_or_start("followup_cpu", manifest, {}, {}, Path(temporary))
            launch.assert_not_called()

    def test_known_live_orphan_is_not_replaced(self):
        with patch.object(overnight, "identity_matches", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "scientific child"):
                overnight.reject_orphan_children(None, {"child_pid": 3, "child_creation_token": "3"})

    def test_all_five_queue_commands_remain_inside_the_project(self):
        manifest = overnight.ROOT / "results_revision/fixture/revision_manifest.json"
        for name in overnight.QUEUE_NAMES:
            command = overnight.command_for(name, manifest)
            self.assertEqual(command[0], sys.executable)
            self.assertIn(str(manifest), command)
            self.assertTrue(Path(command[6]).is_relative_to(overnight.ROOT))
            self.assertNotIn("--copy-figures", command)

    def failed_state(self, root, name, failure=None):
        path = overnight.queue_state_path(root, name)
        state = {"status": "failed", "failure": failure or overnight.BOOTSTRAP_METADATA_FAILURE,
                 "source_manifest": str((root / "revision_manifest.json").resolve()),
                 "queue_pid": 2, "queue_creation_token": "2", "completed_steps": ["preserved_step"]}
        path.write_text(json.dumps(state), encoding="utf-8")
        return path, state

    def test_explicit_metadata_recovery_archives_failures_and_preserves_completed_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "revision_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            paths = {name: self.failed_state(root, name) for name in ("followup_gpu", "analysis")}
            with patch.object(overnight, "is_complete", return_value=True), \
                    patch.object(overnight, "identity_matches", return_value=False), \
                    patch.object(overnight, "reject_orphan_children"):
                result = overnight.prepare_explicit_metadata_recovery(manifest, paths, {}, {})
            for name, (path, original) in paths.items():
                archive = Path(result["archived_states"][name]["path"])
                self.assertEqual(json.loads(archive.read_text()), original)
                current = json.loads(path.read_text())
                self.assertEqual(current["completed_steps"], ["preserved_step"])
                self.assertNotIn("failure", current)
                self.assertFalse(current["explicit_recovery"]["automatic_retry"])

    def test_explicit_recovery_cannot_retry_a_different_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "revision_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            path, original = self.failed_state(root, "analysis", "Scientific fitting failure")
            with patch.object(overnight, "is_complete", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "generic retry"):
                    overnight.prepare_explicit_metadata_recovery(manifest, ["analysis"], {}, {})
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertFalse((root / "operational_recovery").exists())

    def test_explicit_recovery_cannot_mutate_an_alive_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "revision_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            path, original = self.failed_state(root, "analysis")
            with patch.object(overnight, "is_complete", return_value=True), \
                    patch.object(overnight, "identity_matches", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "still alive"):
                    overnight.prepare_explicit_metadata_recovery(manifest, ["analysis"], {}, {})
            self.assertEqual(json.loads(path.read_text()), original)

    def test_explicit_recovery_checks_all_targets_before_any_state_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "revision_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            path, original = self.failed_state(root, "followup_gpu")
            self.failed_state(root, "analysis", "Unrelated analysis failure")
            with patch.object(overnight, "is_complete", return_value=True), \
                    patch.object(overnight, "identity_matches", return_value=False), \
                    patch.object(overnight, "reject_orphan_children"):
                with self.assertRaisesRegex(RuntimeError, "generic retry"):
                    overnight.prepare_explicit_metadata_recovery(manifest, ["followup_gpu", "analysis"], {}, {})
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertFalse((root / "operational_recovery").exists())

    def attention_failure_fixture(self, root):
        manifest = root / "revision_manifest.json"
        baseline = root / "baseline"
        baseline.mkdir()
        (baseline / "config.json").write_text(json.dumps({"test_samples": 2}), encoding="utf-8")
        manifest.write_text(json.dumps({"baseline_runs": {"baf_base/fttransformer": {
            "run_dir": str(baseline)}}}), encoding="utf-8")
        logs = root / "logs"
        logs.mkdir()
        log = logs / "followup_baf_ft_attention_20261006_100429.log"
        log.write_text(overnight.ATTENTION_DRIFT_FAILURE, encoding="utf-8")
        path, state = self.failed_state(root, "followup_gpu", f"Follow-up baf_ft_attention failed (1); inspect {log}")
        state.update(current_step="baf_ft_attention", last_child_exit_code=1,
                     completed_steps=["baf_absence_sensitivity_transformer", "baf_ft_smotenc"])
        path.write_text(json.dumps(state), encoding="utf-8")
        directory = root / "derived/interpretability/attention"
        directory.mkdir(parents=True)
        summary = {"n_test_samples": 2, "n_numerical_features": 2, "n_categorical_features": 1,
                   "score_verification": {"compared_test_rows": 2, "absolute_tolerance": 1e-5},
                   "extractor_code_sha256": overnight.file_sha256(overnight.ROOT / "src/attention_analysis.py"),
                   "model_implementation_sha256": overnight.file_sha256(overnight.ROOT / "src/models/fttransformer.py")}
        summary_path = directory / "attention_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        np.save(directory / "attention_cls_weights.npy", np.full((2, 4), .25, dtype=np.float32))
        return manifest, path, state, summary_path

    def test_verified_attention_recovery_preserves_failure_and_prior_completed_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path, original, unused = self.attention_failure_fixture(root)
            with patch.object(overnight, "identity_matches", return_value=False), \
                    patch.object(overnight, "reject_orphan_children"), \
                    patch.object(overnight, "step_complete", return_value=True) as complete:
                report = overnight.prepare_verified_attention_recovery(manifest, {}, {})
            archive = Path(report["archived_states"]["followup_gpu"]["path"])
            self.assertEqual(json.loads(archive.read_text()), original)
            recovered = json.loads(path.read_text())
            self.assertEqual(recovered["completed_steps"], original["completed_steps"])
            self.assertNotIn("failure", recovered)
            self.assertEqual(complete.call_count, 3)
            self.assertEqual(report["verified_test_rows"], 2)
            self.assertFalse(recovered["explicit_recovery"]["automatic_retry"])

    def test_attention_recovery_refuses_any_unverified_gpu_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path, original, unused = self.attention_failure_fixture(root)
            with patch.object(overnight, "identity_matches", return_value=False), \
                    patch.object(overnight, "reject_orphan_children"), \
                    patch.object(overnight, "step_complete", side_effect=[True, True, False]):
                with self.assertRaisesRegex(RuntimeError, "not complete"):
                    overnight.prepare_verified_attention_recovery(manifest, {}, {})
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertFalse((root / "operational_recovery").exists())

    def test_attention_recovery_refuses_changed_code_or_partial_population(self):
        for changed in ("code", "population"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, path, original, summary_path = self.attention_failure_fixture(root)
                summary = json.loads(summary_path.read_text())
                if changed == "code":
                    summary["model_implementation_sha256"] = "wrong source"
                else:
                    summary["score_verification"]["compared_test_rows"] = 1
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with patch.object(overnight, "identity_matches", return_value=False), \
                        patch.object(overnight, "reject_orphan_children"), \
                        patch.object(overnight, "step_complete", return_value=True):
                    with self.assertRaisesRegex(RuntimeError, "Full-population"):
                        overnight.prepare_verified_attention_recovery(manifest, {}, {})
                self.assertEqual(json.loads(path.read_text()), original)

    def test_attention_recovery_refuses_live_queue_or_unnormalised_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path, original, summary_path = self.attention_failure_fixture(root)
            with patch.object(overnight, "identity_matches", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "still alive"):
                    overnight.prepare_verified_attention_recovery(manifest, {}, {})
            np.save(summary_path.parent / "attention_cls_weights.npy", np.full((2, 4), .2, dtype=np.float32))
            with patch.object(overnight, "identity_matches", return_value=False), \
                    patch.object(overnight, "reject_orphan_children"), \
                    patch.object(overnight, "step_complete", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "normalised"):
                    overnight.prepare_verified_attention_recovery(manifest, {}, {})
            self.assertEqual(json.loads(path.read_text()), original)

    def test_attention_recovery_cannot_be_used_for_a_different_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path, original, unused = self.attention_failure_fixture(root)
            original["failure"] = "unrelated failed training"
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "generic retry"):
                overnight.prepare_verified_attention_recovery(manifest, {}, {})
            self.assertEqual(json.loads(path.read_text()), original)


if __name__ == "__main__":
    unittest.main()
