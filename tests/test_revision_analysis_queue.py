"""Bounded saved-evidence queue checks; never launch analysis or training jobs."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_revision_analysis as analysis
from experiment_protocol import file_sha256
from revision_resources import keep_system_awake


class RevisionAnalysisQueueTests(unittest.TestCase):
    def test_system_awake_request_does_not_require_the_display_and_is_restored(self):
        request = Mock(return_value=0x80000000)
        kernel = SimpleNamespace(SetThreadExecutionState=request)
        with patch("revision_resources.os.name", "nt"):
            with patch("revision_resources.ctypes.WinDLL", return_value=kernel, create=True):
                with keep_system_awake():
                    self.assertEqual(request.call_args.args, (0x80000001,))
                self.assertEqual([call.args for call in request.call_args_list],
                                 [(0x80000001,), (0x80000000,)])

    def test_system_awake_request_is_restored_after_an_exception(self):
        request = Mock(return_value=0x80000000)
        kernel = SimpleNamespace(SetThreadExecutionState=request)
        with patch("revision_resources.os.name", "nt"):
            with patch("revision_resources.ctypes.WinDLL", return_value=kernel, create=True):
                with self.assertRaisesRegex(ValueError, "fixture"):
                    with keep_system_awake():
                        raise ValueError("fixture")
                self.assertEqual(request.call_args.args, (0x80000000,))

    def test_system_awake_request_fails_clearly_if_windows_refuses_it(self):
        request = Mock(return_value=0)
        kernel = SimpleNamespace(SetThreadExecutionState=request)
        with patch("revision_resources.os.name", "nt"):
            with patch("revision_resources.ctypes.WinDLL", return_value=kernel, create=True):
                with self.assertRaisesRegex(RuntimeError, "Cannot establish"):
                    with keep_system_awake():
                        self.fail("The queue must not start with an unverified sleep guard.")

    def test_non_windows_awake_context_is_a_safe_noop(self):
        with patch("revision_resources.os.name", "posix"):
            with patch("revision_resources.ctypes.WinDLL", create=True) as kernel:
                with keep_system_awake():
                    pass
                kernel.assert_not_called()

    def test_analysis_resume_rejects_a_different_manifest(self):
        with tempfile.TemporaryDirectory(prefix="fraud_analysis_queue_") as directory:
            root = Path(directory)
            manifest = root / "revision_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            (root / "analysis_state.json").write_text(json.dumps({
                "source_manifest": str(root / "another_manifest.json")}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another source manifest"):
                analysis.execute_queue(SimpleNamespace(manifest=manifest), {})

    def test_missing_outputs_are_not_declared_complete(self):
        with tempfile.TemporaryDirectory(prefix="fraud_analysis_queue_") as directory:
            root = Path(directory)
            manifest = root / "revision_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            self.assertFalse(analysis.validate_diagnostics(manifest, root))
            self.assertFalse(analysis.validate_thresholds(manifest, root, "ulb_2013"))
            self.assertFalse(analysis.validate_transfer(manifest, root))
            self.assertFalse(analysis.validate_paired(manifest, root))
            self.assertFalse(analysis.validate_sampler_audit(manifest, root))
            self.assertFalse(analysis.variant_shap_ready(manifest, root))
            self.assertFalse(analysis.validate_generation(manifest, root))
            self.assertFalse(analysis.validate_attention_text(manifest, root))

    def test_variant_names_match_the_actual_prepared_cohort_api(self):
        from revision_transfer import VARIANTS
        self.assertEqual(tuple(VARIANTS), analysis.VARIANTS)

    def test_partial_variant_shap_is_waited_for_not_replaced(self):
        with tempfile.TemporaryDirectory(prefix="fraud_analysis_queue_") as directory:
            root = Path(directory)
            path = root / "derived/interpretability/variant_stability/shap_variant_stability.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "in_progress"}), encoding="utf-8")
            with patch("run_revision_analysis.partition_directory") as prepare:
                self.assertFalse(analysis.variant_shap_ready(root / "manifest", root))
                prepare.assert_not_called()

    def test_prepared_partition_requires_an_explicit_pin(self):
        with tempfile.TemporaryDirectory(prefix="fraud_analysis_queue_") as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicitly pinned"):
                analysis.partition_directory(manifest)

    def test_completed_step_with_invalid_outputs_cannot_be_skipped(self):
        with patch("run_revision_analysis.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "lost its evidence"):
                analysis.run_analysis_step("unused.py", [], "fixture", lambda: False,
                                           Path("output"), Path("state"), {"completed_steps": ["fixture"]})
            popen.assert_not_called()

    def test_complete_outputs_resume_without_launching_a_child(self):
        state = {}
        with patch("run_revision_analysis.subprocess.Popen") as popen:
            with patch("run_revision_analysis.write_state"):
                analysis.run_analysis_step("unused.py", [], "fixture", lambda: True,
                                           Path("output"), Path("state"), state)
            popen.assert_not_called()
            self.assertEqual(state["completed_steps"], ["fixture"])

    def test_paired_consumer_checks_new_source_hash_maps_without_new_jobs(self):
        from test_revision_paired_analysis import PairedSourceVerificationTests
        with tempfile.TemporaryDirectory(prefix="fraud_paired_queue_") as directory:
            root = Path(directory)
            report, primary, sensitivity, paths = PairedSourceVerificationTests.fixture(root)
            manifest = root / "revision_manifest.json"
            manifest.write_text(json.dumps(primary), encoding="utf-8")
            sensitivity_file = root / "sensitivity/revision_manifest.json"
            sensitivity_file.write_text(json.dumps(sensitivity), encoding="utf-8")
            output = root / "derived/paired/paired_analysis.json"
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps(report), encoding="utf-8")
            with patch("run_revision_analysis.subprocess.Popen") as popen:
                self.assertTrue(analysis.validate_paired(manifest, root))
                for label, name in (("rf/second", "y_test_scores.npy"),
                                    ("ft_control/second", "test_row_indices.npy"),
                                    ("cb_control", "completed.json")):
                    path = paths[label] / name
                    original = path.read_bytes()
                    path.write_bytes(b"changed after the paired analysis")
                    try:
                        with self.subTest(source=label, artefact=name), self.assertRaises(ValueError):
                            analysis.validate_paired(manifest, root)
                    finally:
                        path.write_bytes(original)
                popen.assert_not_called()

    def test_generation_rejects_changed_frozen_manifest(self):
        with tempfile.TemporaryDirectory(prefix="fraud_analysis_queue_") as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            qa = root / "derived/reporting/generation_qa.json"
            qa.parent.mkdir(parents=True)
            qa.write_text(json.dumps({"required_runs": 72, "manifest_sha256": "another digest"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen complete"):
                analysis.validate_generation(manifest, root)

    def test_generation_rejects_an_artefact_outside_its_isolated_root(self):
        with tempfile.TemporaryDirectory(prefix="fraud_analysis_queue_") as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            qa = root / "derived/reporting/generation_qa.json"
            qa.parent.mkdir(parents=True)
            qa.write_text(json.dumps({"required_runs": 72, "manifest_sha256": file_sha256(manifest),
                                      "output_sha256": {"../../../outside.tex": "irrelevant"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                analysis.validate_generation(manifest, root)

    def test_planned_queue_has_no_fit_or_active_thesis_writes(self):
        with tempfile.TemporaryDirectory(prefix="fraud_analysis_queue_") as directory:
            root = Path(directory)
            manifest = root / "revision_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            owner = {"pid": 123, "creation_token": 4}
            with patch("run_revision_analysis.wait_for_pins") as waits:
                with patch("run_revision_analysis.wait_until"):
                    with patch("run_revision_analysis.partition_directory", return_value=root / "prepared"):
                        with patch("run_revision_analysis.pin_variant_shap"):
                            with patch("run_revision_analysis.run_analysis_step") as steps:
                                analysis.execute_queue(SimpleNamespace(manifest=manifest), owner)
            scripts = [call.args[0] for call in steps.call_args_list]
            self.assertEqual(scripts, ["revision_lgbm_diagnostics.py", "revision_thresholds.py", "revision_thresholds.py",
                                       "revision_transfer.py", "revision_paired_analysis.py", "revision_sampler_audit.py",
                                       "revision_attention_text.py",
                                       "generate_results.py"])
            self.assertNotIn("main.py", scripts)
            self.assertNotIn("main_transformer.py", scripts)
            self.assertEqual(len(waits.call_args_list[1].args[1]), 12)
            transfer = steps.call_args_list[3].args[1]
            self.assertEqual(transfer[transfer.index("--bootstrap-iterations") + 1], "1000")
            self.assertEqual(transfer[transfer.index("--output-dir") + 1], root / "derived/transfer")
            export = steps.call_args_list[-1].args[1]
            self.assertEqual(export[export.index("--output-root") + 1], root / "derived/reporting")
            state = json.loads((root / "analysis_state.json").read_text())
            self.assertEqual(state["status"], "analysis_ready_for_human_review")
            self.assertEqual(manifest.read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
