"""Operator-only transfer recovery fixtures; no training or real inference."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_revision_overnight as overnight
import run_revision_analysis as analysis


class TransferRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        manifest = root / "revision_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        baseline = root / "baseline"
        baseline.mkdir()
        config = {"test_samples": 2, "protocol_version": overnight.PROTOCOL_VERSION}
        (baseline / "config.json").write_text(json.dumps(config), encoding="utf-8")
        names = ("model.pt", "preprocessors.joblib", "y_test.npy", "y_test_scores.npy", "test_row_indices.npy")
        for name in names:
            (baseline / name).write_bytes(name.encode())
        log = root / "logs/analysis_transfer_six_models_20261006_235034_037097.log"
        log.parent.mkdir()
        log.write_text(overnight.TRANSFER_CPU_FAILURE, encoding="utf-8")
        state = {"status": "failed", "current_step": "transfer_six_models", "last_child_exit_code": 1,
                 "failure": f"Analysis transfer_six_models failed; inspect {log}",
                 "source_manifest": str(manifest.resolve()), "current_command": ["--device", "cpu"],
                 "queue_pid": 2, "queue_creation_token": "2", "completed_steps": ["thresholds_ulb_2013"]}
        state_path = root / "analysis_state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        report = {"status": "complete", "new_fits": 0, "source_files_modified": False,
                  "source_run": str(baseline.resolve()), "protocol_version": overnight.PROTOCOL_VERSION,
                  "batch_size": 2048, "labels_and_indices_exactly_equal": True,
                  "model_implementation_sha256": overnight.file_sha256(overnight.ROOT / "src/models/fttransformer.py"),
                  "score_absolute_tolerance": 1e-6, "score_relative_tolerance": 1e-5,
                  "cuda_verification": {"device": "cuda", "rows": 2, "scores_bitwise_equal": True,
                                        "maximum_absolute_score_difference": 0,
                                        "rows_outside_existing_tolerance": 0, "decisions_changed_at_frozen_threshold": 0},
                  "source_artefacts_sha256": {name: overnight.file_sha256(baseline / name)
                                             for name in ("config.json", *names)}}
        report_path = root / "audit/transfer_cpu_cuda_20261007/ft_base_verification.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return manifest, baseline, config, state_path, state, report_path, report

    def recover(self, fixture, **overrides):
        manifest, baseline, config, unused_path, unused_state, unused_report_path, unused_report = fixture
        partial = {"models": {name: {} for name in ("logreg", "rf", "lgbm", "catboost")}, "reuse_provenance": {}}
        with patch.object(overnight, "identity_matches", return_value=overrides.get("alive", False)), \
                patch.object(overnight, "reject_orphan_children") as orphan, \
                patch.object(overnight, "queue_complete", return_value=overrides.get("complete", True)), \
                patch("revision_transfer.resolve_transfer_source", return_value=(baseline.resolve(), config)), \
                patch("revision_transfer.verify_partial_transfer", return_value=partial,
                      side_effect=overrides.get("partial_error")), \
                patch.object(analysis, "partition_directory", return_value=manifest.parent / "partitions"), \
                patch("torch.cuda.is_available", return_value=overrides.get("cuda", True)):
            if overrides.get("orphan_error"):
                orphan.side_effect = RuntimeError("Live orphan")
            return overnight.prepare_verified_transfer_recovery(manifest, {}, {})

    def unchanged(self, root, fixture):
        self.assertEqual(json.loads(fixture[3].read_text()), fixture[4])
        self.assertFalse((root / "operational_recovery").exists())

    def test_known_failure_is_archived_without_touching_scientific_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            before = {path.name: path.read_bytes() for path in fixture[1].iterdir()}
            result = self.recover(fixture)
            self.assertEqual(json.loads(Path(result["archived_states"]["analysis"]["path"]).read_text()), fixture[4])
            current = json.loads(fixture[3].read_text())
            self.assertEqual(current["completed_steps"], fixture[4]["completed_steps"])
            self.assertNotIn("failure", current)
            self.assertFalse(current["explicit_transfer_recovery"]["automatic_retry"])
            self.assertEqual({path.name: path.read_bytes() for path in fixture[1].iterdir()}, before)

    def test_different_failure_and_malformed_device_are_not_retryable(self):
        for changed in ("failure", "device"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                fixture[4]["failure"] = "Unrelated failure" if changed == "failure" else fixture[4]["failure"]
                if changed == "device":
                    fixture[4]["current_command"] = ["--device"]
                fixture[3].write_text(json.dumps(fixture[4]), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "generic retry"):
                    self.recover(fixture)
                self.unchanged(root, fixture)

    def test_live_parent_or_orphan_is_not_replaced(self):
        for options in ({"alive": True}, {"orphan_error": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                with self.assertRaises(RuntimeError):
                    self.recover(fixture, **options)
                self.unchanged(root, fixture)

    def test_missing_dependency_partial_corruption_or_cuda_refuses_before_writes(self):
        for options in ({"complete": False}, {"partial_error": ValueError("Tampered partial")}, {"cuda": False}):
            with self.subTest(options=str(options)), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.recover(fixture, **options)
                self.unchanged(root, fixture)

    def test_population_decision_and_tolerance_changes_are_not_accepted(self):
        for field, value in (("rows", 1), ("maximum_absolute_score_difference", 1e-8),
                             ("decisions_changed_at_frozen_threshold", 1), ("scores_bitwise_equal", False),
                             ("score_absolute_tolerance", 1e-4)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                if field == "score_absolute_tolerance":
                    fixture[6][field] = value
                else:
                    fixture[6]["cuda_verification"][field] = value
                fixture[5].write_text(json.dumps(fixture[6]), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "full-population"):
                    self.recover(fixture)
                self.unchanged(root, fixture)

    def test_changed_source_hash_cannot_authorise_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            (fixture[1] / "y_test_scores.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "protected FT source"):
                self.recover(fixture)
            self.unchanged(root, fixture)

    def test_ordinary_analysis_is_not_implicitly_recovered(self):
        root = Path("fixture").resolve()
        arguments = analysis.transfer_step_arguments(root / "revision_manifest.json", root, root / "partitions", {})
        self.assertEqual(arguments[-2:], ["--device", "cpu"])
        self.assertNotIn("--resume-verified-partial", arguments)

    def test_recovered_analysis_requires_hash_pinned_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            self.recover(fixture)
            state = json.loads(fixture[3].read_text())
            with patch("revision_transfer.verify_partial_transfer", return_value={"reuse_provenance": {}}):
                arguments = analysis.transfer_step_arguments(fixture[0], root, root / "partitions", state)
            self.assertEqual(arguments[arguments.index("--device") + 1], "cuda")
            self.assertIn("--resume-verified-partial", arguments)
            fixture[5].write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                analysis.transfer_step_arguments(fixture[0], root, root / "partitions", state)

    def test_changed_approved_partial_provenance_is_not_dispatched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            self.recover(fixture)
            state = json.loads(fixture[3].read_text())
            with patch("revision_transfer.verify_partial_transfer", return_value={"reuse_provenance": {"rf": "changed"}}):
                with self.assertRaisesRegex(ValueError, "provenance changed"):
                    analysis.transfer_step_arguments(fixture[0], root, root / "partitions", state)

    def test_later_resume_revalidates_completed_transfer_without_partial_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            self.recover(fixture)
            state = json.loads(fixture[3].read_text())
            completed = root / "derived/transfer/transfer_manifest.json"
            completed.parent.mkdir(parents=True)
            completed.write_text("{}", encoding="utf-8")
            with patch.object(analysis, "validate_transfer", return_value=True) as full, \
                    patch("revision_transfer.verify_partial_transfer") as partial:
                arguments = analysis.transfer_step_arguments(fixture[0], root, root / "partitions", state)
            full.assert_called_once()
            partial.assert_not_called()
            self.assertNotIn("--resume-verified-partial", arguments)


if __name__ == "__main__":
    unittest.main()
