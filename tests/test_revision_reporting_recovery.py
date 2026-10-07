"""Final-export-only recovery fixtures; no real scientific jobs or evidence."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_revision_overnight as overnight


class ReportingRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        manifest = root / "revision_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        qa_path = root / "derived/reporting/generation_qa.json"
        qa_path.parent.mkdir(parents=True)
        qa_path.write_text(json.dumps({"output_sha256": {"tables\\ulb\\baseline.tex": "fixture"}}),
                           encoding="utf-8")
        error_log = root / "logs/overnight_supervisor_20261007T0842385510422Z.error.log"
        error_log.parent.mkdir()
        error_log.write_text(f"ValueError: {overnight.REPORTING_PATH_FAILURE}\n", encoding="utf-8")
        prior = {"status": "failed", "failure": overnight.REPORTING_PATH_FAILURE,
                 "manifest_path": str(manifest.resolve()), "active_thesis_sources_modified": False,
                 "primary_pins_present": 43, "primary_pins_required": 43, "queues_finished": 4,
                 "supervisor": {"pid": 1, "creation_token": "1"}, "queues": {}}
        state = {"status": "analysis_ready_for_human_review", "current_step": "isolated_reporting_export",
                 "last_child_exit_code": 0, "completed_steps": list(overnight.REPORTING_COMPLETED_STEPS),
                 "source_manifest": str(manifest.resolve()),
                 "frozen_manifest_sha256": overnight.file_sha256(manifest),
                 "queue_pid": 2, "queue_creation_token": "2", "child_pid": None}
        (root / "analysis_state.json").write_text(json.dumps(state), encoding="utf-8")
        (root / "overnight_status.json").write_text(json.dumps(prior), encoding="utf-8")
        return manifest, qa_path, prior, state

    def guard_patches(self, fixture, **overrides):
        manifest, qa_path, unused_prior, unused_state = fixture
        stack = ExitStack()
        stack.enter_context(patch.object(overnight, "REPORTING_MANIFEST_SHA256", overnight.file_sha256(manifest)))
        stack.enter_context(patch.object(overnight, "REPORTING_QA_SHA256", overnight.file_sha256(qa_path)))
        stack.enter_context(patch.object(overnight, "identity_matches", **overrides.get("identity", {"return_value": False})))
        orphan = stack.enter_context(patch.object(overnight, "reject_orphan_children",
                                                  **overrides.get("orphan", {})))
        complete = stack.enter_context(patch.object(overnight, "queue_complete",
                                                    **overrides.get("complete", {"return_value": True})))
        launch = stack.enter_context(patch.object(overnight.subprocess, "Popen"))
        return stack, complete, orphan, launch

    def test_preserves_original_qa_and_completed_analysis_and_archives_old_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            manifest, qa_path, prior, unused_state = fixture
            originals = {path: path.read_bytes() for path in
                         (manifest, qa_path, root / "analysis_state.json", root / "overnight_status.json")}
            stack, complete, unused_orphan, launch = self.guard_patches(fixture)
            with stack:
                report = overnight.prepare_verified_reporting_recovery(manifest, {}, prior)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)
            for name, source in (("analysis", root / "analysis_state.json"),
                                 ("supervisor", root / "overnight_status.json"), ("generation_qa", qa_path)):
                archived = report["archived_states"][name]
                self.assertEqual(Path(archived["path"]).read_bytes(), originals[source])
                self.assertEqual(archived["sha256"], overnight.file_sha256(source))
            self.assertEqual(complete.call_args_list, [call(manifest, name) for name in overnight.QUEUE_NAMES])
            self.assertEqual(report["queues"], [])
            self.assertFalse(report["analysis_queue_restarted"])
            self.assertFalse(report["reporting_generator_reexecuted"])
            self.assertFalse(report["scientific_outputs_modified"])
            self.assertFalse(report["automatic_retry"])
            launch.assert_not_called()

    def test_refuses_an_unrelated_failure_or_incomplete_analysis_before_writes(self):
        for mutation in ("failure", "status", "step", "exit", "steps", "source", "frozen_hash"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                manifest, unused_qa, prior, state = fixture
                if mutation == "failure":
                    prior["failure"] = "Unrelated scientific error"
                elif mutation == "status":
                    state["status"] = "failed"
                elif mutation == "step":
                    state["current_step"] = "transfer_six_models"
                elif mutation == "exit":
                    state["last_child_exit_code"] = 1
                elif mutation == "steps":
                    state["completed_steps"].pop()
                elif mutation == "source":
                    state["source_manifest"] = str(root / "other_manifest.json")
                else:
                    state["frozen_manifest_sha256"] = "changed"
                state_path = root / "analysis_state.json"
                state_path.write_text(json.dumps(state), encoding="utf-8")
                original = state_path.read_bytes()
                stack, complete, unused_orphan, launch = self.guard_patches(fixture)
                with stack, self.assertRaisesRegex(RuntimeError, "generic retry"):
                    overnight.prepare_verified_reporting_recovery(manifest, {}, prior)
                self.assertEqual(state_path.read_bytes(), original)
                self.assertFalse((root / "operational_recovery").exists())
                complete.assert_not_called()
                launch.assert_not_called()

    def test_refuses_changed_frozen_manifest_or_original_qa(self):
        for name in ("manifest", "qa"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                manifest, qa_path, prior, unused_state = fixture
                stack, unused_complete, unused_orphan, launch = self.guard_patches(fixture)
                with stack:
                    target = manifest if name == "manifest" else qa_path
                    target.write_bytes(target.read_bytes() + b"\n")
                    with self.assertRaisesRegex(RuntimeError, "original reporting QA changed"):
                        overnight.prepare_verified_reporting_recovery(manifest, {}, prior)
                self.assertFalse((root / "operational_recovery").exists())
                launch.assert_not_called()

    def test_refuses_a_live_supervisor_or_queue_and_unknown_process_identity(self):
        for target in ("supervisor", "analysis", "unknown"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                manifest, unused_qa, prior, unused_state = fixture

                def identity(candidate):
                    if target == "unknown":
                        raise RuntimeError("Cannot verify queue PID; refuse duplicate work")
                    return bool(candidate and candidate.get("pid") == (1 if target == "supervisor" else 2))

                stack, unused_complete, unused_orphan, launch = self.guard_patches(fixture, identity={"side_effect": identity})
                with stack, self.assertRaisesRegex(RuntimeError, "still alive|refuse duplicate work"):
                    overnight.prepare_verified_reporting_recovery(manifest, {}, prior)
                self.assertFalse((root / "operational_recovery").exists())
                launch.assert_not_called()

    def test_refuses_an_orphan_or_unverified_dependency_without_archiving(self):
        for problem in ("orphan", "dependency"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                manifest, unused_qa, prior, unused_state = fixture
                changes = ({"orphan": {"side_effect": RuntimeError("Previous queue still has children")}}
                           if problem == "orphan" else {"complete": {"side_effect": [True, True, False]}})
                stack, unused_complete, unused_orphan, launch = self.guard_patches(fixture, **changes)
                with stack, self.assertRaisesRegex(RuntimeError, "children|not verified"):
                    overnight.prepare_verified_reporting_recovery(manifest, {}, prior)
                self.assertFalse((root / "operational_recovery").exists())
                launch.assert_not_called()

    def test_refuses_pid_records_without_creation_identity_before_writes(self):
        for name in ("supervisor", "analysis", "attachment"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                manifest, unused_qa, prior, state = fixture
                references = {}
                if name == "supervisor":
                    prior["supervisor"].pop("creation_token")
                elif name == "analysis":
                    state.pop("queue_creation_token")
                    (root / "analysis_state.json").write_text(json.dumps(state), encoding="utf-8")
                else:
                    references = {"queues": {"classical": {"pid": 3}}}
                stack, unused_complete, unused_orphan, launch = self.guard_patches(fixture)
                with stack, self.assertRaisesRegex(RuntimeError, "complete process identity"):
                    overnight.prepare_verified_reporting_recovery(manifest, references, prior)
                self.assertFalse((root / "operational_recovery").exists())
                launch.assert_not_called()

    def test_refuses_inputs_changing_during_queue_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            manifest, qa_path, prior, unused_state = fixture

            def changed(unused_manifest, name):
                if name == "analysis":
                    qa_path.write_bytes(qa_path.read_bytes() + b"\n")
                return True

            stack, unused_complete, unused_orphan, launch = self.guard_patches(fixture, complete={"side_effect": changed})
            with stack, self.assertRaisesRegex(RuntimeError, "changed during recovery"):
                overnight.prepare_verified_reporting_recovery(manifest, {}, prior)
            self.assertFalse((root / "operational_recovery").exists())
            launch.assert_not_called()

    def test_final_export_completion_never_enters_a_queue_launch_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            manifest, unused_qa, prior, unused_state = fixture
            with patch.object(overnight, "prepare_verified_reporting_recovery", return_value={"queues": []}), \
                    patch.object(overnight, "finish_exports", return_value={"fixture": "verified"}) as finish, \
                    patch.object(overnight, "attach_or_start") as attach, \
                    patch.object(overnight.subprocess, "Popen") as launch:
                overnight.complete_verified_reporting_recovery(manifest, {}, prior, {"fixture": "owner"})
            state = json.loads((root / "overnight_status.json").read_text())
            self.assertEqual(state["status"], "ready_for_thesis_revision")
            self.assertEqual(state["queues_finished"], 5)
            self.assertEqual(state["primary_pins_present"], 43)
            self.assertFalse(state["active_thesis_sources_modified"])
            self.assertFalse(state["automatic_failed_job_retries"])
            finish.assert_called_once_with(manifest)
            attach.assert_not_called()
            launch.assert_not_called()

    def test_failed_final_export_keeps_failed_status_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            manifest, unused_qa, prior, unused_state = fixture
            with patch.object(overnight, "prepare_verified_reporting_recovery", return_value={"queues": []}), \
                    patch.object(overnight, "finish_exports", side_effect=RuntimeError("Integrity failure")), \
                    patch.object(overnight, "attach_or_start") as attach, \
                    patch.object(overnight.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "Integrity failure"):
                    overnight.complete_verified_reporting_recovery(manifest, {}, prior, {})
            state = json.loads((root / "overnight_status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["failure"], "Integrity failure")
            self.assertFalse(state["active_thesis_sources_modified"])
            attach.assert_not_called()
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
