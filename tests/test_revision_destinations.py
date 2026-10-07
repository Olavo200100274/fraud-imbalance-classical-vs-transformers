"""Operational path guards tested without loading data, fitting or archiving."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import experiment_protocol as protocol
from save_load import finalise_run, save_run


class RevisionDestinationTests(unittest.TestCase):
    def test_roots_and_descendants_of_all_preserved_directories_are_rejected(self):
        for name in ("results", "results_thesis", "Overleaf"):
            for suffix in ("", "nested/new-output"):
                with self.subTest(name=name, suffix=suffix):
                    path = PROJECT_ROOT / name / suffix
                    with self.assertRaisesRegex(ValueError, "preserved"):
                        protocol.validate_revision_destinations(path)
                    with self.assertRaisesRegex(ValueError, "preserved"):
                        protocol.validate_revision_destinations(manifest_path=path / "manifest.json")

    def test_resolved_traversal_and_windows_case_equivalence_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "preserved"):
            protocol.validate_revision_destinations(PROJECT_ROOT / "temporary/../results/forbidden")
        if os.name == "nt":
            with self.assertRaisesRegex(ValueError, "preserved"):
                protocol.validate_revision_destinations(Path(str(PROJECT_ROOT / "RESULTS_THESIS/forbidden").upper()))

    def test_repository_root_is_not_an_output_destination(self):
        with self.assertRaisesRegex(ValueError, "repository root"):
            protocol.validate_revision_destinations(PROJECT_ROOT)

    def test_custom_root_defaults_to_its_own_manifest(self):
        custom = PROJECT_ROOT / "results_revision/reproduction-fixture"
        root, manifest = protocol.validate_revision_destinations(custom)
        self.assertEqual(root, custom.resolve())
        self.assertEqual(manifest, custom.resolve() / "revision_manifest.json")
        self.assertFalse(custom.exists())

    def test_snapshot_capture_rejects_archive_before_any_creation(self):
        with patch("experiment_protocol.zipfile.ZipFile") as archive:
            with self.assertRaisesRegex(ValueError, "preserved"):
                protocol.capture_source_provenance(PROJECT_ROOT / "results_thesis/new-output")
            archive.assert_not_called()

    def test_save_rejects_protected_manifest_before_any_artefact_write(self):
        with patch("save_load.os.makedirs") as mkdir, patch("save_load._save_json") as write:
            with self.assertRaisesRegex(ValueError, "preserved"):
                save_run(None, {}, {}, [], [], {}, "fixture", manifest_path=PROJECT_ROOT / "results/manifest.json")
            mkdir.assert_not_called()
            write.assert_not_called()

    def test_completion_rejects_protected_destination_before_writing_a_marker(self):
        with patch("save_load._save_json") as write:
            with self.assertRaisesRegex(ValueError, "preserved"):
                finalise_run(PROJECT_ROOT / "results/dataset/model/none/run_fixture")
            write.assert_not_called()

    def test_manifest_registration_rejects_archive_before_config_read(self):
        with self.assertRaisesRegex(ValueError, "preserved"):
            protocol.record_completed_run(PROJECT_ROOT / "results_revision/fixture_missing",
                                          PROJECT_ROOT / "Overleaf/manifest.json")

    def test_both_training_clis_reject_bad_manifest_before_snapshot_lock_or_dataset(self):
        import main
        import main_transformer
        for module in (main, main_transformer):
            args = SimpleNamespace(results_root=protocol.DEFAULT_RESULTS_ROOT,
                                   run_manifest=PROJECT_ROOT / "results_thesis/manifest.json")
            with self.subTest(module=module.__name__):
                with patch.object(module, "parse_args", return_value=args):
                    with patch.object(module, "capture_source_provenance") as snapshot:
                        with patch.object(module, "baf_training_resource") as lock:
                            with patch.object(module, "load_dataset") as load:
                                with self.assertRaisesRegex(ValueError, "preserved"):
                                    module.main()
                                snapshot.assert_not_called()
                                lock.assert_not_called()
                                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
