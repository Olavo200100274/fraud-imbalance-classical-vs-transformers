"""Bounded compatibility/CLI-isolation fixtures; no model scoring or fitting."""

from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cross_domain
import revision_transfer as transfer


class CrossDomainCompatibilityTests(unittest.TestCase):
    def test_legacy_entry_point_only_delegates_and_preserves_arguments(self):
        arguments = ["cross_domain.py", "--manifest", "explicit.json", "--models", "lgbm",
                     "--partitions-dir", "prepared", "--output-dir", "new-output"]
        with patch("sys.argv", arguments), patch("revision_transfer.main", return_value="delegated") as delegate:
            with patch("cross_domain.find_latest_run", side_effect=AssertionError("No latest-run selection")):
                with patch("cross_domain.evaluate_cross_domain", side_effect=AssertionError("No historical scoring")):
                    with patch("cross_domain.save_json", side_effect=AssertionError("No historical writes")):
                        self.assertEqual(cross_domain.main(), "delegated")
                        self.assertEqual(sys.argv, arguments)
            delegate.assert_called_once_with()

    def test_historical_imported_helpers_remain_available(self):
        for name in ("find_latest_run", "score_model", "score_torch_model", "evaluate_cross_domain", "load_json"):
            self.assertTrue(callable(getattr(cross_domain, name)))

    def test_custom_manifest_preparation_uses_its_own_transfer_root(self):
        with tempfile.TemporaryDirectory(prefix="fraud_transfer_cli_") as directory:
            manifest = Path(directory) / "custom/revision_manifest.json"
            with patch("sys.argv", ["cross_domain.py", "--manifest", str(manifest), "--prepare-only"]):
                with patch("revision_transfer.exclusive_resource", return_value=nullcontext()):
                    with patch("revision_transfer.prepare_partitions") as prepare:
                        with patch("revision_transfer.evaluate_transfer") as evaluate:
                            cross_domain.main()
            prepare.assert_called_once_with(output_root=manifest.resolve().parent / "transfer", manifest_path=manifest)
            evaluate.assert_not_called()

    def test_custom_manifest_implicit_preparation_and_evaluation_stay_isolated(self):
        with tempfile.TemporaryDirectory(prefix="fraud_transfer_cli_") as directory:
            manifest = Path(directory) / "custom/revision_manifest.json"
            root = manifest.resolve().parent / "transfer"
            prepared = root / "partitions/fixture"
            with patch("sys.argv", ["cross_domain.py", "--manifest", str(manifest), "--models", "lgbm"]):
                with patch("revision_transfer.exclusive_resource", return_value=nullcontext()):
                    with patch("revision_transfer.resolve_transfer_source"):
                        with patch("revision_transfer.prepare_partitions", return_value=prepared) as prepare:
                            with patch("revision_transfer.evaluate_transfer") as evaluate:
                                cross_domain.main()
            prepare.assert_called_once_with(output_root=root, manifest_path=manifest)
            self.assertEqual(evaluate.call_args.args, (prepared, ["lgbm"]))
            self.assertEqual(evaluate.call_args.kwargs["output_root"], root)

    def test_explicit_prepared_cohort_and_destination_are_not_replaced(self):
        with tempfile.TemporaryDirectory(prefix="fraud_transfer_cli_") as directory:
            manifest = Path(directory) / "custom/revision_manifest.json"
            prepared, destination = Path(directory) / "prepared", Path(directory) / "new-output"
            with patch("sys.argv", ["cross_domain.py", "--manifest", str(manifest), "--models", "lgbm",
                                    "--partitions-dir", str(prepared), "--output-dir", str(destination)]):
                with patch("revision_transfer.exclusive_resource", return_value=nullcontext()):
                    with patch("revision_transfer.resolve_transfer_source"):
                        with patch("revision_transfer.prepare_partitions") as prepare:
                            with patch("revision_transfer.evaluate_transfer") as evaluate:
                                cross_domain.main()
            prepare.assert_not_called()
            self.assertEqual(evaluate.call_args.args[0], prepared)
            self.assertEqual(evaluate.call_args.kwargs["output_dir"], destination)

    def test_archive_destination_is_rejected_before_data_or_models_are_loaded(self):
        destination = transfer.PROJECT_ROOT / "results/forbidden-transfer"
        with patch("sys.argv", ["cross_domain.py", "--output-dir", str(destination)]):
            with patch("revision_transfer.load_dataset") as load:
                with patch("revision_transfer.resolve_transfer_source") as resolve:
                    with patch("revision_transfer.evaluate_transfer") as evaluate:
                        with self.assertRaisesRegex(ValueError, "preserved results"):
                            cross_domain.main()
                        load.assert_not_called()
                        resolve.assert_not_called()
                        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
