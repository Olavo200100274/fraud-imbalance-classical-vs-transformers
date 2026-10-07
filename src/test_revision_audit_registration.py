"""Metadata-only historical registration fixtures; no dataset or model fits."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import revision_audit as audit
from experiment_protocol import file_sha256


class HistoricalRegistrationTests(unittest.TestCase):
    def fixture(self, root):
        for strategy, name in audit.HISTORICAL_ULB_LGBM.items():
            run = root / "results/ulb_2013/lgbm" / strategy / name
            run.mkdir(parents=True)
            config = {"dataset": "ulb_2013", "model": "lgbm", "strategy": strategy,
                      "train_samples": 227845, "test_samples": 56962,
                      "train_fraud": 394, "test_fraud": 98, "split_seed": 42,
                      "split_ratio": "80/20 stratified", "best_params": {"classifier__num_leaves": 4},
                      "n_trials": 50 if strategy == "none" else 0,
                      "dataset_hash_sha256": audit.ULB_RAW_SHA256}
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return {"historical_baseline_runs": {"ulb_2013": {"lgbm":
                   "results/ulb_2013/lgbm/none/run_20260309_181852"}},
                "historical_runs": {"baf_base/fixture/none": {"run_dir": "preserved", "config_sha256": "unchanged"}},
                "runs": {"corrected": {"run_dir": "new"}}, "baseline_runs": {"corrected": "new"}}

    def test_seven_explicit_sources_are_diagnostic_only_and_idempotent(self):
        with tempfile.TemporaryDirectory(prefix="fraud_historical_pin_") as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            protected = copy.deepcopy({key: manifest[key] for key in ("runs", "baseline_runs")})
            with patch("revision_audit.ROOT", root):
                pins = audit.historical_ulb_lgbm_pins(manifest)
            self.assertEqual(len(pins), 7)
            self.assertTrue(all(pin["role"] == "historical_ulb_diagnostic_only" for pin in pins.values()))
            audit.merge_historical_pins(manifest, pins)
            once = copy.deepcopy(manifest)
            audit.merge_historical_pins(manifest, pins)
            self.assertEqual(manifest, once)
            self.assertEqual({key: manifest[key] for key in protected}, protected)
            self.assertEqual(manifest["historical_runs"]["baf_base/fixture/none"]["run_dir"], "preserved")

    def test_missing_explicit_run_has_no_latest_fallback(self):
        with tempfile.TemporaryDirectory(prefix="fraud_historical_pin_") as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            path = root / "results/ulb_2013/lgbm/ros" / audit.HISTORICAL_ULB_LGBM["ros"] / "config.json"
            path.rename(path.with_name("not_config.json"))
            with patch("revision_audit.ROOT", root):
                with self.assertRaises(FileNotFoundError):
                    audit.historical_ulb_lgbm_pins(manifest)

    def test_identity_counts_hash_and_baseline_mismatches_fail(self):
        for field, value in (("strategy", "weights"), ("train_samples", 226980),
                             ("test_samples", 56746), ("dataset_hash_sha256", "changed"),
                             ("sample_fraction", 0.05)):
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix="fraud_historical_pin_") as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                path = root / "results/ulb_2013/lgbm/none" / audit.HISTORICAL_ULB_LGBM["none"] / "config.json"
                config = json.loads(path.read_text(encoding="utf-8"))
                config[field] = value
                path.write_text(json.dumps(config), encoding="utf-8")
                with patch("revision_audit.ROOT", root):
                    with self.assertRaises(ValueError):
                        audit.historical_ulb_lgbm_pins(manifest)
        with tempfile.TemporaryDirectory(prefix="fraud_historical_pin_") as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest["historical_baseline_runs"]["ulb_2013"]["lgbm"] = "results/different"
            with patch("revision_audit.ROOT", root):
                with self.assertRaisesRegex(ValueError, "baseline pin"):
                    audit.historical_ulb_lgbm_pins(manifest)

    def test_preflight_merge_preserves_extra_ulb_pins_and_rejects_conflicts(self):
        manifest = {"historical_runs": {"ulb_2013/lgbm/none": {"run_dir": "old", "config_sha256": "fixed"}}}
        audit.merge_historical_pins(manifest, {"baf_base/lgbm/none": {"run_dir": "baf", "config_sha256": "baf"}})
        self.assertEqual(set(manifest["historical_runs"]), {"ulb_2013/lgbm/none", "baf_base/lgbm/none"})
        unchanged = copy.deepcopy(manifest)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            audit.merge_historical_pins(manifest, {"ulb_2013/lgbm/none": {"run_dir": "other", "config_sha256": "changed"}})
        self.assertEqual(manifest, unchanged)

    def test_flag_registers_only_and_never_calls_preflight(self):
        with tempfile.TemporaryDirectory(prefix="fraud_historical_pin_") as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            path = root / "revision_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            originals = {name: file_sha256(root / "results/ulb_2013/lgbm" / strategy / name / "config.json")
                         for strategy, name in audit.HISTORICAL_ULB_LGBM.items()}
            with patch("revision_audit.ROOT", root), patch("revision_audit.preflight") as preflight:
                with patch("sys.argv", ["revision_audit.py", "--manifest", str(path), "--register-historical-ulb-lgbm"]):
                    audit.main()
                preflight.assert_not_called()
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["historical_runs"]), 8)
            self.assertEqual(saved["runs"], manifest["runs"])
            self.assertEqual(saved["baseline_runs"], manifest["baseline_runs"])
            for strategy, name in audit.HISTORICAL_ULB_LGBM.items():
                self.assertEqual(file_sha256(root / "results/ulb_2013/lgbm" / strategy / name / "config.json"), originals[name])


class ReportRegistrationTests(unittest.TestCase):
    def fixture(self, root):
        manifest = HistoricalRegistrationTests().fixture(root)
        with patch("revision_audit.ROOT", root):
            audit.merge_historical_pins(manifest, audit.historical_ulb_lgbm_pins(manifest))
        manifest.update(status="training_in_progress", unrelated={"preserved": True},
                        completion_checks={"protocol_tests": True})
        revision = root / "results_revision/fixture"
        (revision / "audit").mkdir(parents=True)
        manifest_path = revision / "revision_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        source = root / "Overleaf/Chapters/fixture.tex"
        source.parent.mkdir(parents=True)
        source.write_text("Scientific claim fixture.\n", encoding="utf-8")
        citation = {"schema_version": 1, "source_snapshot": {"hash_algorithm": "SHA-256", "files": [
            {"path": str(source.relative_to(root)), "sha256": file_sha256(source)}]},
            "remaining_items": [{"id": "overleaf_pdf_validation", "severity": "Pending final compilation"}]}
        historical = {"status": "complete", "role": "historical_original_protocol_diagnostic_only",
                      "source_manifest": str(manifest_path), "source_manifest_snapshot_sha256": "older training snapshot",
                      "fit_or_resampling_performed": False, "threshold_selection_performed": False,
                      "test_labels_used_for_fitting": False, "historical_hpo": {"baseline_trials": 50},
                      "split_audit": {"raw_file_sha256": audit.ULB_RAW_SHA256,
                                      "dev_rows": 227845, "test_rows": 56962, "dev_fraud": 394,
                                      "test_fraud": 98, "split_seed": 42, "split_ratio": "80/20 stratified",
                                      "all_saved_label_sequences_match_raw_test": True}, "strategies": {}}
        for strategy in audit.HISTORICAL_ULB_LGBM:
            reference = manifest["historical_runs"][f"ulb_2013/lgbm/{strategy}"]
            historical["strategies"][strategy] = {"source_run": reference["run_dir"],
                "source_config_sha256": reference["config_sha256"], "rows": 56962, "fraud": 98,
                "best_params": {"classifier__num_leaves": 4}}
        citation_path = revision / "audit/citation_claim_review.json"
        historical_path = revision / "audit/historical_lgbm_score_diagnostics.json"
        citation_path.write_text(json.dumps(citation), encoding="utf-8")
        historical_path.write_text(json.dumps(historical), encoding="utf-8")
        return manifest_path, manifest, citation_path, citation, historical_path, historical, source

    def run_registration(self, root, path, *flags):
        with patch("revision_audit.ROOT", root), patch("revision_audit.preflight") as preflight:
            with patch("revision_audit.load_dataset") as data, patch("revision_audit.subprocess.run") as archive:
                with patch("sys.argv", ["revision_audit.py", "--manifest", str(path), *map(str, flags)]):
                    audit.main()
                preflight.assert_not_called()
                data.assert_not_called()
                archive.assert_not_called()

    def test_both_reports_register_atomically_preserving_status_and_all_other_keys(self):
        with tempfile.TemporaryDirectory(prefix="fraud_report_pin_") as directory:
            root = Path(directory)
            path, original, citation_path, _, historical_path, _, source = self.fixture(root)
            source_hash = file_sha256(source)
            flags = ["--register-citation-claims", citation_path,
                     "--register-historical-lgbm-diagnostics", historical_path]
            self.run_registration(root, path, *flags)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual({key: saved[key] for key in original}, original)
            self.assertEqual(set(saved) - set(original), {"citation_claim_review", "historical_lgbm_diagnostics"})
            for key, report_path in (("citation_claim_review", citation_path),
                                     ("historical_lgbm_diagnostics", historical_path)):
                self.assertEqual(saved[key], {"path": str(report_path.resolve()), "sha256": file_sha256(report_path)})
            self.assertEqual(file_sha256(source), source_hash)
            self.assertNotIn("pdf_complete", saved.get("completion_checks", {}))
            self.run_registration(root, path, *flags)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), saved)

    def test_stale_citation_snapshot_fails_without_mutating_manifest(self):
        with tempfile.TemporaryDirectory(prefix="fraud_report_pin_") as directory:
            root = Path(directory)
            path, _, citation_path, _, _, _, source = self.fixture(root)
            before = path.read_bytes()
            source.write_text("Changed after the finite claim review.\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.run_registration(root, path, "--register-citation-claims", citation_path)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_suffix(".json.lock").exists())

    def test_missing_duplicate_and_escaping_citation_sources_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fraud_report_pin_") as directory:
            root = Path(directory)
            _, _, _, citation, _, _, _ = self.fixture(root)
            variants = []
            missing = copy.deepcopy(citation)
            missing["source_snapshot"]["files"] = []
            variants.append(missing)
            repeated = copy.deepcopy(citation)
            repeated["source_snapshot"]["files"] *= 2
            variants.append(repeated)
            escaped = copy.deepcopy(citation)
            escaped["source_snapshot"]["files"][0]["path"] = "../outside.tex"
            variants.append(escaped)
            for item in variants:
                with self.subTest(report=item), patch("revision_audit.ROOT", root):
                    with self.assertRaises(ValueError):
                        audit.validate_citation_claim_review(item)

    def test_report_outside_isolated_revision_root_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fraud_report_pin_") as directory:
            root = Path(directory)
            path, _, _, citation, _, _, _ = self.fixture(root)
            outside = root / "outside_citation_review.json"
            outside.write_text(json.dumps(citation), encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "inside the revision root"):
                self.run_registration(root, path, "--register-citation-claims", outside)
            self.assertEqual(path.read_bytes(), before)

    def test_corrected_incomplete_or_fitted_diagnostic_cannot_be_registered_as_historical(self):
        with tempfile.TemporaryDirectory(prefix="fraud_report_pin_") as directory:
            root = Path(directory)
            path, manifest, _, _, _, historical, _ = self.fixture(root)
            for field, value in (("role", "corrected_primary_diagnostic"),
                                 ("protocol_version", "thesis_revision_20261005_v1"),
                                 ("status", "auditing"), ("fit_or_resampling_performed", True),
                                 ("threshold_selection_performed", True), ("test_labels_used_for_fitting", True)):
                report = copy.deepcopy(historical)
                report[field] = value
                with self.subTest(field=field), patch("revision_audit.ROOT", root):
                    with self.assertRaisesRegex(ValueError, "original-protocol"):
                        audit.validate_historical_lgbm_diagnostics(report, manifest, path)

    def test_missing_strategy_changed_pin_and_wrong_raw_split_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fraud_report_pin_") as directory:
            root = Path(directory)
            path, manifest, _, _, _, historical, _ = self.fixture(root)
            cases = []
            incomplete = copy.deepcopy(historical)
            del incomplete["strategies"]["weights"]
            cases.append((incomplete, manifest))
            changed = copy.deepcopy(historical)
            changed["strategies"]["smote"]["source_config_sha256"] = "changed"
            cases.append((changed, manifest))
            wrong_source = copy.deepcopy(historical)
            wrong_source["strategies"]["ros"]["source_run"] = "results/another_run"
            cases.append((wrong_source, manifest))
            missing_pin = copy.deepcopy(manifest)
            del missing_pin["historical_runs"]["ulb_2013/lgbm/rus"]
            cases.append((historical, missing_pin))
            wrong_split = copy.deepcopy(historical)
            wrong_split["split_audit"]["test_rows"] = 56746
            cases.append((wrong_split, manifest))
            for report, pins in cases:
                with self.subTest(report=report), patch("revision_audit.ROOT", root):
                    with self.assertRaises(ValueError):
                        audit.validate_historical_lgbm_diagnostics(report, pins, path)

    def test_corrected_config_and_changed_parameter_sources_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fraud_report_pin_") as directory:
            root = Path(directory)
            path, manifest, _, _, _, historical, _ = self.fixture(root)
            key = "ulb_2013/lgbm/smote"
            config_path = Path(manifest["historical_runs"][key]["run_dir"]) / "config.json"
            original = json.loads(config_path.read_text(encoding="utf-8"))
            for field, value in (("protocol_version", "thesis_revision_20261005_v1"),
                                 ("best_params", {"classifier__num_leaves": 6})):
                config = {**original, field: value}
                config_path.write_text(json.dumps(config), encoding="utf-8")
                manifest["historical_runs"][key]["config_sha256"] = file_sha256(config_path)
                historical["strategies"]["smote"]["source_config_sha256"] = file_sha256(config_path)
                with self.subTest(field=field), patch("revision_audit.ROOT", root):
                    with self.assertRaises(ValueError):
                        audit.validate_historical_lgbm_diagnostics(historical, manifest, path)

    def test_invalid_historical_report_cannot_partially_register_a_valid_citation(self):
        with tempfile.TemporaryDirectory(prefix="fraud_report_pin_") as directory:
            root = Path(directory)
            path, _, citation_path, _, historical_path, historical, _ = self.fixture(root)
            historical["status"] = "auditing"
            historical_path.write_text(json.dumps(historical), encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "original-protocol"):
                self.run_registration(root, path, "--register-citation-claims", citation_path,
                                      "--register-historical-lgbm-diagnostics", historical_path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
