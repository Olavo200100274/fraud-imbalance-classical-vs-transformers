from pathlib import Path
from contextlib import ExitStack
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiment_protocol import file_sha256
from revision_thesis_assets import (extract_tabular, parameter_value, replace_labelled_tabular,
                                   table_map, verify_source_record, verify_transfer_record, verify_shap_records,
                                   verify_paired_record, verify_bootstrap_compatibility_record, verify_generation,
                                   normalise_output_hashes)


class ThesisAssetTests(unittest.TestCase):
    def test_table_map_has_complete_non_shap_grid(self):
        self.assertEqual(len(table_map()), 27)
        self.assertNotIn("tab:shap_consistency", table_map())
        self.assertEqual(len(set(table_map().values())), 27)

    def test_generated_caption_is_not_transferred(self):
        source = r"\caption{Old caption}\label{tab:x}\begin{tabular}{c}old\end{tabular}tail"
        generated = r"\caption{Different}\begin{tabular}{r}new\end{tabular}"
        revised = replace_labelled_tabular(source, "tab:x", extract_tabular(generated))
        self.assertIn(r"\caption{Old caption}", revised)
        self.assertNotIn("Different", revised)
        self.assertTrue(revised.endswith("tail"))

    def test_duplicate_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one explicit"):
            replace_labelled_tabular(r"\label{tab:x}\label{tab:x}", "tab:x", r"\begin{tabular}{c}\end{tabular}")

    def test_intervening_document_unit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "intervenes"):
            replace_labelled_tabular(r"\label{tab:x}\section{Next}\begin{tabular}{c}\end{tabular}",
                                     "tab:x", r"\begin{tabular}{c}\end{tabular}")

    def test_multiple_generated_tables_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            extract_tabular(r"\begin{tabular}{c}\end{tabular}\begin{tabular}{c}\end{tabular}")

    def test_next_table_cannot_supply_missing_tabular(self):
        for boundary in (r"\end{table}\begin{table}", r"\end{table*}\begin{table*}"):
            with self.subTest(boundary=boundary), self.assertRaisesRegex(ValueError, "intervenes"):
                replace_labelled_tabular(r"\label{tab:x}" + boundary + r"\begin{tabular}{c}\end{tabular}",
                                         "tab:x", r"\begin{tabular}{r}\end{tabular}")

    def test_parameter_values_keep_numeric_resolution(self):
        self.assertEqual(parameter_value(176), "176")
        self.assertEqual(parameter_value(None), "None")
        self.assertAlmostEqual(float(parameter_value(0.04930723286708181)), 0.04930723286708181, places=9)
        with self.assertRaises(ValueError):
            parameter_value(True)

    def test_source_cv_and_pr_files_are_revalidated(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            record = {"run_dir": str(run)}
            fields = {"config.json": "config_sha256", "metrics_test.json": "metrics_sha256",
                      "metrics_cv.json": "metrics_cv_sha256", "pr_curve_data.json": "pr_curve_data_sha256",
                      "y_test.npy": "y_test_sha256", "y_test_scores.npy": "y_test_scores_sha256",
                      "test_row_indices.npy": "test_row_indices_sha256"}
            for filename, field in fields.items():
                (run / filename).write_bytes(b"synthetic immutable source")
                record[field] = file_sha256(run / filename)
            verify_source_record(record)
            (run / "metrics_cv.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after generation"):
                verify_source_record(record)

    def test_nullable_historical_files_must_be_absent(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            record = {"run_dir": str(run)}
            fields = {"config.json": "config_sha256", "metrics_test.json": "metrics_sha256",
                      "metrics_cv.json": "metrics_cv_sha256", "y_test.npy": "y_test_sha256",
                      "y_test_scores.npy": "y_test_scores_sha256"}
            for filename, field in fields.items():
                (run / filename).write_bytes(b"fixture")
                record[field] = file_sha256(run / filename)
            record.update(pr_curve_data_sha256=None, test_row_indices_sha256=None)
            verify_source_record(record)
            (run / "pr_curve_data.json").write_bytes(b"unrecorded insertion")
            with self.assertRaisesRegex(ValueError, "not explicitly hashed"):
                verify_source_record(record)

    def test_transfer_original_arrays_are_revalidated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audit = root / "source_audit.json"
            audit.write_bytes(b"fixture")
            record = {"path": str(audit), "sha256": file_sha256(audit), "variants": {}}
            arrays = ("y_test.npy", "y_test_scores.npy", "test_row_indices.npy",
                      "original_y_test.npy", "original_y_test_scores.npy", "original_test_row_indices.npy")
            for number in range(1, 6):
                name = f"baf_var{number}"
                cohort = root / name
                cohort.mkdir()
                metrics = cohort / "metrics_test.json"
                metrics.write_bytes(b"fixture metrics")
                variant = {"path": str(metrics), "sha256": file_sha256(metrics), "artefacts": {}}
                for filename in arrays:
                    target = cohort / filename
                    target.write_bytes(b"synthetic array")
                    variant["artefacts"][filename] = {"path": str(target), "sha256": file_sha256(target)}
                record["variants"][name] = variant
            verify_transfer_record(record)
            (root / "baf_var3/original_y_test_scores.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "transfer artefact changed"):
                verify_transfer_record(record)

    def test_shap_matrices_and_partition_manifest_are_revalidated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def evidence(target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"immutable synthetic evidence")
                return {"path": str(target), "sha256": file_sha256(target)}
            qa = {"compatible_shap_sources": {}}
            for model in ("logreg", "lgbm", "catboost"):
                run = root / model
                artefacts = {name: evidence(run / name) for name in
                             ("config.json", "model.joblib", "shap_values.npy", "shap_global.json",
                              "shap_local_cases.json", "y_test.npy", "y_test_scores.npy")}
                qa["compatible_shap_sources"][model] = {"run_dir": str(run), "artefacts": artefacts}
            stability = evidence(root / "stability/shap_variant_stability.json")
            stability["partition_manifest"] = evidence(root / "partitions/partition_manifest.json")
            stability["artefacts"] = {
                f"baf_var{number}/{name}": evidence(root / "stability" / f"baf_var{number}" / name)
                for number in range(1, 6) for name in ("shap_values.npy", "y_test.npy", "test_row_indices.npy")}
            qa["variant_shap_stability"] = stability
            verify_shap_records(qa)
            (root / "stability/baf_var5/shap_values.npy").write_bytes(b"changed matrix")
            with self.assertRaisesRegex(ValueError, "SHAP source artefact changed"):
                verify_shap_records(qa)

    def test_paired_handoff_verifies_report_manifest_and_nested_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path = root / "revision_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            sensitivity_path = root / "sensitivity/revision_manifest.json"
            sensitivity_path.parent.mkdir()
            sensitivity_path.write_text("{}", encoding="utf-8")
            paired_path = root / "derived/paired/paired_analysis.json"
            paired_path.parent.mkdir(parents=True)
            report = {"sensitivity_manifest_sha256": file_sha256(sensitivity_path)}
            paired_path.write_text(json.dumps(report), encoding="utf-8")
            record = {"path": str(paired_path), "sha256": file_sha256(paired_path),
                      "sensitivity_manifest": {"path": str(sensitivity_path), "sha256": file_sha256(sensitivity_path)}}
            with patch("revision_paired_analysis.verify_paired_report") as verify:
                verify_paired_record(record, manifest_path)
                verify.assert_called_once_with(report, {}, {})
            with patch("revision_paired_analysis.verify_paired_report", side_effect=ValueError("nested source changed")):
                with self.assertRaisesRegex(ValueError, "nested source changed"):
                    verify_paired_record(record, manifest_path)
            sensitivity_path.write_text('{"changed": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after generation"):
                verify_paired_record(record, manifest_path)

    def test_paired_handoff_cannot_substitute_another_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "revision_manifest.json"
            with self.assertRaisesRegex(ValueError, "explicit comparison sources"):
                verify_paired_record({"path": str(Path(folder) / "another.json")}, path)

    def test_bootstrap_handoff_requires_pinned_qa_and_a_fresh_sidecar_verification(self):
        key = "ulb_2013/fttransformer/none"
        with tempfile.TemporaryDirectory(prefix="fraud_bootstrap_handoff_") as folder:
            root = Path(folder)
            run = root / "run"
            run.mkdir()
            (run / "config.json").write_text("{}", encoding="utf-8")
            reference = {"path": str(root / "compatibility.json"), "sha256": "verified report digest"}
            manifest = root / "revision_manifest.json"
            manifest.write_text(json.dumps({"bootstrap_compatibility": {key: reference}}), encoding="utf-8")
            qa = {"source_runs": {key: {"run_dir": str(run)}}, "bootstrap_compatibility": {key: reference}}
            verify = Mock(return_value=reference)
            module = SimpleNamespace(KEY=key, verify_bootstrap_compatibility=verify)
            with patch.dict("sys.modules", {"revision_bootstrap_compatibility": module}):
                verify_bootstrap_compatibility_record(qa, manifest)
                verify.assert_called_once_with({"bootstrap_compatibility": {key: reference}}, run.resolve(), {},
                                               expected_iterations=1000)
                with self.assertRaisesRegex(ValueError, "lacks the exact"):
                    verify_bootstrap_compatibility_record({"source_runs": qa["source_runs"]}, manifest)
                with self.assertRaisesRegex(ValueError, "differs from the current"):
                    verify_bootstrap_compatibility_record({"source_runs": qa["source_runs"],
                        "bootstrap_compatibility": {key: {**reference, "sha256": "changed QA digest"}}}, manifest)
                verify.return_value = {**reference, "sha256": "changed verified digest"}
                with self.assertRaisesRegex(ValueError, "changed after generation"):
                    verify_bootstrap_compatibility_record(qa, manifest)
                verify.side_effect = ValueError("sidecar bytes changed")
                with self.assertRaisesRegex(ValueError, "sidecar bytes changed"):
                    verify_bootstrap_compatibility_record(qa, manifest)

    def test_explicit_bootstrap_budget_cannot_be_hidden_by_a_compatibility_record(self):
        key = "ulb_2013/fttransformer/none"
        with tempfile.TemporaryDirectory(prefix="fraud_bootstrap_handoff_") as folder:
            root = Path(folder)
            verify = Mock(side_effect=AssertionError("No fallback for explicit metadata"))
            module = SimpleNamespace(KEY=key, verify_bootstrap_compatibility=verify)
            qa = {"source_runs": {key: {"run_dir": str(root)}}}
            with patch.dict("sys.modules", {"revision_bootstrap_compatibility": module}):
                for value in (None, 0, 999):
                    (root / "config.json").write_text(json.dumps({"bootstrap_iterations": value}), encoding="utf-8")
                    with self.subTest(value=value), self.assertRaisesRegex(ValueError, "explicitly incompatible"):
                        verify_bootstrap_compatibility_record(qa, root / "unused_manifest.json")
                (root / "config.json").write_text(json.dumps({"bootstrap_iterations": 1000}), encoding="utf-8")
                verify_bootstrap_compatibility_record(qa, root / "unused_manifest.json")
                with self.assertRaisesRegex(ValueError, "must not use"):
                    verify_bootstrap_compatibility_record({**qa, "bootstrap_compatibility": {key: {}}},
                                                        root / "unused_manifest.json")
                verify.assert_not_called()

    def test_full_generation_handoff_invokes_the_sidecar_guard_before_sources(self):
        with tempfile.TemporaryDirectory(prefix="fraud_bootstrap_handoff_") as folder:
            root = Path(folder)
            manifest = root / "revision_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            qa = {"required_runs": 72, "manifest_path": str(manifest), "manifest_sha256": file_sha256(manifest),
                  "source_runs": {str(i): {} for i in range(72)},
                  "threshold_studies": {str(i): {} for i in range(12)},
                  "transfer_sources": {str(i): {} for i in range(6)}}
            (root / "generation_qa.json").write_text(json.dumps(qa), encoding="utf-8")
            with patch("revision_thesis_assets.verify_bootstrap_compatibility_record",
                       side_effect=ValueError("missing sidecar QA")) as verify:
                with patch("revision_thesis_assets.verify_source_record") as sources:
                    with self.assertRaisesRegex(ValueError, "missing sidecar QA"):
                        verify_generation(root, manifest)
                    verify.assert_called_once_with(qa, manifest)
                    sources.assert_not_called()


class GenerationOutputPathTests(unittest.TestCase):
    @staticmethod
    def fixture(root, separator="/"):
        manifest = root / "revision_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        derived = root / "reporting"
        derived.mkdir()
        threshold = root / "threshold.json"
        threshold.write_text("{}", encoding="utf-8")
        qa = {"required_runs": 72, "manifest_path": str(manifest),
              "manifest_sha256": file_sha256(manifest),
              "source_runs": {str(i): {} for i in range(72)},
              "threshold_studies": {str(i): {"path": str(threshold), "sha256": file_sha256(threshold)}
                                    for i in range(12)},
              "transfer_sources": {str(i): {} for i in range(6)}, "output_sha256": {}}
        for name in (*table_map().values(), "figures/ulb/pr_curve.pdf"):
            path = derived / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic generated output")
            qa["output_sha256"][name.replace("/", separator)] = file_sha256(path)
        (derived / "generation_qa.json").write_text(json.dumps(qa), encoding="utf-8")
        return derived, manifest, qa

    @staticmethod
    def verify_fixture(derived, manifest):
        # Isolate output compatibility; existing tests exercise these source guards.
        with ExitStack() as patches:
            for name in ("verify_bootstrap_compatibility_record", "verify_source_record",
                         "verify_transfer_record", "verify_shap_records", "verify_paired_record"):
                patches.enter_context(patch(f"revision_thesis_assets.{name}"))
            return verify_generation(derived, manifest)

    @staticmethod
    def save_qa(derived, qa):
        (derived / "generation_qa.json").write_text(json.dumps(qa), encoding="utf-8")

    def test_both_separator_styles_return_canonical_keys_and_preserve_saved_files(self):
        for separator in ("/", "\\"):
            with self.subTest(separator=separator), tempfile.TemporaryDirectory(prefix="fraud_output_paths_") as folder:
                root = Path(folder)
                derived, manifest, original_qa = self.fixture(root, separator)
                previous = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
                verified = self.verify_fixture(derived, manifest)
                expected = {name.replace("\\", "/"): digest
                            for name, digest in original_qa["output_sha256"].items()}
                self.assertEqual(verified["output_sha256"], expected)
                self.assertIn("figures/ulb/pr_curve.pdf", verified["output_sha256"])
                self.assertEqual(normalise_output_hashes(original_qa["output_sha256"]), expected)
                self.assertEqual(json.loads((derived / "generation_qa.json").read_text(encoding="utf-8")), original_qa)
                self.assertEqual({path: path.read_bytes() for path in root.rglob("*") if path.is_file()}, previous)

    def test_missing_required_table_key_is_rejected_even_when_file_exists(self):
        with tempfile.TemporaryDirectory(prefix="fraud_output_paths_") as folder:
            derived, manifest, qa = self.fixture(Path(folder), "\\")
            del qa["output_sha256"]["tables\\ulb\\baseline.tex"]
            self.save_qa(derived, qa)
            with self.assertRaisesRegex(ValueError, "required generated table is missing"):
                self.verify_fixture(derived, manifest)

    def test_empty_output_hash_map_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fraud_output_paths_") as folder:
            derived, manifest, qa = self.fixture(Path(folder))
            qa["output_sha256"] = {}
            self.save_qa(derived, qa)
            with self.assertRaisesRegex(ValueError, "identify every generated output"):
                self.verify_fixture(derived, manifest)

    def test_missing_output_file_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fraud_output_paths_") as folder:
            derived, manifest, qa = self.fixture(Path(folder))
            qa["output_sha256"]["figures/ulb/missing.pdf"] = "missing file digest"
            self.save_qa(derived, qa)
            with self.assertRaises(FileNotFoundError):
                self.verify_fixture(derived, manifest)

    def test_changed_output_hash_is_rejected_for_both_separator_styles(self):
        for separator in ("/", "\\"):
            with self.subTest(separator=separator), tempfile.TemporaryDirectory(prefix="fraud_output_paths_") as folder:
                derived, manifest, unused = self.fixture(Path(folder), separator)
                (derived / "tables/ulb/baseline.tex").write_bytes(b"changed after QA")
                with self.assertRaisesRegex(ValueError, "paths or hashes are invalid"):
                    self.verify_fixture(derived, manifest)

    def test_traversal_absolute_and_drive_paths_are_rejected(self):
        invalid = ("../outside.tex", "..\\outside.tex", "tables/../../outside.tex",
                   "tables\\..\\..\\outside.tex", "/absolute.tex", "\\rooted.tex",
                   "C:/absolute.tex", "C:\\absolute.tex", "C:drive_relative.tex",
                   "./C:/absolute.tex", ".\\C:drive_relative.tex",
                   "\\\\server\\share\\output.tex", "//server/share/output.tex")
        with tempfile.TemporaryDirectory(prefix="fraud_output_paths_") as folder:
            derived, manifest, qa = self.fixture(Path(folder))
            original_outputs = qa["output_sha256"]
            for name in (*invalid, str(derived / "tables/ulb/baseline.tex")):
                with self.subTest(name=name):
                    qa["output_sha256"] = {**original_outputs, name: "invalid path digest"}
                    self.save_qa(derived, qa)
                    with self.assertRaisesRegex(ValueError, "must be relative"):
                        self.verify_fixture(derived, manifest)

    def test_normalised_alias_collisions_are_rejected_even_with_matching_hashes(self):
        aliases = ("tables\\ulb\\baseline.tex", "./tables/ulb/baseline.tex", "tables//ulb/baseline.tex")
        with tempfile.TemporaryDirectory(prefix="fraud_output_paths_") as folder:
            derived, manifest, qa = self.fixture(Path(folder))
            original_outputs = qa["output_sha256"]
            digest = original_outputs["tables/ulb/baseline.tex"]
            for alias in aliases:
                with self.subTest(alias=alias):
                    qa["output_sha256"] = {**original_outputs, alias: digest}
                    self.save_qa(derived, qa)
                    with self.assertRaisesRegex(ValueError, "aliases collide"):
                        self.verify_fixture(derived, manifest)

    def test_resolved_output_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fraud_output_paths_") as folder:
            derived, manifest, qa = self.fixture(Path(folder))
            baseline = derived / "tables/ulb/baseline.tex"
            qa["output_sha256"]["tables/ulb/alias.tex"] = file_sha256(baseline)
            self.save_qa(derived, qa)
            resolve = Path.resolve
            def alias_resolution(path, *args, **kwargs):
                if path == derived / "tables/ulb/alias.tex":
                    return resolve(baseline, *args, **kwargs)
                return resolve(path, *args, **kwargs)
            with patch.object(Path, "resolve", alias_resolution):
                with self.assertRaisesRegex(ValueError, "aliases collide"):
                    self.verify_fixture(derived, manifest)

    def test_relative_output_resolving_outside_the_root_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fraud_output_paths_") as folder:
            root = Path(folder)
            derived, manifest, unused = self.fixture(root, "\\")
            baseline = derived / "tables/ulb/baseline.tex"
            resolve = Path.resolve
            def escaped_resolution(path, *args, **kwargs):
                if path == baseline:
                    return resolve(root / "outside.tex", *args, **kwargs)
                return resolve(path, *args, **kwargs)
            with patch.object(Path, "resolve", escaped_resolution):
                with self.assertRaisesRegex(ValueError, "paths or hashes are invalid"):
                    self.verify_fixture(derived, manifest)


if __name__ == "__main__":
    unittest.main()
