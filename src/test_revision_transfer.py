"""Small, isolated checks for the corrected frozen-transfer protocol."""

import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import revision_transfer as transfer
from experiment_protocol import array_sha256, file_sha256


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ExactOverlapTests(unittest.TestCase):
    def test_every_candidate_pair_is_checked_including_duplicate_base_profiles(self):
        base = pd.DataFrame({"number": [1, 1, 2], "category": ["A", "A", "B"]})
        target = pd.DataFrame({"number": [1, 2, 3], "category": ["A", "B", "C"]})
        excluded, pairs, audit = transfer.exact_overlap(base, target)
        np.testing.assert_array_equal(excluded, [True, True, False])
        self.assertEqual({tuple(row) for row in pairs.tolist()}, {(0, 0), (0, 1), (1, 2)})
        self.assertEqual(audit["elementwise_verified_pairs"], 3)
        self.assertFalse(audit["target_labels_used_for_exclusion"])

    def test_hash_collision_is_not_treated_as_equality(self):
        base = pd.DataFrame({"number": [1, 2], "category": ["A", "B"]})
        target = pd.DataFrame({"number": [1, 7], "category": ["A", "Z"]})
        excluded, pairs, audit = transfer.exact_overlap(
            base, target, base_hashes=np.zeros(2, dtype=np.uint64),
            target_hashes=np.zeros(2, dtype=np.uint64),
        )
        np.testing.assert_array_equal(excluded, [True, False])
        np.testing.assert_array_equal(pairs, [[0, 0]])
        self.assertEqual(audit["hash_candidate_pairs"], 4)
        self.assertEqual(audit["rejected_hash_candidate_pairs"], 3)

    def test_equal_numeric_values_and_signed_zero_hash_consistently(self):
        base = pd.DataFrame({"number": [0, 1], "category": ["A", "B"]})
        target = pd.DataFrame({"number": [-0.0, 1.0], "category": ["A", "B"]})
        excluded, _, _ = transfer.exact_overlap(base, target)
        np.testing.assert_array_equal(excluded, [True, True])

    def test_missing_predictors_are_rejected(self):
        frame = pd.DataFrame({"number": [np.nan]})
        with self.assertRaisesRegex(ValueError, "Missing values"):
            transfer.exact_overlap(frame, frame)

    def test_large_integer_does_not_get_silently_rounded(self):
        frame = pd.DataFrame({"number": [2**53 + 3]})
        with self.assertRaisesRegex(ValueError, "exact float64"):
            transfer.predictor_hashes(frame)


class PartitionTests(unittest.TestCase):
    @staticmethod
    def _loader(name, return_metadata=False):
        columns = [f"f{column:02d}" for column in range(30)]
        dev = pd.DataFrame(np.arange(30)[None, :] + np.arange(4)[:, None] * 100,
                           columns=columns, index=np.arange(10, 14))
        if name == "baf_base":
            test = pd.DataFrame(np.arange(30)[None, :] + np.arange(4, 8)[:, None] * 100,
                                columns=columns, index=np.arange(14, 18))
            y_test = pd.Series([0, 1, 0, 1], index=test.index)
        else:
            test = pd.DataFrame(np.arange(30)[None, :] + np.array([0, 1, 5, 6, 7, 8])[:, None] * 100,
                                columns=columns, index=np.arange(20, 26))
            # One overlapping label deliberately disagrees. Both rows must
            # still be excluded because labels do not define profile overlap.
            y_test = pd.Series([1, 1, 0, 1, 0, 1], index=test.index)
        y_dev = pd.Series([0, 1, 0, 1], index=dev.index)
        metadata = {"dev_indices": dev.index.to_numpy(), "test_indices": test.index.to_numpy(),
                    "raw_file_sha256": name + "_raw_hash", "feature_columns": columns,
                    "protocol_version": transfer.PROTOCOL_VERSION, "split_seed": 42}
        result = (dev, test, y_dev, y_test, metadata)
        return result if return_metadata else result[:4]

    def test_prepare_without_any_model_loading_and_preserve_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            _write_json(manifest, {})
            with patch.object(transfer.joblib, "load", side_effect=AssertionError("No model loading")):
                destination = transfer.prepare_partitions(root / "transfer", manifest, loader=self._loader)
            document = transfer.load_partition_manifest(destination)
            self.assertEqual(document["status"], "complete")
            for variant in transfer.VARIANTS:
                directory = destination / variant
                audit = json.loads((directory / "partition_audit.json").read_text(encoding="utf-8"))
                self.assertEqual(audit["original_population"]["rows"], 6)
                self.assertEqual(audit["kept_population"]["rows"], 4)
                self.assertEqual(audit["excluded_population"]["rows"], 2)
                self.assertEqual(audit["matched_pair_label_disagreements"], 1)
                np.testing.assert_array_equal(np.load(directory / "kept_test_row_indices.npy"), [22, 23, 24, 25])
                np.testing.assert_array_equal(np.load(directory / "excluded_test_row_indices.npy"), [20, 21])
                np.testing.assert_array_equal(np.load(directory / "kept_test_y.npy"), [0, 1, 0, 1])
            # Corruption is detected, not silently accepted as prepared data.
            with (destination / "baf_var1" / "kept_test_positions.npy").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "changed"):
                transfer.load_partition_manifest(destination)

    def test_no_preserved_output_directory_is_allowed(self):
        for name in ("results", "results_thesis", "Overleaf", "Article 1", "Article 2"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                transfer.validate_output_root(transfer.PROJECT_ROOT / name / "new")

    def test_target_and_variant_only_features_are_rejected(self):
        frame = pd.DataFrame(np.zeros((2, 30)), columns=[f"f{column}" for column in range(30)])
        for name in transfer.FORBIDDEN_PREDICTORS:
            with self.subTest(name=name), self.assertRaises(ValueError):
                transfer.validate_schema(frame.rename(columns={"f0": name}))


class SourceSelectionTests(unittest.TestCase):
    @staticmethod
    def _run(root, model, *, corrected=False):
        run = root / ("results_revision" if corrected else "results") / model / "run"
        config = {"dataset": "baf_base", "model": model, "strategy": "none",
                  "dataset_hash_sha256": "raw_hash", "protocol_version": transfer.PROTOCOL_VERSION}
        _write_json(run / "config.json", config)
        _write_json(run / "metrics_test.json", {"threshold": 0.5})
        for name in ("model.joblib", "model.pt", "preprocessors.joblib",
                     "test_row_indices.npy", "y_test.npy", "y_test_scores.npy"):
            (run / name).write_bytes(b"test fixture")
        _write_json(run / "completed.json", {"status": "complete",
                                             "protocol_version": transfer.PROTOCOL_VERSION})
        return run

    def test_corrected_ft_pin_is_required_even_when_historical_ft_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical = self._run(root, "fttransformer")
            manifest = root / "manifest.json"
            _write_json(manifest, {"historical_baseline_runs": {"baf_base": {"fttransformer": str(historical)}}})
            with patch.object(transfer, "PROJECT_ROOT", root), self.assertRaisesRegex(FileNotFoundError, "fallback"):
                transfer.resolve_transfer_source(manifest, "fttransformer")

    def test_flat_ft_pin_and_its_config_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._run(root, "fttransformer", corrected=True)
            manifest = root / "manifest.json"
            value = {"baseline_runs": {"baf_base/fttransformer": {
                "run_dir": str(run), "config_sha256": file_sha256(run / "config.json")}}}
            _write_json(manifest, value)
            with patch.object(transfer, "PROJECT_ROOT", root):
                selected, _ = transfer.resolve_transfer_source(manifest, "fttransformer")
                self.assertEqual(selected, run)
                value["baseline_runs"]["baf_base/fttransformer"]["config_sha256"] = "incorrect"
                manifest.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "config hash"):
                    transfer.resolve_transfer_source(manifest, "fttransformer")

    def test_corrected_ft_pin_requires_config_hash_and_completed_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._run(root, "fttransformer", corrected=True)
            manifest = root / "manifest.json"
            with patch.object(transfer, "PROJECT_ROOT", root):
                for reference in (str(run), {"run_dir": str(run)}):
                    _write_json(manifest, {"baseline_runs": {"baf_base/fttransformer": reference}})
                    with self.assertRaisesRegex(ValueError, "SHA-256 pin"):
                        transfer.resolve_transfer_source(manifest, "fttransformer")
                _write_json(manifest, {"baseline_runs": {"baf_base/fttransformer": {
                    "run_dir": str(run), "config_sha256": file_sha256(run / "config.json")}}})
                for completion in ({"status": "in_progress", "protocol_version": transfer.PROTOCOL_VERSION},
                                   {"status": "complete", "protocol_version": "historical"}):
                    _write_json(run / "completed.json", completion)
                    with self.assertRaisesRegex(ValueError, "completion marker"):
                        transfer.resolve_transfer_source(manifest, "fttransformer")

    def test_corrected_classical_pin_precedes_historical_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical = self._run(root, "logreg")
            corrected = self._run(root, "logreg", corrected=True)
            config = json.loads((corrected / "config.json").read_text(encoding="utf-8"))
            config["threshold_exact"] = 0.123456789
            _write_json(corrected / "config.json", config)
            manifest = root / "manifest.json"
            _write_json(manifest, {
                "historical_baseline_runs": {"baf_base": {"logreg": str(historical)}},
                "baseline_runs": {"baf_base/logreg": {"run_dir": str(corrected),
                                                       "config_sha256": file_sha256(corrected / "config.json")}},
            })
            with patch.object(transfer, "PROJECT_ROOT", root):
                selected, selected_config = transfer.resolve_transfer_source(manifest, "logreg")
                self.assertEqual(selected, corrected)
                self.assertEqual(selected_config["threshold_exact"], 0.123456789)
                _write_json(corrected / "completed.json", {"status": "in_progress"})
                with self.assertRaisesRegex(ValueError, "completion marker"):
                    transfer.resolve_transfer_source(manifest, "logreg")

    def test_intervention_is_not_accepted_as_a_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._run(root, "logreg")
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            config["strategy"] = "smote"
            _write_json(run / "config.json", config)
            manifest = root / "manifest.json"
            _write_json(manifest, {"historical_baseline_runs": {"baf_base": {"logreg": str(run)}}})
            with self.assertRaisesRegex(ValueError, "baseline"):
                transfer.resolve_transfer_source(manifest, "logreg")


class FrozenPredictionTests(unittest.TestCase):
    def test_existing_explicit_output_is_rejected_before_partition_or_data_reading(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "partial"
            destination.mkdir()
            with patch.object(transfer, "load_partition_manifest", side_effect=AssertionError("No partition reading")), \
                    patch.object(transfer, "load_dataset", side_effect=AssertionError("No data reading")), \
                    self.assertRaisesRegex(FileExistsError, "already exist"):
                transfer.evaluate_transfer("not_read", ["logreg"], output_dir=destination)

    def test_cli_passes_explicit_new_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "new_output"
            with patch.object(sys, "argv", ["transfer", "--models", "logreg", "--partitions-dir", "prepared",
                                             "--output-dir", str(destination)]), \
                    patch.object(transfer, "resolve_transfer_source"), \
                    patch.object(transfer, "exclusive_resource"), \
                    patch.object(transfer, "evaluate_transfer") as evaluation:
                transfer.main()
                self.assertEqual(evaluation.call_args.kwargs["output_dir"], destination)

    def test_small_end_to_end_evaluation_saves_aligned_full_scores_without_refitting(self):
        class PredictOnlyModel:
            def predict_proba(self, frame):
                scores = frame["f00"].to_numpy() / 1000
                return np.column_stack((1 - scores, scores))

            def fit(self, *args, **kwargs):
                raise AssertionError("Frozen transfer must not fit")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = SourceSelectionTests._run(root, "logreg")
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            config["dataset_hash_sha256"] = "baf_base_raw_hash"
            _write_json(run / "config.json", config)
            _, base_test, _, labels, _ = PartitionTests._loader("baf_base", return_metadata=True)
            np.save(run / "y_test.npy", labels.to_numpy())
            np.save(run / "y_test_scores.npy", PredictOnlyModel().predict_proba(base_test)[:, 1])
            np.save(run / "test_row_indices.npy", base_test.index.to_numpy())
            manifest = root / "manifest.json"
            _write_json(manifest, {"historical_baseline_runs": {"baf_base": {"logreg": str(run)}}})
            partitions = transfer.prepare_partitions(root / "transfer", manifest, loader=PartitionTests._loader)
            with patch.object(transfer, "PROJECT_ROOT", root), \
                    patch.object(transfer, "load_dataset", side_effect=PartitionTests._loader), \
                    patch.object(transfer.joblib, "load", return_value=PredictOnlyModel()):
                destination = transfer.evaluate_transfer(partitions, ["logreg"], manifest_path=manifest,
                                                          output_root=root / "transfer")
                explicit = transfer.evaluate_transfer(partitions, ["logreg"], manifest_path=manifest,
                                                       output_dir=root / "explicit_output")
                self.assertEqual(explicit, root / "explicit_output")
                self.assertEqual(json.loads((explicit / "transfer_manifest.json").read_text(encoding="utf-8"))["status"],
                                 "complete")
            summary = json.loads((destination / "transfer_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete")
            for variant in transfer.VARIANTS:
                directory = destination / "logreg" / variant
                np.testing.assert_array_equal(np.load(directory / "test_row_indices.npy"), [22, 23, 24, 25])
                np.testing.assert_array_equal(np.load(directory / "y_test.npy"), [0, 1, 0, 1])
                np.testing.assert_array_equal(np.load(directory / "y_test_scores.npy"), [0.5, 0.6, 0.7, 0.8])
                record = json.loads((directory / "metrics_test.json").read_text(encoding="utf-8"))
                self.assertEqual(record["threshold_used"], 0.5)
                self.assertEqual(record["original_population_same_frozen_model"]["role"],
                                 "newly_scored_original_population_not_a_retuned_or_historical_run")

    def test_base_validation_rejects_changed_scores_before_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            labels = np.array([0, 1, 0, 1])
            scores = np.array([0.1, 0.9, 0.2, 0.8])
            np.save(run / "y_test.npy", labels)
            np.save(run / "y_test_scores.npy", scores)
            metadata = {"raw_file_sha256": "raw_hash", "test_indices": np.arange(4)}
            scorer = transfer.FrozenScorer("logreg", run, {"dataset_hash_sha256": "raw_hash"},
                                          0.5, "test threshold", lambda frame: scores.copy(), {})
            frame = pd.DataFrame({"number": np.arange(4)})
            result = transfer.validate_base_predictions(scorer, frame, labels, metadata)
            self.assertTrue(result["scores_bitwise_equal"])
            scorer.predictor = lambda frame: scores + 0.001
            with self.assertRaisesRegex(ValueError, "reproduce"):
                transfer.validate_base_predictions(scorer, frame, labels, metadata)


class VerifiedPartialRecoveryTests(unittest.TestCase):
    """Temporary synthetic artefacts only; no fitting or real inference."""

    class PredictOnly:
        def predict_proba(self, frame):
            scores = frame["f00"].to_numpy() / 1000
            return np.column_stack((1 - scores, scores))

        def fit(self, *args, **kwargs):
            raise AssertionError("Recovery must never fit")

    @staticmethod
    def intervals(*args, **kwargs):
        return {key: [0., 1.] for key in ("PR-AUC_ci", "ROC-AUC_ci", "F2_ci")}

    def fixture(self, root, *, approve=False):
        manifest = root / "revision_manifest.json"
        dev, base, y_dev, labels, metadata = PartitionTests._loader("baf_base", return_metadata=True)
        document = {"baseline_runs": {}, "protocol_version": transfer.PROTOCOL_VERSION}
        for model in transfer.MODEL_ORDER:
            run = SourceSelectionTests._run(root, model, corrected=True)
            config = json.loads((run / "config.json").read_text())
            config.update(threshold_exact=.5, train_samples=len(dev), test_samples=len(base), missing_policy="preserve",
                          dataset_hash_sha256=metadata["raw_file_sha256"], data_provenance={
                              "raw_file_sha256": metadata["raw_file_sha256"], "feature_columns": list(base.columns),
                              "dev_indices_sha256": array_sha256(metadata["dev_indices"]),
                              "test_indices_sha256": array_sha256(metadata["test_indices"])})
            _write_json(run / "config.json", config)
            scores = self.PredictOnly().predict_proba(base)[:, 1]
            np.save(run / "y_test.npy", labels.to_numpy())
            np.save(run / "y_test_scores.npy", scores)
            np.save(run / "test_row_indices.npy", metadata["test_indices"])
            _write_json(run / "metrics_test.json", transfer.compute_all_metrics(labels, scores, .5))
            document["baseline_runs"][f"baf_base/{model}"] = {
                "run_dir": str(run), "config_sha256": file_sha256(run / "config.json")}
        _write_json(manifest, document)
        with patch("builtins.print"):
            partitions = transfer.prepare_partitions(root / "transfer", manifest, loader=PartitionTests._loader)
        document["transfer_partitions"] = {"path": str(partitions / "partition_manifest.json"),
                                            "sha256": file_sha256(partitions / "partition_manifest.json")}
        _write_json(manifest, document)
        destination = root / "derived/transfer"
        with patch.object(transfer, "PROJECT_ROOT", root), \
                patch.object(transfer, "load_dataset", side_effect=PartitionTests._loader), \
                patch.object(transfer.joblib, "load", return_value=self.PredictOnly()), \
                patch.object(transfer, "bootstrap_ci", side_effect=self.intervals), patch("builtins.print"):
            transfer.evaluate_transfer(partitions, list(transfer.PARTIAL_MODELS), manifest_path=manifest,
                                       output_dir=destination, bootstrap_iterations=1000)
        # Simulate the known failure before the global six-family marker.
        (destination / "transfer_manifest.json").unlink()
        log = root / "logs/analysis_transfer_six_models_20261006_235034_037097.log"
        log.parent.mkdir()
        log.write_text("Frozen Base TEST predictions do not reproduce the source's saved score arrays.", encoding="utf-8")
        state = {"status": "failed", "current_step": "transfer_six_models", "last_child_exit_code": 1,
                 "failure": f"Analysis transfer_six_models failed; inspect {log}", "source_manifest": str(manifest),
                 "current_command": ["revision_transfer.py", "--manifest", str(manifest), "--models", "all",
                                     "--output-dir", str(destination), "--bootstrap-iterations", "1000", "--device", "cpu"]}
        _write_json(root / "analysis_state.json", state)
        if approve:
            with patch.object(transfer, "PROJECT_ROOT", root):
                verified = transfer.verify_partial_transfer(destination, manifest, partitions)
            state["explicit_transfer_recovery"] = {"verified_reuse": verified["reuse_provenance"]}
            state["status"] = "explicit_verified_transfer_recovery_authorised"
            _write_json(root / "analysis_state.json", state)
        return manifest, partitions, destination

    @staticmethod
    def hashes(directory):
        return {str(path.relative_to(directory)): file_sha256(path)
                for path in directory.rglob("*") if path.is_file()}

    @staticmethod
    def scorer(model, manifest_path, **kwargs):
        run, config, unused = kwargs["selected_source"]
        predictor = lambda frame: frame["f00"].to_numpy() / 1000
        return transfer.FrozenScorer(model, run, config, .5, "config.threshold_exact", predictor,
                                     {"config.json": file_sha256(run / "config.json")})

    def test_metadata_verifier_reads_complete_four_without_model_loading_or_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, partitions, directory = self.fixture(root)
            before = self.hashes(directory)
            with patch.object(transfer, "PROJECT_ROOT", root), \
                    patch.object(transfer.joblib, "load", side_effect=AssertionError("No model loading")), \
                    patch.object(transfer, "bootstrap_ci", side_effect=AssertionError("No bootstrap")):
                report = transfer.verify_partial_transfer(directory, manifest, partitions)
            self.assertEqual(set(report["models"]), set(transfer.PARTIAL_MODELS))
            self.assertEqual(sum(len(v["variant_metrics"]) for v in report["reuse_provenance"].values()), 20)
            self.assertEqual(before, self.hashes(directory))

    def test_changed_source_artefacts_are_rejected_without_fitting(self):
        names = ("config.json", "metrics_test.json", "completed.json", "y_test.npy",
                 "y_test_scores.npy", "test_row_indices.npy", "model.joblib")
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, partitions, directory = self.fixture(root)
                record = json.loads((directory / "logreg/source_audit.json").read_text())
                source = Path(record["source_run"]) / name
                source.write_bytes(source.read_bytes() + b"changed")
                with patch.object(transfer, "PROJECT_ROOT", root), self.assertRaises(ValueError):
                    transfer.verify_partial_transfer(directory, manifest, partitions)

    def test_changed_variant_array_or_metrics_hash_is_rejected(self):
        names = ("metrics_test.json", "y_test.npy", "y_test_scores.npy", "test_row_indices.npy",
                 "original_y_test.npy", "original_y_test_scores.npy", "original_test_row_indices.npy")
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, partitions, directory = self.fixture(root)
                path = directory / "rf/baf_var1" / name
                path.write_bytes(path.read_bytes() + b"changed")
                with patch.object(transfer, "PROJECT_ROOT", root), self.assertRaisesRegex(ValueError, "changed"):
                    transfer.verify_partial_transfer(directory, manifest, partitions)

    def test_wrong_source_and_unknown_or_incomplete_model_are_rejected(self):
        for mutation in ("source", "unknown", "missing", "ft_partial"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, partitions, directory = self.fixture(root)
                if mutation == "source":
                    path = directory / "catboost/source_audit.json"
                    record = json.loads(path.read_text())
                    record["source_run"] = str(root / "wrong")
                    _write_json(path, record)
                elif mutation == "missing":
                    (directory / "lgbm/baf_var3/metrics_test.json").unlink()
                else:
                    (directory / ("unknown" if mutation == "unknown" else "fttransformer")).mkdir()
                with patch.object(transfer, "PROJECT_ROOT", root), self.assertRaises(ValueError):
                    transfer.verify_partial_transfer(directory, manifest, partitions)

    def test_approved_pin_rejects_consistently_rehashed_changed_timer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, partitions, directory = self.fixture(root, approve=True)
            path = directory / "logreg/baf_var1/metrics_test.json"
            record = json.loads(path.read_text())
            record["evaluation_seconds"] += 1
            _write_json(path, record)
            audit = directory / "logreg/source_audit.json"
            source = json.loads(audit.read_text())
            source["variants"]["baf_var1"]["metrics_test_sha256"] = file_sha256(path)
            _write_json(audit, source)
            with patch.object(transfer, "PROJECT_ROOT", root), self.assertRaisesRegex(ValueError, "approved reuse"):
                transfer.verify_partial_transfer(directory, manifest, partitions)

    def test_wrong_bootstrap_budget_or_partition_pin_is_rejected(self):
        for mutation in ("budget", "partition"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, partitions, directory = self.fixture(root)
                path = root / "analysis_state.json" if mutation == "budget" else manifest
                document = json.loads(path.read_text())
                if mutation == "budget":
                    command = document["current_command"]
                    command[command.index("--bootstrap-iterations") + 1] = "999"
                else:
                    document["transfer_partitions"]["sha256"] = "wrong"
                _write_json(path, document)
                with patch.object(transfer, "PROJECT_ROOT", root), self.assertRaises(ValueError):
                    transfer.verify_partial_transfer(directory, manifest, partitions)

    def cli_arguments(self, manifest, partitions, directory):
        return ["transfer", "--manifest", str(manifest), "--models", "all", "--partitions-dir", str(partitions),
                "--output-dir", str(directory), "--resume-verified-partial", str(directory),
                "--device", "cuda", "--batch-size", "2048", "--bootstrap-iterations", "1000"]

    def test_cli_refuses_unavailable_cuda_before_csv_or_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, partitions, directory = self.fixture(root, approve=True)
            before = self.hashes(directory)
            with patch.object(sys, "argv", self.cli_arguments(manifest, partitions, directory)), \
                    patch("torch.cuda.is_available", return_value=False), \
                    patch.object(transfer, "load_dataset") as loading, \
                    patch.object(transfer, "exclusive_resource") as resource:
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    transfer.main()
            loading.assert_not_called()
            resource.assert_not_called()
            self.assertEqual(before, self.hashes(directory))

    def test_cli_requires_approval_and_original_batch(self):
        for mutation in ("approval", "batch"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, partitions, directory = self.fixture(root, approve=mutation == "batch")
                arguments = self.cli_arguments(manifest, partitions, directory)
                if mutation == "batch":
                    arguments[arguments.index("--batch-size") + 1] = "32"
                with patch.object(sys, "argv", arguments), patch("torch.cuda.is_available", return_value=True), \
                        patch.object(transfer, "load_dataset") as loading, \
                        patch.object(transfer, "exclusive_resource") as resource, self.assertRaises(ValueError):
                    transfer.main()
                loading.assert_not_called()
                resource.assert_not_called()

    def test_failed_ft_base_validation_never_changes_partial_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, partitions, directory = self.fixture(root, approve=True)
            before = self.hashes(directory)
            with patch.object(transfer, "PROJECT_ROOT", root), patch("torch.cuda.is_available", return_value=True), \
                    patch.object(transfer, "load_dataset", side_effect=PartitionTests._loader), \
                    patch.object(transfer, "load_frozen_scorer", side_effect=self.scorer), \
                    patch.object(transfer, "validate_base_predictions", side_effect=ValueError("Base failed")), \
                    patch.object(transfer, "bootstrap_ci", side_effect=AssertionError("No bootstrap")):
                with self.assertRaisesRegex(ValueError, "Base failed"):
                    transfer.evaluate_transfer(partitions, list(transfer.MODEL_ORDER), manifest_path=manifest,
                                               output_dir=directory, resume_verified_partial=directory,
                                               device="cuda", batch_size=2048, bootstrap_iterations=1000)
            self.assertEqual(before, self.hashes(directory))

    def test_approved_recovery_scores_only_two_missing_models_and_preserves_all_twenty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, partitions, directory = self.fixture(root, approve=True)
            before = self.hashes(directory)
            state = json.loads((root / "analysis_state.json").read_text())
            approved = state["explicit_transfer_recovery"]["verified_reuse"]
            with patch.object(transfer, "PROJECT_ROOT", root), patch("torch.cuda.is_available", return_value=True), \
                    patch.object(transfer, "load_dataset", side_effect=PartitionTests._loader), \
                    patch.object(transfer, "load_frozen_scorer", side_effect=self.scorer) as loading, \
                    patch.object(transfer, "bootstrap_ci", side_effect=self.intervals) as bootstrap, patch("builtins.print"):
                output = transfer.evaluate_transfer(partitions, list(transfer.MODEL_ORDER), manifest_path=manifest,
                                                     output_dir=directory, resume_verified_partial=directory,
                                                     device="cuda", batch_size=2048, bootstrap_iterations=1000)
            self.assertEqual(output, directory)
            self.assertEqual([call.args[0] for call in loading.call_args_list], ["fttransformer", "ocsvm"])
            self.assertEqual(bootstrap.call_count, 10)
            for name, digest in before.items():
                self.assertEqual(file_sha256(directory / name), digest)
            summary = json.loads((directory / "transfer_manifest.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(set(summary["models"]), set(transfer.MODEL_ORDER))
            self.assertEqual(summary["verified_partial_reuse"]["models"], approved)
            self.assertEqual(summary["models"]["fttransformer"]["inference_execution"]["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
