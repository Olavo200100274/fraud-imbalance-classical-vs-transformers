"""Recompute frozen-LightGBM SHAP stability on corrected transfer populations.

Sources, the completed disjoint partition directory and the new output directory
must be explicit. No model, preprocessor or threshold is fitted. LightGBM's
native ``pred_contrib`` computes exact tree-path-dependent TreeSHAP in raw
log-odds, matching the saved Base TreeExplainer values. This equivalence is
checked numerically before any Variant is analysed.

All retained rows are explained in bounded-memory chunks. Encoded SHAP matrices
are stored as float32; original-feature importance is accumulated in float64
before storage and groups categorical one-hot columns by summing absolute
contributions. Top-feature membership and ordering are distinct descriptive
properties, not evidence of causality or an attention-based explanation.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np

from data import load_dataset
from generate_revision_interpretability import (file_digest, group_contributions,
                                                load_artefacts, read_manifest,
                                                validate_output_directory)
from revision_transfer import load_partition_manifest


VARIANTS = ("baf_var1", "baf_var2", "baf_var3", "baf_var4", "baf_var5")


def contribution_matrix(native: np.ndarray, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Separate exact encoded attributions from the native expected raw score."""
    native = np.asarray(native)
    if native.ndim != 2 or native.shape[1] != feature_count + 1 or not np.isfinite(native).all():
        raise ValueError("Native TreeSHAP matrix must contain finite feature contributions and one expected value")
    return native[:, :-1], native[:, -1]


def ranked_importance(features: list[str], values: np.ndarray, top_k: int) -> tuple[list[str], dict]:
    """Keep complete actual magnitudes and derive deterministic top-feature sets."""
    values = np.asarray(values)
    if len(features) != len(values) or len(set(features)) != len(features) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Original-feature importance must be unique, finite and non-negative")
    if not 1 <= top_k <= len(features):
        raise ValueError("top_k must be within the original-feature count")
    order = np.argsort(-values, kind="stable")
    return [features[index] for index in order[:top_k]], {features[index]: float(values[index]) for index in order}


def generate(manifest_path: Path, partitions_dir: Path, output_dir: Path,
             *, batch_size: int = 4096, threads: int = 2, top_k: int = 10) -> dict:
    if batch_size < 1 or threads < 1:
        raise ValueError("Batch size and thread count must be positive")
    output_dir = validate_output_directory(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; partial or completed scientific artefacts are not overwritten")
    partitions_dir = partitions_dir.resolve()
    prepared = load_partition_manifest(partitions_dir)
    if prepared.get("matching_policy") != "any_exact_common_predictor_match_with_Base_DEV":
        raise ValueError("The partition manifest does not identify the corrected common-predictor population")
    run = read_manifest(manifest_path)["lgbm"]
    base = load_artefacts("lgbm", run)
    pipeline = joblib.load(run / "model.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    booster = classifier.booster_
    names = list(preprocessor.get_feature_names_out())
    if names != base.feature_names:
        raise ValueError("Saved Base SHAP and frozen LightGBM feature schemas disagree")
    dev, base_test, dev_y, base_y, metadata = load_dataset("baf_base", return_metadata=True)
    del dev, dev_y
    if metadata["raw_file_sha256"] != prepared["base"]["data_provenance"]["raw_file_sha256"]:
        raise ValueError("Base raw data changed after partition preparation")
    if not np.array_equal(metadata["test_indices"], np.load(partitions_dir / "base_test_row_indices.npy", allow_pickle=False)):
        raise ValueError("Base TEST row positions differ from the completed partition evidence")
    if not np.array_equal(np.asarray(base_y), base.labels):
        raise ValueError("Base TEST labels differ from the saved compatible SHAP evidence")
    sample_positions = np.linspace(0, len(base_test) - 1, min(128, len(base_test)), dtype=int)
    transformed = preprocessor.transform(base_test.iloc[sample_positions])
    native = booster.predict(transformed, pred_contrib=True, num_threads=threads)
    contributions, expected = contribution_matrix(native, len(names))
    equivalence_error = float(np.max(np.abs(contributions - base.shap_values[sample_positions])))
    expected_error = float(np.max(np.abs(expected - base.expected_logit)))
    if equivalence_error > 1e-10 or expected_error > 1e-10:
        raise ValueError("Native frozen TreeSHAP does not reproduce the saved Base explainer")
    base_top, base_importance = ranked_importance(base.original_features, base.global_importance, top_k)
    report = {"status": "in_progress", "model": "lgbm", "source_run": str(run),
              "source_artefacts_sha256": {name: file_digest(run / name) for name in
                                         ("config.json", "model.joblib", "shap_values.npy", "y_test.npy", "y_test_scores.npy")},
              "partition_directory": str(partitions_dir), "partition_manifest_sha256": file_digest(partitions_dir / "partition_manifest.json"),
              "source_manifest": str(manifest_path.resolve()), "source_code_sha256": file_digest(Path(__file__)),
              "explainer": "Exact native tree-path-dependent TreeSHAP, raw log-odds",
              "categorical_grouping": "Mean of the per-row sum of absolute one-hot contributions",
              "model_fitting": False, "variant_threshold_selection": False,
              "base_explainer_max_absolute_difference": equivalence_error,
              "base_expected_value_max_absolute_difference": expected_error,
              "expected_raw_logit": base.expected_logit,
              "top_k": top_k, "variant_keys": ["baf_base", *VARIANTS],
              "encoded_feature_names": names, "original_feature_names": base.original_features,
              "rankings": {"baf_base": base_top}, "global_importance": {"baf_base": base_importance},
              "populations": {"baf_base": {"rows": len(base.labels), "positive_rows": int(base.labels.sum())}},
              "variants": {}}
    del base_test, base_y, base
    gc.collect()
    output_dir.mkdir(parents=True, exist_ok=False)
    for variant in VARIANTS:
        start = time.perf_counter()
        partition_dir = partitions_dir / variant
        audit = json.loads((partition_dir / "partition_audit.json").read_text(encoding="utf-8"))
        dev, test, dev_y, labels, metadata = load_dataset(variant, return_metadata=True)
        del dev, dev_y
        if list(test.columns) != prepared["feature_columns"]:
            raise ValueError("Variant predictors differ from the common 30-feature projection")
        if metadata["raw_file_sha256"] != audit["data_provenance"]["raw_file_sha256"]:
            raise ValueError("Variant raw data changed after preparation")
        original_indices = np.load(partition_dir / "original_test_row_indices.npy", allow_pickle=False)
        original_labels = np.load(partition_dir / "original_test_y.npy", allow_pickle=False)
        kept = np.load(partition_dir / "kept_test_positions.npy", allow_pickle=False)
        if not np.array_equal(metadata["test_indices"], original_indices) or not np.array_equal(np.asarray(labels), original_labels):
            raise ValueError("Variant TEST membership/order differs from the corrected partition evidence")
        selected = test.iloc[kept]
        del test, labels
        variant_dir = output_dir / variant
        variant_dir.mkdir(exist_ok=False)
        shap_path = variant_dir / "shap_values.npy"
        saved = np.lib.format.open_memmap(shap_path, mode="w+", dtype=np.float32, shape=(len(kept), len(names)))
        grouped_sum = np.zeros(len(report["original_feature_names"]), dtype=np.float64)
        additivity_error = 0.0
        baseline_error = 0.0
        for first in range(0, len(selected), batch_size):
            last = min(first + batch_size, len(selected))
            transformed = preprocessor.transform(selected.iloc[first:last])
            native = booster.predict(transformed, pred_contrib=True, num_threads=threads)
            contributions, expected = contribution_matrix(native, len(names))
            logits = booster.predict(transformed, raw_score=True, num_threads=threads)
            additivity_error = max(additivity_error, float(np.max(np.abs(contributions.sum(axis=1) + expected - logits))))
            baseline_error = max(baseline_error, float(np.max(np.abs(expected - report["expected_raw_logit"]))))
            if additivity_error > 1e-7 or baseline_error > 1e-10:
                raise ValueError("Corrected Variant TreeSHAP fails exact raw-logit additivity or changes the frozen expected value")
            grouped, features = group_contributions(contributions, names, absolute=True)
            if features != report["original_feature_names"]:
                raise ValueError("Original-feature grouping changed between TreeSHAP chunks")
            grouped_sum += grouped.sum(axis=0)
            saved[first:last] = contributions
            if first == 0 or last == len(selected) or last % (batch_size * 10) == 0:
                print(f"{variant}: {last:,}/{len(selected):,} retained rows explained", flush=True)
        saved.flush()
        del saved
        ranking, magnitudes = ranked_importance(features, grouped_sum / len(selected), top_k)
        np.save(variant_dir / "test_row_indices.npy", original_indices[kept], allow_pickle=False)
        np.save(variant_dir / "y_test.npy", original_labels[kept], allow_pickle=False)
        report["rankings"][variant] = ranking
        report["global_importance"][variant] = magnitudes
        report["populations"][variant] = audit["kept_population"]
        report["variants"][variant] = {"encoded_matrix_storage_dtype": "float32",
                                       "importance_accumulation_dtype": "float64",
                                       "max_logit_additivity_error": additivity_error,
                                       "max_expected_value_variation": baseline_error,
                                       "seconds": time.perf_counter() - start,
                                       "artefacts_sha256": {name: file_digest(variant_dir / name) for name in
                                                           ("shap_values.npy", "test_row_indices.npy", "y_test.npy")}}
        del selected
        gc.collect()
    report["jaccard_matrix"] = {}
    for first in report["variant_keys"]:
        report["jaccard_matrix"][first] = {}
        for second in report["variant_keys"]:
            left, right = set(report["rankings"][first]), set(report["rankings"][second])
            report["jaccard_matrix"][first][second] = len(left & right) / len(left | right)
    report["status"] = "complete"
    output = output_dir / "shap_variant_stability.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Corrected SHAP stability: {output}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    generate(args.manifest, args.partitions_dir, args.output_dir,
             batch_size=args.batch_size, threads=args.threads, top_k=args.top_k)


if __name__ == "__main__":
    main()
