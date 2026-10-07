"""Audit the seven pinned original ULB LightGBM runs without fitting.

This is historical diagnostic evidence, not revised primary performance.
The original stratified seed-42 TEST is reconstructed before deduplication.
Existing score-group and chunked-loading helpers are reused, while corrected
run selection, its queue and its immutable report remain completely separate.
"""

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from data import get_dataset_info
from experiment_protocol import (
    DEFAULT_MANIFEST_PATH, DEFAULT_RESULTS_ROOT, PROJECT_ROOT,
    array_sha256, file_sha256,
)
from revision_lgbm_diagnostics import (
    STRATEGIES, _json, _read_test_rows, _save_array, _write_json,
    score_diagnostics,
)


def select_historical_runs(manifest):
    """Require all seven historical pins and verify their immutable configs."""
    selected = {}
    for strategy in STRATEGIES:
        key = f"ulb_2013/lgbm/{strategy}"
        reference = manifest.get("historical_runs", {}).get(key)
        if not isinstance(reference, dict) or not reference.get("config_sha256"):
            raise FileNotFoundError(f"Missing explicit historical source pin: {key}.")
        run = Path(reference["run_dir"])
        run = (run if run.is_absolute() else PROJECT_ROOT / run).resolve()
        config_hash = file_sha256(run / "config.json")
        if config_hash != reference["config_sha256"]:
            raise ValueError(f"Historical config changed after pinning: {key}.")
        config = _json(run / "config.json")
        if (config.get("dataset") != "ulb_2013" or config.get("model") != "lgbm"
                or config.get("strategy") != strategy or config.get("split_seed") != 42
                or config.get("split_ratio") != "80/20 stratified"
                or config.get("sample_fraction") not in (None, 1, 1.0)
                or config.get("protocol_version") is not None):
            raise ValueError(f"The source is not an original full ULB run: {key}.")
        if (reference.get("dataset_hash_sha256") is not None
                and reference["dataset_hash_sha256"] != config.get("dataset_hash_sha256")):
            raise ValueError(f"Historical pin/raw hash disagreement: {key}.")
        selected[strategy] = (run, config)
    baseline_config = selected["none"][1]
    if baseline_config.get("n_trials") != 50:
        raise ValueError("The pinned historical baseline must document its original 50-trial search.")
    for _, config in selected.values():
        if config.get("best_params") != baseline_config.get("best_params"):
            raise ValueError("The seven historical runs do not use the same baseline-selected hyperparameters.")
    return selected


def reconstruct_original_test(csv_path, selected):
    """Reproduce the raw 80/20 split and verify every saved label sequence."""
    raw_hash = file_sha256(csv_path)
    raw_labels = pd.read_csv(csv_path, usecols=["Class"])["Class"].to_numpy()
    if set(np.unique(raw_labels)) != {0, 1}:
        raise ValueError("Original ULB labels must contain exactly the two binary classes.")
    all_indices = np.arange(len(raw_labels), dtype=np.int64)
    dev_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, stratify=raw_labels, random_state=42,
    )
    test_labels = raw_labels[test_indices]
    for strategy, (run, config) in selected.items():
        if (config.get("dataset_hash_sha256") != raw_hash
                or config.get("train_samples") != len(dev_indices)
                or config.get("test_samples") != len(test_indices)
                or config.get("train_fraud") != int(raw_labels[dev_indices].sum())
                or config.get("test_fraud") != int(test_labels.sum())
                or not np.array_equal(np.load(run / "y_test.npy", allow_pickle=False), test_labels)):
            raise ValueError(f"Original raw split/labels/config mismatch for {strategy}.")
        saved_indices_path = run / "test_row_indices.npy"
        if saved_indices_path.exists() and not np.array_equal(
                np.load(saved_indices_path, allow_pickle=False), test_indices):
            raise ValueError(f"Historical saved indices disagree with the raw seed-42 split for {strategy}.")
    return dev_indices, test_indices, test_labels, {
        "raw_file": str(Path(csv_path).resolve()), "raw_file_sha256": raw_hash,
        "raw_rows": len(raw_labels), "raw_fraud": int(raw_labels.sum()),
        "split_seed": 42, "split_ratio": "80/20 stratified",
        "dev_rows": len(dev_indices), "dev_fraud": int(raw_labels[dev_indices].sum()),
        "test_rows": len(test_indices), "test_fraud": int(test_labels.sum()),
        "test_prevalence": float(test_labels.mean()),
        "test_row_indices_sha256": array_sha256(test_indices),
        "dev_row_indices_sha256": array_sha256(dev_indices),
        "deduplication": "Not applied: original historical CSV population and split.",
        "index_provenance": "Reconstructed from raw CSV row positions and recorded stratified seed-42 split; not falsely attributed to an original saved indices artefact.",
        "all_saved_label_sequences_match_raw_test": True,
    }


def _assert_output_scope(output_root):
    root = Path(output_root).resolve()
    for name in ("results", "results_thesis", "Overleaf", "Article 1", "Article 2"):
        protected = (PROJECT_ROOT / name).resolve()
        if root == protected or protected in root.parents:
            raise ValueError("Historical diagnostics must not modify preserved results or documents.")
    return root / "audit"


def run_historical_diagnostics(manifest_path=DEFAULT_MANIFEST_PATH,
                               output_root=DEFAULT_RESULTS_ROOT, *, threads=2,
                               csv_path=None):
    """Validate frozen historical models and write a distinct immutable audit."""
    if threads < 1 or threads > 2:
        raise ValueError("Historical diagnostics accept one or two operational threads only.")
    audit_root = _assert_output_scope(output_root)
    final_report = audit_root / "historical_lgbm_score_diagnostics.json"
    if final_report.exists():
        raise FileExistsError("The historical audit already exists and must not be replaced.")
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    selected = select_historical_runs(manifest)
    csv_path = Path(csv_path or get_dataset_info("ulb")[0]).resolve()
    dev_indices, indices, labels, split_audit = reconstruct_original_test(csv_path, selected)
    frame = _read_test_rows(csv_path, indices, labels)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = audit_root / "historical_lgbm_diagnostics" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "auditing", "role": "historical_original_protocol_diagnostic_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_snapshot_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_code_sha256": file_sha256(__file__),
        "reused_diagnostic_helpers_sha256": file_sha256(Path(__file__).with_name("revision_lgbm_diagnostics.py")),
        "audit_directory": str(directory), "threads": threads,
        "fit_or_resampling_performed": False, "threshold_selection_performed": False,
        "test_labels_used_for_fitting": False,
        "split_audit": split_audit,
        "historical_hpo": {"baseline_trials": selected["none"][1]["n_trials"],
                           "intervention_parameters": "Fixed at the historical baseline-selected values; no new search."},
        "scientific_boundary": "The original non-deduplicated population has repeated predictor profiles and is superseded for primary evaluation. These frozen-score diagnostics explain historical AP/ROC behaviour; they are not corrected primary results, a deduplication ablation or a causal decomposition of the full treatment effect.",
        "artefacts": {"reconstructed_test_row_indices": _save_array(directory, "test_row_indices.npy", indices),
                      "reconstructed_dev_row_indices": _save_array(directory, "dev_row_indices.npy", dev_indices)},
        "strategies": {},
    }
    saved_scores = {}
    with threadpool_limits(limits=threads):
        for strategy, (run, config) in selected.items():
            scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
            if scores.shape != labels.shape or np.any((scores < 0) | (scores > 1)):
                raise ValueError(f"Historical probability shape or range is invalid for {strategy}.")
            record = score_diagnostics(labels, scores)
            metrics_path = run / "metrics_test.json"
            stored_metrics = _json(metrics_path)
            for key, measured in (("PR-AUC", record["average_precision"]),
                                  ("ROC-AUC", record["roc_auc"])):
                if abs(stored_metrics[key] - measured) > 5.1e-7:
                    raise ValueError(f"Historical rounded {key} is inconsistent with scores for {strategy}.")
            model = joblib.load(run / "model.joblib")
            transformed = model.named_steps["preprocessor"].transform(frame)
            classifier = model.named_steps["classifier"]
            predicted = classifier.predict_proba(transformed, num_threads=threads)[:, 1]
            if not np.allclose(scores, predicted, rtol=1e-10, atol=1e-12):
                raise ValueError(f"Frozen historical model does not reproduce stored probabilities for {strategy}.")
            raw_margins = classifier.booster_.predict(transformed, raw_score=True, num_threads=threads)
            transformed_array = np.asarray(transformed)
            target_dir = directory / strategy
            target_dir.mkdir(exist_ok=False)
            classifier_params = classifier.get_params()
            saved_indices_path = run / "test_row_indices.npy"
            record.update({
                "source_run": str(run), "source_config_sha256": file_sha256(run / "config.json"),
                "source_model_sha256": file_sha256(run / "model.joblib"),
                "source_scores_sha256": file_sha256(run / "y_test_scores.npy"),
                "source_labels_sha256": file_sha256(run / "y_test.npy"),
                "source_metrics_sha256": file_sha256(metrics_path),
                "source_indices_artefact_present": saved_indices_path.exists(),
                "source_indices_sha256": file_sha256(saved_indices_path) if saved_indices_path.exists() else None,
                "reconstructed_indices_artefact": report["artefacts"]["reconstructed_test_row_indices"],
                "best_params": config["best_params"],
                "classifier_parameter_audit": {key: classifier_params.get(key) for key in
                                               ("num_leaves", "n_estimators", "min_child_samples", "learning_rate",
                                                "class_weight", "scale_pos_weight", "is_unbalance")},
                "trees": classifier.booster_.num_trees(),
                "saved_probabilities_bitwise_reproduced": bool(np.array_equal(scores, predicted)),
                "maximum_probability_difference": float(np.max(np.abs(scores - predicted))),
                "raw_margin_unique_values": len(np.unique(raw_margins)),
                "raw_margin_minimum": float(raw_margins.min()),
                "raw_margin_maximum": float(raw_margins.max()),
                "artefacts": {"raw_margins": _save_array(target_dir, "raw_margins.npy", raw_margins)},
            })
            for group in record["top_exact_score_groups"][:5]:
                positions = np.asarray(group["test_positions"], dtype=np.int64)
                leaf_batches = [np.asarray(classifier.booster_.predict(
                    transformed_array[positions[start:start + 128]], pred_leaf=True, num_threads=threads,
                ), dtype=np.int32) for start in range(0, len(positions), 128)]
                leaf_vectors = np.vstack(leaf_batches)
                unique_paths = np.unique(leaf_vectors, axis=0)
                group.update({
                    "raw_margin_unique_values": len(np.unique(raw_margins[positions])),
                    "unique_full_leaf_vectors": len(unique_paths),
                    "all_members_share_one_full_leaf_vector": len(unique_paths) == 1,
                    "leaf_equality_checked_across_trees": int(leaf_vectors.shape[1]),
                    "original_test_row_indices": indices[positions].tolist(),
                    "leaf_vectors_artefact": _save_array(target_dir, f"top_group_{group['group']}_leaf_vectors.npy", leaf_vectors),
                })
                del leaf_batches, leaf_vectors, unique_paths
            report["strategies"][strategy] = record
            saved_scores[strategy] = scores
            print(f"Historical {strategy}: AP={record['average_precision']:.12g}; "
                  f"ROC={record['roc_auc']:.12g}; top={record['top_exact_score_groups'][0]['rows']}; "
                  f"unique={record['unique_exact_scores']}", flush=True)
            del model, transformed, transformed_array, classifier, predicted, raw_margins
            gc.collect()
    report["exact_score_comparisons"] = {}
    for first, second in (("smote", "smote_tomek"), ("ros", "weights")):
        report["exact_score_comparisons"][f"{first}_vs_{second}"] = {
            "bitwise_equal": bool(np.array_equal(saved_scores[first], saved_scores[second])),
            "maximum_absolute_difference": float(np.max(np.abs(saved_scores[first] - saved_scores[second]))),
        }
    report["status"] = "complete"
    _write_json(directory / "historical_lgbm_score_diagnostics.json", report)
    _write_json(final_report, report)
    return final_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run_historical_diagnostics(args.manifest, args.results_root, threads=args.threads)


if __name__ == "__main__":
    main()
