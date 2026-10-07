"""Saved-score and frozen-tree diagnostics for all seven corrected ULB runs.

No fitting, resampling, threshold selection or artificial tie breaking occurs.
Only explicitly pinned completed runs are accepted. ULB TEST rows are read in
CSV chunks using their saved original indices, avoiding a full dataset copy.
Historical score diagnostics are separate context, not a causal deduplication
ablation: corrected HPO, split membership and preprocessing may also differ.
"""

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from data import get_dataset_info
from experiment_protocol import (
    DEFAULT_MANIFEST_PATH, DEFAULT_RESULTS_ROOT, PROJECT_ROOT, PROTOCOL_VERSION,
    array_sha256, file_sha256,
)


STRATEGIES = ("none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_array_hashes(run):
    """Pin the bytes of the source scores, labels and original TEST indices."""
    return {
        "source_scores_sha256": file_sha256(Path(run) / "y_test_scores.npy"),
        "source_labels_sha256": file_sha256(Path(run) / "y_test.npy"),
        "source_indices_sha256": file_sha256(Path(run) / "test_row_indices.npy"),
    }


def score_diagnostics(labels, scores, *, top_groups=10):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all()
            or set(np.unique(labels)) != {0, 1}):
        raise ValueError("Binary TEST labels and finite scores must align.")
    order = np.argsort(-scores, kind="stable")
    ranked_scores, ranked_labels = scores[order], labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ranked_scores) != 0) + 1]
    ends = np.r_[starts[1:], len(scores)]
    sizes = ends - starts
    frauds = np.add.reduceat(ranked_labels, starts)
    cumulative_rows, cumulative_frauds = np.cumsum(sizes), np.cumsum(frauds)
    precisions = cumulative_frauds / cumulative_rows
    recalls = cumulative_frauds / labels.sum()
    contributions = frauds / labels.sum() * precisions
    ap, roc = average_precision_score(labels, scores), roc_auc_score(labels, scores)
    if not np.isclose(contributions.sum(), ap, rtol=1e-13, atol=1e-13):
        raise ValueError("Exact score-group AP decomposition does not reproduce saved-score AP.")
    top = []
    for group in range(min(top_groups, len(starts))):
        positions = order[starts[group]:ends[group]]
        top.append({"group": group, "score": float(ranked_scores[starts[group]]),
                    "rows": int(sizes[group]), "fraud": int(frauds[group]),
                    "non_fraud": int(sizes[group] - frauds[group]),
                    "rank_start": int(starts[group] + 1), "rank_end": int(ends[group]),
                    "cumulative_precision": float(precisions[group]), "cumulative_recall": float(recalls[group]),
                    "average_precision_contribution": float(contributions[group]),
                    "test_positions": positions.tolist()})
    largest = int(np.argmax(sizes))
    near_top = {}
    for tolerance in (1e-12, 1e-8, 1e-6):
        mask = scores.max() - scores <= tolerance
        near_top[str(tolerance)] = {"rows": int(mask.sum()), "fraud": int(labels[mask].sum()),
                                    "non_fraud": int(mask.sum() - labels[mask].sum())}
    result = {
        "rows": len(scores), "fraud": int(labels.sum()), "fraud_prevalence": float(labels.mean()),
        "average_precision": float(ap), "roc_auc": float(roc), "unique_exact_scores": len(starts),
        "exact_zero_scores": int(np.sum(scores == 0)), "exact_one_scores": int(np.sum(scores == 1)),
        "scores_at_least_0_99": int(np.sum(scores >= 0.99)), "scores_at_least_0_999": int(np.sum(scores >= 0.999)),
        "minimum_score": float(scores.min()), "maximum_score": float(scores.max()),
        "quantiles": {str(q): float(np.quantile(scores, q)) for q in (0, 0.5, 0.9, 0.99, 0.999, 1)},
        "largest_exact_group": {"score": float(ranked_scores[starts[largest]]), "rows": int(sizes[largest]),
                                "fraud": int(frauds[largest]), "non_fraud": int(sizes[largest] - frauds[largest])},
        "top_exact_score_groups": top, "near_maximum_groups": near_top,
        "ap_decomposition_matches": True,
    }
    return result


def _read_test_rows(csv_path, original_indices, labels, chunk_rows=16384):
    indices = np.asarray(original_indices, dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0):
        raise ValueError("Saved TEST row indices must be unique non-negative CSV positions.")
    pieces = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):
        selected = chunk.index.isin(indices)
        if selected.any():
            pieces.append(chunk.loc[selected].copy())
    frame = pd.concat(pieces).loc[indices]
    if not np.array_equal(frame["Class"].to_numpy(), labels):
        raise ValueError("Saved TEST labels do not match their explicit raw CSV positions.")
    return frame.drop(columns="Class")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _save_array(directory, name, values):
    destination = directory / name
    with destination.open("xb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
    return {"path": str(destination), "file_sha256": file_sha256(destination),
            "array_sha256": array_sha256(np.asarray(values))}


def run_diagnostics(manifest_path=DEFAULT_MANIFEST_PATH, output_root=DEFAULT_RESULTS_ROOT, *, threads=2):
    if threads < 1:
        raise ValueError("The operational thread limit must be positive.")
    manifest = _json(manifest_path)
    selected = {}
    for strategy in STRATEGIES:
        key = f"ulb_2013/lgbm/{strategy}"
        reference = manifest.get("runs", {}).get(key)
        if not isinstance(reference, dict) or not reference.get("config_sha256"):
            raise FileNotFoundError(f"All seven corrected strategies must be pinned; missing {key}.")
        run = Path(reference["run_dir"])
        run = (run if run.is_absolute() else Path(manifest_path).resolve().parent / run).resolve()
        if file_sha256(run / "config.json") != reference["config_sha256"]:
            raise ValueError("A corrected source config changed after explicit pinning.")
        config, completion = _json(run / "config.json"), _json(run / "completed.json")
        if (config.get("dataset") != "ulb_2013" or config.get("model") != "lgbm"
                or config.get("strategy") != strategy or config.get("sample_fraction") not in (None, 1, 1.0)
                or config.get("protocol_version") != PROTOCOL_VERSION
                or completion.get("status") != "complete" or completion.get("protocol_version") != PROTOCOL_VERSION):
            raise ValueError("Score diagnostics require completed full corrected ULB LGBM runs.")
        selected[strategy] = (run, config)
    csv_path, _ = get_dataset_info("ulb")
    raw_hash = file_sha256(csv_path)
    baseline_dir = selected["none"][0]
    labels = np.load(baseline_dir / "y_test.npy", allow_pickle=False)
    indices = np.load(baseline_dir / "test_row_indices.npy", allow_pickle=False)
    frame = _read_test_rows(csv_path, indices, labels)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    audit_root = Path(output_root).resolve() / "audit"
    for name in ("results", "results_thesis", "Overleaf", "Article 1", "Article 2"):
        if (PROJECT_ROOT / name).resolve() in audit_root.parents:
            raise ValueError("A diagnostic must not write into preserved result/document directories.")
    directory = audit_root / "lgbm_diagnostics" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "auditing", "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(Path(manifest_path).resolve()), "source_manifest_sha256": file_sha256(manifest_path),
        "audit_directory": str(directory),
        "source_code_sha256": file_sha256(__file__), "raw_file_sha256": raw_hash,
        "test_row_indices_sha256": array_sha256(indices), "threads": threads,
        "labels_used_for_fitting": False, "fit_or_resampling_performed": False,
        "exact_tie_analysis": "Stored full-precision probabilities; no rounding, perturbation or artificial tie breaking.",
        "scientific_boundary": "Score and tree-leaf equivalence identify a predictive-resolution mechanism compatible with AP/ROC behaviour; they do not isolate the total causal effect of imbalance treatment, deduplication or HPO.",
        "strategies": {}, "historical_context": {},
    }
    saved_arrays = {}
    with threadpool_limits(limits=threads):
        for strategy, (run, config) in selected.items():
            scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
            if (config["dataset_hash_sha256"] != raw_hash
                    or not np.array_equal(np.load(run / "y_test.npy", allow_pickle=False), labels)
                    or not np.array_equal(np.load(run / "test_row_indices.npy", allow_pickle=False), indices)):
                raise ValueError("All strategy comparisons require the same exact corrected TEST and raw file.")
            saved_arrays[strategy] = scores
            record = score_diagnostics(labels, scores)
            saved_metrics = _json(run / "metrics_test.json")
            for key, recomputed in (("PR-AUC", record["average_precision"]), ("ROC-AUC", record["roc_auc"])):
                if abs(saved_metrics[key] - recomputed) > 5.1e-7:
                    raise ValueError("Saved rounded metrics do not reproduce their score arrays.")
            model = joblib.load(run / "model.joblib")
            transformed = model.named_steps["preprocessor"].transform(frame)
            classifier = model.named_steps["classifier"]
            predicted = classifier.predict_proba(transformed, num_threads=threads)[:, 1]
            if not np.allclose(scores, predicted, rtol=1e-10, atol=1e-12):
                raise ValueError("The explicitly pinned frozen LGBM model does not reproduce its saved probabilities.")
            raw_margin = classifier.booster_.predict(transformed, raw_score=True, num_threads=threads)
            transformed_array = np.asarray(transformed)
            target_dir = directory / strategy
            target_dir.mkdir(exist_ok=False)
            record.update({
                "source_run": str(run), "source_config_sha256": file_sha256(run / "config.json"),
                "source_model_sha256": file_sha256(run / "model.joblib"), "best_params": config["best_params"],
                **source_array_hashes(run),
                "trees": classifier.booster_.num_trees(), "num_leaves": classifier.get_params().get("num_leaves"),
                "saved_probabilities_bitwise_reproduced": bool(np.array_equal(scores, predicted)),
                "maximum_probability_difference": float(np.max(np.abs(scores - predicted))),
                "raw_margin_unique_values": len(np.unique(raw_margin)),
                "raw_margin_minimum": float(raw_margin.min()), "raw_margin_maximum": float(raw_margin.max()),
                "artefacts": {"raw_margins": _save_array(target_dir, "raw_margins.npy", raw_margin),
                              "test_row_indices": _save_array(target_dir, "test_row_indices.npy", indices)},
            })
            for group in record["top_exact_score_groups"][:5]:
                positions = np.asarray(group["test_positions"], dtype=np.int64)
                leaves = []
                for start in range(0, len(positions), 128):
                    leaf_batch = classifier.booster_.predict(
                        transformed_array[positions[start:start + 128]], pred_leaf=True, num_threads=threads,
                    )
                    leaves.append(np.asarray(leaf_batch, dtype=np.int32))
                leaf_vectors = np.vstack(leaves)
                unique_paths = np.unique(leaf_vectors, axis=0)
                group["raw_margin_unique_values"] = len(np.unique(raw_margin[positions]))
                group["unique_full_leaf_vectors"] = len(unique_paths)
                group["all_members_share_one_full_leaf_vector"] = len(unique_paths) == 1
                group["leaf_equality_checked_across_trees"] = int(leaf_vectors.shape[1])
                group["original_test_row_indices"] = indices[positions].tolist()
                group["leaf_vectors_artefact"] = _save_array(target_dir, f"top_group_{group['group']}_leaf_vectors.npy", leaf_vectors)
                del leaves, leaf_vectors, unique_paths
            report["strategies"][strategy] = record
            print(f"{strategy}: AP={record['average_precision']:.12g}; ROC={record['roc_auc']:.12g}; "
                  f"top={record['top_exact_score_groups'][0]['rows']} rows; unique={record['unique_exact_scores']}", flush=True)
            del model, transformed, transformed_array, predicted, raw_margin
            gc.collect()
    for first, second in (("smote", "smote_tomek"), ("ros", "weights")):
        report.setdefault("exact_score_comparisons", {})[f"{first}_vs_{second}"] = {
            "bitwise_equal": bool(np.array_equal(saved_arrays[first], saved_arrays[second])),
            "maximum_absolute_difference": float(np.max(np.abs(saved_arrays[first] - saved_arrays[second]))),
        }
    for strategy in STRATEGIES:
        reference = manifest.get("historical_runs", {}).get(f"ulb_2013/lgbm/{strategy}")
        if reference is None:
            continue
        run = Path(reference["run_dir"])
        run = run if run.is_absolute() else PROJECT_ROOT / run
        historical_labels = np.load(run / "y_test.npy", allow_pickle=False)
        historical_scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
        report["historical_context"][strategy] = {
            "role": "Separate original-protocol context; not paired with the corrected TEST.",
            "source_run": str(run), "source_config_sha256": file_sha256(run / "config.json"),
            "diagnostics": score_diagnostics(historical_labels, historical_scores),
        }
    report["status"] = "complete"
    _write_json(directory / "lgbm_score_diagnostics.json", report)
    # This explicit report is created once after all seven strategies pass.
    # A rerun uses a new timestamp and must not silently replace the audit.
    _write_json(audit_root / "lgbm_score_diagnostics.json", report)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run_diagnostics(args.manifest, args.results_root, threads=args.threads)


if __name__ == "__main__":
    main()
