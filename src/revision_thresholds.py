"""Derive four operating rules from pinned, saved validation evidence only.

No model fitting, threshold reconstruction, TEST-based selection or source-run
mutation is permitted. Classical baselines use the median of five per-fold
thresholds; FT-Transformer uses its saved single-holdout scores. Missing saved
validation evidence is an error, including for historical BAF runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.metrics import compute_all_metrics
from generate_revision_interpretability import ROOT, file_digest, validate_output_directory
from threshold_study import STRATEGIES, apply_threshold, compute_fold_thresholds


MODELS = ("logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm")


def derive_thresholds(labels: np.ndarray, scores: np.ndarray, fold_ids: np.ndarray,
                      model: str) -> dict:
    """Apply the declared fold aggregation to saved validation predictions."""
    labels, scores, fold_ids = map(np.asarray, (labels, scores, fold_ids))
    if labels.ndim != 1 or scores.ndim != 1 or fold_ids.ndim != 1:
        raise ValueError("Validation evidence must consist of one-dimensional arrays")
    if not len(labels) or len(labels) != len(scores) or len(labels) != len(fold_ids):
        raise ValueError("Validation labels, scores and fold identifiers must align")
    if not np.isfinite(scores).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Validation scores must be finite and labels binary")
    folds = np.unique(fold_ids)
    expected = 1 if model == "fttransformer" else 5
    if not np.array_equal(folds, np.arange(1, expected + 1)):
        raise ValueError(f"{model} requires {expected} saved validation fold(s)")
    per_fold = {rule: [] for rule in STRATEGIES}
    sizes = []
    for fold in folds:
        mask = fold_ids == fold
        if np.unique(labels[mask]).size != 2:
            raise ValueError(f"Validation fold {fold} does not contain both classes")
        thresholds = compute_fold_thresholds(labels[mask], scores[mask])
        for rule in STRATEGIES:
            per_fold[rule].append(float(thresholds[rule]))
        sizes.append({"fold": int(fold), "samples": int(mask.sum()),
                      "fraud_samples": int(labels[mask].sum())})
    return {"thresholds_per_fold": per_fold,
            "thresholds_median": {rule: float(np.median(values)) for rule, values in per_fold.items()},
            "validation_folds": sizes,
            "aggregation": "single validation holdout" if expected == 1 else "median of five per-fold thresholds"}


def verify_primary_threshold(derived: float, exact: float) -> None:
    """Fail if saved validation evidence cannot reproduce the baseline rule."""
    if not np.isfinite(derived) or not np.isfinite(exact) or not np.isclose(derived, exact, rtol=0, atol=1e-12):
        raise ValueError(f"Derived max-F2 threshold {derived!r} differs from baseline threshold_exact {exact!r}")


def resolve_run(manifest: dict, dataset: str, model: str) -> Path:
    entry = manifest.get("baseline_runs", {}).get(f"{dataset}/{model}")
    if entry is None:
        raise KeyError(f"No pinned baseline_runs entry for {dataset}/{model}")
    run = Path(entry["run_dir"] if isinstance(entry, dict) else entry)
    run = run.resolve() if run.is_absolute() else (ROOT / run).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    if isinstance(entry, dict) and entry.get("config_sha256"):
        if file_digest(run / "config.json") != entry["config_sha256"]:
            raise ValueError(f"Pinned baseline configuration changed: {run}")
    return run


def standard_json(value):
    """Use null, not non-standard Infinity tokens, for unattainable cut-offs."""
    if isinstance(value, dict):
        return {key: standard_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [standard_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def study_saved_run(run_dir: Path, dataset: str, model: str) -> dict:
    """Derive DEV thresholds, verify the primary threshold, then score TEST."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("dataset") != dataset or config.get("model") != model:
        raise ValueError("Pinned run has the wrong dataset or model")
    if config.get("strategy") not in ("none", "n/a") or config.get("missing_policy", "preserve") != "preserve":
        raise ValueError("Threshold comparison requires a primary preserve-policy baseline")
    if "threshold_exact" not in config:
        raise ValueError("Pinned baseline lacks threshold_exact; no implicit reconstruction is allowed")
    required = ("y_val.npy", "y_val_scores.npy", "validation_fold_ids.npy",
                "validation_row_indices.npy", "test_row_indices.npy", "y_test.npy", "y_test_scores.npy")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Saved validation/split evidence is missing: {missing}; refit explicitly, not in this script")
    evidence = {name: np.load(run_dir / name) for name in required}
    val_indices = evidence["validation_row_indices.npy"]
    test_indices = evidence["test_row_indices.npy"]
    if len(val_indices) != len(evidence["y_val.npy"]) or len(test_indices) != len(evidence["y_test.npy"]):
        raise ValueError("Saved split indices and labels have different lengths")
    if len(np.unique(val_indices)) != len(val_indices) or len(np.unique(test_indices)) != len(test_indices):
        raise ValueError("Saved validation or TEST indices are repeated")
    if np.intersect1d(val_indices, test_indices).size:
        raise ValueError("Validation and TEST row indices overlap")
    thresholds = derive_thresholds(evidence["y_val.npy"], evidence["y_val_scores.npy"],
                                   evidence["validation_fold_ids.npy"], model)
    verify_primary_threshold(thresholds["thresholds_median"]["max_f2"], float(config["threshold_exact"]))
    y_test, scores_test = evidence["y_test.npy"], evidence["y_test_scores.npy"]
    if len(y_test) != len(scores_test) or not np.isfinite(scores_test).all() or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Invalid saved TEST labels or scores")
    results = {}
    for rule, threshold in thresholds["thresholds_median"].items():
        result = apply_threshold(y_test, scores_test, threshold)
        result["threshold_exact"] = threshold
        metrics = compute_all_metrics(y_test, scores_test, threshold)
        result.update({key: metrics[key] for key in ("TP", "FP", "TN", "FN", "PR-AUC", "ROC-AUC")})
        result["selection_state"] = "finite threshold" if np.isfinite(threshold) else "no finite median precision-constrained threshold"
        results[rule] = result
    primary = json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))
    for key in ("F1", "F2", "TP", "FP", "TN", "FN", "alert_rate"):
        if abs(float(results["max_f2"][key]) - float(primary[key])) > 1e-6:
            raise ValueError(f"The saved baseline TEST {key} does not reproduce under the exact max-F2 threshold")
    return {"model": model, "dataset": dataset, "source_run": str(run_dir.resolve()),
            "selection_population": "DEV validation only; TEST scores used after threshold selection",
            "precision_constraint_scope": "each validation fold; neither a pooled validation nor TEST guarantee after median aggregation",
            "fixed_rule_scale": "anomaly-score reference, not a probability" if model == "ocsvm" else "fraud probability",
            "test_samples": len(y_test), "test_fraud_samples": int(y_test.sum()),
            "test_prevalence": float(y_test.mean()), **thresholds, "test_results": results,
            "source_sha256": {name: file_digest(run_dir / name) for name in ("config.json", "metrics_test.json", *required)}}


def generate(manifest_path: Path, dataset: str, output_dir: Path, models=MODELS) -> dict:
    output_dir = validate_output_directory(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Complete all checks before writing outputs, avoiding partial updates.
    studies = {model: study_saved_run(resolve_run(manifest, dataset, model), dataset, model) for model in models}
    output_dir.mkdir(parents=True, exist_ok=True)
    for model, study in studies.items():
        model_dir = output_dir / model
        model_dir.mkdir(exist_ok=True)
        (model_dir / "threshold_study.json").write_text(json.dumps(standard_json(study), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"manifest_path": str(manifest_path.resolve()), "dataset": dataset,
               "models": list(models), "new_fits": 0, "all_primary_thresholds_reproduced": True,
               "studies": {model: str(output_dir / model / "threshold_study.json") for model in models}}
    (output_dir / "threshold_qa.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ulb", "ulb_2013", "baf_base"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    dataset = "ulb_2013" if args.dataset == "ulb" else args.dataset
    print(json.dumps(generate(args.manifest, dataset, args.output_dir, args.models), indent=2))


if __name__ == "__main__":
    main()
