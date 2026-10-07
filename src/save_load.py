import joblib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from experiment_protocol import (
    DEFAULT_RESULTS_ROOT, PROTOCOL_VERSION, array_sha256, file_sha256,
    record_completed_run, software_versions, validate_revision_destinations,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

def save_run(
    model,
    metrics_cv,
    metrics_test,
    y_test,
    y_test_scores,
    config,
    model_name,
    strategy="none",
    dataset="ulb_2013",
    bootstrap_ci=None,
    model_type="sklearn",
    results_root=None,
    split_metadata=None,
    validation_evidence=None,
    tuning_trials=None,
    sampler_diagnostics=None,
    manifest_path=None,
    defer_completion=False,
):
    """
    Persist a complete run with all artefacts.

    Directory layout
    ----------------
    results_revision/20261005/<dataset>/<model_name>/<strategy>/run_<timestamp>/
        config.json            – full reproducibility config
        model.joblib           – serialised pipeline (preprocess + model)
        metrics_cv.json        – cross-validation metrics (per fold + aggregated)
        metrics_test.json      – holdout test metrics + optional bootstrap CI
        confusion_matrix.png   – confusion-matrix heatmap
        pr_curve.png           – precision-recall curve
        pr_curve_data.json     – raw PR-curve points for LaTeX pgfplots
        roc_curve.png          – ROC curve
        roc_curve_data.json    – raw ROC-curve points for LaTeX pgfplots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Operational isolation only; thresholds and every saved value are unchanged.
    output_root, manifest_file = validate_revision_destinations(results_root, manifest_path)
    test_labels = np.asarray(y_test)
    test_scores = np.asarray(y_test_scores)
    if len(test_labels) != len(test_scores) or not np.isfinite(test_scores).all():
        raise ValueError("TEST labels and finite prediction scores must align.")
    threshold = config.get("threshold_exact", metrics_test["threshold"])
    test_predictions = (test_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions, labels=[0, 1]).ravel()
    recomputed_counts = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    if any(metrics_test[key] != value for key, value in recomputed_counts.items()):
        raise ValueError("Saved TEST confusion counts disagree with the exact decision threshold.")
    run_dir = os.path.join(
        str(output_root), dataset, model_name, strategy, f"run_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=False)

    provenance = None
    if split_metadata is not None:
        provenance = {
            key: value for key, value in split_metadata.items()
            if key not in ("dev_indices", "test_indices")
        }
        for split in ("dev", "test"):
            indices = np.asarray(split_metadata[f"{split}_indices"], dtype=np.int64)
            np.save(os.path.join(run_dir, f"{split}_row_indices.npy"), indices)
        if len(split_metadata["test_indices"]) != len(y_test):
            raise ValueError("Saved TEST labels do not align with the recorded split indices.")

    validation_summary = None
    if validation_evidence is not None:
        validation_labels = np.asarray(validation_evidence["y_val"])
        validation_scores = np.asarray(validation_evidence["y_val_scores"])
        validation_indices = np.asarray(validation_evidence["row_indices"], dtype=np.int64)
        fold_ids = np.asarray(
            validation_evidence.get("fold_ids", np.ones(len(validation_labels))),
            dtype=np.int64,
        )
        if not (len(validation_labels) == len(validation_scores) == len(validation_indices) == len(fold_ids)):
            raise ValueError("Validation predictions, labels and original indices must align.")
        if not np.isfinite(validation_scores).all():
            raise ValueError("Validation scores contain non-finite values.")
        for filename, array in (
            ("y_val.npy", validation_labels), ("y_val_scores.npy", validation_scores),
            ("validation_row_indices.npy", validation_indices),
            ("validation_fold_ids.npy", fold_ids),
        ):
            np.save(os.path.join(run_dir, filename), array)
        validation_summary = {
            "role": validation_evidence.get("role", "validation_only"),
            "rows": len(validation_labels), "folds": np.unique(fold_ids).tolist(),
            "row_indices_sha256": array_sha256(validation_indices),
            "scores_sha256": array_sha256(validation_scores),
            "labels_sha256": array_sha256(validation_labels),
        }

    # 1. Config (reproducibility)
    full_config = {
        **config,
        "timestamp": timestamp,
        "python_version": sys.version,
        "platform": platform.platform(),
        "protocol_version": PROTOCOL_VERSION,
        "software_versions": software_versions(),
        "source_files_sha256": {
            str(path.relative_to(_PROJECT_ROOT)): file_sha256(path)
            for path in sorted((_PROJECT_ROOT / "src").rglob("*.py"))
        },
        "source_files_sha256_scope": "on_disk_at_save_not_execution_proof",
        "data_provenance": provenance,
        "validation_evidence": validation_summary,
    }
    _save_json(full_config, os.path.join(run_dir, "config.json"))

    # 2. Model
    if model_type == "torch":
        import torch
        torch.save(model, os.path.join(run_dir, "model.pt"))
    else:
        joblib.dump(model, os.path.join(run_dir, "model.joblib"))

    # 3. Cross-validation metrics
    _save_json(metrics_cv, os.path.join(run_dir, "metrics_cv.json"))

    # 4. Test metrics (+ bootstrap CI if provided)
    test_out = dict(metrics_test)
    if bootstrap_ci is not None:
        test_out["bootstrap_ci"] = bootstrap_ci
    _save_json(test_out, os.path.join(run_dir, "metrics_test.json"))

    # 5. Confusion matrix plot
    _save_confusion_matrix(y_test, test_predictions, model_name, strategy, run_dir)

    # 6. PR curve plot
    _save_pr_curve(y_test, y_test_scores, model_name, strategy, run_dir)

    # 7. PR curve raw data
    _save_pr_data(y_test, y_test_scores, run_dir)

    # 8. ROC curve plot
    _save_roc_curve(y_test, y_test_scores, model_name, strategy, run_dir)

    # 9. ROC curve raw data
    _save_roc_data(y_test, y_test_scores, run_dir)

    # 10. Raw predictions (for threshold study, DeLong tests, etc.)
    np.save(os.path.join(run_dir, "y_test.npy"), np.asarray(y_test))
    np.save(os.path.join(run_dir, "y_test_scores.npy"), np.asarray(y_test_scores))

    if tuning_trials is not None:
        _save_json(tuning_trials, os.path.join(run_dir, "optuna_trials.json"))
    if sampler_diagnostics is not None:
        _save_json(sampler_diagnostics, os.path.join(run_dir, "sampler_diagnostics.json"))
    if not defer_completion:
        finalise_run(run_dir, manifest_file)

    print(f"  Run saved → {run_dir}")
    return run_dir


def finalise_run(run_dir, manifest_path=None, required_artefacts=()):
    """Guard destinations and promote only complete model-specific artefacts."""
    run_dir = Path(run_dir).resolve()
    run_dir, manifest_file = validate_revision_destinations(
        run_dir, manifest_path or run_dir.parents[3] / "revision_manifest.json")
    if not all((run_dir / filename).is_file() for filename in required_artefacts):
        raise ValueError("Required model-specific artefacts are missing; run remains incomplete.")
    _save_json({"status": "complete", "protocol_version": PROTOCOL_VERSION},
               run_dir / "completed.json")
    record_completed_run(run_dir, manifest_file)


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────

def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(_make_serializable(obj), f, indent=2)


def _make_serializable(obj):
    """Recursively convert numpy types to Python-native types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_make_serializable(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _save_confusion_matrix(y_test, y_pred, model_name, strategy, run_dir):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"{model_name} ({strategy}) — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def _save_pr_curve(y_test, y_scores, model_name, strategy, run_dir):
    precisions, recalls, _ = precision_recall_curve(y_test, y_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} ({strategy}) — Precision-Recall Curve")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pr_curve.png"), dpi=150)
    plt.close()


def _save_pr_data(y_test, y_scores, run_dir):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    data = {
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "thresholds": thresholds.tolist(),
    }
    with open(os.path.join(run_dir, "pr_curve_data.json"), "w") as f:
        json.dump(data, f)


def _save_roc_curve(y_test, y_scores, model_name, strategy, run_dir):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_val = roc_auc_score(y_test, y_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC-AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ({strategy}) — ROC Curve")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "roc_curve.png"), dpi=150)
    plt.close()


def _save_roc_data(y_test, y_scores, run_dir):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }
    with open(os.path.join(run_dir, "roc_curve_data.json"), "w") as f:
        json.dump(data, f)
