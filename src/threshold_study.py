"""
Threshold Sensitivity Study
============================

The command-line interface derives the four rules from explicitly pinned saved
validation predictions. It never refits a model, chooses thresholds on TEST,
or overwrites source-run artefacts. Missing validation evidence is an error.

Usage:
    python src/threshold_study.py --dataset ulb --manifest <manifest.json> \\
        --output-dir <revision-derived-directory>

The older classical reconstruction helpers are retained for source-history
traceability, but are not invoked by the CLI. FT reconstruction is disabled:
its thresholds must come from the saved best-checkpoint validation scores.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from data import load_dataset, DATASET_REGISTRY
from preprocess import get_preprocessor
from evaluation.metrics import (
    find_threshold_maximizing_f1,
    find_threshold_maximizing_f2,
    find_threshold_at_min_precision,
    compute_all_metrics,
)


# ── Constants (must match main.py exactly) ────────────────────────────────
SPLIT_SEED = 42
CV_SPLITS = 5
CV = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SPLIT_SEED)

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

# Per-dataset model lists
MODEL_ORDER_BY_DATASET = {
    "ulb":      ["logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm"],
    "baf_base": ["logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm"],
}

# Dataset label used in results/ directory names
DATASET_LABEL = {
    "ulb":      "ulb_2013",
    "baf_base": "baf_base",
}

# Threshold strategies
STRATEGIES = ["fixed_05", "max_f1", "max_f2", "prec_ge_05"]
STRATEGY_LABELS = {
    "fixed_05":   "Fixed (0.5)",
    "max_f1":     "max-F1",
    "max_f2":     "max-F2",
    "prec_ge_05": "Prec≥0.5",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def find_latest_run(dataset_label, model_name):
    """Return Path to the latest run directory for a given model."""
    base = RESULTS_ROOT / dataset_label / model_name / "none"
    if not base.exists():
        return None
    runs = sorted(base.iterdir())
    return runs[-1] if runs else None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    """Save dict as JSON, converting numpy types."""
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(v) for v in o]
        return o

    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=2)


def compute_fold_thresholds(y_val, y_val_scores):
    """
    Compute τ for all 4 strategies on a single validation fold.

    Returns dict: strategy_name → threshold value.
    """
    tau_f1, _ = find_threshold_maximizing_f1(y_val, y_val_scores)
    tau_f2, _ = find_threshold_maximizing_f2(y_val, y_val_scores)
    tau_prec, _ = find_threshold_at_min_precision(y_val, y_val_scores, min_precision=0.5)

    return {
        "fixed_05":   0.5,
        "max_f1":     tau_f1,
        "max_f2":     tau_f2,
        "prec_ge_05": tau_prec,
    }


def apply_threshold(y_true, y_scores, threshold):
    """Compute F2, F1, Recall, Precision, Alert Rate for a given threshold."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if np.isinf(threshold):
        # No threshold satisfies the constraint → no predictions
        return {
            "F2": 0.0,
            "F1": 0.0,
            "Recall": 0.0,
            "Precision": 0.0,
            "alert_rate": 0.0,
            "threshold": float("inf"),
        }

    m = compute_all_metrics(y_true, y_scores, threshold)

    tp, fp, fn = m["TP"], m["FP"], m["FN"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "F2": m["F2"],
        "F1": m["F1"],
        "Recall": round(rec, 6),
        "Precision": round(prec, 6),
        "alert_rate": m["alert_rate"],
        "threshold": m["threshold"],
    }


# ── Per-model study ──────────────────────────────────────────────────────

def run_threshold_study_supervised(model_name, X_train, X_test, y_train, y_test,
                                    preprocessor, run_dir):
    """
    Threshold study for a supervised model.

    Steps:
      1. Load saved model (model.joblib)
      2. Rebuild 5-fold CV with same params → per-fold τ for each strategy
      3. τ_final = median(per-fold τ) for each strategy
      4. Score test set with saved model
      5. Apply each τ_final → compute all metrics
    """
    print(f"\n{'─' * 50}")
    print(f"  {model_name.upper()} — Threshold Study")
    print(f"{'─' * 50}")

    # Load saved model and config
    final_model = joblib.load(run_dir / "model.joblib")
    config = load_json(run_dir / "config.json")
    best_params = config["best_params"]

    print(f"  Run dir  : {run_dir}")
    print(f"  Params   : {best_params}")

    # ── Step 1: 5-fold CV to compute per-fold thresholds ──────────
    print(f"\n  [1/3] Computing per-fold thresholds ...")

    per_fold = {s: [] for s in STRATEGIES}

    for i, (train_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Rebuild fold model with best params
        fold_model = clone(final_model)
        fold_model.fit(X_fold_train, y_fold_train)
        y_val_scores = fold_model.predict_proba(X_fold_val)[:, 1]

        fold_taus = compute_fold_thresholds(y_fold_val, y_val_scores)

        for s in STRATEGIES:
            per_fold[s].append(fold_taus[s])

        print(f"    Fold {i+1}: "
              f"F1-τ={fold_taus['max_f1']:.6f}  "
              f"F2-τ={fold_taus['max_f2']:.6f}  "
              f"Prec-τ={fold_taus['prec_ge_05']:.6f}")

    # Median τ per strategy
    tau_final = {}
    for s in STRATEGIES:
        tau_final[s] = float(np.median(per_fold[s]))

    print(f"\n  Median thresholds:")
    for s in STRATEGIES:
        print(f"    {STRATEGY_LABELS[s]:<15s} : τ = {tau_final[s]:.6f}")

    # ── Step 2: Score test set ────────────────────────────────────
    print(f"\n  [2/3] Scoring test set ...")
    y_test_scores = final_model.predict_proba(X_test)[:, 1]

    # ── Step 3: Apply each threshold & compute metrics ────────────
    print(f"\n  [3/3] Applying thresholds to test set ...")
    results = {}
    for s in STRATEGIES:
        res = apply_threshold(y_test, y_test_scores, tau_final[s])
        results[s] = res
        print(f"    {STRATEGY_LABELS[s]:<15s} : "
              f"F2={res['F2']:.4f}  F1={res['F1']:.4f}  "
              f"Recall={res['Recall']:.4f}  Alert={res['alert_rate']:.4%}")

    return {
        "model": model_name,
        "thresholds_per_fold": {s: per_fold[s] for s in STRATEGIES},
        "thresholds_median": tau_final,
        "test_results": results,
    }


def run_threshold_study_ocsvm(X_train, X_test, y_train, y_test,
                               preprocessor, run_dir):
    """
    Threshold study for OCSVM (anomaly detector).

    Same protocol but:
      - Trained only on class-0 samples per fold
      - Scores = negated decision_function
    """
    print(f"\n{'─' * 50}")
    print(f"  OCSVM — Threshold Study")
    print(f"{'─' * 50}")

    final_model = joblib.load(run_dir / "model.joblib")
    print(f"  Run dir  : {run_dir}")

    # ── Step 1: 5-fold CV ─────────────────────────────────────────
    print(f"\n  [1/3] Computing per-fold thresholds ...")

    per_fold = {s: [] for s in STRATEGIES}

    for i, (train_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Train only on class 0
        X_fold_train_0 = X_fold_train[y_fold_train == 0]
        fold_model = clone(final_model)
        fold_model.fit(X_fold_train_0)

        # Negate: higher → more anomalous
        y_val_scores = -fold_model.decision_function(X_fold_val)

        fold_taus = compute_fold_thresholds(y_fold_val, y_val_scores)

        for s in STRATEGIES:
            per_fold[s].append(fold_taus[s])

        print(f"    Fold {i+1}: "
              f"F1-τ={fold_taus['max_f1']:.6f}  "
              f"F2-τ={fold_taus['max_f2']:.6f}  "
              f"Prec-τ={fold_taus['prec_ge_05']:.6f}")

    tau_final = {}
    for s in STRATEGIES:
        tau_final[s] = float(np.median(per_fold[s]))

    print(f"\n  Median thresholds:")
    for s in STRATEGIES:
        print(f"    {STRATEGY_LABELS[s]:<15s} : τ = {tau_final[s]:.6f}")

    # ── Step 2: Score test set ────────────────────────────────────
    print(f"\n  [2/3] Scoring test set ...")
    y_test_scores = -final_model.decision_function(X_test)

    # ── Step 3: Apply each threshold ──────────────────────────────
    print(f"\n  [3/3] Applying thresholds to test set ...")
    results = {}
    for s in STRATEGIES:
        res = apply_threshold(y_test, y_test_scores, tau_final[s])
        results[s] = res
        print(f"    {STRATEGY_LABELS[s]:<15s} : "
              f"F2={res['F2']:.4f}  F1={res['F1']:.4f}  "
              f"Recall={res['Recall']:.4f}  Alert={res['alert_rate']:.4%}")

    return {
        "model": "ocsvm",
        "thresholds_per_fold": {s: per_fold[s] for s in STRATEGIES},
        "thresholds_median": tau_final,
        "test_results": results,
    }


def run_threshold_study_fttransformer(X_train, y_train, y_test, run_dir):
    """Derive thresholds from saved validation evidence; never reconstruct a fit."""
    from revision_thresholds import study_saved_run
    config = load_json(run_dir / "config.json")
    return study_saved_run(run_dir, config["dataset"], "fttransformer")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Threshold Sensitivity Study"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_LABEL.keys()),
        required=True,
        help="Dataset to use: ulb | baf_base",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to analyse (default: all). Valid: logreg rf lgbm catboost ocsvm",
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Pinned revision baseline runs with saved validation evidence")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New derived output directory; source runs are never modified")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    from revision_thresholds import generate
    selected_models = MODEL_ORDER_BY_DATASET[args.dataset] if "all" in args.models else args.models
    print(json.dumps(generate(args.manifest, DATASET_LABEL[args.dataset],
                              args.output_dir, selected_models), indent=2))


if __name__ == "__main__":
    main()
