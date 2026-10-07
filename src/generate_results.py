"""
Generate thesis-ready tables (LaTeX) and figures (PDF) from actual results.

Revision usage requires explicitly pinned sources and an isolated output root.
The complete grid and derived evidence are checked before any file is written.
Historical helpers remain for traceability, but the CLI cannot overwrite archives.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TABLES_BASE = ROOT / "results_thesis" / "tables"
FIGURES_BASE = ROOT / "results_thesis" / "figures"
RUN_MANIFEST = None
THRESHOLD_STUDY_ROOT = None
CROSS_DOMAIN_ROOT = None
FULL_PRECISION_CACHE = {}

def _tables_dir(filename):
    d = TABLES_BASE / filename
    d.mkdir(parents=True, exist_ok=True)
    return d

def _figures_dir(filename):
    d = FIGURES_BASE / filename
    d.mkdir(parents=True, exist_ok=True)
    return d

def _emit(out: Path):
    """Print generated file path."""
    print(f"  → {out}")

# ── display ordering & names ─────────────────────────────────────────────
MODEL_ORDER = ["logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm"]
MODEL_LABELS = {
    "logreg": "LR",
    "rf": "RF",
    "lgbm": "LGBM",
    "catboost": "CatBoost",
    "fttransformer": "FT-Trans.",
    "ocsvm": "OCSVM",
}
MODEL_COLORS = {
    "logreg":  "#9467bd",
    "rf":      "#2ca02c",
    "lgbm":    "#1f77b4",
    "catboost": "#d62728",
    "fttransformer": "#ff7f0e",
    "ocsvm":   "#7f7f7f",
}


# ── helpers ──────────────────────────────────────────────────────────────
def find_latest_run(dataset, model, strategy="none"):
    """Resolve an explicit run, or the sole historical run; never guess by date."""
    if RUN_MANIFEST is not None:
        key = f"{dataset}/{model}/{strategy}"
        selected = RUN_MANIFEST.get("runs", {}).get(key)
        if selected is None and strategy == "none":
            selected = RUN_MANIFEST.get("baseline_runs", {}).get(f"{dataset}/{model}")
        if selected is None and dataset == "baf_base" and model != "fttransformer":
            selected = RUN_MANIFEST.get("historical_runs", {}).get(key)
        if selected is None:
            return None
        path = Path(selected["run_dir"] if isinstance(selected, dict) else selected)
        path = path if path.is_absolute() else ROOT / path
        if not path.is_dir():
            raise FileNotFoundError(f"Manifest-selected run does not exist: {path}")
        if isinstance(selected, dict) and selected.get("config_sha256"):
            from generate_revision_interpretability import file_digest
            if file_digest(path / "config.json") != selected["config_sha256"]:
                raise ValueError(f"Manifest-selected configuration hash changed: {path}")
        return path
    base = RESULTS_DIR / dataset / model / strategy
    if not base.exists():
        return None
    runs = sorted(path for path in base.iterdir() if path.is_dir() and path.name.startswith("run_"))
    if len(runs) > 1:
        raise ValueError(f"Multiple runs at {base}; provide an explicit --manifest")
    return runs[0] if runs else None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def full_precision_metrics(run):
    """Recover metric precision without re-fitting or selecting a TEST threshold.

    Ranking metrics use the complete saved scores. Operating metrics use the
    saved integer confusion matrix, avoiding the loss of precision in six-place
    JSON fields. Current exact thresholds must reproduce those counts. Older
    runs may only retain a rounded threshold, so their original counts, not a
    newly rounded operating decision, remain authoritative.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    run = Path(run).resolve()
    if run in FULL_PRECISION_CACHE:
        return FULL_PRECISION_CACHE[run]
    config = load_json(run / "config.json")
    metrics = load_json(run / "metrics_test.json")
    labels = np.load(run / "y_test.npy", allow_pickle=False)
    scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError(f"Invalid full-precision metric evidence: {run}")
    tp, fp, tn, fn = (int(metrics[name]) for name in ("TP", "FP", "TN", "FN"))
    if tp + fp + tn + fn != len(labels) or tp + fn != int(labels.sum()):
        raise ValueError(f"Saved confusion counts do not match the TEST population: {run}")
    threshold = float(config.get("threshold_exact", metrics["threshold"]))
    if "threshold_exact" in config:
        decisions = scores >= threshold
        counts = (int(np.sum(decisions & (labels == 1))), int(np.sum(decisions & (labels == 0))),
                  int(np.sum(~decisions & (labels == 0))), int(np.sum(~decisions & (labels == 1))))
        if counts != (tp, fp, tn, fn):
            raise ValueError(f"Exact saved threshold changes the operating confusion matrix: {run}")
    full = dict(metrics)
    full["PR-AUC"] = float(average_precision_score(labels, scores))
    full["ROC-AUC"] = float(roc_auc_score(labels, scores))
    for name in ("PR-AUC", "ROC-AUC"):
        if not np.isclose(full[name], metrics[name], atol=1e-6, rtol=0):
            raise ValueError(f"Saved {name} does not match full score arrays: {run}")
    full["F1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    full["F2"] = 5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0.0
    full["alert_rate"] = (tp + fp) / len(labels)
    full["FP/TP"] = fp / tp if tp else float("inf")
    full["threshold"] = threshold
    k = int(metrics["k_used"])
    if not 1 <= k <= len(labels):
        raise ValueError(f"Invalid saved workload size: {run}")
    top_positives = int(labels[np.argsort(scores)[::-1][:k]].sum())
    full["precision_at_k"] = top_positives / k
    full["recall_at_k"] = top_positives / int(labels.sum())
    if config["model"] == "ocsvm":
        full["brier_score"] = None  # Anomaly scores are not calibrated probabilities.
    else:
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Supervised probability scores are outside [0, 1]: {run}")
        full["brier_score"] = float(brier_score_loss(labels, scores))
    FULL_PRECISION_CACHE[run] = full
    return full


def validated_bootstrap_interval(metrics, name):
    """Require a real saved finite interval; never replace absent evidence by zero."""
    try:
        interval = metrics["bootstrap_ci"][name]
    except KeyError as error:
        raise ValueError(f"Missing saved bootstrap interval: {name}") from error
    if (len(interval) != 2 or not np.isfinite(interval).all()
            or not 0 <= interval[0] <= interval[1] <= 1):
        raise ValueError(f"Invalid saved bootstrap interval {name}: {interval}")
    return interval


def computational_cost_basis(config):
    """Keep historical final-fit/search costs separate from validation recovery."""
    recovered = config.get("validation_only_recovery") is True
    if recovered:
        for name in ("historical_tuning_time_s", "historical_train_time_s",
                     "historical_cost_source_run", "validation_recovery_time_s", "train_time_source"):
            if name not in config:
                raise ValueError(f"Recovered baseline lacks cost provenance: {name}")
        recovery_time = config["validation_recovery_time_s"]
        if not np.isfinite(recovery_time) or recovery_time < 0:
            raise ValueError("Validation recovery time must be a finite non-negative separate cost")
    tuning = config["historical_tuning_time_s"] if recovered else config.get("tuning_time_s")
    train = config["historical_train_time_s"] if recovered else config.get("train_time_s")
    infer = config.get("infer_time_s")
    infer_historical = "historical" in str(config.get("infer_time_source", "")).lower()
    checkpoint = config.get("optuna_sampler_checkpoint") or {}
    sampler_recovery = config.get("optuna_sampler_recovery") or {}
    resumed = bool(checkpoint.get("loaded_from") or sampler_recovery.get("loaded_from")
                   or sampler_recovery.get("reconstructed_trials", 0) > 0)
    for name, value in (("tuning", tuning), ("training", train), ("inference", infer)):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"Invalid recorded {name} cost: {value}")
    if train is None or infer is None:
        raise ValueError("Recorded training and inference times are required")
    return {"tuning_time_s": tuning, "train_time_s": train, "infer_time_s": infer,
            "tuning_historical": recovered and tuning is not None and tuning > 0,
            "tuning_partial_wall_time": resumed and not recovered and tuning is not None and tuning > 0,
            "train_historical": recovered, "infer_historical": infer_historical,
            "validation_recovery_time_s": config.get("validation_recovery_time_s"),
            "historical_cost_source_run": config.get("historical_cost_source_run"),
            "tuning_source": ("historical recorded search" if recovered and tuning else
                              "resumed invocation only; earlier search wall time excluded" if resumed and tuning else
                              "current recorded search" if tuning else "no new search; fixed or untuned"),
            "scope": {"tuning": ("historical_recorded_search" if recovered and tuning else
                                   "resumed_invocation_only_not_full_trial_budget" if resumed and tuning else
                                   "current_search_invocation" if tuning else "no_new_search"),
                      "final_fit": "historical_saved_full_DEV_fit" if recovered else "current_full_DEV_fit",
                      "inference": "historical_saved_TEST_inference" if infer_historical else "current_TEST_inference",
                      "validation_recovery": "current_fixed_HP_validation_only" if recovered else "not_applicable"},
            "resume_evidence": {"optuna_sampler_checkpoint": checkpoint, "optuna_sampler_recovery": sampler_recovery},
            "train_time_source": config.get("train_time_source", "current full-DEV fit"),
            "infer_time_source": config.get("infer_time_source", "current TEST inference")}


def verify_saved_shap_matrix(values, encoded_names, original_names, declared_importance,
                             expected_logit, scores, batch_size=4096):
    """Check stored float32 attributions against full importance and raw logits."""
    from generate_revision_interpretability import group_contributions
    if values.shape != (len(scores), len(encoded_names)) or set(declared_importance) != set(original_names):
        raise ValueError("Stored SHAP dimensions or original-feature magnitude keys disagree")
    if not np.isfinite(scores).all() or np.any((scores <= 0) | (scores >= 1)):
        raise ValueError("SHAP log-odds verification requires probabilities strictly between zero and one")
    sums = np.zeros(len(original_names), dtype=np.float64)
    additivity_error = 0.0
    for start in range(0, len(scores), batch_size):
        stop = min(start + batch_size, len(scores))
        chunk = np.asarray(values[start:stop], dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError("Stored SHAP matrix contains non-finite attributions")
        grouped, names = group_contributions(chunk, encoded_names, absolute=True)
        if names != original_names:
            raise ValueError("Stored SHAP categorical grouping differs from the declared schema")
        sums += grouped.sum(axis=0)
        logits = np.log(scores[start:stop]) - np.log1p(-scores[start:stop])
        additivity_error = max(additivity_error, float(np.max(np.abs(expected_logit + chunk.sum(axis=1) - logits))))
    calculated = sums / len(scores)
    declared = np.array([declared_importance[name] for name in original_names])
    if not np.allclose(calculated, declared, rtol=2e-7, atol=1e-8):
        raise ValueError("Declared SHAP importance is not reproduced by the complete stored attribution matrix")
    # Encoded contributions are deliberately stored as float32 after exact
    # float64 importance accumulation; verification allows only that rounding.
    if additivity_error > 1e-5:
        raise ValueError("Stored SHAP matrix does not reproduce the frozen model log-odds")
    return {"maximum_absolute_importance_difference": float(np.max(np.abs(calculated - declared))),
            "maximum_stored_matrix_logit_additivity_error": additivity_error,
            "storage_verification_logit_tolerance": 1e-5}


def verify_transfer_evidence(directory, record, model):
    """Reproduce corrected transfer metrics from immutable, aligned score arrays."""
    from experiment_protocol import array_sha256
    from evaluation.metrics import compute_all_metrics
    from generate_revision_interpretability import file_digest
    from sklearn.metrics import average_precision_score, roc_auc_score

    directory = Path(directory).resolve()
    required = ("y_test.npy", "y_test_scores.npy", "test_row_indices.npy",
                "original_y_test.npy", "original_y_test_scores.npy", "original_test_row_indices.npy")
    artefacts = record.get("artefacts", {})
    if set(artefacts) != set(required):
        raise ValueError("Transfer evidence must retain corrected and original labels, scores and row indices")
    arrays, snapshots = {}, {}
    for name in required:
        path = directory / name
        evidence = artefacts[name]
        digest = file_digest(path)
        if digest != evidence.get("file_sha256"):
            raise ValueError(f"Transfer array file hash changed: {path}")
        values = np.load(path, allow_pickle=False)
        if (list(values.shape) != evidence.get("shape") or str(values.dtype) != evidence.get("dtype")
                or array_sha256(values) != evidence.get("array_sha256")):
            raise ValueError(f"Transfer array content or schema differs: {path}")
        arrays[name] = values
        snapshots[name] = {"path": str(path), "sha256": digest}
    labels, scores, indices = (arrays[name] for name in required[:3])
    if (labels.ndim != 1 or scores.shape != labels.shape or indices.shape != labels.shape
            or set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all()
            or not np.issubdtype(indices.dtype, np.integer) or len(np.unique(indices)) != len(indices)):
        raise ValueError("Corrected transfer TEST arrays are not finite, binary or position-aligned")
    if model != "ocsvm" and np.any((scores < 0) | (scores > 1)):
        raise ValueError("Supervised transfer scores must be probabilities in [0, 1]")
    original_labels, original_scores, original_indices = (arrays[name] for name in required[3:])
    if (original_labels.ndim != 1 or original_scores.shape != original_labels.shape
            or original_indices.shape != original_labels.shape or set(np.unique(original_labels)) != {0, 1}
            or not np.isfinite(original_scores).all() or not np.issubdtype(original_indices.dtype, np.integer)
            or len(np.unique(original_indices)) != len(original_indices)):
        raise ValueError("Original transfer TEST arrays are not finite, binary or position-aligned")
    if model != "ocsvm" and np.any((original_scores < 0) | (original_scores > 1)):
        raise ValueError("Original supervised transfer scores must be probabilities in [0, 1]")
    positions = {int(value): position for position, value in enumerate(original_indices)}
    try:
        kept_positions = np.array([positions[int(value)] for value in indices], dtype=np.int64)
    except KeyError as error:
        raise ValueError("A corrected transfer index is absent from its original TEST population") from error
    if (not np.array_equal(labels, original_labels[kept_positions])
            or not np.array_equal(scores, original_scores[kept_positions])):
        raise ValueError("Corrected transfer labels or scores differ from the retained original observations")
    population = record["population"]
    if (population["rows"] != len(labels) or population["positive_rows"] != int(labels.sum())
            or population["negative_rows"] != len(labels) - int(labels.sum())):
        raise ValueError("Corrected transfer population counts do not match the saved labels")
    threshold = float(record["threshold_used"])
    if not np.isfinite(threshold):
        raise ValueError("Corrected transfer requires a finite frozen Base threshold")
    calculated = compute_all_metrics(labels, scores, threshold)
    metrics = dict(record["metrics"])
    for name in ("PR-AUC", "ROC-AUC", "F1", "F2", "TP", "FP", "TN", "FN", "alert_rate",
                 "precision_at_k", "recall_at_k", "k_used"):
        if abs(float(metrics[name]) - float(calculated[name])) > 1e-6:
            raise ValueError(f"Corrected transfer metric does not reproduce its saved scores: {name}")
    average_precision = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))
    exact = record["metrics_full_precision"]
    for name, value in (("average_precision", average_precision), ("roc_auc", roc_auc), ("threshold", threshold)):
        if not np.isclose(exact[name], value, rtol=0, atol=1e-12):
            raise ValueError(f"Corrected transfer full-precision metric differs: {name}")
    tp, fp, fn = (int(calculated[name]) for name in ("TP", "FP", "FN"))
    metrics.update({"PR-AUC": average_precision, "ROC-AUC": roc_auc,
                    "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                    "F2": 5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0.0,
                    "alert_rate": (tp + fp) / len(labels)})
    return metrics, snapshots


def collect_paired_evidence(manifest_root):
    """Verify fixed-policy controls without fitting or recalculating intervals."""
    from experiment_protocol import file_sha256
    from revision_paired_analysis import verify_paired_report
    paired_path = manifest_root / "derived/paired/paired_analysis.json"
    sensitivity_path = manifest_root / "sensitivity/revision_manifest.json"
    paired_digest = file_sha256(paired_path)
    sensitivity_digest = file_sha256(sensitivity_path)
    report = load_json(paired_path)
    if report.get("sensitivity_manifest_sha256") != sensitivity_digest:
        raise ValueError("The predefined sensitivity manifest changed after paired analysis")
    verify_paired_report(report, RUN_MANIFEST, load_json(sensitivity_path))
    if (file_sha256(paired_path) != paired_digest
            or file_sha256(sensitivity_path) != sensitivity_digest):
        raise ValueError("Paired evidence changed during verification")
    return {"path": str(paired_path.resolve()), "sha256": paired_digest,
            "sensitivity_manifest": {"path": str(sensitivity_path.resolve()),
                                     "sha256": sensitivity_digest}}


def verify_test_index_provenance(run, config, manifest):
    """Bind saved TEST row positions to both original provenance pins.

    This is an operational source-integrity check only. It does not reconstruct
    a split, fit a model or alter any metric or operating threshold.
    """
    from experiment_protocol import array_sha256
    indices = np.load(Path(run) / "test_row_indices.npy", allow_pickle=False)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) != config["test_samples"]):
        raise ValueError("Saved TEST indices have an invalid shape, type or population size.")
    digest = array_sha256(indices)
    source_digest = (config.get("data_provenance") or {}).get("test_indices_sha256")
    manifest_digest = manifest.get("datasets", {}).get(config["dataset"], {}).get("test_indices_sha256")
    if not source_digest or not manifest_digest or digest != source_digest or digest != manifest_digest:
        raise ValueError("Saved TEST indices differ from their configuration or manifest provenance pin.")


def collect_bootstrap_compatibility(manifest, source_runs):
    """Capture only the verified missing-field exception, never a default budget."""
    from revision_bootstrap_compatibility import KEY, verify_bootstrap_compatibility
    source = source_runs.get(KEY)
    if source is None:
        return {}
    run = Path(source["run_dir"])
    run = (run if run.is_absolute() else ROOT / run).resolve()
    config = load_json(run / "config.json")
    if "bootstrap_iterations" in config:
        if config["bootstrap_iterations"] != 1000:
            raise ValueError("The ULB FT baseline has an explicitly incompatible bootstrap budget.")
        return {}
    reference = verify_bootstrap_compatibility(manifest, run, config, expected_iterations=1000)
    return {KEY: reference}


def preflight_revision():
    """Reject incomplete or stale revision sources before generating any output."""
    from experiment_protocol import PROTOCOL_VERSION
    from generate_revision_interpretability import file_digest
    if RUN_MANIFEST is None:
        raise ValueError("Revision preflight requires an explicit manifest")
    manifest_root = Path(RUN_MANIFEST.get("results_root", ""))
    if not str(manifest_root) or str(manifest_root) == ".":
        raise ValueError("The revision manifest must declare results_root")
    manifest_root = manifest_root.resolve() if manifest_root.is_absolute() else (ROOT / manifest_root).resolve()
    if RESULTS_DIR.resolve() != manifest_root:
        raise ValueError("--results-root must match the manifest's revision results_root")
    if THRESHOLD_STUDY_ROOT is None or CROSS_DOMAIN_ROOT is None:
        raise ValueError("Revision preflight requires threshold and cross-domain roots")
    for source_root in (THRESHOLD_STUDY_ROOT.resolve(), CROSS_DOMAIN_ROOT.resolve()):
        if manifest_root not in source_root.parents:
            raise ValueError("Derived threshold and transfer sources must belong to the isolated revision")
    missing = []
    selected = {}
    for dataset in ("ulb_2013", "baf_base"):
        for model in MODEL_ORDER:
            for strategy in (["none"] if model == "ocsvm" else BALANCE_ORDER):
                key = f"{dataset}/{model}/{strategy}"
                run = find_latest_run(dataset, model, strategy)
                if run is None:
                    missing.append(key)
                else:
                    selected[key] = run.resolve()
    if missing:
        raise FileNotFoundError("Revision grid is incomplete; no files generated. Missing pins: " + ", ".join(missing))
    report = {"revision_id": RUN_MANIFEST.get("revision_id"), "required_runs": len(selected),
              "metric_precision_basis": {"ranking": "Complete saved TEST score arrays",
                                         "operating": "Integer saved confusion counts; exact current thresholds verified",
                                         "strategy_population_sd": "Full-precision AP and count-derived F2, ddof=0; not random-seed variability"},
              "source_runs": {}, "threshold_studies": {}, "transfer_sources": {}}
    execution_note = RUN_MANIFEST.get("execution_provenance_notes")
    if execution_note:
        note_path = Path(execution_note["path"])
        note_path = note_path.resolve() if note_path.is_absolute() else (ROOT / note_path).resolve()
        note_digest = file_digest(note_path)
        if note_digest != execution_note["sha256"]:
            raise ValueError("Execution provenance notes changed after their manifest pin")
        report["execution_provenance_notes"] = {"path": str(note_path), "sha256": note_digest}
    labels_by_dataset = {}
    raw_hashes_by_dataset = {}
    for key, run in selected.items():
        dataset, model, strategy = key.split("/")
        config = load_json(run / "config.json")
        normalised_strategy = "none" if config.get("strategy") == "n/a" else config.get("strategy")
        if (config.get("dataset"), config.get("model"), normalised_strategy) != (dataset, model, strategy):
            raise ValueError(f"Pinned source identity disagrees with {key}: {run}")
        if config.get("missing_policy", "preserve") != "preserve":
            raise ValueError(f"Primary tables cannot silently include a missingness sensitivity run: {key}")
        provenance = config.get("data_provenance") or {}
        if config.get("sample_fraction") not in (None, 1, 1.0) or provenance.get("sample_fraction") not in (None, 1, 1.0):
            raise ValueError(f"Primary tables cannot include a subsample diagnostic: {key}")
        dataset_hash = config.get("dataset_hash_sha256")
        if not dataset_hash or raw_hashes_by_dataset.setdefault(dataset, dataset_hash) != dataset_hash:
            raise ValueError(f"Raw dataset hashes differ within the {dataset} comparison")
        current_required = dataset == "ulb_2013" or model == "fttransformer"
        if current_required and (manifest_root not in run.parents or config.get("protocol_version") != PROTOCOL_VERSION):
            raise ValueError(f"Corrected {key} cannot fall back to a historical or different-protocol run")
        if dataset == "ulb_2013" and provenance.get("deduplication", {}).get("policy") != "exact_predictor_deduplication_before_all_splits":
            raise ValueError(f"Corrected ULB provenance does not declare pre-split exact-profile deduplication: {key}")
        required = ["metrics_cv.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy"]
        if strategy == "none":
            required.append("pr_curve_data.json")
        if current_required:
            required += ["completed.json", "test_row_indices.npy"]
            if load_json(run / "completed.json").get("status") != "complete":
                raise ValueError(f"Pinned revision run is not complete: {run}")
        for name in required:
            if not (run / name).is_file():
                raise FileNotFoundError(run / name)
        labels = np.load(run / "y_test.npy", allow_pickle=False)
        scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
        if labels.ndim != 1 or scores.shape != labels.shape or set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
            raise ValueError(f"Invalid TEST evidence for {key}")
        if len(labels) != config["test_samples"] or int(labels.sum()) != config["test_fraud"]:
            raise ValueError(f"TEST population and configuration disagree for {key}")
        if config.get("protocol_version") == PROTOCOL_VERSION:
            verify_test_index_provenance(run, config, RUN_MANIFEST)
        previous = labels_by_dataset.setdefault(dataset, labels)
        if not np.array_equal(previous, labels):
            raise ValueError(f"TEST label order is not shared across the {dataset} comparison")
        metrics = load_json(run / "metrics_test.json")
        if strategy == "none":
            for interval_name in ("PR-AUC_ci", "ROC-AUC_ci", "F2_ci"):
                validated_bootstrap_interval(metrics, interval_name)
            computational_cost_basis(config)
        from evaluation.metrics import compute_all_metrics
        exact = float(config.get("threshold_exact", metrics["threshold"]))
        recomputed = compute_all_metrics(labels, scores, exact)
        for name in ("PR-AUC", "ROC-AUC", "F1", "F2", "TP", "FP", "TN", "FN", "alert_rate", "precision_at_k", "recall_at_k", "k_used"):
            if abs(float(metrics[name]) - float(recomputed[name])) > 1e-6:
                raise ValueError(f"Saved {key} {name} fails numerical verification")
        report["source_runs"][key] = {"run_dir": str(run), "config_sha256": file_digest(run / "config.json"),
                                      "metrics_sha256": file_digest(run / "metrics_test.json"),
                                      "metrics_cv_sha256": file_digest(run / "metrics_cv.json"),
                                      "pr_curve_data_sha256": (file_digest(run / "pr_curve_data.json")
                                                               if (run / "pr_curve_data.json").is_file() else None),
                                      "y_test_sha256": file_digest(run / "y_test.npy"),
                                      "y_test_scores_sha256": file_digest(run / "y_test_scores.npy"),
                                      "test_row_indices_sha256": (file_digest(run / "test_row_indices.npy")
                                                                  if (run / "test_row_indices.npy").is_file() else None),
                                      "test_samples": len(labels), "test_fraud_samples": int(labels.sum())}
        if strategy == "none":
            report["source_runs"][key]["computational_cost_basis"] = computational_cost_basis(config)
    report["bootstrap_compatibility"] = collect_bootstrap_compatibility(RUN_MANIFEST, report["source_runs"])
    for dataset in ("ulb_2013", "baf_base"):
        for model in MODEL_ORDER:
            path = THRESHOLD_STUDY_ROOT / dataset / model / "threshold_study.json"
            study = load_json(path)
            baseline = selected[f"{dataset}/{model}/none"]
            if Path(study["source_run"]).resolve() != baseline:
                raise ValueError(f"Threshold evidence does not belong to the pinned {dataset}/{model} baseline")
            if set(study.get("test_results", {})) != set(STRATEGY_ORDER):
                raise ValueError(f"Four threshold rules are required: {path}")
            config = load_json(baseline / "config.json")
            if not np.isclose(study["thresholds_median"]["max_f2"], config["threshold_exact"], rtol=0, atol=1e-12):
                raise ValueError(f"Primary validation threshold changed: {path}")
            report["threshold_studies"][f"{dataset}/{model}"] = {"path": str(path), "sha256": file_digest(path)}
    transfer_manifest = load_json(CROSS_DOMAIN_ROOT / "transfer_manifest.json")
    if transfer_manifest.get("status") != "complete" or transfer_manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("A completed corrected transfer manifest is required")
    if transfer_manifest.get("threshold_selection_on_variants") is not False:
        raise ValueError("Corrected transfer must freeze Base thresholds before Variant evaluation")
    from revision_transfer import load_partition_manifest
    partition_root = Path(transfer_manifest["partition_directory"]).resolve()
    if file_digest(partition_root / "partition_manifest.json") != transfer_manifest["partition_manifest_sha256"]:
        raise ValueError("The corrected transfer partition manifest changed")
    load_partition_manifest(partition_root)
    transfer_all = collect_cross_domain()
    for model in MODEL_ORDER:
        path = CROSS_DOMAIN_ROOT / model / "source_audit.json"
        transfer = transfer_all.get(model, {})
        source_audit = load_json(path)
        if source_audit != transfer_manifest.get("models", {}).get(model):
            raise ValueError(f"Transfer source audit disagrees with its completed manifest: {model}")
        baseline = selected[f"baf_base/{model}/none"]
        if Path(source_audit["source_run"]).resolve() != baseline:
            raise ValueError(f"Transfer does not use the explicitly pinned corrected {model} baseline")
        baseline_config = load_json(baseline / "config.json")
        if float(source_audit["threshold_used"]) != float(baseline_config["threshold_exact"]):
            raise ValueError(f"Transfer does not freeze the exact primary {model} threshold")
        if any(variant not in transfer for variant in VARIANT_ORDER):
            raise ValueError(f"Corrected transfer requires Base and all five Variants: {path}")
        for variant in VARIANT_ORDER:
            if not all(name in transfer[variant]["metrics"] for name in ("PR-AUC", "ROC-AUC", "F2")):
                raise ValueError(f"Incomplete transfer metrics: {path} / {variant}")
        variants = {}
        for variant in VARIANT_ORDER[1:]:
            metrics_path = CROSS_DOMAIN_ROOT / model / variant / "metrics_test.json"
            record = load_json(metrics_path)
            unused_metrics, snapshots = verify_transfer_evidence(metrics_path.parent, record, model)
            partition_dir = partition_root / variant
            for filename, prepared_name in (("y_test.npy", "kept_test_y.npy"),
                                            ("test_row_indices.npy", "kept_test_row_indices.npy")):
                if not np.array_equal(np.load(metrics_path.parent / filename, allow_pickle=False),
                                      np.load(partition_dir / prepared_name, allow_pickle=False)):
                    raise ValueError(f"Corrected transfer differs from its predeclared cohort: {model}/{variant}")
            variants[variant] = {"path": str(metrics_path), "sha256": file_digest(metrics_path),
                                  "artefacts": snapshots}
        report["transfer_sources"][model] = {"path": str(path), "sha256": file_digest(path), "variants": variants}
    report["compatible_shap_sources"] = {}
    for model in SHAP_MODELS:
        run = find_shap_run(model)
        artefacts = {}
        for name in ("config.json", "model.joblib", "shap_values.npy", "shap_global.json", "shap_local_cases.json",
                     "y_test.npy", "y_test_scores.npy"):
            if not (run / name).is_file():
                raise FileNotFoundError(run / name)
            artefacts[name] = {"path": str((run / name).resolve()), "sha256": file_digest(run / name)}
        report["compatible_shap_sources"][model] = {"run_dir": str(run.resolve()), "artefacts": artefacts}
    stability_pin = RUN_MANIFEST.get("interpretability", {}).get("variant_stability")
    if not stability_pin:
        raise ValueError("Corrected Variant SHAP stability must be explicitly pinned before final generation")
    stability_path = Path(stability_pin)
    stability_path = stability_path.resolve() if stability_path.is_absolute() else (ROOT / stability_path).resolve()
    if manifest_root not in stability_path.parents:
        raise ValueError("Variant SHAP stability must come from the isolated corrected revision")
    stability = load_json(stability_path)
    stability_digest = file_digest(stability_path)
    pinned_stability_digest = RUN_MANIFEST.get("interpretability", {}).get("variant_stability_sha256")
    if pinned_stability_digest and stability_digest != pinned_stability_digest:
        raise ValueError("Corrected SHAP stability differs from its explicit manifest hash")
    if stability.get("status") != "complete":
        raise ValueError("Corrected SHAP stability is not complete")
    if Path(stability["partition_directory"]).resolve() != Path(transfer_manifest["partition_directory"]).resolve():
        raise ValueError("Transfer and SHAP stability use different corrected Variant populations")
    if stability["partition_manifest_sha256"] != transfer_manifest["partition_manifest_sha256"]:
        raise ValueError("Transfer and SHAP partition evidence hashes differ")
    if set(stability.get("variant_keys", [])) != set(VARIANT_ORDER):
        raise ValueError("Corrected SHAP stability must cover Base and all five Variants")
    if Path(stability["source_run"]).resolve() != find_shap_run("lgbm").resolve():
        raise ValueError("Corrected Variant SHAP does not use the approved frozen compatible LGBM source")
    for filename, expected_hash in stability["source_artefacts_sha256"].items():
        if file_digest(find_shap_run("lgbm") / filename) != expected_hash:
            raise ValueError("The frozen LGBM Variant SHAP source changed after its analysis")
    stability_verification, stability_artefacts = {}, {}
    for variant in VARIANT_ORDER:
        magnitudes = stability["global_importance"][variant]
        if len(magnitudes) != len(stability["original_feature_names"]) or any(not np.isfinite(value) or value < 0 for value in magnitudes.values()):
            raise ValueError("Corrected SHAP requires complete, finite original-feature magnitudes")
        expected_ranking = sorted(magnitudes, key=magnitudes.get, reverse=True)[:stability["top_k"]]
        if stability["rankings"][variant] != expected_ranking:
            raise ValueError("Corrected SHAP top-feature order disagrees with its complete magnitudes")
        if variant == "baf_base":
            continue
        population = transfer_all["lgbm"][variant]["population"]
        if stability["populations"][variant] != population:
            raise ValueError("Corrected transfer and SHAP report different retained populations")
        variant_dir = stability_path.parent / variant
        for filename, expected_hash in stability["variants"][variant]["artefacts_sha256"].items():
            if file_digest(variant_dir / filename) != expected_hash:
                raise ValueError("Corrected Variant SHAP artefact changed after generation")
            stability_artefacts[f"{variant}/{filename}"] = {"path": str((variant_dir / filename).resolve()),
                                                            "sha256": expected_hash}
        values = np.load(variant_dir / "shap_values.npy", mmap_mode="r", allow_pickle=False)
        if values.shape != (population["rows"], len(stability["encoded_feature_names"])):
            raise ValueError("Corrected SHAP matrix dimensions do not match the retained cohort")
        for filename in ("y_test.npy", "test_row_indices.npy"):
            if not np.array_equal(np.load(variant_dir / filename, allow_pickle=False),
                                  np.load(CROSS_DOMAIN_ROOT / "lgbm" / variant / filename, allow_pickle=False)):
                raise ValueError("Corrected SHAP and transfer rows/labels do not align")
        scores = np.load(CROSS_DOMAIN_ROOT / "lgbm" / variant / "y_test_scores.npy", allow_pickle=False)
        stability_verification[variant] = verify_saved_shap_matrix(
            values, stability["encoded_feature_names"], stability["original_feature_names"],
            magnitudes, stability["expected_raw_logit"], scores)
    for first in VARIANT_ORDER:
        for second in VARIANT_ORDER:
            value = stability["jaccard_matrix"][first][second]
            expected = len(set(stability["rankings"][first]) & set(stability["rankings"][second])) / len(set(stability["rankings"][first]) | set(stability["rankings"][second]))
            if not np.isclose(value, expected, rtol=0, atol=1e-12):
                raise ValueError("SHAP stability does not reproduce its saved top-feature sets")
    report["variant_shap_stability"] = {"path": str(stability_path), "sha256": stability_digest,
                                       "stored_matrix_verification": stability_verification,
                                       "artefacts": stability_artefacts,
                                       "partition_manifest": {"path": str(partition_root / "partition_manifest.json"),
                                                              "sha256": transfer_manifest["partition_manifest_sha256"]}}
    report["paired_analysis"] = collect_paired_evidence(manifest_root)
    return report


# ══════════════════════════════════════════════════════════════════════════
#  COLLECT DATA
# ══════════════════════════════════════════════════════════════════════════

def collect_baseline(dataset):
    """Collect metrics_test, config, and pr_curve_data for all models."""
    data = {}
    for model in MODEL_ORDER:
        run_dir = find_latest_run(dataset, model)
        if run_dir is None:
            print(f"  [SKIP] {model} — no results found for {dataset}")
            continue
        data[model] = {
            "metrics": full_precision_metrics(run_dir),
            "config": load_json(run_dir / "config.json"),
            "pr_data": load_json(run_dir / "pr_curve_data.json"),
        }
    return data


def collect_threshold_study(dataset):
    """Collect threshold_study.json for all models that have it."""
    data = {}
    for model in MODEL_ORDER:
        run_dir = find_latest_run(dataset, model)
        if run_dir is None:
            continue
        if THRESHOLD_STUDY_ROOT is not None:
            ts_path = THRESHOLD_STUDY_ROOT / dataset / model / "threshold_study.json"
        elif RUN_MANIFEST is not None:
            raise ValueError("Revision generation requires --threshold-study-root")
        else:
            ts_path = run_dir / "threshold_study.json"
        if ts_path.exists():
            data[model] = load_json(ts_path)
    return data


# ══════════════════════════════════════════════════════════════════════════
#  TABLE 1 — Baseline comparison
# ══════════════════════════════════════════════════════════════════════════

def generate_baseline_table(data, dataset_label, filename):
    """Generate LaTeX table: PR-AUC, F1, F2, Precision, Recall, τ, Brier."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Baseline comparison on "
        + dataset_label
        + r" (strategy\,=\,None, threshold\,=\,max-$F_2$). Brier is not reported for OCSVM because its anomaly scores are not probabilities.}"
    )
    lines.append(r"\label{tab:baseline_" + filename + "}")
    lines.append(r"\begin{tabular}{l c c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & PR-AUC & ROC-AUC & $F_1$ & $F_2$ & Precision & Recall & $\tau$ & Brier \\"
    )
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in data:
            continue
        m = data[model]["metrics"]
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        tau = m["threshold"]
        if abs(tau) >= 10:
            tau_str = f"{tau:.2f}"
        else:
            tau_str = f"{tau:.4f}"
        brier_str = "---" if model == "ocsvm" else f"{m['brier_score']:.5f}"
        roc_auc = m["ROC-AUC"]

        label = MODEL_LABELS[model]
        # Pad label to 10 chars for alignment
        lines.append(
            f"{label:<10s} & {m['PR-AUC']:.4f} & {roc_auc:.4f} & {m['F1']:.4f} & {m['F2']:.4f} "
            f"& {prec:.4f} & {rec:.4f} & {tau_str} & {brier_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / "baseline.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  TABLE 2 — Operational metrics
# ══════════════════════════════════════════════════════════════════════════

def generate_ops_table(data, dataset_label, filename):
    """Generate LaTeX table: Alert Rate, FP/TP, P@k, R@k, k."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Operational metrics on "
        + dataset_label
        + r" (strategy\,=\,None, threshold\,=\,max-$F_2$).}"
    )
    lines.append(r"\label{tab:ops_" + filename + "}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Alert Rate & FP/TP & P@$k$ & R@$k$ & $k$ \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in data:
            continue
        m = data[model]["metrics"]
        label = MODEL_LABELS[model]
        alert_rate = m["alert_rate"]
        fp_tp = m["FP/TP"]
        fp_tp_str = f"{fp_tp:.4f}" if fp_tp < 100 else f"{fp_tp:.1f}"

        lines.append(
            f"{label:<10s} & {alert_rate:.5f} & {fp_tp_str} "
            f"& {m['precision_at_k']:.4f} & {m['recall_at_k']:.4f} "
            f"& {m['k_used']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / "ops.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  TABLE 3 — Bootstrap CIs
# ══════════════════════════════════════════════════════════════════════════

def generate_ci_table(data, dataset_label, filename):
    """Generate LaTeX table: PR-AUC and ROC-AUC with 95% Bootstrap CI."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{95\% Bootstrap CI for PR-AUC and ROC-AUC on "
        + dataset_label
        + r" (strategy\,=\,None, 1\,000 iterations).}"
    )
    lines.append(r"\label{tab:ci_" + filename + "}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & PR-AUC & 95\% CI & ROC-AUC & 95\% CI \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in data:
            continue
        m = data[model]["metrics"]
        ci_pr = validated_bootstrap_interval(m, "PR-AUC_ci")
        ci_roc = validated_bootstrap_interval(m, "ROC-AUC_ci")
        roc_auc = m["ROC-AUC"]
        label = MODEL_LABELS[model]
        lines.append(
            f"{label:<10s} & {m['PR-AUC']:.3f} "
            f"& [{ci_pr[0]:.3f}, {ci_pr[1]:.3f}] "
            f"& {roc_auc:.3f} "
            f"& [{ci_roc[0]:.3f}, {ci_roc[1]:.3f}] \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / "ci.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  TABLE 4 — Computational cost
# ══════════════════════════════════════════════════════════════════════════

def generate_cost_table(data, dataset_label, filename):
    """Generate recorded costs, marking reused historical measurements explicitly."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Computational cost on "
        + dataset_label
        + r" (strategy\,=\,None). $\dagger$ marks a historical recorded measurement reused with the frozen model; it is not a new search or final fit. "
        + r"$\ddagger$ marks wall time for a resumed search invocation only, excluding earlier trials' elapsed time. "
        + r"A dash denotes no new hyperparameter search for fixed or untuned models. "
        + r"Validation-only recovery costs are recorded separately and are not included in final-fit time.}"
    )
    lines.append(r"\label{tab:cost_" + filename + "}")
    lines.append(r"\begin{tabular}{l r r r}")
    lines.append(r"\toprule")
    lines.append(r"Model & Tuning (s) & Train (s) & Inference (s) \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in data:
            continue
        c = data[model]["config"]
        label = MODEL_LABELS[model]
        cost = computational_cost_basis(c)
        tuning, train, infer = (cost[name] for name in ("tuning_time_s", "train_time_s", "infer_time_s"))

        if tuning is not None and tuning > 0:
            tuning_str = f"{tuning:,.0f}" + (r"$^{\dagger}$" if cost["tuning_historical"] else "")
            if cost["tuning_partial_wall_time"]:
                tuning_str += r"$^{\ddagger}$"
        else:
            tuning_str = "---"
        train_str = f"{train:.1f}" + (r"$^{\dagger}$" if cost["train_historical"] else "")
        infer_str = f"{infer:.3f}" + (r"$^{\dagger}$" if cost["infer_historical"] else "")

        lines.append(
            f"{label:<10s} & {tuning_str} & {train_str} & {infer_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / "cost.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — PR Curves (all models overlaid)
# ══════════════════════════════════════════════════════════════════════════

def generate_pr_curve(data, dataset_label, filename, fraud_rate=0.0017):
    """Generate PR curve PDF with all models overlaid."""
    fig, ax = plt.subplots(figsize=(6.6, 3.5))

    for model in MODEL_ORDER:
        if model not in data:
            continue
        pr = data[model]["pr_data"]
        precisions = np.array(pr["precisions"])
        recalls = np.array(pr["recalls"])
        prauc = data[model]["metrics"]["PR-AUC"]
        label = f"{MODEL_LABELS[model]} ({prauc:.3f})"

        style = {"linewidth": 2.0}
        if model == "ocsvm":
            style["linestyle"] = "--"
            style["linewidth"] = 1.5

        ax.plot(recalls, precisions,
                color=MODEL_COLORS[model],
                label=label, **style)

    # Random baseline
    ax.axhline(y=fraud_rate, color="black", linestyle=":",
               linewidth=0.8, alpha=0.6, label=f"Random ({fraud_rate:.4f})")

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f"Precision–Recall Curves — {dataset_label} (strategy = None)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out = _figures_dir(filename) / "pr_curves_baseline.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Bar chart: PR-AUC comparison
# ══════════════════════════════════════════════════════════════════════════

def generate_prauc_bar(data, dataset_label, filename):
    """Generate horizontal bar chart of PR-AUC with CI error bars."""
    models = [m for m in MODEL_ORDER if m in data]
    labels = [MODEL_LABELS[m] for m in models]
    praucs = [data[m]["metrics"]["PR-AUC"] for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    # CI error bars
    ci_low = [data[m]["metrics"]["bootstrap_ci"]["PR-AUC_ci"][0] for m in models]
    ci_high = [data[m]["metrics"]["bootstrap_ci"]["PR-AUC_ci"][1] for m in models]
    err_low = [p - lo for p, lo in zip(praucs, ci_low)]
    err_high = [hi - p for p, hi in zip(praucs, ci_high)]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y_pos = np.arange(len(models))

    ax.barh(y_pos, praucs, color=colors, edgecolor="white", height=0.6,
            xerr=[err_low, err_high], capsize=4, error_kw={"linewidth": 1.2})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("PR-AUC", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_title(f"PR-AUC — {dataset_label} (strategy = None)", fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Value annotations
    for i, (v, hi) in enumerate(zip(praucs, ci_high)):
        ax.text(hi + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

    out = _figures_dir(filename) / "prauc_bar_baseline.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Confusion matrix grid
# ══════════════════════════════════════════════════════════════════════════

def generate_confusion_grid(data, dataset_label, filename):
    """Generate a grid of confusion matrices for all models."""
    models = [m for m in MODEL_ORDER if m in data]
    n = len(models)
    rows, columns = int(np.ceil(n / 2)), 2
    fig, axes = plt.subplots(rows, columns, figsize=(6.6, 2.0 * rows), squeeze=False)
    axes = axes.ravel()
    for unused in axes[n:]:
        unused.set_visible(False)

    for ax, model in zip(axes, models):
        m = data[model]["metrics"]
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])

        ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() * 0.5 else "black"
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"], fontsize=8)
        ax.set_yticklabels(["Legit", "Fraud"], fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

    fig.suptitle(f"Confusion Matrices — {dataset_label} (strategy = None)",
                 fontsize=10, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .96))
    out = _figures_dir(filename) / "confusion_grid_baseline.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  THRESHOLD STUDY TABLES
# ══════════════════════════════════════════════════════════════════════════

STRATEGY_ORDER = ["fixed_05", "max_f1", "max_f2", "prec_ge_05"]
STRATEGY_COL_HEADERS = {
    "fixed_05":   r"Fixed (0.5)",
    "max_f1":     r"max-$F_1$",
    "max_f2":     r"max-$F_2$",
    "prec_ge_05": r"Prec\,$\geq$\,0.5",
}


def _generate_threshold_table(ts_data, metric_key, dataset_label, filename,
                                caption_metric, label_suffix, fmt=".3f"):
    """Generic helper: one threshold-study table for a given metric."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Threshold sensitivity on "
        + dataset_label
        + r"~--- " + caption_metric
        + r" by threshold strategy (strategy\,=\,None).}"
    )
    lines.append(r"\label{tab:threshold_" + filename + "_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")

    col_headers = " & ".join(STRATEGY_COL_HEADERS[s] for s in STRATEGY_ORDER)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in ts_data:
            continue
        label = MODEL_LABELS[model]
        tr = ts_data[model]["test_results"]
        vals = []
        for s in STRATEGY_ORDER:
            v = tr[s][metric_key]
            if metric_key == "alert_rate":
                vals.append(f"{v * 100:.2f}\\%")
            else:
                vals.append(f"{v:{fmt}}")
        lines.append(f"{label:<10s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / f"threshold_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_threshold_tables(ts_data, dataset_label, filename):
    """Generate all 4 threshold study LaTeX tables for a dataset."""
    _generate_threshold_table(
        ts_data, "F2", dataset_label, filename, "$F_2$", "f2"
    )
    _generate_threshold_table(
        ts_data, "F1", dataset_label, filename, "$F_1$", "f1"
    )
    _generate_threshold_table(
        ts_data, "Recall", dataset_label, filename, "Recall", "recall"
    )
    _generate_threshold_table(
        ts_data, "alert_rate", dataset_label, filename, r"Alert Rate (\%)", "alert"
    )


# ══════════════════════════════════════════════════════════════════════════
#  FACTORIAL TABLES  —  Model × Imbalance Strategy
# ══════════════════════════════════════════════════════════════════════════

BALANCE_ORDER = ["none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights"]
BALANCE_LABELS = {
    "none":       "None",
    "rus":        "RUS",
    "ros":        "ROS",
    "smote":      "SMOTE",
    "smote_tomek": "SM+T",
    "smoteenn":   "SMOTEENN",
    "weights":    "Weights",
}
# Models eligible for factorial (OCSVM excluded — anomaly detection paradigm)
FACTORIAL_MODELS = ["logreg", "rf", "lgbm", "catboost", "fttransformer"]


def collect_factorial(dataset):
    """
    Collect metrics_test.json for every (model, strategy) combination.

    Returns dict[model][strategy] = metrics_test dict, or None if missing.
    """
    data = {}
    for model in FACTORIAL_MODELS:
        data[model] = {}
        for strat in BALANCE_ORDER:
            run_dir = find_latest_run(dataset, model, strategy=strat)
            if run_dir is None:
                data[model][strat] = None
                continue
            mt_path = run_dir / "metrics_test.json"
            if mt_path.exists():
                data[model][strat] = full_precision_metrics(run_dir)
            else:
                data[model][strat] = None
    return data


def generate_factorial_table(fdata, metric_key, dataset_label, filename,
                             caption_metric, label_suffix, fmt=".3f"):
    """One factorial table: rows = models, columns = strategies."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{" + caption_metric + r" by model $\times$ strategy on "
        + dataset_label + r".}"
    )
    lines.append(r"\label{tab:factorial_" + filename + "_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l" + " c" * len(BALANCE_ORDER) + "}")
    lines.append(r"\toprule")

    col_headers = " & ".join(BALANCE_LABELS[s] for s in BALANCE_ORDER)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in FACTORIAL_MODELS:
        label = MODEL_LABELS[model]
        vals = []
        # Find baseline value for bolding the best
        baseline_val = None
        if fdata[model]["none"] is not None:
            baseline_val = fdata[model]["none"].get(metric_key)

        row_vals = []
        for strat in BALANCE_ORDER:
            mt = fdata[model].get(strat)
            if mt is None:
                row_vals.append(("---", None))
            else:
                v = mt[metric_key]
                row_vals.append((f"{v:{fmt}}", v))

        # Find the best value in this row
        numeric_vals = [rv[1] for rv in row_vals if rv[1] is not None]
        best_val = max(numeric_vals) if numeric_vals else None

        formatted = []
        for text, v in row_vals:
            if v is not None and best_val is not None and abs(v - best_val) < 1e-6:
                formatted.append(r"\textbf{" + text + "}")
            else:
                formatted.append(text)

        lines.append(f"{label:<10s} & " + " & ".join(formatted) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / f"factorial_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_factorial_tables(fdata, dataset_label, filename):
    """Generate PR-AUC, ROC-AUC, F2 and F1 factorial tables."""
    generate_factorial_table(
        fdata, "PR-AUC", dataset_label, filename, "PR-AUC", "prauc"
    )
    generate_factorial_table(
        fdata, "ROC-AUC", dataset_label, filename, "ROC-AUC", "rocauc"
    )
    generate_factorial_table(
        fdata, "F2", dataset_label, filename, "$F_2$", "f2"
    )
    generate_factorial_table(
        fdata, "F1", dataset_label, filename, "$F_1$", "f1"
    )


def generate_transformer_robustness_table(fdata, dataset_label, filename):
    """Compare FT-Transformer and CatBoost across all seven strategies.

    The final row is the population standard deviation (``ddof=0``) across the
    complete set of designed strategy conditions, computed from complete saved
    score arrays and integer confusion counts before table-display rounding.
    """
    models = ("fttransformer", "catboost")
    if any(fdata[model].get(strategy) is None
           for model in models for strategy in BALANCE_ORDER):
        print(f"  Incomplete {dataset_label} robustness grid — skipping.")
        return

    row_labels = {
        "none": "None",
        "rus": "RUS",
        "ros": "ROS",
        "smote": "SMOTE",
        "smote_tomek": "SM+Tomek",
        "smoteenn": "SMOTEENN",
        "weights": "Weights",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (r"\caption{FT-Transformer and CatBoost performance across imbalance "
         rf"strategies on {dataset_label}. The final row is the population "
         r"standard deviation across the seven complete strategy values, "
         r"computed from complete saved scores and integer confusion counts before table-display rounding; it is not variability across random seeds.}"),
        rf"\label{{tab:transformer_robustness_{filename}}}",
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{FT-Transformer} & \multicolumn{2}{c}{CatBoost} \\",
        r"Strategy & PR-AUC & $F_2$ & PR-AUC & $F_2$ \\",
        r"\midrule",
    ]

    for strategy in BALANCE_ORDER:
        ft_metrics = fdata["fttransformer"][strategy]
        cb_metrics = fdata["catboost"][strategy]
        lines.append(
            f"{row_labels[strategy]:<9s} & {ft_metrics['PR-AUC']:.3f} & "
            f"{ft_metrics['F2']:.3f} & {cb_metrics['PR-AUC']:.3f} & "
            f"{cb_metrics['F2']:.3f} " + r"\\"
        )

    def population_sd(model, metric):
        values = [fdata[model][strategy][metric] for strategy in BALANCE_ORDER]
        return float(np.std(values, ddof=0))

    lines.extend([
        r"\midrule",
        (r"\textit{Pop.\ std.\ dev.} & "
         f"\\textit{{{population_sd('fttransformer', 'PR-AUC'):.3f}}} & "
         f"\\textit{{{population_sd('fttransformer', 'F2'):.3f}}} & "
         f"\\textit{{{population_sd('catboost', 'PR-AUC'):.3f}}} & "
         f"\\textit{{{population_sd('catboost', 'F2'):.3f}}} " + r"\\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out = _tables_dir(filename) / "transformer_robustness.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_factorial_heatmap(fdata, dataset_label, filename):
    """
    Generate a heatmap of ΔPR-AUC (strategy − baseline) for each model.
    Green = improvement, red = degradation.
    """
    import seaborn as sns

    # Non-baseline strategies only for the heatmap
    strats = [s for s in BALANCE_ORDER if s != "none"]
    strat_labels = [BALANCE_LABELS[s] for s in strats]
    model_labels = [MODEL_LABELS[m] for m in FACTORIAL_MODELS]

    matrix = np.full((len(FACTORIAL_MODELS), len(strats)), np.nan)
    for i, model in enumerate(FACTORIAL_MODELS):
        baseline = fdata[model].get("none")
        if baseline is None:
            continue
        base_val = baseline["PR-AUC"]
        for j, strat in enumerate(strats):
            mt = fdata[model].get(strat)
            if mt is not None:
                matrix[i, j] = mt["PR-AUC"] - base_val

    # Also build F2 heatmap
    matrix_f2 = np.full((len(FACTORIAL_MODELS), len(strats)), np.nan)
    for i, model in enumerate(FACTORIAL_MODELS):
        baseline = fdata[model].get("none")
        if baseline is None:
            continue
        base_val = baseline["F2"]
        for j, strat in enumerate(strats):
            mt = fdata[model].get(strat)
            if mt is not None:
                matrix_f2[i, j] = mt["F2"] - base_val

    for mat, metric_name, suffix in [
        (matrix, "PR-AUC", "prauc"),
        (matrix_f2, "$F_2$", "f2"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
        vmax = max(vmax, 0.01)  # avoid zero range

        sns.heatmap(
            mat, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            vmin=-vmax, vmax=vmax,
            xticklabels=strat_labels, yticklabels=model_labels,
            ax=ax, linewidths=0.5, cbar_kws={"label": f"Δ{metric_name}"},
        )
        ax.set_title(
            f"Δ{metric_name} (strategy − None) — {dataset_label}",
            fontsize=13,
        )
        ax.set_ylabel("")

        out = _figures_dir(filename) / f"heatmap_{suffix}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  CROSS-DOMAIN GENERALIZATION  (BAF Base → Variants I–V)
# ══════════════════════════════════════════════════════════════════════════

VARIANT_ORDER = ["baf_base", "baf_var1", "baf_var2", "baf_var3", "baf_var4", "baf_var5"]
VARIANT_LABELS = {
    "baf_base": "Base",
    "baf_var1": "Var I",
    "baf_var2": "Var II",
    "baf_var3": "Var III",
    "baf_var4": "Var IV",
    "baf_var5": "Var V",
}


def collect_cross_domain():
    """Load pinned corrected evaluations, or historical flat records in legacy mode."""
    data = {}
    for model in MODEL_ORDER:
        run_dir = find_latest_run("baf_base", model)
        if run_dir is None:
            continue
        if CROSS_DOMAIN_ROOT is not None:
            model_dir = CROSS_DOMAIN_ROOT / model
            source = load_json(model_dir / "source_audit.json")
            validation = source["base_test_prediction_validation"]
            if not validation.get("labels_exactly_equal") or validation.get("decisions_changed_at_frozen_threshold") != 0:
                raise ValueError(f"Frozen Base prediction verification failed: {model}")
            base_metrics = full_precision_metrics(Path(source["source_run"]))
            for name in ("PR-AUC", "ROC-AUC", "F2", "TP", "FP", "TN", "FN"):
                if abs(float(base_metrics[name]) - float(validation["recomputed_metrics"][name])) > 1e-6:
                    raise ValueError(f"Frozen Base transfer metrics differ from its saved baseline: {model}/{name}")
            transfer = {"baf_base": {"metrics": base_metrics,
                                     "threshold_used": source["threshold_used"]}}
            for variant in VARIANT_ORDER[1:]:
                metrics_path = model_dir / variant / "metrics_test.json"
                from generate_revision_interpretability import file_digest
                if file_digest(metrics_path) != source["variants"][variant]["metrics_test_sha256"]:
                    raise ValueError(f"Corrected transfer metrics changed: {model}/{variant}")
                record = load_json(metrics_path)
                if not np.isclose(record["threshold_used"], source["threshold_used"], rtol=0, atol=0):
                    raise ValueError(f"Variant threshold differs from frozen Base threshold: {model}/{variant}")
                record["metrics"], unused_snapshots = verify_transfer_evidence(metrics_path.parent, record, model)
                transfer[variant] = record
            data[model] = transfer
            continue
        elif RUN_MANIFEST is not None:
            raise ValueError("Revision generation requires --cross-domain-root")
        else:
            cd_path = run_dir / "cross_domain.json"
        if cd_path.exists():
            data[model] = load_json(cd_path)
    return data


def generate_cross_domain_table(cd_data, metric_key, caption_metric,
                                 label_suffix, fmt=".4f"):
    """Cross-domain table: rows = models, columns = Base + Variants."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cross-domain " + caption_metric
        + r": trained on BAF Base, evaluated on each Variant (no retraining).}"
    )
    lines.append(r"\label{tab:crossdomain_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l" + " c" * len(VARIANT_ORDER) + "}")
    lines.append(r"\toprule")

    col_headers = " & ".join(VARIANT_LABELS[v] for v in VARIANT_ORDER)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in cd_data:
            continue
        label = MODEL_LABELS[model]
        vals = []
        numeric = []
        for var in VARIANT_ORDER:
            if var in cd_data[model]:
                m = cd_data[model][var]["metrics"]
                v = m[metric_key]
                vals.append((f"{v:{fmt}}", v))
                numeric.append(v)
            else:
                vals.append(("---", None))

        best = max(numeric) if numeric else None
        formatted = []
        for text, v in vals:
            if v is not None and best is not None and abs(v - best) < 1e-6:
                formatted.append(r"\textbf{" + text + "}")
            else:
                formatted.append(text)

        lines.append(f"{label:<10s} & " + " & ".join(formatted) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / f"crossdomain_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_cross_domain_tables(cd_data):
    """Generate PR-AUC and F2 cross-domain tables."""
    generate_cross_domain_table(cd_data, "PR-AUC", "PR-AUC", "prauc")
    generate_cross_domain_table(cd_data, "F2", "$F_2$", "f2")
    generate_cross_domain_table(cd_data, "ROC-AUC", "ROC-AUC", "rocauc")


# ══════════════════════════════════════════════════════════════════════════
#  BAF VARIANTS IN-DOMAIN  (train & test on same variant)
# ══════════════════════════════════════════════════════════════════════════

def collect_variants_indomain():
    """Collect baseline metrics for all BAF variants (in-domain runs)."""
    data = {}
    for var in VARIANT_ORDER:
        data[var] = {}
        for model in MODEL_ORDER:
            run_dir = find_latest_run(var, model)
            if run_dir is None:
                data[var][model] = None
                continue
            mt_path = run_dir / "metrics_test.json"
            if mt_path.exists():
                data[var][model] = load_json(mt_path)
            else:
                data[var][model] = None
    return data


def generate_variants_indomain_table(vi_data, metric_key, caption_metric,
                                      label_suffix, fmt=".4f"):
    """In-domain table: rows = models, columns = Base + Variants."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{In-domain " + caption_metric
        + r" across BAF datasets (each model trained and tested on the same dataset).}"
    )
    lines.append(r"\label{tab:indomain_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l" + " c" * len(VARIANT_ORDER) + "}")
    lines.append(r"\toprule")

    col_headers = " & ".join(VARIANT_LABELS[v] for v in VARIANT_ORDER)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        label = MODEL_LABELS[model]
        vals = []
        numeric = []
        for var in VARIANT_ORDER:
            mt = vi_data[var].get(model)
            if mt is not None:
                v = mt[metric_key]
                vals.append((f"{v:{fmt}}", v))
                numeric.append(v)
            else:
                vals.append(("---", None))

        best = max(numeric) if numeric else None
        formatted = []
        for text, v in vals:
            if v is not None and best is not None and abs(v - best) < 1e-6:
                formatted.append(r"\textbf{" + text + "}")
            else:
                formatted.append(text)

        lines.append(f"{label:<10s} & " + " & ".join(formatted) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / f"indomain_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_variants_indomain_tables(vi_data):
    """Generate in-domain summary tables for PR-AUC and F2."""
    generate_variants_indomain_table(vi_data, "PR-AUC", "PR-AUC", "prauc")
    generate_variants_indomain_table(vi_data, "F2", "$F_2$", "f2")


# ══════════════════════════════════════════════════════════════════════════
#  CROSS vs IN-DOMAIN DELTA TABLE
# ══════════════════════════════════════════════════════════════════════════

def generate_delta_table(cd_data, vi_data, metric_key, caption_metric,
                          label_suffix, fmt="+.4f"):
    """Δ table (cross-domain − in-domain) for Variants I–V only."""
    var_keys = [v for v in VARIANT_ORDER if v != "baf_base"]

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{$\Delta$" + caption_metric
        + r" (cross-domain $-$ in-domain) on BAF Variants. "
        + r"Negative values indicate degradation under distribution shift.}"
    )
    lines.append(r"\label{tab:delta_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l" + " c" * len(var_keys) + "}")
    lines.append(r"\toprule")

    col_headers = " & ".join(VARIANT_LABELS[v] for v in var_keys)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in cd_data:
            continue
        label = MODEL_LABELS[model]
        vals = []
        for var in var_keys:
            cd_metrics = cd_data[model].get(var, {}).get("metrics")
            id_metrics = vi_data.get(var, {}).get(model)
            if cd_metrics is not None and id_metrics is not None:
                delta = cd_metrics[metric_key] - id_metrics[metric_key]
                vals.append(f"{delta:{fmt}}")
            else:
                vals.append("---")
        lines.append(f"{label:<10s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / f"delta_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_delta_tables(cd_data, vi_data):
    """Generate Δ tables for PR-AUC and F2."""
    generate_delta_table(cd_data, vi_data, "PR-AUC", "PR-AUC", "prauc")
    generate_delta_table(cd_data, vi_data, "F2", "$F_2$", "f2")


# ══════════════════════════════════════════════════════════════════════════
#  SHAP ANALYSIS TABLES & FIGURES
# ══════════════════════════════════════════════════════════════════════════

SHAP_MODELS = ["logreg", "lgbm", "catboost"]
SHAP_LABELS = {
    "logreg": "LR",
    "lgbm": "LGBM",
    "catboost": "CatBoost",
    "fttransformer": "FT-Trans.",
}


def find_shap_run(model):
    """Use the explicitly approved compatible-model SHAP source, not a new pin."""
    if RUN_MANIFEST is None:
        return find_latest_run("baf_base", model)
    selected = RUN_MANIFEST.get("interpretability", {}).get("baseline_runs", {}).get(model)
    if selected is None:
        raise ValueError(f"No explicit compatible SHAP source for {model}")
    path = Path(selected["run_dir"] if isinstance(selected, dict) else selected)
    return path if path.is_absolute() else ROOT / path

VARIANT_LABEL_MAP = {
    "baf_base": "Base",
    "baf_var1": "Var.~I",
    "baf_var2": "Var.~II",
    "baf_var3": "Var.~III",
    "baf_var4": "Var.~IV",
    "baf_var5": "Var.~V",
}


def collect_shap_data():
    """Collect shap_global.json, shap_consistency.json, shap_variant_stability.json."""
    global_data = {}
    for model in SHAP_MODELS:
        run_dir = find_shap_run(model)
        if run_dir is None:
            continue
        gpath = run_dir / "shap_global.json"
        if gpath.exists():
            with open(gpath) as f:
                global_data[model] = json.load(f)

    # Recompute compatible-model consistency from complete original-feature
    # matrices, never from truncated dictionaries or an invalid FT panel.
    from generate_revision_interpretability import jaccard_matrix, load_artefacts
    artefacts = {model: load_artefacts(model, find_shap_run(model)) for model in SHAP_MODELS}
    consistency = {"models": SHAP_MODELS, **jaccard_matrix(artefacts)}
    lgbm_dir = find_shap_run("lgbm")

    # Variant stability
    stability = None
    if lgbm_dir:
        if RUN_MANIFEST is not None:
            selected = RUN_MANIFEST.get("interpretability", {}).get("variant_stability")
            if not selected:
                raise ValueError("Revision generation requires corrected Variant SHAP stability")
            spath = Path(selected)
            spath = spath if spath.is_absolute() else ROOT / spath
        else:
            spath = lgbm_dir / "shap_variant_stability.json"
        if spath.exists():
            with open(spath) as f:
                stability = json.load(f)

    # Local cases
    local_data = {}
    for model in SHAP_MODELS:
        run_dir = find_shap_run(model)
        if run_dir is None:
            continue
        lpath = run_dir / "shap_local_cases.json"
        if lpath.exists():
            with open(lpath) as f:
                local_data[model] = json.load(f)

    return global_data, consistency, stability, local_data


def generate_shap_global_bar(global_data):
    """Draw each compatible model's actual top-15 from complete saved SHAP values."""
    if not global_data:
        return
    from generate_revision_interpretability import load_artefacts, plot_global
    artefacts = {model: load_artefacts(model, find_shap_run(model)) for model in SHAP_MODELS}
    out = _figures_dir("baf") / "shap_global.pdf"
    plot_global(artefacts, out)
    _emit(out)


def generate_shap_beeswarm(global_data):
    """Display actual encoded-column SHAP values and transformed feature values."""
    lgbm_dir = find_shap_run("lgbm")
    if lgbm_dir is None:
        return

    shap_path = lgbm_dir / "shap_values.npy"
    if not shap_path.exists():
        return

    try:
        import shap as shap_lib
        shap_values = np.load(shap_path)

        # Load test data feature names from preprocessor
        import joblib
        pipeline = joblib.load(lgbm_dir / "model.joblib")
        preprocessor = pipeline.named_steps["preprocessor"]

        from data import load_dataset
        _, X_test, _, _ = load_dataset("baf_base")
        X_transformed = preprocessor.transform(X_test)
        feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(X_transformed, "values"):
            X_transformed_np = X_transformed.values
        else:
            X_transformed_np = X_transformed

        # Create SHAP Explanation object
        explanation = shap_lib.Explanation(
            values=shap_values,
            data=X_transformed_np,
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(6.6, 6.5))
        random_state = np.random.get_state()
        try:
            np.random.seed(42)  # Cosmetic jitter only; never a model or case-selection seed.
            shap_lib.plots.beeswarm(explanation, max_display=15, show=False, ax=ax, plot_size=None)
        finally:
            np.random.set_state(random_state)
        ax.set_title("SHAP Beeswarm — LGBM on BAF Base", fontsize=10.5)
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_xlabel("SHAP contribution (log-odds)", fontsize=10)
        for colour_axis in fig.axes[1:]:
            colour_axis.tick_params(labelsize=10)
            colour_axis.set_ylabel(colour_axis.get_ylabel(), fontsize=10)
        plt.tight_layout()

        out = _figures_dir("baf") / "shap_beeswarm.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        _emit(out)
    except Exception as e:
        if RUN_MANIFEST is not None:
            raise RuntimeError("Required revised SHAP beeswarm could not be generated") from e
        print(f"  [WARN] Beeswarm plot failed: {e}")


def generate_shap_consistency_table(consistency):
    """Generate Jaccard cross-model consistency table."""
    if consistency is None:
        return

    models = [model for model in consistency["models"] if model in SHAP_MODELS]
    jm = consistency["jaccard_matrix"]

    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{Jaccard similarity of top-10 SHAP features across model pairs (BAF Base).}",
        r"\label{tab:shap_consistency}",
    ]

    col_spec = "l " + " ".join(["c"] * len(models))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = " & ".join(SHAP_LABELS.get(m, m) for m in models)
    lines.append(f"          & {header} \\\\")
    lines.append(r"\midrule")

    for m1 in models:
        label = SHAP_LABELS.get(m1, m1)
        vals = []
        for m2 in models:
            if m1 == m2:
                vals.append("---")
            else:
                v = jm[m1][m2]
                vals.append(f"{v:.2f}")
        lines.append(f"{label:<10s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / "shap_consistency.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_shap_stability_table(stability):
    """Generate Jaccard cross-variant stability table."""
    if stability is None:
        return

    var_keys = stability["variant_keys"]
    jm = stability["jaccard_matrix"]

    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{Jaccard similarity of top-10 SHAP features for LGBM across BAF Variants.}",
        r"\label{tab:shap_stability}",
    ]

    col_spec = "l " + " ".join(["c"] * len(var_keys))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = " & ".join(VARIANT_LABEL_MAP.get(v, v) for v in var_keys)
    lines.append(f"         & {header} \\\\")
    lines.append(r"\midrule")

    for v1 in var_keys:
        label = VARIANT_LABEL_MAP.get(v1, v1)
        vals = []
        for v2 in var_keys:
            if v1 == v2:
                vals.append("---")
            else:
                v = jm[v1][v2]
                vals.append(f"{v:.2f}")
        lines.append(f"{label:<10s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / "shap_stability.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_shap_waterfall(local_data):
    """Generate complete additive log-odds waterfalls for selected LGBM cases."""
    model = "lgbm"
    if model not in local_data:
        return

    from generate_revision_interpretability import apply_operating_baseline, load_artefacts, plot_local, select_local_cases
    artefact = load_artefacts(model, find_shap_run(model))
    if RUN_MANIFEST is not None:
        apply_operating_baseline(artefact, find_latest_run("baf_base", model))
    out = _figures_dir("baf") / "shap_waterfall_tp_fp_fn.pdf"
    plot_local(artefact, select_local_cases(artefact), out)
    _emit(out)


def generate_shap_dependence(global_data):
    """Generate dependence plots for top-3 features (LGBM on BAF Base)."""
    lgbm_dir = find_shap_run("lgbm")
    if lgbm_dir is None or "lgbm" not in global_data:
        return

    shap_path = lgbm_dir / "shap_values.npy"
    if not shap_path.exists():
        return

    try:
        import joblib
        shap_values = np.load(shap_path)
        pipeline = joblib.load(lgbm_dir / "model.joblib")
        preprocessor = pipeline.named_steps["preprocessor"]

        from data import load_dataset
        _, X_test, _, _ = load_dataset("baf_base")
        X_transformed = preprocessor.transform(X_test)
        feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(X_transformed, "values"):
            X_transformed_np = X_transformed.values
        else:
            X_transformed_np = X_transformed

        # Top-3 features from global importance
        top3 = list(global_data["lgbm"].keys())[:3]

        # The global report groups one-hot columns back to their source
        # categorical features.  A grouped name therefore cannot be plotted
        # directly against a single transformed column.  Do not emit a
        # partially empty and potentially misleading figure when this occurs.
        missing_features = [name for name in top3 if name not in feature_names]
        if missing_features:
            print(
                "  [WARN] SHAP dependence plot skipped: grouped top-feature "
                f"names are absent from the transformed matrix: {missing_features}"
            )
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, feat_name in zip(axes, top3):
            idx = feature_names.index(feat_name)
            ax.scatter(X_transformed_np[:, idx], shap_values[:, idx],
                       alpha=0.05, s=3, c=shap_values[:, idx],
                       cmap="coolwarm", rasterized=True)
            ax.set_xlabel(feat_name, fontsize=11)
            ax.set_ylabel("SHAP value", fontsize=11)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.grid(alpha=0.2)

        plt.suptitle("SHAP Dependence Plots — LGBM on BAF Base (top-3 features)",
                     fontsize=13)
        plt.tight_layout()

        out = _figures_dir("baf") / "shap_dependence.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        _emit(out)
    except Exception as e:
        print(f"  [WARN] Dependence plots failed: {e}")


def generate_all_shap():
    """Generate all SHAP tables and figures."""
    global_data, consistency, stability, local_data = collect_shap_data()

    if not global_data:
        print("\n  No SHAP results found — skipping.")
        return

    print(f"\n── BAF Base — SHAP Analysis ({len(global_data)} models) ──")
    generate_shap_global_bar(global_data)
    generate_shap_beeswarm(global_data)
    generate_shap_consistency_table(consistency)
    generate_shap_stability_table(stability)
    generate_shap_waterfall(local_data)
    generate_shap_dependence(global_data)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    global TABLES_BASE, FIGURES_BASE, RUN_MANIFEST, RESULTS_DIR, THRESHOLD_STUDY_ROOT, CROSS_DOMAIN_ROOT
    parser = argparse.ArgumentParser(description="Generate tables and figures from selected saved runs")
    parser.add_argument("--manifest", type=Path, required=True, help="Explicit corrected and historical BAF source pins")
    parser.add_argument("--output-root", type=Path, required=True, help="Write derived revision artefacts here")
    parser.add_argument("--results-root", type=Path, required=True, help="Explicit corrected results root")
    parser.add_argument("--threshold-study-root", type=Path, required=True, help="Derived <dataset>/<model>/threshold_study.json root")
    parser.add_argument("--cross-domain-root", type=Path, required=True, help="Completed transfer/evaluations/<explicit-timestamp> directory")
    args = parser.parse_args()
    manifest_digest = None
    if args.manifest:
        from generate_revision_interpretability import file_digest
        manifest_digest = file_digest(args.manifest.resolve())
        RUN_MANIFEST = load_json(args.manifest)
        if not args.output_root:
            parser.error("--manifest requires --output-root to preserve historical outputs")
    if args.results_root:
        if not args.manifest:
            parser.error("--results-root requires --manifest")
        RESULTS_DIR = args.results_root.resolve()
    THRESHOLD_STUDY_ROOT = args.threshold_study_root.resolve() if args.threshold_study_root else None
    CROSS_DOMAIN_ROOT = args.cross_domain_root.resolve() if args.cross_domain_root else None
    if args.output_root:
        from generate_revision_interpretability import validate_output_directory
        output_root = validate_output_directory(args.output_root)
        TABLES_BASE = output_root / "tables"
        FIGURES_BASE = output_root / "figures"
    preflight_report = preflight_revision() if RUN_MANIFEST is not None else None
    if preflight_report is not None:
        from generate_revision_interpretability import file_digest
        if file_digest(args.manifest.resolve()) != manifest_digest:
            raise ValueError("The source manifest changed during preflight; no files generated")
        preflight_report["manifest_path"] = str(args.manifest.resolve())
        preflight_report["manifest_sha256"] = manifest_digest
    print("=" * 60)
    print("  Generating thesis tables and figures from actual results")
    print("=" * 60)

    # ── ULB 2013 ──
    print("\n── ULB Credit Card 2013 ──")
    ulb = collect_baseline("ulb_2013")
    if ulb:
        print(f"  Found {len(ulb)} models: {', '.join(ulb.keys())}")
        generate_baseline_table(ulb, "ULB 2013", "ulb")
        generate_ops_table(ulb, "ULB 2013", "ulb")
        generate_ci_table(ulb, "ULB 2013", "ulb")
        generate_cost_table(ulb, "ULB 2013", "ulb")
        first_ulb = next(iter(ulb.values()))["metrics"]
        ulb_prevalence = (first_ulb["TP"] + first_ulb["FN"]) / sum(first_ulb[key] for key in ("TP", "FP", "TN", "FN"))
        generate_pr_curve(ulb, "ULB 2013", "ulb", fraud_rate=ulb_prevalence)
        generate_prauc_bar(ulb, "ULB 2013", "ulb")
        generate_confusion_grid(ulb, "ULB 2013", "ulb")

    # ── ULB 2013 — Threshold Study ──
    ulb_ts = collect_threshold_study("ulb_2013")
    if ulb_ts:
        print(f"\n── ULB 2013 — Threshold Study ({len(ulb_ts)} models) ──")
        generate_threshold_tables(ulb_ts, "ULB 2013", "ulb")
    else:
        print("\n  No ULB threshold study results found — skipping.")

    # ── ULB 2013 — Factorial (Model × Strategy) ──
    ulb_fact = collect_factorial("ulb_2013")
    n_combos = sum(
        1
        for m in ulb_fact
        for s in ulb_fact[m]
        if s != "none" and ulb_fact[m][s] is not None
    )
    if n_combos > 0:
        print(f"\n── ULB 2013 — Factorial ({n_combos} combos) ──")
        generate_factorial_tables(ulb_fact, "ULB 2013", "ulb")
        generate_transformer_robustness_table(ulb_fact, "ULB 2013", "ulb")
        generate_factorial_heatmap(ulb_fact, "ULB 2013", "ulb")
    else:
        print("\n  No ULB factorial results found — skipping.")

    # ── BAF Base (if exists) ──
    print("\n── BAF Base ──")
    baf = collect_baseline("baf_base")
    if baf:
        print(f"  Found {len(baf)} models: {', '.join(baf.keys())}")
        generate_baseline_table(baf, "BAF Base", "baf")
        generate_ops_table(baf, "BAF Base", "baf")
        generate_ci_table(baf, "BAF Base", "baf")
        generate_cost_table(baf, "BAF Base", "baf")
        first_baf = next(iter(baf.values()))["metrics"]
        baf_prevalence = (first_baf["TP"] + first_baf["FN"]) / sum(first_baf[key] for key in ("TP", "FP", "TN", "FN"))
        generate_pr_curve(baf, "BAF Base", "baf", fraud_rate=baf_prevalence)
        generate_prauc_bar(baf, "BAF Base", "baf")
        generate_confusion_grid(baf, "BAF Base", "baf")
    else:
        print("  No BAF results found yet — skipping.")

    # ── BAF Base — Threshold Study ──
    baf_ts = collect_threshold_study("baf_base")
    if baf_ts:
        print(f"\n── BAF Base — Threshold Study ({len(baf_ts)} models) ──")
        generate_threshold_tables(baf_ts, "BAF Base", "baf")
    else:
        print("\n  No BAF threshold study results found — skipping.")

    # ── BAF Base — Factorial (Model × Strategy) ──
    baf_fact = collect_factorial("baf_base")
    n_baf_combos = sum(
        1
        for m in baf_fact
        for s in baf_fact[m]
        if s != "none" and baf_fact[m][s] is not None
    )
    if n_baf_combos > 0:
        print(f"\n── BAF Base — Factorial ({n_baf_combos} combos) ──")
        generate_factorial_tables(baf_fact, "BAF Base", "baf")
        generate_transformer_robustness_table(baf_fact, "BAF Base", "baf")
        generate_factorial_heatmap(baf_fact, "BAF Base", "baf")
    else:
        print("\n  No BAF factorial results found — skipping.")

    # ── BAF Cross-Domain Generalization ──
    cd_data = collect_cross_domain()
    if cd_data:
        print(f"\n── BAF Cross-Domain ({len(cd_data)} models) ──")
        generate_cross_domain_tables(cd_data)
    else:
        print("\n  No BAF cross-domain results found — skipping.")

    # ── BAF Variants In-Domain ──
    vi_data = collect_variants_indomain()
    n_vi = sum(
        1 for v in vi_data for m in vi_data[v] if vi_data[v][m] is not None
    )
    # Subtract Base models (already reported as baselines)
    n_vi_variants = sum(
        1 for v in vi_data if v != "baf_base"
        for m in vi_data[v] if vi_data[v][m] is not None
    )
    if n_vi_variants > 0:
        print(f"\n── BAF Variants In-Domain ({n_vi_variants} runs) ──")
        generate_variants_indomain_tables(vi_data)
        # If we also have cross-domain data, generate delta tables
        if cd_data:
            print(f"\n── BAF Δ Cross-vs-In-Domain ──")
            generate_delta_tables(cd_data, vi_data)
    else:
        print("\n  No BAF variant in-domain results found — skipping delta tables.")

    # ── SHAP Analysis ──
    generate_all_shap()

    if preflight_report is not None:
        from generate_revision_interpretability import file_digest
        if file_digest(args.manifest.resolve()) != manifest_digest:
            raise ValueError("The source manifest changed during generation; outputs must not be promoted")
        for source in preflight_report["source_runs"].values():
            for filename, digest_key in (("config.json", "config_sha256"), ("metrics_test.json", "metrics_sha256"),
                                         ("metrics_cv.json", "metrics_cv_sha256"), ("pr_curve_data.json", "pr_curve_data_sha256"),
                                         ("y_test.npy", "y_test_sha256"), ("y_test_scores.npy", "y_test_scores_sha256"),
                                         ("test_row_indices.npy", "test_row_indices_sha256")):
                if source[digest_key] is not None and file_digest(Path(source["run_dir"]) / filename) != source[digest_key]:
                    raise ValueError(f"A source artefact changed during generation: {source['run_dir']} / {filename}")
        for source in preflight_report["transfer_sources"].values():
            if file_digest(Path(source["path"])) != source["sha256"]:
                raise ValueError("A transfer source audit changed during generation")
            for variant in source["variants"].values():
                for evidence in (variant, *variant["artefacts"].values()):
                    if file_digest(Path(evidence["path"])) != evidence["sha256"]:
                        raise ValueError(f"A transfer artefact changed during generation: {evidence['path']}")
        shap_sources = [evidence for source in preflight_report["compatible_shap_sources"].values()
                        for evidence in source["artefacts"].values()]
        stability_source = preflight_report["variant_shap_stability"]
        shap_sources.extend((stability_source, stability_source["partition_manifest"],
                             *stability_source["artefacts"].values()))
        for evidence in shap_sources:
            if file_digest(Path(evidence["path"])) != evidence["sha256"]:
                raise ValueError(f"A SHAP source artefact changed during generation: {evidence['path']}")
        if collect_paired_evidence(RESULTS_DIR) != preflight_report["paired_analysis"]:
            raise ValueError("Paired sensitivity/control evidence changed during generation")
        if (collect_bootstrap_compatibility(RUN_MANIFEST, preflight_report["source_runs"])
                != preflight_report["bootstrap_compatibility"]):
            raise ValueError("Bootstrap compatibility evidence changed during generation")
        preflight_report["output_sha256"] = {
            path.relative_to(TABLES_BASE.parent).as_posix(): file_digest(path)
            for folder in (TABLES_BASE, FIGURES_BASE) for path in sorted(folder.rglob("*")) if path.is_file()
        }
        (TABLES_BASE.parent / "generation_qa.json").write_text(json.dumps(preflight_report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  Done! Tables: {TABLES_BASE}; figures: {FIGURES_BASE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
