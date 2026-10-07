"""Predefined CatBoost SMOTE-NC control with historical fixed hyperparameters.

No HPO or TEST-dependent selection takes place. The final model still uses the
primary standardised numerical plus OHE representation, not native CatBoost
categorical features. The intervention concerns representation during sampling
and nominal-aware resampling; it is not an isolated architecture comparison.

Use --design-only to validate the explicit pins and save the implementation
plan without reading the large dataset or fitting/resampling any model.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from categorical_preprocess import (
    CategoryOneHotRepresentation, MixedCategoryPreprocessor, validate_category_codes,
)
from data import load_dataset
from evaluation.metrics import bootstrap_ci, compute_all_metrics, find_threshold_maximizing_f2
from experiment_protocol import (
    DEFAULT_MANIFEST_PATH, DEFAULT_RESULTS_ROOT, PROJECT_ROOT, PROTOCOL_VERSION,
    array_sha256, file_sha256, sampler_diagnostics, software_versions,
)
from revision_transfer import validate_output_root
from revision_resources import exclusive_resource
from save_load import save_run


SEED = 42
FOLDS = 5
STRATEGY = "smotenc_control"
SOURCE_KEYS = {"baseline": "baf_base/catboost/none", "primary_smote": "baf_base/catboost/smote"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def resolve_source_runs(manifest_path):
    """Select explicit historical references and verify their pinned hashes."""
    manifest = _read_json(manifest_path)
    result = {}
    for role, key in SOURCE_KEYS.items():
        reference = manifest.get("historical_runs", {}).get(key)
        if not isinstance(reference, dict) or not reference.get("config_sha256"):
            raise ValueError(f"An explicit historical run/config SHA-256 pin is required for {key}.")
        run = Path(reference["run_dir"])
        run = (run if run.is_absolute() else PROJECT_ROOT / run).resolve()
        if file_sha256(run / "config.json") != reference["config_sha256"]:
            raise ValueError(f"The historical {role} source config differs from its pin.")
        config = _read_json(run / "config.json")
        expected_strategy = "none" if role == "baseline" else "smote"
        if (config.get("dataset") != "baf_base" or config.get("model") != "catboost"
                or config.get("strategy") != expected_strategy):
            raise ValueError("Historical categorical-control source has the wrong dataset/model/strategy.")
        if config.get("sample_fraction") not in (None, 1, 1.0):
            raise ValueError("The categorical-control reference must use the full Base split.")
        if config.get("missing_policy", "preserve") != "preserve":
            raise ValueError("The categorical-control reference must preserve the primary absence codes.")
        result[role] = {"run_dir": run, "config": config,
                        "config_sha256": reference["config_sha256"]}
    baseline, primary = result["baseline"]["config"], result["primary_smote"]["config"]
    if baseline["dataset_hash_sha256"] != primary["dataset_hash_sha256"]:
        raise ValueError("The baseline and primary SMOTE sources refer to different raw files.")
    if baseline["best_params"] != primary["best_params"]:
        raise ValueError("The primary SMOTE and baseline source hyperparameters differ.")
    if any(not key.startswith("classifier__") for key in baseline["best_params"]):
        raise ValueError("Fixed source hyperparameters must configure the classifier only.")
    if "classifier__iterations" not in baseline["best_params"]:
        raise ValueError("The full fixed iteration budget must be available in the baseline source.")
    for filename in ("metrics_test.json", "y_test.npy", "y_test_scores.npy"):
        if not (result["primary_smote"]["run_dir"] / filename).is_file():
            raise FileNotFoundError(f"The explicitly pinned primary SMOTE source lacks {filename}.")
    return result


def save_design(manifest_path=DEFAULT_MANIFEST_PATH, output_root=DEFAULT_RESULTS_ROOT, *, threads=2):
    if threads < 1:
        raise ValueError("The operational thread limit must be positive.")
    sources = resolve_source_runs(manifest_path)
    root = validate_output_root(output_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / "categorical_control" / "catboost" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    document = {
        "status": "design_validated_not_trained", "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "baf_base", "model": "catboost", "strategy": STRATEGY,
        "source_manifest": str(Path(manifest_path).resolve()), "source_manifest_sha256": file_sha256(manifest_path),
        "sources": {role: {"run_dir": str(record["run_dir"]), "config_sha256": record["config_sha256"]}
                    for role, record in sources.items()},
        "best_params": sources["baseline"]["config"]["best_params"],
        "outer_split": "Historical 80/20 stratified Base split; seed 42; original CSV row indices retained.",
        "inner_validation": "Five stratified shuffled DEV folds; seed 42; no HPO or early stopping.",
        "threshold": "Maximise F2 separately on each DEV validation fold; use their full-precision median.",
        "training_representation": "Original numerical mean imputation/StandardScaler plus fold-fitted ordinal category identifiers.",
        "resampling": "SMOTENC with explicit categorical indices, k_neighbors=5, random_state=42, sampling_strategy=auto.",
        "classifier_representation": "Numerical features followed by OHE; vocabulary fitted on original fold training data before resampling.",
        "unknown_categories": "Ordinal -1 maps to an all-zero OHE block; no validation/TEST vocabulary fitting.",
        "category_integrity": "Reject fractional, non-finite or out-of-vocabulary resampled category codes; never truncate them.",
        "missing_policy": "preserve", "threads": threads, "test_used_for_selection": False,
        "scientific_scope": "Fixed-parameter representation/resampling control, not a native-categorical CatBoost or pure architecture effect.",
        "uncertainty_scope": "Optional paired holdout bootstrap at the two fixed DEV-selected thresholds; not repeated-seed uncertainty.",
        "memory_note": "Full DEV resampling produces 1,582,354 rows; real OHE width is derived from the fitted vocabulary, not hard-coded. Reserve a dedicated RAM slot.",
        "source_code_sha256": {name: file_sha256(PROJECT_ROOT / "src" / name)
                               for name in ("revision_catboost_control.py", "categorical_preprocess.py",
                                            "data.py", "missing_values.py", "preprocess.py")},
        "software_versions": software_versions(),
        "deferred_command": (
            "python src/revision_runtime.py src/revision_catboost_control.py "
            "--bootstrap-iterations 1000 --threads 2"
        ),
    }
    _write_json(directory / "design.json", document)
    print(f"Validated categorical-control design: {directory}", flush=True)
    return directory, sources


def fit_control_model(frame, labels, fixed_params, *, stage, threads=2, classifier_factory=CatBoostClassifier):
    """Fit all preprocessing and resampling on this training partition only."""
    mixed = MixedCategoryPreprocessor().set_output(transform="default")
    training = mixed.fit_transform(frame)
    post = CategoryOneHotRepresentation(
        mixed.numeric_features_, mixed.categorical_features_, mixed.categorical_cardinalities_,
    ).set_output(transform="default")
    post.fit(training)
    # The project globally requests pandas transformer output. SMOTE-NC's
    # internal sparse OHE must explicitly retain its default array output.
    sampler_encoder = OneHotEncoder(handle_unknown="ignore").set_output(transform="default")
    sampler = SMOTENC(categorical_features=mixed.categorical_indices_, categorical_encoder=sampler_encoder, k_neighbors=5,
                      random_state=SEED, sampling_strategy="auto")
    resampled, resampled_labels = sampler.fit_resample(training, np.asarray(labels))
    categorical = validate_category_codes(resampled, mixed.categorical_indices_, mixed.categorical_cardinalities_)
    diagnostic = sampler_diagnostics(sampler, training, labels, resampled, resampled_labels, stage)
    diagnostic.update({
        "categorical_features": mixed.categorical_features_, "categorical_indices": mixed.categorical_indices_,
        "categorical_cardinalities": mixed.categorical_cardinalities_,
        "categorical_codes_integer_and_in_vocabulary": True,
        "unknown_resampled_category_codes": int(np.sum(categorical == -1)),
        "sampler_input_columns": len(mixed.get_feature_names_out()),
        "classifier_input_columns": len(post.get_feature_names_out()),
        "imputation_scaling_and_category_vocab_fitted_on_original_training_only": True,
        "validation_or_test_rows_used_for_resampling": False,
        "sampler_median_continuous_standard_deviation": {str(key): float(value)
                                                       for key, value in sampler.median_std_.items()},
    })
    del categorical, training, sampler
    gc.collect()
    classifier_input = post.transform(resampled)
    del resampled
    gc.collect()
    params = {key.removeprefix("classifier__"): value for key, value in fixed_params.items()}
    classifier = classifier_factory(**{
        "iterations": 2000, "silent": True, "random_state": SEED,
        **params, "thread_count": threads, "allow_writing_files": False,
    })
    classifier.fit(classifier_input, resampled_labels)
    del classifier_input, resampled_labels
    gc.collect()
    return Pipeline([("mixed_preprocessor", mixed), ("one_hot_representation", post),
                     ("classifier", classifier)]), diagnostic


def _paired_bootstrap(labels, control_scores, primary_scores, control_threshold, primary_threshold, iterations):
    rng = np.random.RandomState(SEED)
    values = {"PR-AUC": [], "ROC-AUC": [], "F2": []}
    for _ in range(iterations):
        positions = rng.choice(len(labels), len(labels), replace=True)
        target = labels[positions]
        if len(np.unique(target)) != 2:
            continue
        control, primary = control_scores[positions], primary_scores[positions]
        values["PR-AUC"].append(average_precision_score(target, control) - average_precision_score(target, primary))
        values["ROC-AUC"].append(roc_auc_score(target, control) - roc_auc_score(target, primary))
        values["F2"].append(fbeta_score(target, control >= control_threshold, beta=2, zero_division=0)
                            - fbeta_score(target, primary >= primary_threshold, beta=2, zero_division=0))
    return {key: {"ci_95_percent": np.percentile(sample, [2.5, 97.5]).tolist(),
                  "valid_replicates": len(sample)} for key, sample in values.items()}


def compare_primary_smote(source, labels, control_scores, threshold, metadata, iterations=0):
    """Check paired TEST alignment and retain the historical primary evidence."""
    run = source["run_dir"]
    if source["config"]["dataset_hash_sha256"] != metadata["raw_file_sha256"]:
        raise ValueError("Primary SMOTE source and categorical control raw-file hashes differ.")
    historical_labels = np.load(run / "y_test.npy", allow_pickle=False)
    primary_scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
    if not np.array_equal(labels, historical_labels):
        raise ValueError("Primary SMOTE and categorical control TEST labels or row order differ.")
    if len(primary_scores) != len(labels) or not np.isfinite(primary_scores).all():
        raise ValueError("Primary SMOTE scores are not finite/aligned with the shared TEST.")
    index_file = run / "test_row_indices.npy"
    if index_file.is_file() and not np.array_equal(np.load(index_file, allow_pickle=False), metadata["test_indices"]):
        raise ValueError("Primary SMOTE original TEST row indices differ from the control.")
    saved_metrics = _read_json(run / "metrics_test.json")
    primary_threshold = float(source["config"].get("threshold_exact", saved_metrics["threshold"]))
    primary_metrics = compute_all_metrics(labels, primary_scores, primary_threshold)
    if any(saved_metrics[key] != primary_metrics[key] for key in ("TP", "FP", "TN", "FN")):
        raise ValueError("The historical saved threshold does not reproduce its TEST confusion counts.")
    control_metrics = compute_all_metrics(labels, control_scores, threshold)
    exact = {
        "PR-AUC": float(average_precision_score(labels, control_scores) - average_precision_score(labels, primary_scores)),
        "ROC-AUC": float(roc_auc_score(labels, control_scores) - roc_auc_score(labels, primary_scores)),
        "F2": float(fbeta_score(labels, control_scores >= threshold, beta=2, zero_division=0)
                    - fbeta_score(labels, primary_scores >= primary_threshold, beta=2, zero_division=0)),
    }
    record = {
        "comparison": "SMOTENC control minus primary OHE-space SMOTE; fixed historical HP, same Base TEST",
        "source_run": str(run), "source_config_sha256": source["config_sha256"],
        "raw_file_sha256": metadata["raw_file_sha256"],
        "shared_test_row_indices_sha256": array_sha256(metadata["test_indices"]),
        "test_labels_exactly_equal": True,
        "historical_test_indices_explicitly_saved": index_file.is_file(),
        "alignment_basis": "Same raw CSV, deterministic seed-42 stratified split, and exact saved ordered TEST labels; explicit indices additionally checked when available.",
        "source_artefacts_sha256": {name: file_sha256(run / name) for name in ("config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy")},
        "primary_threshold": primary_threshold,
        "primary_threshold_precision": "full precision" if "threshold_exact" in source["config"] else "historical saved six-decimal threshold; confusion counts reproduced",
        "control_threshold_full_precision": threshold, "primary_metrics": primary_metrics,
        "control_metrics": control_metrics, "difference_full_precision": exact,
        "bootstrap_iterations": iterations, "bootstrap_seed": SEED,
        "test_used_for_policy_or_hyperparameter_selection": False,
        "scientific_scope": "Representation and nominal-aware resampling intervention; not a native-categorical architecture effect.",
    }
    if iterations:
        record["paired_difference_bootstrap"] = _paired_bootstrap(
            labels, control_scores, primary_scores, threshold, primary_threshold, iterations,
        )
    return record


def run_control(manifest_path=DEFAULT_MANIFEST_PATH, output_root=DEFAULT_RESULTS_ROOT, *, threads=2, bootstrap_iterations=0):
    if bootstrap_iterations < 0:
        raise ValueError("Bootstrap iterations must be non-negative.")
    audit_dir, sources = save_design(manifest_path, output_root, threads=threads)
    config_source = sources["baseline"]["config"]
    fixed_params = config_source["best_params"]
    dev, test, y_dev, y_test, metadata = load_dataset("baf_base", return_metadata=True)
    if metadata["raw_file_sha256"] != config_source["dataset_hash_sha256"]:
        raise ValueError("The categorical control raw CSV differs from the fixed-parameter source.")
    if len(dev) != config_source["train_samples"] or len(test) != config_source["test_samples"]:
        raise ValueError("Categorical-control DEV/TEST membership differs from the historical source.")
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_scores = np.full(len(y_dev), np.nan, dtype=np.float64)
    oof_fold_ids = np.zeros(len(y_dev), dtype=np.int64)
    results, diagnostics = [], []
    with threadpool_limits(limits=threads):
        for fold, (train_positions, validation_positions) in enumerate(folds.split(dev, y_dev), start=1):
            train, validation = dev.iloc[train_positions], dev.iloc[validation_positions]
            train_labels, validation_labels = y_dev.iloc[train_positions], y_dev.iloc[validation_positions]
            start = time.perf_counter()
            model, diagnostic = fit_control_model(train, train_labels, fixed_params, stage=f"fold_{fold}", threads=threads)
            scores = model.predict_proba(validation)[:, 1]
            tau, _ = find_threshold_maximizing_f2(validation_labels, scores)
            metrics = compute_all_metrics(validation_labels, scores, tau)
            metrics.update({"fold": fold, "threshold_exact": tau,
                            "train_row_indices_sha256": array_sha256(train.index.to_numpy(dtype=np.int64)),
                            "validation_row_indices_sha256": array_sha256(validation.index.to_numpy(dtype=np.int64)),
                            "train_rows": len(train), "validation_rows": len(validation),
                            "elapsed_seconds": time.perf_counter() - start})
            oof_scores[validation_positions], oof_fold_ids[validation_positions] = scores, fold
            results.append(metrics)
            diagnostics.append(diagnostic)
            fold_dir = audit_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=False)
            _write_json(fold_dir / "metrics_validation.json", metrics)
            _write_json(fold_dir / "sampler_diagnostics.json", diagnostic)
            for name, values in (("y_val.npy", np.asarray(validation_labels)), ("y_val_scores.npy", scores),
                                 ("validation_row_indices.npy", validation.index.to_numpy(dtype=np.int64))):
                with (fold_dir / name).open("xb") as stream:
                    np.save(stream, values, allow_pickle=False)
            joblib.dump(model, fold_dir / "model.joblib")
            print(f"Fold {fold}: AP={metrics['PR-AUC']:.6f}; tau={tau:.12g}; resampled n={diagnostic['rows_after']:,}", flush=True)
            del model, train, validation, train_labels, validation_labels, scores
            gc.collect()
        if not np.isfinite(oof_scores).all() or np.any(oof_fold_ids == 0):
            raise ValueError("Every original DEV row must have exactly one held-out fold prediction.")
        threshold = float(np.median([result["threshold_exact"] for result in results]))
        start = time.perf_counter()
        final_model, diagnostic = fit_control_model(dev, y_dev, fixed_params, stage="final_dev", threads=threads)
        training_seconds = time.perf_counter() - start
        diagnostics.append(diagnostic)
        start = time.perf_counter()
        scores_test = final_model.predict_proba(test)[:, 1]
        inference_seconds = time.perf_counter() - start
    metrics_test = compute_all_metrics(y_test, scores_test, threshold)
    comparison = compare_primary_smote(sources["primary_smote"], np.asarray(y_test), scores_test,
                                       threshold, metadata, bootstrap_iterations)
    ci = bootstrap_ci(y_test, scores_test, threshold, n_bootstrap=bootstrap_iterations) if bootstrap_iterations else None
    aggregate = {}
    for key in ("PR-AUC", "ROC-AUC", "F1", "F2", "brier_score", "precision_at_k", "recall_at_k"):
        aggregate[key + "_mean"] = round(float(np.mean([result[key] for result in results])), 6)
        aggregate[key + "_std"] = round(float(np.std([result[key] for result in results])), 6)
    aggregate["threshold_median"] = round(threshold, 6)
    aggregate["threshold_per_fold"] = [result["threshold_exact"] for result in results]
    config = {
        "dataset": "baf_base", "dataset_file": metadata["raw_file"], "dataset_hash_sha256": metadata["raw_file_sha256"],
        "split_seed": SEED, "split_ratio": "80/20 stratified", "sample_fraction": None,
        "train_samples": len(dev), "test_samples": len(test), "train_fraud": int(y_dev.sum()), "test_fraud": int(y_test.sum()),
        "model": "catboost", "strategy": STRATEGY, "missing_policy": "preserve", "cv_folds": FOLDS,
        "scoring": "average_precision (PR-AUC)", "best_params": fixed_params,
        "best_params_source": "Explicit historical Base baseline; no retuning or early stopping.",
        "baseline_run": str(sources["baseline"]["run_dir"]), "baseline_config_sha256": sources["baseline"]["config_sha256"],
        "threshold_rule": "maximise F2 on validation, take median across folds", "threshold_exact": threshold,
        "categorical_sampler": "SMOTENC; seed42; k_neighbors5; auto balance; original training vocabularies",
        "classifier_representation": "Standardised numerical plus OHE; not native CatBoost categorical features.",
        "classifier_input_columns": diagnostic["classifier_input_columns"], "threads": threads,
        "allow_writing_files": False, "train_time_s": training_seconds, "infer_time_s": inference_seconds,
        "bootstrap_iterations": bootstrap_iterations, "audit_directory": str(audit_dir),
        "test_used_for_selection": False,
        "scientific_scope": "Representation/resampling control; not an isolated architecture effect.",
    }
    run_dir = Path(save_run(
        model=final_model, metrics_cv={"per_fold": results, "aggregated": aggregate}, metrics_test=metrics_test,
        y_test=np.asarray(y_test), y_test_scores=scores_test, config=config, model_name="catboost",
        strategy=STRATEGY, dataset="baf_base", bootstrap_ci=ci, results_root=output_root,
        split_metadata=metadata, sampler_diagnostics=diagnostics, manifest_path=manifest_path,
        validation_evidence={"y_val": np.asarray(y_dev), "y_val_scores": oof_scores,
                             "row_indices": dev.index.to_numpy(dtype=np.int64), "fold_ids": oof_fold_ids,
                             "role": "fivefold_out_of_fold_categorical_control"},
    ))
    _write_json(run_dir / "primary_smote_comparison.json", comparison)
    _write_json(audit_dir / "completed_control.json", {
        "status": "complete", "run_dir": str(run_dir), "config_sha256": file_sha256(run_dir / "config.json"),
        "comparison_file_sha256": file_sha256(run_dir / "primary_smote_comparison.json"),
        "metrics_test": metrics_test, "difference_full_precision": comparison["difference_full_precision"],
    })
    print(f"Completed CatBoost categorical control: {run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1 or args.bootstrap_iterations < 0:
        parser.error("Threads must be positive and bootstrap iterations non-negative.")
    if args.design_only:
        save_design(args.manifest, args.results_root, threads=args.threads)
    else:
        with exclusive_resource(args.results_root, "baf_training_ram", "CatBoost categorical-aware control"):
            run_control(args.manifest, args.results_root, threads=args.threads,
                        bootstrap_iterations=args.bootstrap_iterations)


if __name__ == "__main__":
    main()
