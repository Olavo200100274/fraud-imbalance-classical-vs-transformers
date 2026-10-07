"""
Fraud Detection Pipeline — Leakage-Free Evaluation Protocol
============================================================
Supports baseline (strategy=None) and imbalance handling strategies
on multiple fraud detection datasets (ULB 2013, BAF Base, etc.).

For each supervised model:
  strategy=none (baseline):
    1. Hyperparameter tuning  — Optuna (TPE), scoring = PR-AUC (average_precision)
    2. Threshold selection    — 5-fold CV, maximise F2 per fold, τ = median
    3. Final training         — best params on full training partition
    4. Test evaluation        — holdout test with median τ
    5. Persist                — config + model + metrics_cv + metrics_test + plots

  strategy=<resampling|weights>:
    1. Load best params       — from the baseline run (no re-tuning)
    2. Threshold selection    — 5-fold CV with resampling inside each fold
    3. Final training         — best params on resampled full training partition
    4. Test evaluation        — holdout test with median τ
    5. Persist

For OCSVM (anomaly-detection baseline):
  - Trained only on class-0 (legitimate) samples
  - No imbalance strategies apply (anomaly-detection paradigm)
  - Scoring = negated decision_function
  - Same threshold selection + evaluation protocol

Usage:
  python main.py --models logreg --strategy none          # baseline
  python main.py --models logreg rf --strategy smote      # single strategy
  python main.py --models all --strategy all              # full factorial
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

from data import load_dataset, get_dataset_info, DATASET_REGISTRY
from preprocess import get_preprocessor
from revision_resources import baf_training_resource
from models.logreg import get_pipeline_and_params as get_logreg
from models.rf import get_pipeline_and_params as get_rf
from models.lgbm import get_pipeline_and_params as get_lgbm
from models.catboost import get_pipeline_and_params as get_catboost
from models.ocsvm import get_pipeline_and_params as get_ocsvm
from evaluation.metrics import (
    find_threshold_maximizing_f2,
    compute_all_metrics,
    bootstrap_ci,
)
from save_load import save_run
from experiment_protocol import (
    DEFAULT_RESULTS_ROOT, PROTOCOL_VERSION, resolve_baseline_run,
    sampler_diagnostics, optuna_trials_record, file_sha256, array_sha256,
    load_sampler_checkpoint, save_sampler_checkpoint,
    capture_source_provenance, validate_revision_destinations,
)
from strategies.balancing import (
    STRATEGY_NAMES,
    RESAMPLING_STRATEGIES,
    get_sampler,
    apply_class_weights,
)


# ── Constants ─────────────────────────────────────────────────────────────
SPLIT_SEED = 42
CV_SPLITS = 5
BOOTSTRAP_ITERATIONS = 1000
EARLY_STOP_MODELS = {"lgbm", "catboost"}
EARLY_STOP_ROUNDS = 50

# Inner CV — same folds for tuning AND threshold selection
CV = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SPLIT_SEED)

SUPERVISED_MODELS = {
    "logreg": get_logreg,
    "rf": get_rf,
    "lgbm": get_lgbm,
    "catboost": get_catboost,
}


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fraud Detection — Leakage-Free Evaluation Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_REGISTRY.keys()),
        required=True,
        help="Dataset to use: " + " | ".join(DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["logreg", "rf", "lgbm", "catboost", "ocsvm", "all"],
        required=True,
        help="Models to run: logreg | rf | lgbm | catboost | ocsvm | all",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="none",
        choices=STRATEGY_NAMES + ["all"],
        help=(
            "Imbalance handling strategy. "
            "'none' = baseline (default). "
            "'all' = run every non-baseline strategy. "
            "OCSVM is always run with strategy=none regardless."
        ),
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help=(
            "Stratified subsample fraction (0 < sample <= 1). "
            "E.g. --sample 0.01 uses 1%% of the dataset (preserving "
            "fraud prevalence). Useful for smoke-testing the pipeline."
        ),
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter tuning (default: 50).",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--baseline-run", type=Path, default=None,
                        help="Explicit baseline for intervention runs; otherwise use the manifest.")
    parser.add_argument("--fixed-params-run", type=Path, default=None,
                        help="Explicit BAF baseline HP for a fixed-parameter sensitivity run; skip HPO.")
    parser.add_argument("--missing-policy", choices=["preserve", "nan_indicators"], default="preserve")
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--recover-interrupted-trials", action="store_true",
                        help="Mark interrupted RUNNING Optuna trials failed; use only when no matching job is running.")
    parser.add_argument("--resume-optuna-study", type=Path, default=None,
                        help="Explicit existing SQLite study after a reviewed operational code change.")
    parser.add_argument("--recover-validation-only", action="store_true",
                        help="Recover BAF/preserve validation scores using historical HP and the frozen final model.")
    return parser.parse_args()


# ── Dataset fingerprint (reproducibility) ─────────────────────────────────
def _file_hash(path, algorithm="sha256"):
    """Return hex digest of a file for reproducibility logging."""
    h = hashlib.new(algorithm)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


# ── supervised models ─────────────────────────────────────────────────────
def run_supervised(model_name, X_train, X_test, y_train, y_test,
                   preprocessor, dataset_hash, dataset_name, dataset_file,
                   n_trials=50, context=None):
    """Full leakage-free protocol for a supervised classifier (strategy=None)."""

    context = dict(context or {})
    get_fn = SUPERVISED_MODELS[model_name]
    pipeline, suggest_fn = get_fn(preprocessor)

    print(f"\n{'=' * 60}")
    print(f"  {model_name.upper()} — Strategy: None")
    print(f"{'=' * 60}")

    fixed_run = context.get("fixed_params_run")
    recovery_only = context.get("recover_validation_only", False)
    if recovery_only and fixed_run is None:
        raise ValueError("Validation-only recovery requires an explicit fixed-parameter baseline.")
    tuning_trials = None
    if fixed_run is not None:
        if dataset_name != "baf_base":
            raise ValueError("Historical fixed-parameter borrowing is restricted to BAF analyses.")
        run_dir = resolve_baseline_run(dataset_name, model_name, explicit_run=fixed_run)
        source_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if source_config.get("dataset_hash_sha256") != dataset_hash:
            raise ValueError("The fixed-parameter source refers to a different raw dataset.")
        best_params = source_config["best_params"]
        best_pipeline = clone(pipeline).set_params(**best_params)
        tuning_time, n_pruned, n_trials = 0.0, 0, 0
        _uses_early_stop = model_name in EARLY_STOP_MODELS
        context["selected_baseline_run"] = str(run_dir)
        context["baseline_config_sha256"] = file_sha256(run_dir / "config.json")
        if recovery_only:
            _validate_recovery_source(context, source_config, X_train, X_test, y_train, y_test, dataset_name)
        print(f"  Fixed hyperparameters loaded from {run_dir}; no new tuning.")
    else:
        # ── 1. Hyperparameter tuning (Optuna + pruning, scoring = PR-AUC) ─
        print(f"\n[1/4] Hyperparameter tuning ({model_name}, {n_trials} trials) ...")
        t0 = time.time()

        # Pre-compute fold-level preprocessed data — avoids re-fitting
        # the preprocessor in every trial (same folds for tuning & threshold).
        folds_data = []
        for train_idx, val_idx in CV.split(X_train, y_train):
            fold_pre = clone(preprocessor)
            X_ft = fold_pre.fit_transform(X_train.iloc[train_idx])
            X_fv = fold_pre.transform(X_train.iloc[val_idx])
            folds_data.append((
                X_ft, X_fv,
                y_train.iloc[train_idx], y_train.iloc[val_idx],
            ))

        _base_clf = pipeline.named_steps["classifier"]
        _uses_early_stop = model_name in EARLY_STOP_MODELS

        def objective(trial):
            params = suggest_fn(trial)
            trial.set_user_attr("pipeline_params", params)
            clf_params = {k.replace("classifier__", ""): v for k, v in params.items()}

            scores = []
            best_iters = []
            for step, (X_ft, X_fv, y_ft, y_fv) in enumerate(folds_data):
                clf = clone(_base_clf)
                clf.set_params(**clf_params)

                _can_early_stop = (
                    _uses_early_stop and len(np.unique(y_fv)) > 1
                )
                if _can_early_stop:
                    if model_name == "lgbm":
                        import lightgbm as _lgb
                        clf.fit(
                            X_ft, y_ft,
                            eval_set=[(X_fv, y_fv)],
                            callbacks=[
                                _lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                                _lgb.log_evaluation(0),
                            ],
                        )
                    else:  # catboost
                        clf.fit(
                            X_ft, y_ft,
                            eval_set=[(X_fv, y_fv)],
                            early_stopping_rounds=EARLY_STOP_ROUNDS,
                        )
                    best_iters.append(clf.best_iteration_)
                else:
                    clf.fit(X_ft, y_ft)

                score = average_precision_score(
                    y_fv, clf.predict_proba(X_fv)[:, 1],
                )
                scores.append(score)

                # Report running mean — enables pruning of unpromising trials
                trial.report(np.mean(scores), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if best_iters:
                trial.set_user_attr("best_iterations", best_iters)
            return np.mean(scores)

        def _log_trial(study, trial):
            if trial.value is not None:
                print(
                    f"  Trial {trial.number + 1:>3}/{n_trials}: "
                    f"CV PR-AUC = {trial.value:.4f}  "
                    f"(best: {study.best_value:.4f})"
                )
            else:
                print(f"  Trial {trial.number + 1:>3}/{n_trials}: PRUNED")

        tuning_dir = Path(context.get("results_root") or DEFAULT_RESULTS_ROOT) / "tuning_cache"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        study_signature = hashlib.sha256(json.dumps({
            "protocol": PROTOCOL_VERSION, "dataset_sha256": dataset_hash,
            "dataset": dataset_name, "model": model_name,
            "sample_fraction": context.get("sample_fraction"),
            "missing_policy": context.get("missing_policy", "preserve"),
            "dev_indices_sha256": array_sha256(X_train.index.to_numpy(dtype=np.int64)),
            "features": list(X_train.columns),
            "source_sha256": {
                str(path.relative_to(Path(__file__).parent)): file_sha256(path)
                for path in (
                    Path(__file__), Path(__file__).parent / "data.py",
                    Path(__file__).parent / "preprocess.py",
                    Path(__file__).parent / "models" / f"{model_name}.py",
                )
            },
        }, sort_keys=True).encode("utf-8")).hexdigest()
        storage_path = tuning_dir / f"{dataset_name}_{model_name}_{study_signature[:16]}.sqlite3"
        explicit_storage = context.get("resume_optuna_study")
        if explicit_storage is not None:
            storage_path = Path(explicit_storage).resolve()
            if (storage_path.parent != tuning_dir.resolve()
                    or not storage_path.name.startswith(f"{dataset_name}_{model_name}_")
                    or not storage_path.is_file()):
                raise ValueError("The explicit Optuna checkpoint must match this dataset/model and output root.")
        sampler_path = storage_path.with_suffix(".sampler.joblib")
        tpe_sampler, sampler_checkpoint = load_sampler_checkpoint(sampler_path)
        if tpe_sampler is None:
            tpe_sampler = optuna.samplers.TPESampler(seed=SPLIT_SEED)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{storage_path.resolve().as_posix()}",
        )
        selected_study_name = f"{PROTOCOL_VERSION}_{study_signature}"
        if explicit_storage is not None:
            summaries = optuna.study.get_all_study_summaries(storage=storage)
            if len(summaries) != 1 or not summaries[0].study_name.startswith(f"{PROTOCOL_VERSION}_"):
                storage.remove_session()
                storage.engine.dispose()
                raise ValueError("The explicit checkpoint must contain exactly one compatible protocol study.")
            selected_study_name = summaries[0].study_name
        study = optuna.create_study(
            direction="maximize",
            sampler=tpe_sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=1,
            ),
            storage=storage,
            study_name=selected_study_name,
            load_if_exists=True,
        )
        study_metadata = {
            "dataset": dataset_name, "model": model_name, "raw_sha256": dataset_hash,
            "sample_fraction": context.get("sample_fraction"),
            "missing_policy": context.get("missing_policy", "preserve"),
            "dev_indices_sha256": array_sha256(X_train.index.to_numpy(dtype=np.int64)),
            "features": list(X_train.columns),
        }
        previous_metadata = study.user_attrs.get("data_protocol")
        if previous_metadata is not None and previous_metadata != study_metadata:
            storage.remove_session()
            storage.engine.dispose()
            raise ValueError("The resumed Optuna study has incompatible dataset/split/representation metadata.")
        study.set_user_attr("data_protocol", study_metadata)
        if study.trials and sampler_checkpoint is None:
            storage.remove_session()
            storage.engine.dispose()
            raise ValueError("Existing trials cannot be resumed without their persisted TPE state.")
        interrupted = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        if interrupted and not context.get("recover_interrupted_trials"):
            storage.remove_session()
            storage.engine.dispose()
            raise RuntimeError(
                "The persisted study contains RUNNING trials. Confirm that no matching job is "
                "active before using --recover-interrupted-trials."
            )
        for trial in interrupted:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        remaining_trials = max(0, n_trials - len(study.trials))
        print(f"  Optuna checkpoint: {len(study.trials)} existing, {remaining_trials} remaining trials")
        def _checkpoint_sampler(study, trial):
            save_sampler_checkpoint(study, trial, sampler_path)

        try:
            study.optimize(objective, n_trials=remaining_trials,
                           callbacks=[_checkpoint_sampler, _log_trial])
        except BaseException:
            storage.remove_session()
            storage.engine.dispose()
            raise
        context["optuna_storage"] = str(storage_path.resolve())
        context["optuna_study_name"] = study.study_name
        context["optuna_study_signature"] = selected_study_name[len(PROTOCOL_VERSION) + 1:]
        context["optuna_runtime_signature"] = study_signature
        context["optuna_sampler_checkpoint"] = sampler_checkpoint

        n_pruned = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ])
        tuning_time = time.time() - t0
        best_params = study.best_trial.user_attrs["pipeline_params"]
        best_pipeline = clone(pipeline)
        best_pipeline.set_params(**best_params)

        # For early-stop models, set n_estimators/iterations to optimal value
        if _uses_early_stop:
            best_iters = study.best_trial.user_attrs.get("best_iterations", [])
            if best_iters:
                iter_key = "iterations" if model_name == "catboost" else "n_estimators"
                best_n = int(np.median(best_iters))
                if model_name == "catboost":
                    best_n += 1  # 0-based → count
                best_pipeline.named_steps["classifier"].set_params(**{iter_key: best_n})
                best_params[f"classifier__{iter_key}"] = best_n
                print(f"  Early stopping: {iter_key} = {best_n} (median of fold best iters)")

        print(f"\n  Best params : {best_params}")
        print(f"  CV PR-AUC   : {study.best_value:.4f}")
        print(f"  Trials      : {n_trials} total, {n_pruned} pruned")
        print(f"  Tuning time : {tuning_time:.1f}s")

        tuning_trials = optuna_trials_record(study)
        storage.remove_session()
        storage.engine.dispose()

    # ── 2. Threshold selection (5-fold CV, maximise F2 → median τ) ────
    #    Also collect per-fold validation metrics for metrics_cv.json
    print(f"\n[2/4] Threshold selection ({model_name}) ...")
    validation_start = time.time()
    fold_results = []
    oof_scores = np.full(len(y_train), np.nan, dtype=np.float64)
    oof_fold_ids = np.zeros(len(y_train), dtype=np.int64)
    resampling_records = []

    for i, (train_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_model = clone(best_pipeline)
        fold_model.fit(X_fold_train, y_fold_train)

        y_val_scores = fold_model.predict_proba(X_fold_val)[:, 1]
        oof_scores[val_idx] = y_val_scores
        oof_fold_ids[val_idx] = i + 1
        tau, _ = find_threshold_maximizing_f2(y_fold_val, y_val_scores)
        fold_metrics = compute_all_metrics(y_fold_val, y_val_scores, tau)
        fold_metrics["threshold_exact"] = tau
        fold_metrics["fold"] = i + 1
        fold_metrics["train_row_indices_sha256"] = array_sha256(X_fold_train.index.to_numpy(dtype=np.int64))
        fold_metrics["validation_row_indices_sha256"] = array_sha256(X_fold_val.index.to_numpy(dtype=np.int64))

        fold_results.append(fold_metrics)
        print(f"  Fold {i + 1}: τ={tau:.6f}  PR-AUC={fold_metrics['PR-AUC']:.4f}"
              f"  F1={fold_metrics['F1']:.4f}  F2={fold_metrics['F2']:.4f}")

    # Aggregate CV metrics
    cv_keys = ["PR-AUC", "ROC-AUC", "F1", "F2", "brier_score", "precision_at_k", "recall_at_k"]
    cv_summary = {}
    for k in cv_keys:
        vals = [fr[k] for fr in fold_results]
        cv_summary[f"{k}_mean"] = round(float(np.mean(vals)), 6)
        cv_summary[f"{k}_std"] = round(float(np.std(vals)), 6)

    thresholds = [fr["threshold_exact"] for fr in fold_results]
    tau_final = float(np.median(thresholds))
    cv_summary["threshold_median"] = round(tau_final, 6)
    cv_summary["threshold_per_fold"] = [round(float(t), 6) for t in thresholds]

    metrics_cv = {
        "per_fold": fold_results,
        "aggregated": cv_summary,
    }

    print(f"  ── CV Aggregated ──")
    print(f"  PR-AUC  : {cv_summary['PR-AUC_mean']:.4f} ± {cv_summary['PR-AUC_std']:.4f}")
    print(f"  ROC-AUC : {cv_summary['ROC-AUC_mean']:.4f} ± {cv_summary['ROC-AUC_std']:.4f}")
    print(f"  F1      : {cv_summary['F1_mean']:.4f} ± {cv_summary['F1_std']:.4f}")
    print(f"  F2      : {cv_summary['F2_mean']:.4f} ± {cv_summary['F2_std']:.4f}")
    print(f"  Median threshold: τ = {tau_final:.6f}")
    validation_recovery_time = time.time() - validation_start

    # ── 3. Final training on full training partition ──────────────────
    print(f"\n[3/4] Training final model ({model_name}) ...")
    t0 = time.time()
    if recovery_only:
        final_model = joblib.load(run_dir / "model.joblib")
        train_time = source_config["train_time_s"]
        context.update(_recovery_cost_metadata(run_dir, source_config, validation_recovery_time))
        print("  Reusing the historical full-DEV model; no new final fit.")
    else:
        final_model = clone(best_pipeline)
        final_model.fit(X_train, y_train)
        train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # ── 4. Evaluation on holdout test set ─────────────────────────────
    print(f"\n[4/4] Evaluating on holdout test ({model_name}) ...")
    t0 = time.time()
    y_test_scores = final_model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t0
    if recovery_only:
        historical_scores = np.load(run_dir / "y_test_scores.npy", allow_pickle=False)
        maximum_difference = float(np.max(np.abs(y_test_scores - historical_scores)))
        if not np.allclose(y_test_scores, historical_scores, rtol=0, atol=1e-12):
            raise ValueError("Recovered frozen-model TEST scores disagree with the historical scores.")
        context["recovery_evidence"].update({
            "test_scores_recomputed": True, "test_scores_max_absolute_difference": maximum_difference,
            "test_score_comparison_absolute_tolerance": 1e-12,
        })

    metrics_test = compute_all_metrics(y_test, y_test_scores, tau_final)

    # Bootstrap CI on test
    bootstrap_iterations = context.get("bootstrap_iterations", BOOTSTRAP_ITERATIONS)
    print(f"  Bootstrap CI: {bootstrap_iterations} iterations" if bootstrap_iterations
          else "  Bootstrap CI skipped for this run.")
    ci = (bootstrap_ci(y_test, y_test_scores, tau_final, n_bootstrap=bootstrap_iterations)
          if bootstrap_iterations else None)

    _print_results(metrics_test, ci)

    # ── Config (full reproducibility record) ──────────────────────────
    config = {
        "dataset": dataset_name,
        "dataset_file": dataset_file,
        "dataset_hash_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "model": model_name,
        "strategy": "none",
        "cv_folds": CV_SPLITS,
        "scoring": "average_precision (PR-AUC)",
        "tuning": ("fixed hyperparameters; no new tuning" if fixed_run else "Optuna (TPE sampler + MedianPruner)"),
        "n_trials": n_trials,
        "n_pruned": n_pruned,
        "early_stopping_rounds": EARLY_STOP_ROUNDS if _uses_early_stop else None,
        "threshold_rule": "maximise F2 on validation, take median across folds",
        "threshold_exact": tau_final,
        "best_params": best_params,
        "tuning_time_s": round(tuning_time, 2),
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 4),
    }

    # ── Save everything ──────────────────────────────────────────────
    _save_completed_run(
        context=context,
        validation_evidence={
            "y_val": np.asarray(y_train), "y_val_scores": oof_scores,
            "row_indices": X_train.index.to_numpy(dtype=np.int64),
            "fold_ids": oof_fold_ids, "role": "fivefold_out_of_fold",
        },
        tuning_trials=tuning_trials,
        model=final_model,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        y_test=np.asarray(y_test),
        y_test_scores=y_test_scores,
        config=config,
        model_name=model_name,
        strategy="none",
        dataset=dataset_name,
        bootstrap_ci=ci,
    )

    return final_model, metrics_test


# ── Load baseline best_params ─────────────────────────────────────────────
def _load_baseline_params(model_name, dataset, context=None):
    """Load explicitly pinned baseline parameters, without timestamp discovery."""
    context = context if context is not None else {}
    run_dir = resolve_baseline_run(
        dataset, model_name, results_root=context.get("results_root"),
        manifest_path=context.get("manifest_path"),
        explicit_run=context.get("baseline_run"),
    )
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("dataset_hash_sha256") != context.get("dataset_hash", config.get("dataset_hash_sha256")):
        raise ValueError("Baseline and intervention raw dataset hashes differ.")
    if config.get("sample_fraction") != context.get("sample_fraction"):
        raise ValueError("Baseline and intervention sample fractions differ.")
    if config.get("missing_policy", "preserve") != context.get("missing_policy", "preserve"):
        raise ValueError("Baseline and intervention missing-value policies differ.")
    provenance = config.get("data_provenance") or {}
    intended_split = context.get("split_metadata") or {}
    if dataset == "ulb_2013" and config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Corrected ULB interventions require a duplicate-safe revised baseline.")
    for split in ("dev", "test"):
        key = f"{split}_indices_sha256"
        if intended_split.get(key) and provenance.get(key) != intended_split[key]:
            raise ValueError(f"Baseline and intervention {split.upper()} row indices differ.")
    context["selected_baseline_run"] = str(run_dir)
    context["baseline_config_sha256"] = file_sha256(run_dir / "config.json")
    print(f"  Pinned baseline: {run_dir}")
    return config["best_params"]


def _save_completed_run(context=None, **kwargs):
    """Save immutable corrected artefacts and their complete selection evidence."""
    context = context or {}
    config = kwargs["config"]
    config.update({
        "missing_policy": context.get("missing_policy", "preserve"),
        "sample_fraction": context.get("sample_fraction"),
        "baseline_run": context.get("selected_baseline_run"),
        "baseline_config_sha256": context.get("baseline_config_sha256"),
        "bootstrap_iterations": context.get("bootstrap_iterations", BOOTSTRAP_ITERATIONS),
        "optuna_storage": context.get("optuna_storage"),
        "optuna_study_name": context.get("optuna_study_name"),
        "optuna_study_signature": context.get("optuna_study_signature"),
        "optuna_runtime_signature": context.get("optuna_runtime_signature"),
        "optuna_sampler_checkpoint": context.get("optuna_sampler_checkpoint"),
        **(context.get("recovery_evidence") or {}),
        **(context.get("source_provenance") or {}),
    })
    return save_run(
        **kwargs, results_root=context.get("results_root"),
        manifest_path=context.get("manifest_path"),
        split_metadata=context.get("split_metadata"),
    )


# ── Supervised model with imbalance strategy ──────────────────────────────
def run_supervised_strategy(model_name, strategy, X_train, X_test,
                            y_train, y_test, preprocessor, dataset_hash,
                            dataset_name, dataset_file, context=None):
    """
    Leakage-free protocol for a supervised classifier with an imbalance
    handling strategy (resampling or class weights).

    Differences from baseline:
    - No Optuna tuning — best_params are loaded from the baseline run.
    - Resampling is applied INSIDE each CV fold (no leakage).
    - Class weights are injected into the classifier before training.
    """

    context = dict(context or {})
    get_fn = SUPERVISED_MODELS[model_name]
    pipeline, _ = get_fn(preprocessor)
    best_params = _load_baseline_params(model_name, dataset=dataset_name, context=context)

    # Set baseline hyperparameters (no re-tuning)
    pipeline.set_params(**best_params)

    # Inject class weights if strategy == "weights"
    if strategy == "weights":
        apply_class_weights(pipeline)

    print(f"\n{'=' * 60}")
    print(f"  {model_name.upper()} — Strategy: {strategy}")
    print(f"{'=' * 60}")

    # ── 1. Threshold selection (5-fold CV, maximise F2 → median τ) ────
    #    Resampling applied INSIDE each fold to prevent leakage.
    print(f"\n[1/3] Threshold selection ({model_name}, {strategy}) ...")
    fold_results = []
    oof_scores = np.full(len(y_train), np.nan, dtype=np.float64)
    oof_fold_ids = np.zeros(len(y_train), dtype=np.int64)
    resampling_records = []

    for i, (train_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Apply resampling ONLY to the fold training data
        if strategy in RESAMPLING_STRATEGIES:
            sampler = get_sampler(strategy)
            # Clone preprocessor per fold to avoid state leakage
            fold_pre = clone(pipeline.named_steps["preprocessor"])
            X_fold_pre = fold_pre.fit_transform(X_fold_train)
            X_fold_res, y_fold_res = sampler.fit_resample(X_fold_pre, y_fold_train)
            resampling_records.append(sampler_diagnostics(
                sampler, X_fold_pre, y_fold_train, X_fold_res, y_fold_res, f"fold_{i + 1}",
            ))

            # Train only the classifier step (data already preprocessed)
            fold_clf = clone(pipeline.named_steps["classifier"])
            fold_clf.fit(X_fold_res, y_fold_res)

            # Score validation data through the fold preprocessor
            X_val_pre = fold_pre.transform(X_fold_val)
            y_val_scores = fold_clf.predict_proba(X_val_pre)[:, 1]
        else:
            # strategy == "weights" — no resampling, just class weights
            fold_model = clone(pipeline)
            fold_model.fit(X_fold_train, y_fold_train)
            y_val_scores = fold_model.predict_proba(X_fold_val)[:, 1]

        oof_scores[val_idx] = y_val_scores
        oof_fold_ids[val_idx] = i + 1
        tau, _ = find_threshold_maximizing_f2(y_fold_val, y_val_scores)
        fold_metrics = compute_all_metrics(y_fold_val, y_val_scores, tau)
        fold_metrics["threshold_exact"] = tau
        fold_metrics["fold"] = i + 1
        fold_metrics["train_row_indices_sha256"] = array_sha256(X_fold_train.index.to_numpy(dtype=np.int64))
        fold_metrics["validation_row_indices_sha256"] = array_sha256(X_fold_val.index.to_numpy(dtype=np.int64))

        fold_results.append(fold_metrics)
        print(f"  Fold {i + 1}: τ={tau:.6f}  PR-AUC={fold_metrics['PR-AUC']:.4f}"
              f"  F1={fold_metrics['F1']:.4f}  F2={fold_metrics['F2']:.4f}")

    # Aggregate CV metrics
    cv_keys = ["PR-AUC", "ROC-AUC", "F1", "F2", "brier_score", "precision_at_k", "recall_at_k"]
    cv_summary = {}
    for k in cv_keys:
        vals = [fr[k] for fr in fold_results]
        cv_summary[f"{k}_mean"] = round(float(np.mean(vals)), 6)
        cv_summary[f"{k}_std"] = round(float(np.std(vals)), 6)

    thresholds = [fr["threshold_exact"] for fr in fold_results]
    tau_final = float(np.median(thresholds))
    cv_summary["threshold_median"] = round(tau_final, 6)
    cv_summary["threshold_per_fold"] = [round(float(t), 6) for t in thresholds]

    metrics_cv = {
        "per_fold": fold_results,
        "aggregated": cv_summary,
    }

    print(f"  ── CV Aggregated ──")
    print(f"  PR-AUC  : {cv_summary['PR-AUC_mean']:.4f} ± {cv_summary['PR-AUC_std']:.4f}")
    print(f"  ROC-AUC : {cv_summary['ROC-AUC_mean']:.4f} ± {cv_summary['ROC-AUC_std']:.4f}")
    print(f"  F1      : {cv_summary['F1_mean']:.4f} ± {cv_summary['F1_std']:.4f}")
    print(f"  F2      : {cv_summary['F2_mean']:.4f} ± {cv_summary['F2_std']:.4f}")
    print(f"  Median threshold: τ = {tau_final:.6f}")

    # ── 2. Final training on full training partition ──────────────────
    print(f"\n[2/3] Training final model ({model_name}, {strategy}) ...")
    t0 = time.time()

    if strategy in RESAMPLING_STRATEGIES:
        # Preprocess → resample → train classifier only
        final_preprocessor = clone(pipeline.named_steps["preprocessor"])
        X_train_pre = final_preprocessor.fit_transform(X_train)
        sampler = get_sampler(strategy)
        X_train_res, y_train_res = sampler.fit_resample(X_train_pre, y_train)

        final_clf = clone(pipeline.named_steps["classifier"])
        final_clf.fit(X_train_res, y_train_res)

        # Compose a Pipeline from the fitted components for saving
        final_model = Pipeline([
            ("preprocessor", final_preprocessor),
            ("classifier", final_clf),
        ])
    else:
        # strategy == "weights" — class weights already set in pipeline
        final_model = clone(pipeline)
        final_model.fit(X_train, y_train)

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    if strategy in RESAMPLING_STRATEGIES:
        resampling_records.append(sampler_diagnostics(
            sampler, X_train_pre, y_train, X_train_res, y_train_res, "final_dev",
        ))
        print(f"  Resampled training: {len(X_train_res):,} samples "
              f"(originally {len(X_train):,})")

    # ── 3. Evaluation on holdout test set ─────────────────────────────
    print(f"\n[3/3] Evaluating on holdout test ({model_name}, {strategy}) ...")
    t0 = time.time()
    y_test_scores = final_model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t0

    metrics_test = compute_all_metrics(y_test, y_test_scores, tau_final)

    # Bootstrap CI on test
    bootstrap_iterations = context.get("bootstrap_iterations", BOOTSTRAP_ITERATIONS)
    print(f"  Bootstrap CI: {bootstrap_iterations} iterations" if bootstrap_iterations
          else "  Bootstrap CI skipped for this run.")
    ci = (bootstrap_ci(y_test, y_test_scores, tau_final, n_bootstrap=bootstrap_iterations)
          if bootstrap_iterations else None)

    _print_results(metrics_test, ci)

    # ── Config (full reproducibility record) ──────────────────────────
    config = {
        "dataset": dataset_name,
        "dataset_file": dataset_file,
        "dataset_hash_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "model": model_name,
        "strategy": strategy,
        "cv_folds": CV_SPLITS,
        "scoring": "average_precision (PR-AUC)",
        "threshold_rule": "maximise F2 on validation, take median across folds",
        "threshold_exact": tau_final,
        "best_params": best_params,
        "best_params_source": "loaded from baseline run (no re-tuning)",
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 4),
    }

    # ── Save everything ──────────────────────────────────────────────
    _save_completed_run(
        context=context,
        sampler_diagnostics=resampling_records,
        validation_evidence={
            "y_val": np.asarray(y_train), "y_val_scores": oof_scores,
            "row_indices": X_train.index.to_numpy(dtype=np.int64),
            "fold_ids": oof_fold_ids, "role": "fivefold_out_of_fold",
        },
        model=final_model,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        y_test=np.asarray(y_test),
        y_test_scores=y_test_scores,
        config=config,
        model_name=model_name,
        strategy=strategy,
        dataset=dataset_name,
        bootstrap_ci=ci,
    )

    return final_model, metrics_test
def run_ocsvm(X_train, X_test, y_train, y_test, preprocessor, dataset_hash,
              dataset_name, dataset_file, context=None):
    """
    Leakage-free protocol for One-Class SVM.

    * Trained only on class-0 (legitimate) samples.
    * Scoring = negated decision_function (higher → more anomalous).
    * No hyperparameter grid (fixed config).
    * Same threshold selection + evaluation protocol.
    """
    context = dict(context or {})
    recovery_only = context.get("recover_validation_only", False)
    print(f"\n{'=' * 60}")
    print(f"  OCSVM — Anomaly Detection Baseline")
    print(f"{'=' * 60}")

    pipeline, _ = get_ocsvm(preprocessor)
    ocsvm_params = {"kernel": "rbf", "nu": 0.01, "gamma": "scale"}
    if context.get("fixed_params_run"):
        source_run = resolve_baseline_run(
            dataset_name, "ocsvm", explicit_run=context["fixed_params_run"],
        )
        source_config = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
        if source_config.get("dataset_hash_sha256") != dataset_hash:
            raise ValueError("The OCSVM source refers to a different raw dataset.")
        ocsvm_params = source_config["best_params"]
        pipeline.named_steps["classifier"].set_params(**ocsvm_params)
        context["selected_baseline_run"] = str(source_run)
        context["baseline_config_sha256"] = file_sha256(source_run / "config.json")
        if recovery_only:
            _validate_recovery_source(context, source_config, X_train, X_test, y_train, y_test, dataset_name)
    elif recovery_only:
        raise ValueError("OCSVM validation-only recovery requires a fixed historical source.")

    # ── 1. Threshold selection (5-fold CV, max F2 → median τ) ─────────
    print("\n[1/3] Threshold selection (ocsvm) ...")
    validation_start = time.time()
    fold_results = []
    oof_scores = np.full(len(y_train), np.nan, dtype=np.float64)
    oof_fold_ids = np.zeros(len(y_train), dtype=np.int64)
    resampling_records = []

    for i, (train_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        X_fold_train_0 = X_fold_train[y_fold_train == 0]

        fold_model = clone(pipeline)
        fold_model.fit(X_fold_train_0)

        # negate: higher score → more anomalous → more likely fraud
        y_val_scores = -fold_model.decision_function(X_fold_val)
        oof_scores[val_idx] = y_val_scores
        oof_fold_ids[val_idx] = i + 1
        tau, _ = find_threshold_maximizing_f2(y_fold_val, y_val_scores)
        fold_metrics = compute_all_metrics(y_fold_val, y_val_scores, tau)
        fold_metrics["threshold_exact"] = tau
        fold_metrics["fold"] = i + 1
        fold_metrics["train_row_indices_sha256"] = array_sha256(X_fold_train.index.to_numpy(dtype=np.int64))
        fold_metrics["validation_row_indices_sha256"] = array_sha256(X_fold_val.index.to_numpy(dtype=np.int64))

        fold_results.append(fold_metrics)
        print(f"  Fold {i + 1}: τ={tau:.6f}  PR-AUC={fold_metrics['PR-AUC']:.4f}"
              f"  F1={fold_metrics['F1']:.4f}  F2={fold_metrics['F2']:.4f}")

    cv_keys = ["PR-AUC", "ROC-AUC", "F1", "F2", "brier_score", "precision_at_k", "recall_at_k"]
    cv_summary = {}
    for k in cv_keys:
        vals = [fr[k] for fr in fold_results]
        cv_summary[f"{k}_mean"] = round(float(np.mean(vals)), 6)
        cv_summary[f"{k}_std"] = round(float(np.std(vals)), 6)

    thresholds = [fr["threshold_exact"] for fr in fold_results]
    tau_final = float(np.median(thresholds))
    cv_summary["threshold_median"] = round(tau_final, 6)
    cv_summary["threshold_per_fold"] = [round(float(t), 6) for t in thresholds]

    metrics_cv = {
        "per_fold": fold_results,
        "aggregated": cv_summary,
    }

    print(f"  ── CV Aggregated ──")
    print(f"  PR-AUC  : {cv_summary['PR-AUC_mean']:.4f} ± {cv_summary['PR-AUC_std']:.4f}")
    print(f"  ROC-AUC : {cv_summary['ROC-AUC_mean']:.4f} ± {cv_summary['ROC-AUC_std']:.4f}")
    print(f"  F1      : {cv_summary['F1_mean']:.4f} ± {cv_summary['F1_std']:.4f}")
    print(f"  F2      : {cv_summary['F2_mean']:.4f} ± {cv_summary['F2_std']:.4f}")
    print(f"  Median threshold: τ = {tau_final:.6f}")
    validation_recovery_time = time.time() - validation_start

    # ── 2. Final training (class 0 only) ──────────────────────────────
    print("\n[2/3] Training final model (ocsvm) ...")
    t0 = time.time()
    X_train_0 = X_train[y_train == 0]
    if recovery_only:
        final_model = joblib.load(source_run / "model.joblib")
        train_time = source_config["train_time_s"]
        context.update(_recovery_cost_metadata(source_run, source_config, validation_recovery_time))
        print("  Reusing the historical OCSVM final model; no new full-DEV fit.")
    else:
        final_model = clone(pipeline)
        final_model.fit(X_train_0)
        train_time = time.time() - t0
    print(f"  Training time : {train_time:.1f}s")
    print(f"  Trained on    : {len(X_train_0):,} legitimate samples")

    # ── 3. Evaluation on holdout test set ─────────────────────────────
    print("\n[3/3] Evaluating on holdout test (ocsvm) ...")
    t0 = time.time()
    if recovery_only:
        y_test_scores = np.load(source_run / "y_test_scores.npy", allow_pickle=False)
        infer_time = source_config["infer_time_s"]
        context["recovery_evidence"].update({
            "test_scores_recomputed": False,
            "infer_time_source": "historical saved full-TEST scoring measurement",
        })
        print("  Reusing validated historical OCSVM TEST scores; no new TEST scoring.")
    else:
        y_test_scores = -final_model.decision_function(X_test)
        infer_time = time.time() - t0

    metrics_test = compute_all_metrics(y_test, y_test_scores, tau_final)

    bootstrap_iterations = context.get("bootstrap_iterations", BOOTSTRAP_ITERATIONS)
    print(f"  Bootstrap CI: {bootstrap_iterations} iterations" if bootstrap_iterations
          else "  Bootstrap CI skipped for this run.")
    ci = (bootstrap_ci(y_test, y_test_scores, tau_final, n_bootstrap=bootstrap_iterations)
          if bootstrap_iterations else None)

    _print_results(metrics_test, ci)

    config = {
        "dataset": dataset_name,
        "dataset_file": dataset_file,
        "dataset_hash_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "model": "ocsvm",
        "strategy": "n/a",
        "cv_folds": CV_SPLITS,
        "threshold_rule": "maximise F2 on validation, take median across folds",
        "threshold_exact": tau_final,
        "best_params": ocsvm_params,
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 4),
        "note": "Trained only on class-0 (legitimate) samples. Scores = negated decision_function.",
    }

    _save_completed_run(
        context=context,
        validation_evidence={
            "y_val": np.asarray(y_train), "y_val_scores": oof_scores,
            "row_indices": X_train.index.to_numpy(dtype=np.int64),
            "fold_ids": oof_fold_ids, "role": "fivefold_out_of_fold",
        },
        model=final_model,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        y_test=np.asarray(y_test),
        y_test_scores=y_test_scores,
        config=config,
        model_name="ocsvm",
        strategy="none",
        dataset=dataset_name,
        bootstrap_ci=ci,
    )

    return final_model, metrics_test


# ── helpers ───────────────────────────────────────────────────────────────
def _print_results(m, ci=None):
    print("\n  ── Test Results ──")
    print(f"  PR-AUC       : {m['PR-AUC']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['PR-AUC_ci'][0]:.4f} – {ci['PR-AUC_ci'][1]:.4f}]")
    else:
        print()
    print(f"  ROC-AUC      : {m['ROC-AUC']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['ROC-AUC_ci'][0]:.4f} \u2013 {ci['ROC-AUC_ci'][1]:.4f}]")
    else:
        print()
    print(f"  F1           : {m['F1']:.4f}")
    print(f"  F2           : {m['F2']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['F2_ci'][0]:.4f} \u2013 {ci['F2_ci'][1]:.4f}]")
    else:
        print()
    print(f"  Brier score  : {m['brier_score']:.4f}")
    print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    print(f"  Alert rate   : {m['alert_rate']:.4%}")
    print(f"  FP/TP        : {m['FP/TP']:.2f}")
    print(f"  Prec@k       : {m['precision_at_k']:.4f}  (k={m['k_used']})")
    print(f"  Recall@k     : {m['recall_at_k']:.4f}  (k={m['k_used']})")
    print(f"  Threshold    : {m['threshold']:.6f}")


def _validate_recovery_source(context, config, X_dev, X_test, y_dev, y_test, dataset):
    """Check the historical evaluation population before validation-only fitting."""
    if dataset != "baf_base" or context.get("missing_policy", "preserve") != "preserve":
        raise ValueError("Validation-only recovery is restricted to BAF with preserved absence codes.")
    if config.get("missing_policy", "preserve") != "preserve":
        raise ValueError("The historical source must use the preserved absence-code policy.")
    checks = {
        "split_seed": SPLIT_SEED, "train_samples": len(X_dev), "test_samples": len(X_test),
        "train_fraud": int(y_dev.sum()), "test_fraud": int(y_test.sum()),
        "sample_fraction": context.get("sample_fraction"),
    }
    if any(config.get(key) != value for key, value in checks.items()):
        raise ValueError("The historical source has different split/sample/class-count provenance.")
    run = Path(context["selected_baseline_run"])
    required = ("model.joblib", "y_test.npy", "y_test_scores.npy")
    if not all((run / name).is_file() for name in required):
        raise ValueError("Historical model and TEST arrays must be present for recovery.")
    labels = np.load(run / "y_test.npy", allow_pickle=False)
    scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
    if not np.array_equal(labels, np.asarray(y_test)) or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Historical TEST arrays do not match the reconstructed BAF split.")
    for split, frame in (("dev", X_dev), ("test", X_test)):
        saved_indices = run / f"{split}_row_indices.npy"
        if saved_indices.exists() and not np.array_equal(np.load(saved_indices), frame.index.to_numpy()):
            raise ValueError("Historical and reconstructed row indices differ.")


def _recovery_cost_metadata(run, config, validation_time):
    return {"recovery_evidence": {
        "validation_only_recovery": True,
        "validation_recovery_time_s": round(validation_time, 4),
        "train_time_source": "historical saved full-DEV fit measurement; no new final fit",
        "historical_tuning_time_s": config.get("tuning_time_s"),
        "historical_train_time_s": config.get("train_time_s"),
        "historical_infer_time_s": config.get("infer_time_s"),
        "historical_cost_source_run": str(run),
        "source_final_model_sha256": file_sha256(run / "model.joblib"),
        "source_saved_test_scores_sha256": file_sha256(run / "y_test_scores.npy"),
    }}


# ── entry point ───────────────────────────────────────────────────────────
def main():
    args = parse_args()
    # Operational guard only: fail before snapshots, locks, CSV loading or fits.
    args.results_root, args.run_manifest = validate_revision_destinations(args.results_root, args.run_manifest)
    source_provenance = capture_source_provenance(args.results_root)
    with baf_training_resource(args.dataset, args.results_root, "Classical BAF fitting or validation recovery"):
        _run_main(args, source_provenance)


def _run_main(args, source_provenance):
    if args.recover_validation_only and (args.dataset != "baf_base" or args.missing_policy != "preserve"
            or args.fixed_params_run is None or args.strategy != "none"):
        raise ValueError("--recover-validation-only requires baf_base/preserve/none and --fixed-params-run.")
    dataset_arg = args.dataset
    models = args.models
    strategy_arg = args.strategy
    sample_frac = args.sample

    if "all" in models:
        models = ["logreg", "rf", "lgbm", "catboost", "ocsvm"]

    # Resolve strategy list
    if strategy_arg == "all":
        strategies = [s for s in STRATEGY_NAMES if s != "none"]
    else:
        strategies = [strategy_arg]

    # ── Load data ─────────────────────────────────────────────────────
    dataset_file, dataset_name = get_dataset_info(dataset_arg)
    print(f"Loading {dataset_name} dataset ...")
    X_train, X_test, y_train, y_test, split_metadata = load_dataset(
        dataset_arg, sample=sample_frac, return_metadata=True,
    )
    print(f"  Train : {len(X_train):,} samples  ({y_train.sum()} fraud)")
    print(f"  Test  : {len(X_test):,} samples   ({y_test.sum()} fraud)")

    dataset_hash = _file_hash(dataset_file)
    print(f"  SHA-256 : {dataset_hash[:16]}...")

    # ── Preprocessor ──────────────────────────────────────────────────
    preprocessor = get_preprocessor(X_train, missing_policy=args.missing_policy)
    if args.bootstrap_iterations < 0:
        raise ValueError("--bootstrap-iterations must be non-negative.")
    context = {
        "results_root": args.results_root,
        "manifest_path": args.run_manifest or args.results_root / "revision_manifest.json",
        "baseline_run": args.baseline_run, "fixed_params_run": args.fixed_params_run,
        "missing_policy": args.missing_policy, "sample_fraction": sample_frac,
        "split_metadata": split_metadata, "dataset_hash": dataset_hash,
        "bootstrap_iterations": args.bootstrap_iterations,
        "recover_interrupted_trials": args.recover_interrupted_trials,
        "resume_optuna_study": args.resume_optuna_study,
        "source_provenance": source_provenance,
        "recover_validation_only": args.recover_validation_only,
    }

    # ── Run requested models × strategies ─────────────────────────────
    results = {}
    for strategy in strategies:
        for name in models:
            # OCSVM doesn't support imbalance strategies
            if name == "ocsvm":
                if strategy == "none":
                    _, metrics = run_ocsvm(
                        X_train, X_test, y_train, y_test,
                        preprocessor, dataset_hash,
                        dataset_name, dataset_file, context=context
                    )
                    results[("ocsvm", "none")] = metrics
                else:
                    print(f"\n  Skipping OCSVM with strategy={strategy} "
                          f"(anomaly-detection paradigm)")
                continue

            if strategy == "none":
                _, metrics = run_supervised(
                    name, X_train, X_test, y_train, y_test,
                    preprocessor, dataset_hash,
                    dataset_name, dataset_file,
                    n_trials=args.n_trials, context=context,
                )
            else:
                _, metrics = run_supervised_strategy(
                    name, strategy, X_train, X_test, y_train, y_test,
                    preprocessor, dataset_hash,
                    dataset_name, dataset_file, context=context
                )
            results[(name, strategy)] = metrics

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n{'=' * 110}")
    print(f"  SUMMARY — {dataset_name.upper()}")
    print(f"{'=' * 110}")
    header = (
        f"{'Model':<10} {'Strategy':<12} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} {'F2':>8} "
        f"{'Brier':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>7} "
        f"{'Alert%':>8} {'FP/TP':>7} {'P@k':>6} {'R@k':>6}"
    )
    print(header)
    print("-" * 120)
    for (name, strat), m in results.items():
        print(
            f"{name:<10} {strat:<12} "
            f"{m['PR-AUC']:>8.4f} {m['ROC-AUC']:>8.4f} {m['F1']:>8.4f} {m['F2']:>8.4f} "
            f"{m['brier_score']:>8.4f} "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>7} "
            f"{m['alert_rate']:>7.4%} {m['FP/TP']:>7.2f} "
            f"{m['precision_at_k']:>6.3f} {m['recall_at_k']:>6.3f}"
        )
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
