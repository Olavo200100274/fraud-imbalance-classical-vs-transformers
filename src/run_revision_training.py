"""Resumable, logged training lanes for the predefined thesis correction.

Only completed manifest entries are skipped. Failed or partial experiments are
not promoted into the evidence base. Historical results are read, never replaced.
Run one classical lane and one transformer lane at most; do not overlap the BAF
sensitivity lane with another memory-intensive CPU job on the recorded hardware.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
from experiment_protocol import PROTOCOL_VERSION, file_sha256, array_sha256
from revision_resources import exclusive_resource, queue_resource_name, record_resource_child

ROOT = Path(__file__).resolve().parent.parent
STRATEGIES = ("none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights")
CLASSICAL = ("lgbm", "catboost", "logreg", "rf", "ocsvm")


def read_manifest(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resolve_run(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def is_complete(manifest, dataset, model, strategy, *, expected_missing_policy="preserve",
                expected_n_trials=None, require_validation_recovery=False,
                expected_fixed_params_run=None, expected_bootstrap_iterations=None):
    """Verify the pinned scientific artefacts, not just a completion filename."""
    reference = manifest.get("runs", {}).get(f"{dataset}/{model}/{strategy}")
    if not reference:
        return False
    run = _resolve_run(reference["run_dir"])
    required = ["completed.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy",
                "y_val.npy", "y_val_scores.npy", "config.json", "metrics_cv.json",
                "dev_row_indices.npy", "test_row_indices.npy", "validation_row_indices.npy",
                "validation_fold_ids.npy"]
    if model == "fttransformer":
        required += ["model.pt", "preprocessors.joblib", "validation_model.pt", "validation_preprocessors.joblib"]
    else:
        required += ["model.joblib"]
    if not all((run / filename).is_file() and (run / filename).stat().st_size for filename in required):
        return False
    config = read_manifest(run / "config.json")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("The completed run belongs to another protocol.")
    config_strategy = "none" if config.get("strategy") == "n/a" else config.get("strategy")
    if config.get("dataset") != dataset or config.get("model") != model or config_strategy != strategy:
        raise ValueError("The manifest key does not identify this run.")
    if reference.get("config_sha256") != file_sha256(run / "config.json"):
        raise ValueError("The pinned run configuration has changed.")
    if config.get("sample_fraction") is not None:
        raise ValueError("A smoke-test run cannot fulfil a full-data task.")
    completion = read_manifest(run / "completed.json")
    if completion.get("status") != "complete" or completion.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("The completion marker is not valid for the current protocol.")
    if config.get("missing_policy") != expected_missing_policy:
        raise ValueError("The completed run uses a different missing-value policy.")
    if config.get("split_seed") != 42 or config.get("split_ratio") != "80/20 stratified":
        raise ValueError("The completed run uses a different outer split.")
    if not np.isfinite(config.get("threshold_exact", np.nan)):
        raise ValueError("The baseline/intervention decision threshold must be saved at full precision.")
    if model == "fttransformer":
        if config.get("max_epochs", config.get("scheduler_horizon")) != 200 or config.get("scheduler_horizon") != 200:
            raise ValueError("A shortened FT schedule cannot fulfil a primary or sensitivity task.")
        if config.get("early_stopping_patience", 15) != 15:
            raise ValueError("The completed FT run uses a different early-stopping rule.")
        if not 1 <= config.get("best_epoch", 0) <= 200:
            raise ValueError("The saved FT epoch selection is outside the fixed training horizon.")
    if expected_bootstrap_iterations is not None:
        if "bootstrap_iterations" not in config:
            from revision_bootstrap_compatibility import verify_bootstrap_compatibility
            verify_bootstrap_compatibility(manifest, run, config, expected_bootstrap_iterations)
        elif config["bootstrap_iterations"] != expected_bootstrap_iterations:
            raise ValueError("The completed run uses a different bootstrap budget.")
    if expected_n_trials is not None and config.get("n_trials", 0 if model == "ocsvm" else None) != expected_n_trials:
        raise ValueError("The completed run does not satisfy the prescribed HPO budget.")
    if expected_n_trials and model != "ocsvm":
        trials_file = run / "optuna_trials.json"
        if not trials_file.is_file():
            return False
        trials = read_manifest(trials_file)
        if len(trials) != expected_n_trials or any(trial["state"] not in ("COMPLETE", "PRUNED", "FAIL") for trial in trials):
            raise ValueError("The completed run lacks the prescribed terminal Optuna trial records.")
    if require_validation_recovery and config.get("validation_only_recovery") is not True:
        raise ValueError("The BAF validation-recovery task did not reuse the historical final model.")
    if expected_fixed_params_run is not None:
        source = _resolve_run(expected_fixed_params_run)
        if not config.get("baseline_run") or _resolve_run(config["baseline_run"]) != source:
            raise ValueError("The completed run borrowed a different historical parameter source.")
        if config.get("baseline_config_sha256") != file_sha256(source / "config.json"):
            raise ValueError("The fixed-parameter source no longer matches its recorded SHA-256.")
        if config.get("best_params") != read_manifest(source / "config.json").get("best_params"):
            raise ValueError("The fixed-parameter experiment changed the source hyperparameters.")
    provenance = config.get("data_provenance")
    if not isinstance(provenance, dict) or provenance.get("raw_file_sha256") != config.get("dataset_hash_sha256"):
        raise ValueError("The completed run has inconsistent raw-file provenance.")
    expected_data = manifest.get("datasets", {}).get(dataset)
    if expected_data:
        for field in ("raw_file_sha256", "dev_indices_sha256", "test_indices_sha256", "feature_columns"):
            if provenance.get(field) != expected_data.get(field):
                raise ValueError("The run does not match the dataset/split pinned in this manifest.")
    arrays = {name: np.load(run / (name + ".npy"), allow_pickle=False, mmap_mode="r")
              for name in ("y_test", "y_test_scores", "y_val", "y_val_scores", "dev_row_indices",
                           "test_row_indices", "validation_row_indices", "validation_fold_ids")}
    if any(array.ndim != 1 for array in arrays.values()):
        raise ValueError("The completed run contains a malformed evidence array.")
    if len(arrays["dev_row_indices"]) != config["train_samples"] or len(arrays["test_row_indices"]) != config["test_samples"]:
        raise ValueError("The saved outer index arrays have incorrect population sizes.")
    if len(arrays["y_test"]) != config["test_samples"] or len(arrays["y_test_scores"]) != config["test_samples"]:
        raise ValueError("The TEST score/label arrays do not align with the recorded population.")
    for split in ("dev", "test"):
        if array_sha256(arrays[split + "_row_indices"]) != provenance.get(split + "_indices_sha256"):
            raise ValueError("A saved outer split index array differs from its recorded hash.")
    validation = config.get("validation_evidence", {})
    for name, field in (("validation_row_indices", "row_indices_sha256"),
                        ("y_val_scores", "scores_sha256"), ("y_val", "labels_sha256")):
        if array_sha256(arrays[name]) != validation.get(field):
            raise ValueError("A validation evidence array differs from its recorded hash.")
        if len(arrays[name]) != validation.get("rows"):
            raise ValueError("The validation evidence population is inconsistent.")
    if len(arrays["validation_fold_ids"]) != validation.get("rows"):
        raise ValueError("Validation fold identifiers do not align with the saved scores.")
    if not all(np.isfinite(arrays[name]).all() for name in ("y_test_scores", "y_val_scores")):
        raise ValueError("The completed run contains non-finite prediction scores.")
    if not all(np.isin(arrays[name], [0, 1]).all() for name in ("y_test", "y_val")):
        raise ValueError("The saved labels are not binary.")
    metrics = read_manifest(run / "metrics_test.json")
    if expected_bootstrap_iterations and "bootstrap_ci" not in metrics:
        raise ValueError("The completed baseline lacks its prescribed confidence intervals.")
    positive = arrays["y_test"] == 1
    alert = arrays["y_test_scores"] >= config["threshold_exact"]
    counts = {"TP": int(np.sum(positive & alert)), "FP": int(np.sum(~positive & alert)),
              "FN": int(np.sum(positive & ~alert)), "TN": int(np.sum(~positive & ~alert))}
    if any(metrics.get(key) != value for key, value in counts.items()):
        raise ValueError("The saved exact threshold does not reproduce the TEST confusion matrix.")
    return True


def tasks(lane, manifest_path, results_root):
    if lane == "classical":
        for model in CLASSICAL:
            for strategy in (("none",) if model == "ocsvm" else STRATEGIES):
                yield "ulb", "ulb_2013", model, strategy, [], results_root, manifest_path
    elif lane == "transformer":
        for strategy in STRATEGIES:
            yield "ulb", "ulb_2013", "fttransformer", strategy, [], results_root, manifest_path
        historical = read_manifest(manifest_path)["historical_baseline_runs"]["baf_base"]["fttransformer"]
        for strategy in STRATEGIES:
            extra = ["--fixed-params-run", historical] if strategy == "none" else []
            yield "baf_base", "baf_base", "fttransformer", strategy, extra, results_root, manifest_path
    elif lane == "sensitivity":
        historical = read_manifest(manifest_path)["historical_baseline_runs"]["baf_base"]
        output = results_root / "sensitivity"
        sensitivity_manifest = output / "revision_manifest.json"
        for model in (*CLASSICAL, "fttransformer"):
            extra = ["--missing-policy", "nan_indicators", "--fixed-params-run", historical[model]]
            yield "baf_base", "baf_base", model, "none", extra, output, sensitivity_manifest
    elif lane == "baf-validation":
        historical = read_manifest(manifest_path)["historical_baseline_runs"]["baf_base"]
        for model in CLASSICAL:
            extra = ["--fixed-params-run", historical[model], "--recover-validation-only"]
            yield "baf_base", "baf_base", model, "none", extra, results_root, manifest_path
    else:
        raise ValueError(lane)


def execute_task(command, log_dir, lane, label, model, strategy, output,
                 manifest_path, completion_options):
    """Prevent a second managed runner from dispatching the same experiment."""
    resource = queue_resource_name(output, f"job_{label}_{model}_{strategy}")
    with exclusive_resource(output, resource, f"Managed experiment {label}/{model}/{strategy}", wait=False) as owner:
        # Another runner may have completed the task before we acquired its lock.
        if is_complete(read_manifest(manifest_path), label, model, strategy, **completion_options):
            print(f"SKIP revalidated {label}/{model}/{strategy}", flush=True)
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_dir / f"{lane}_{label}_{model}_{strategy}_{stamp}.log"
        environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            record_resource_child(owner, process.pid)
            print(f"PID {process.pid}; log={log_path}", flush=True)
            result = process.wait()
            record_resource_child(owner)
        if result != 0:
            raise RuntimeError(f"{label}/{model}/{strategy} failed ({result}); inspect {log_path}")
        if not is_complete(read_manifest(manifest_path), label, model, strategy, **completion_options):
            raise RuntimeError("The process exited without complete, full-data evidence.")
        print(f"DONE {label}/{model}/{strategy}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=("classical", "transformer", "sensitivity", "baf-validation"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "results_revision/20261005/revision_manifest.json")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results_revision/20261005")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", choices=(*CLASSICAL, "fttransformer"),
                        help="Restrict a lane to explicitly named model families.")
    parser.add_argument("--resume-lgbm-study", type=Path,
                        help="Explicit checkpoint for the interrupted ULB LGBM baseline only.")
    args = parser.parse_args()
    log_dir = args.results_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for dataset_key, label, model, strategy, extra, output, manifest_path in tasks(
            args.lane, args.manifest, args.results_root):
        if args.models and model not in args.models:
            continue
        expected_policy = "nan_indicators" if args.lane == "sensitivity" else "preserve"
        expected_trials = (50 if label == "ulb_2013" and strategy == "none" and model != "ocsvm"
                           else 0 if strategy == "none" and label == "baf_base" else None)
        fixed_source = extra[extra.index("--fixed-params-run") + 1] if "--fixed-params-run" in extra else None
        if strategy != "none":
            baseline_reference = read_manifest(manifest_path).get("baseline_runs", {}).get(f"{label}/{model}")
            if baseline_reference:
                fixed_source = (baseline_reference["run_dir"] if isinstance(baseline_reference, dict)
                                else baseline_reference)
        completion_options = {"expected_missing_policy": expected_policy,
                              "expected_n_trials": expected_trials,
                              "require_validation_recovery": args.lane == "baf-validation",
                              "expected_fixed_params_run": fixed_source,
                              "expected_bootstrap_iterations": 1000 if strategy == "none" else 0}
        if is_complete(read_manifest(manifest_path), label, model, strategy, **completion_options):
            print(f"SKIP complete {label}/{model}/{strategy}", flush=True)
            continue
        entry_script = "main_transformer.py" if model == "fttransformer" else "main.py"
        command = [sys.executable, "-B", "-u", str(ROOT / "src/revision_runtime.py"),
                   str(ROOT / "src" / entry_script), "--dataset", dataset_key,
                   "--strategy", strategy, "--results-root", str(output),
                   "--run-manifest", str(manifest_path), "--n_trials", "50",
                   "--bootstrap-iterations", "1000" if strategy == "none" else "0"]
        if model != "fttransformer":
            command += ["--models", model]
        command += extra
        if (args.resume_lgbm_study and args.lane == "classical" and model == "lgbm"
                and strategy == "none"):
            command += ["--resume-optuna-study", str(args.resume_lgbm_study)]
        print("RUN " + subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            continue
        execute_task(command, log_dir, args.lane, label, model, strategy, output, manifest_path, completion_options)


if __name__ == "__main__":
    main()
