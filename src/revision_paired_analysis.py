"""Predefined paired TEST analyses of BAF preprocessing and sampling controls.

No model is fitted and no policy or hyperparameter is selected here. Both members
of each comparison use the same original TEST rows; their thresholds were fixed
from DEV before this descriptive sensitivity analysis.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_protocol import file_sha256, PROTOCOL_VERSION, PROJECT_ROOT

MODELS = ("logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm")
PAIRED_SOURCE_FILES = ("config.json", "metrics_test.json", "completed.json",
                       "y_test.npy", "y_test_scores.npy", "test_row_indices.npy")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_source(path):
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("A paired source must have an explicit run path.")
    path = Path(path)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _verify_source_hashes(run, recorded, required):
    """Check recorded file bytes only; never infer or reconstruct predictions."""
    if not isinstance(recorded, dict) or set(recorded) != set(required):
        raise ValueError("A paired source omits or changes its prescribed artefact hash inventory.")
    for name, digest in recorded.items():
        if not (run / name).is_file() or file_sha256(run / name) != digest:
            raise ValueError(f"A paired source artefact changed: {run / name}")


def _verify_current_source(manifest, model, strategy, recorded_run, recorded_hashes):
    key = f"baf_base/{model}/{strategy}"
    pin = manifest.get("runs", {}).get(key)
    if not isinstance(pin, dict) or not pin.get("config_sha256"):
        raise ValueError(f"A paired comparison lacks its explicit corrected source pin: {key}")
    run = _resolve_source(pin.get("run_dir"))
    if _resolve_source(recorded_run) != run:
        raise ValueError("A paired comparison identifies a different fitted source.")
    _verify_source_hashes(run, recorded_hashes, PAIRED_SOURCE_FILES)
    if recorded_hashes["config.json"] != pin["config_sha256"]:
        raise ValueError("A paired configuration differs from its explicit manifest pin.")
    config, completed = read_json(run / "config.json"), read_json(run / "completed.json")
    normalised_strategy = "none" if model == "ocsvm" and config.get("strategy") == "n/a" else config.get("strategy")
    if ((config.get("dataset"), config.get("model"), normalised_strategy) != ("baf_base", model, strategy)
            or config.get("protocol_version") != PROTOCOL_VERSION
            or completed.get("status") != "complete" or completed.get("protocol_version") != PROTOCOL_VERSION
            or not np.isfinite(config.get("threshold_exact", np.nan))):
        raise ValueError("A paired source has an incompatible identity or completion marker.")
    return run, config


def _verify_pair_record(record, primary, second_manifest, model, *, first_strategy="none",
                        second_strategy="none", missingness=False):
    if not isinstance(record, dict):
        raise ValueError("The paired report contains an incomplete comparison record.")
    configs = []
    for role, manifest, strategy in (("first", primary, first_strategy),
                                     ("second", second_manifest, second_strategy)):
        _, config = _verify_current_source(manifest, model, strategy, record.get(f"{role}_run"),
                                            record.get(f"{role}_artefacts_sha256"))
        if (record.get(f"{role}_config_sha256") != record[f"{role}_artefacts_sha256"]["config.json"]
                or record.get(f"{role}_threshold") != config.get("threshold_exact")):
            raise ValueError("The paired report changed its recorded configuration or fixed DEV threshold.")
        configs.append(config)
    if any(configs[0].get(name) != configs[1].get(name) for name in ("dataset_hash_sha256", "best_params")):
        raise ValueError("A predefined paired comparison changed its raw data or fixed hyperparameters.")
    if missingness and (configs[0].get("missing_policy") != "preserve"
                        or configs[1].get("missing_policy") != "nan_indicators"):
        raise ValueError("The paired absence comparison does not use the two predefined policies.")
    if record.get("bootstrap_requested") != 1000 or not record.get("paired_95_percentile_intervals"):
        raise ValueError("The paired report lacks its prescribed 1,000 resamples and intervals.")


def verify_paired_report(report, primary_manifest, sensitivity_manifest):
    """Return True after verifying every frozen paired source and control.

    This read-only consumer checks identities, explicit pins and file hashes.
    It performs no fitting, inference, metric calculation or bootstrap. The
    historical SHA of a changing training manifest is deliberately not used as
    an equality requirement; each source run remains individually pinned.
    """
    if (not isinstance(report, dict) or report.get("status") != "complete"
            or set(report.get("missingness", {})) != set(MODELS)
            or set(report.get("categorical_controls", {})) != {"catboost", "fttransformer"}):
        raise ValueError("The paired report omits a complete predefined comparison.")
    for model in MODELS:
        _verify_pair_record(report["missingness"][model], primary_manifest, sensitivity_manifest,
                            model, missingness=True)
    _verify_pair_record(report["categorical_controls"]["fttransformer"], primary_manifest,
                        primary_manifest, "fttransformer", first_strategy="smote",
                        second_strategy="smotenc_control")
    control_source = report.get("catboost_control_source", {})
    control_hashes = control_source.get("artefacts_sha256")
    if not isinstance(control_hashes, dict):
        raise ValueError("The paired report lacks the frozen CatBoost control artefacts.")
    control, control_config = _verify_current_source(
        primary_manifest, "catboost", "smotenc_control", control_source.get("run_dir"),
        {name: digest for name, digest in control_hashes.items() if name in PAIRED_SOURCE_FILES})
    _verify_source_hashes(control, control_hashes, (*PAIRED_SOURCE_FILES, "primary_smote_comparison.json"))
    comparison = report["categorical_controls"]["catboost"]
    if comparison != read_json(control / "primary_smote_comparison.json"):
        raise ValueError("The paired report differs from its frozen CatBoost comparison.")
    historical_pin = primary_manifest.get("historical_runs", {}).get("baf_base/catboost/smote")
    if not isinstance(historical_pin, dict) or not historical_pin.get("config_sha256"):
        raise ValueError("The CatBoost comparison lacks its explicit historical primary SMOTE pin.")
    source = _resolve_source(historical_pin.get("run_dir"))
    if _resolve_source(comparison.get("source_run")) != source:
        raise ValueError("The CatBoost comparison identifies another historical primary SMOTE source.")
    # The historical archive does not claim saved completion/index artefacts.
    # Verify its four actually recorded files; do not invent missing hashes.
    _verify_source_hashes(source, comparison.get("source_artefacts_sha256"),
                          ("config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy"))
    source_config = read_json(source / "config.json")
    if (comparison.get("source_config_sha256") != historical_pin["config_sha256"]
            or file_sha256(source / "config.json") != historical_pin["config_sha256"]
            or (source_config.get("dataset"), source_config.get("model"), source_config.get("strategy"))
            != ("baf_base", "catboost", "smote")
            or source_config.get("protocol_version") is not None
            or any(source_config.get(name) != control_config.get(name)
                   for name in ("dataset_hash_sha256", "best_params"))
            or comparison.get("control_threshold_full_precision") != control_config.get("threshold_exact")):
        raise ValueError("The CatBoost comparison changed its historical source, parameters or control threshold.")
    if comparison.get("bootstrap_iterations") != 1000 or not comparison.get("paired_difference_bootstrap"):
        raise ValueError("The CatBoost comparison lacks its prescribed paired bootstrap evidence.")
    return True


def load_pin(manifest, model, strategy="none"):
    reference = manifest["runs"][f"baf_base/{model}/{strategy}"]
    run = Path(reference["run_dir"])
    if file_sha256(run / "config.json") != reference["config_sha256"]:
        raise ValueError("A pinned comparison configuration has changed.")
    config = read_json(run / "config.json")
    if config.get("protocol_version") != PROTOCOL_VERSION or read_json(run / "completed.json").get("status") != "complete":
        raise ValueError("A completed corrected-protocol run is required.")
    arrays = {name: np.load(run / (name + ".npy"), allow_pickle=False)
              for name in ("y_test", "y_test_scores", "test_row_indices")}
    labels = arrays["y_test"]
    indices = arrays["test_row_indices"]
    if (labels.ndim != 1 or not np.array_equal(np.unique(labels), [0, 1])
            or arrays["y_test_scores"].shape != labels.shape
            or not np.isfinite(arrays["y_test_scores"]).all()
            or indices.shape != labels.shape or not np.issubdtype(indices.dtype, np.integer)
            or np.unique(indices).size != labels.size):
        raise ValueError("Invalid TEST labels or scores.")
    if (config.get("dataset") != "baf_base" or config.get("model") != model
            or config.get("strategy") not in ({"none", "n/a"} if model == "ocsvm" else {strategy})):
        raise ValueError("A pinned comparison has an unexpected dataset, model or strategy.")
    if not np.isfinite(config.get("threshold_exact", np.nan)):
        raise ValueError("The baseline max-F2 threshold must be finite and explicit.")
    return run, config, arrays


def paired_difference(y, first_scores, first_threshold, second_scores, second_threshold,
                      iterations=1000, seed=42):
    """Return second-minus-first differences and row-paired percentile intervals."""
    y = np.asarray(y)
    if not (y.shape == np.asarray(first_scores).shape == np.asarray(second_scores).shape):
        raise ValueError("Paired scores must describe exactly the same rows.")
    if y.ndim != 1 or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("A paired comparison requires binary labels and both classes.")
    if (not np.isfinite(first_scores).all() or not np.isfinite(second_scores).all()
            or not np.isfinite([first_threshold, second_threshold]).all()):
        raise ValueError("Scores and fixed max-F2 thresholds must be finite.")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
        raise ValueError("Bootstrap iterations must be a non-negative integer.")
    first_predictions = np.asarray(first_scores) >= first_threshold
    second_predictions = np.asarray(second_scores) >= second_threshold

    def metrics(labels, scores, predictions):
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum(~predictions & (labels == 1)))
        denominator = 5 * tp + 4 * fn + fp
        return np.asarray([average_precision_score(labels, scores),
                           roc_auc_score(labels, scores),
                           5 * tp / denominator if denominator else 0.0,
                           float(np.mean(predictions))])

    names = ("average_precision", "roc_auc", "F2", "alert_rate")
    first = metrics(y, first_scores, first_predictions)
    second = metrics(y, second_scores, second_predictions)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y), size=len(y))
        sampled_y = y[indices]
        if np.unique(sampled_y).size != 2:
            continue
        differences.append(metrics(sampled_y, np.asarray(second_scores)[indices], second_predictions[indices])
                           - metrics(sampled_y, np.asarray(first_scores)[indices], first_predictions[indices]))
    intervals = np.quantile(differences, [0.025, 0.975], axis=0) if differences else None
    return {"first_metrics": dict(zip(names, first.tolist())),
            "second_metrics": dict(zip(names, second.tolist())),
            "difference_second_minus_first": dict(zip(names, (second - first).tolist())),
            "paired_95_percentile_intervals": ({name: intervals[:, i].tolist() for i, name in enumerate(names)}
                                                if intervals is not None else None),
            "bootstrap_requested": iterations, "bootstrap_valid": len(differences),
            "bootstrap_seed": seed, "bootstrap_unit": "paired original TEST row",
            "uncertainty_scope": "Conditional on the two fitted models, fixed DEV thresholds and shared TEST split; not training-seed variability."}


def analyse_pair(primary, sensitivity, model, *, primary_strategy="none", sensitivity_strategy="none",
                 iterations=1000, missingness=False):
    first_run, first_config, first_arrays = load_pin(primary, model, primary_strategy)
    second_run, second_config, second_arrays = load_pin(sensitivity, model, sensitivity_strategy)
    for name in ("y_test", "test_row_indices"):
        if not np.array_equal(first_arrays[name], second_arrays[name]):
            raise ValueError("A paired comparison cannot use different TEST rows or label order.")
    for name in ("dataset_hash_sha256", "best_params"):
        if first_config[name] != second_config[name]:
            raise ValueError("The predefined comparison must keep the raw data and hyperparameters fixed.")
    if missingness and (first_config.get("missing_policy", "preserve") != "preserve"
                        or second_config.get("missing_policy") != "nan_indicators"):
        raise ValueError("Expected preserved-code primary versus predefined NaN-and-indicators sensitivity.")
    source_hashes = {role: {filename: file_sha256(run / filename) for filename in PAIRED_SOURCE_FILES}
                     for role, run in (("first", first_run), ("second", second_run))}
    evidence = paired_difference(first_arrays["y_test"], first_arrays["y_test_scores"], first_config["threshold_exact"],
                                 second_arrays["y_test_scores"], second_config["threshold_exact"], iterations)
    for role, run in (("first", first_run), ("second", second_run)):
        if any(file_sha256(run / name) != digest for name, digest in source_hashes[role].items()):
            raise ValueError("A paired source artefact changed during analysis.")
    evidence.update(first_run=str(first_run), second_run=str(second_run),
                    first_config_sha256=file_sha256(first_run / "config.json"),
                    second_config_sha256=file_sha256(second_run / "config.json"),
                    first_artefacts_sha256=source_hashes["first"], second_artefacts_sha256=source_hashes["second"],
                    rows=len(first_arrays["y_test"]), fraud_rows=int(first_arrays["y_test"].sum()),
                    first_threshold=first_config["threshold_exact"], second_threshold=second_config["threshold_exact"],
                    policy_selected_using_TEST=False)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sensitivity-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_iterations < 0:
        parser.error("Bootstrap iterations must be non-negative.")
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; do not overwrite a paired analysis.")
    primary, sensitivity = read_json(args.manifest), read_json(args.sensitivity_manifest)
    # Verify every pin before creating any apparently complete output.
    for model in MODELS:
        load_pin(primary, model)
        load_pin(sensitivity, model)
        analyse_pair(primary, sensitivity, model, iterations=0, missingness=True)
    analyse_pair(primary, primary, "fttransformer", primary_strategy="smote",
                 sensitivity_strategy="smotenc_control", iterations=0)
    control_run, _, _ = load_pin(primary, "catboost", "smotenc_control")
    control_comparison = read_json(control_run / "primary_smote_comparison.json")
    control_hashes = {name: file_sha256(control_run / name)
                      for name in (*PAIRED_SOURCE_FILES, "primary_smote_comparison.json")}
    report = {"status": "in_progress", "missingness": {}, "categorical_controls": {},
              "source_manifest_sha256": file_sha256(args.manifest),
              "sensitivity_manifest_sha256": file_sha256(args.sensitivity_manifest)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for model in MODELS:
        evidence = analyse_pair(primary, sensitivity, model, iterations=args.bootstrap_iterations, missingness=True)
        report["missingness"][model] = evidence
        (args.output_dir / f"missingness_{model}.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(f"Missingness {model}: {evidence['difference_second_minus_first']}", flush=True)
    for model in ("fttransformer",):
        evidence = analyse_pair(primary, primary, model, primary_strategy="smote", sensitivity_strategy="smotenc_control",
                                iterations=args.bootstrap_iterations)
        report["categorical_controls"][model] = evidence
        (args.output_dir / f"categorical_{model}.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    # CatBoost's independently verified control includes its own paired analysis.
    report["categorical_controls"]["catboost"] = control_comparison
    report["catboost_control_source"] = {"run_dir": str(control_run), "artefacts_sha256": control_hashes}
    if any(file_sha256(control_run / name) != digest for name, digest in control_hashes.items()):
        raise ValueError("A CatBoost control source changed during paired analysis.")
    report["status"] = "complete"
    (args.output_dir / "paired_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
