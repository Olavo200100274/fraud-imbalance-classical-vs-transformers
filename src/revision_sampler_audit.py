"""Audit SMOTE/Tomek identity without fitting or changing any predictive model.

New-run fold/DEV observations are kept distinct from a reconstruction of the
historical BAF classical final preprocessing. Historical fold-level sampler
logs did not exist and are not retrospectively invented.
"""

import argparse
from datetime import datetime, timezone
import gc
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.utils.validation import check_X_y

from data import load_dataset
from experiment_protocol import PROTOCOL_VERSION, array_sha256, file_sha256, sampler_diagnostics, software_versions
from revision_resources import exclusive_resource
from strategies.balancing import get_sampler

ROOT = Path(__file__).resolve().parents[1]
SUPERVISED = ("logreg", "rf", "lgbm", "catboost", "fttransformer")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pinned_run(manifest, key, *, historical=False):
    reference = manifest["historical_runs" if historical else "runs"][key]
    run = Path(reference["run_dir"])
    run = run if run.is_absolute() else ROOT / run
    if file_sha256(run / "config.json") != reference["config_sha256"]:
        raise ValueError("A sampler audit source configuration changed.")
    dataset, model, strategy = key.split("/")
    config = read_json(run / "config.json")
    if (config.get("dataset"), config.get("model"), config.get("strategy")) != (dataset, model, strategy):
        raise ValueError("A sampler audit pin has the wrong identity.")
    if not historical and (config.get("protocol_version") != PROTOCOL_VERSION
                           or read_json(run / "completed.json").get("status") != "complete"):
        raise ValueError("A new sampler observation requires a completed corrected-protocol run.")
    return run, config


def compare_stage_records(smote, tomek):
    first = {record["stage"]: record for record in smote}
    second = {record["stage"]: record for record in tomek}
    if len(first) != len(smote) or len(second) != len(tomek) or set(first) != set(second):
        raise ValueError("Sampler observations must identify the same unique training stages.")
    comparisons = {}
    for stage in first:
        a, b = first[stage], second[stage]
        for name in ("rows_before", "class_counts_before", "X_before_sha256", "y_before_sha256"):
            if a[name] != b[name]:
                raise ValueError("SMOTE and Tomek do not start from the same training representation.")
        removed = b["cleaner_removed_rows"]
        if removed < 0 or b["rows_after_smote_before_cleaning"] - b["rows_after"] != removed:
            raise ValueError("Tomek's observed row counts do not account for its removals.")
        equal = a["X_after_sha256"] == b["X_after_sha256"] and a["y_after_sha256"] == b["y_after_sha256"]
        if removed == 0 and (not equal or a["class_counts_after"] != b["class_counts_after"]):
            raise ValueError("Zero Tomek removals should preserve the same seeded SMOTE output.")
        comparisons[stage] = {"rows_before": a["rows_before"], "smote_rows": a["rows_after"],
                              "tomek_rows": b["rows_after"], "cleaner_removed_rows": removed,
                              "resampled_inputs_identical": equal,
                              "smote_class_counts": a["class_counts_after"],
                              "tomek_class_counts": b["class_counts_after"],
                              "smote_X_after_sha256": a["X_after_sha256"],
                              "tomek_X_after_sha256": b["X_after_sha256"]}
    return comparisons


def saved_pair(smote_run, tomek_run):
    first_labels, second_labels = (np.load(path / "y_test.npy", allow_pickle=False) for path in (smote_run, tomek_run))
    if not np.array_equal(first_labels, second_labels):
        raise ValueError("The saved SMOTE/Tomek TEST label orders differ.")
    first_scores, second_scores = (np.load(path / "y_test_scores.npy", allow_pickle=False) for path in (smote_run, tomek_run))
    if first_scores.shape != first_labels.shape or second_scores.shape != first_labels.shape:
        raise ValueError("Saved score arrays do not match their TEST population.")
    return {"smote_run": str(smote_run), "tomek_run": str(tomek_run),
            "TEST_scores_bitwise_identical": bool(np.array_equal(first_scores, second_scores)),
            "maximum_absolute_TEST_score_difference": float(np.max(np.abs(first_scores - second_scores))),
            "source_sha256": {str(path): file_sha256(path)
                              for run in (smote_run, tomek_run) for path in
                              (run / "config.json", run / "y_test.npy", run / "y_test_scores.npy")}}


def reconstruct_historical_baf(manifest):
    dev, test, labels, unused_test_labels, metadata = load_dataset("baf_base", return_metadata=True)
    del test, unused_test_labels
    transformed = None
    common_hash = None
    sources = {}
    for model in SUPERVISED[:-1]:
        run, config = pinned_run(manifest, f"baf_base/{model}/none", historical=True)
        if config["dataset_hash_sha256"] != metadata["raw_file_sha256"]:
            raise ValueError("A historical preprocessor was fitted on different raw BAF data.")
        pipeline = joblib.load(run / "model.joblib")
        values = np.asarray(pipeline.named_steps["preprocessor"].transform(dev))
        digest = array_sha256(values)
        if common_hash is not None and digest != common_hash:
            raise ValueError("The four historical classical full-DEV representations are not identical.")
        if transformed is None:
            transformed, common_hash = values, digest
        sources[model] = {"source_run": str(run), "config_sha256": file_sha256(run / "config.json"),
                          "model_sha256": file_sha256(run / "model.joblib"), "transformed_DEV_sha256": digest}
        del pipeline, values
        gc.collect()
    del dev
    X, y = check_X_y(transformed, np.asarray(labels), accept_sparse=["csr", "csc"])
    composite = get_sampler("smote_tomek", random_state=42)
    # This is the installed composite's own decomposition, inspected and hashed
    # below. No different Tomek setting is chosen to manufacture a discrepancy.
    composite._validate_estimator()
    X_smote, y_smote = composite.smote_.fit_resample(X, y)
    smote_record = sampler_diagnostics(composite.smote_, X, y, X_smote, y_smote, stage="reconstructed_final_dev")
    X_tomek, y_tomek = composite.tomek_.fit_resample(X_smote, y_smote)
    tomek_record = sampler_diagnostics(composite, X, y, X_tomek, y_tomek, stage="reconstructed_final_dev")
    comparisons = compare_stage_records([smote_record], [tomek_record])
    return {"role": "Reconstruction of historical full-DEV preprocessing and sampler only, not original execution logs or model retraining.",
            "historical_fold_sampler_records_available": False,
            "raw_file_sha256": metadata["raw_file_sha256"], "preprocessor_sources": sources,
            "installed_composite_fit_resample_source": inspect.getsource(type(composite)._fit_resample),
            "software_versions": software_versions(), "observations": comparisons,
            "SMOTE_record": smote_record, "SMOTE_Tomek_record": tomek_record}


def generate(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if not output.is_relative_to(manifest_path.parent) or output.exists():
        raise ValueError("Write a new sampler report inside the isolated revision root.")
    manifest = read_json(manifest_path)
    report = {"status": "auditing", "protocol_version": PROTOCOL_VERSION,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_manifest_sha256": file_sha256(manifest_path),
              "model_fitting_performed": False, "new_training_observations": {}, "historical_saved_scores": {}}
    # Require every new observation before starting the large reconstruction.
    for dataset, models in (("ulb_2013", SUPERVISED), ("baf_base", ("fttransformer",))):
        for model in models:
            a, _ = pinned_run(manifest, f"{dataset}/{model}/smote")
            b, _ = pinned_run(manifest, f"{dataset}/{model}/smote_tomek")
            observed = compare_stage_records(read_json(a / "sampler_diagnostics.json"),
                                             read_json(b / "sampler_diagnostics.json"))
            expected_stages = {"inner_training", "full_dev"} if model == "fttransformer" else {"final_dev", *(f"fold_{n}" for n in range(1, 6))}
            if set(observed) != expected_stages:
                raise ValueError("A new run omits expected fold/DEV sampler diagnostics.")
            evidence = saved_pair(a, b)
            evidence["training_observations"] = observed
            evidence["sampler_diagnostics_sha256"] = {str(path): file_sha256(path)
                                                       for path in (a / "sampler_diagnostics.json", b / "sampler_diagnostics.json")}
            report["new_training_observations"][f"{dataset}/{model}"] = evidence
    for model in SUPERVISED[:-1]:
        a, _ = pinned_run(manifest, f"baf_base/{model}/smote", historical=True)
        b, _ = pinned_run(manifest, f"baf_base/{model}/smote_tomek", historical=True)
        report["historical_saved_scores"][model] = saved_pair(a, b)
    with exclusive_resource(manifest_path.parent, "baf_training_ram", "Historical BAF final-sampler reconstruction; no model fitting"):
        report["historical_BAF_reconstruction"] = reconstruct_historical_baf(manifest)
    report["status"] = "complete"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Completed sampler identity audit: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.manifest, args.output)


if __name__ == "__main__":
    main()
